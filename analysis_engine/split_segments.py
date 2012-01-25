import numpy as np
import logging
from datetime import datetime, timedelta

from analysis_engine import settings
from analysis_engine.library import (calculate_timebase, hash_array,
                                     hysteresis, max_abs_value)
from hdfaccess.file import hdf_file
from utilities.filesystem_tools import sha_hash_file

from datastructures import Segment


def mask_slow_airspeed(airspeed):
    """
    :param airspeed: 
    :type airspeed: np.ma.array
    """
    # mask where airspeed drops below min airspeed, using hysteresis
    hysteresisless_spd = hysteresis(airspeed, settings.HYSTERESIS_FPIAS)
    return np.ma.masked_less(hysteresisless_spd, settings.AIRSPEED_THRESHOLD)



def split_segments2(airspeed, dfc):
    speedy_slices = np.ma.notmasked_contiguous(mask_slow_airspeed(airspeed.array))
    # be clever about splitting between speedy slices
    if len(speedy_slices) <= 1:
        return [slice(0, len(airspeed))]
    # more than one speedy section
    dfc_diff = np.ma.diff(dfc.array)
    dfc_mask_one = np.ma.masked_equal(dfc_diff, 1)
    dfc_mask_4094 = np.ma.masked_equal(dfc_mask_one, -4094)
    rate = 1.0 / dfc.frequency
    segment_slices = []
    origin = 0
    start = speedy_slices[0].stop
    for speedy_slice in speedy_slices[1:]:
        stop = speedy_slice.start
        # find DFC split within speedy sections

        # take the biggest jump (not that it means anything, but only one jump is enough!
        index, value = max_abs_value(dfc_mask_4094, slice(start*dfc.frequency,
                                                          stop*dfc.frequency))
        cut_index = index * (1.0/dfc.frequency) # align index again
        if np.ma.is_masked(value) or cut_index == start:

            #TODO: implement and test below:
            '''
            from analysis_engine.library import vstack_params, min_value
            #TODO: Improve by ensuring we were sat still for the longest period
            #TODO: ensure all at same freq or align!
            engine_params = vstack_params(
                hdf.get('Eng (1) N1'), hdf.get('Eng (1) Oil Temp'),
                hdf.get('Eng (2) N1'), hdf.get('Eng (2) Oil Temp'),
                hdf.get('Eng (3) N1'), hdf.get('Eng (3) Oil Temp'),
                hdf.get('Eng (4) N1'), hdf.get('Eng (4) Oil Temp'), 
                #TODO: Add "Turning" to ensure we're staying still
            )
            cut_index, value = min_value(engine_params, slice(start, stop)) 
            
            #Q: Use a threshold for value? if not met, then cut halfway
            # between the two. To improve effectiveness of threshold, you
            # could mask the Engine values when they are above a "turning off"
            # state so that you're sure the minimum is when they were nearly
            # off, otherwise you'll have a masked value and you can use that
            # to cut upon.
            '''
            # no jump, take half way between values
            cut_index = (start + stop) / 2.0
            
        segment_slices.append(slice(origin, cut_index))
        # keep track of slices
        origin = cut_index
        start = speedy_slice.stop
    else:
        # end slice
        segment_slices.append(slice(origin, None))    
    return segment_slices
        
def split_segments(airspeed, dfc=None):
    """
    Splits data looking for dfc jumps (if dfc provided) and changes in airspeed.
    
    :param airspeed: Airspeed data in Knots
    :type airspeed: Parameter
    :param dfc: Data frame counter signal
    :type dfc: Parameter
    :returns: Segments of flight-like data
    :rtype: list of slices
    
    TODO: Currently requires 1Hz Airspeed Data - make multi-hertz friendly
    Q: should we not perfrom a split using DFC if Airspeed is still high to avoid cutting mid-flight?
    """
    # Split by frame count is optional
    if dfc is not None:
        # split hdf where frame counter is reset
        data_slices = _split_by_frame_counter(dfc.array, dfc.frequency)
    else:
        data_slices = [slice(0, len(airspeed.array))]

    hyst_mask_below_min_airspeed = mask_slow_airspeed(airspeed.array)
    segment_slices = []
    for data_slice in data_slices: ## or [slice(0,hdf.size)]: # whole data is a single segment
        # split based on airspeed for fixed wing / rotorspeed for heli
        data_seg_slices = _split_by_flight_data(hyst_mask_below_min_airspeed[data_slice], data_slice.start)
        segment_slices.extend(data_seg_slices)
    return segment_slices
        
def append_segment_info(hdf_segment_path, segment_slice, part):
    """
    Get information about a segment such as type, hash, etc. and return a
    named tuple.
    
    If a valid timestamp can't be found, it creates start_dt as epoch(0)
    i.e. datetime(1970,1,1,1,0). Go-fast dt and Stop dt are relative to this
    point in time.
    
    :param hdf_segment_path: path to HDF segment to analyse
    :type hdf_segment_path: string
    :param segment_slice: Slice of this segment relative to original file.
    :type segment_slice: slice
    :param part: Numeric part this segment was in the original data file (1 indexed)
    :type part: Integer
    :returns: Segment named tuple
    :rtype: Segment
    """
    # build information about a slice
    with hdf_file(hdf_segment_path) as hdf:
        airspeed = hdf['Airspeed'].array
        duration = hdf.duration
        # establish timebase for start of data
        try:
            #TODO: use hdf.get('Year', [])[segment.slice] to provide empty slices.
            start_datetime = calculate_timebase(
                hdf['Year'].array, hdf['Month'].array, hdf['Day'].array,
                hdf['Hour'].array, hdf['Minute'].array, hdf['Second'].array)
        except (KeyError, ValueError):
            logging.exception("Unable to calculate timebase, using epoch 1.1.1970!")
            start_datetime = datetime.fromtimestamp(0)
        stop_datetime = start_datetime + timedelta(seconds=duration)
            
    #end with

    # identify subsection of airspeed, using original array with mask
    # removing all invalid data (ARINC etc)
    segment_type = _identify_segment_type(airspeed)
    logging.info("Segment type: %s", segment_type)
    
    if segment_type != 'GROUND_ONLY':
        # we went fast, so get the index
        spd_above_threshold = np.ma.where(airspeed > settings.AIRSPEED_THRESHOLD)
        go_fast_index = spd_above_threshold[0][0]
        go_fast_datetime = start_datetime + timedelta(seconds=int(go_fast_index))
        # Identification of raw data airspeed hash (including all spikes etc)
        airspeed_hash = hash_array(airspeed.data[airspeed.data > settings.AIRSPEED_THRESHOLD])
    else:
        go_fast_index = None
        go_fast_datetime = None
        # if not go_fast, create hash from entire file
        airspeed_hash = sha_hash_file(hdf_segment_path)
        
    #                ('slice         type          part  path              hash           start_dt        go_fast_dt        stop_dt')
    segment = Segment(segment_slice, segment_type, part, hdf_segment_path, airspeed_hash, start_datetime, go_fast_datetime, stop_datetime)
    return segment




        
def _split_by_frame_counter(dfc_data, dfc_freq=0.25):
    """
    Creates 1Hz slices by creating slices at 4 times the stop and start
    indexes of the 0.25Hz DFC.
    
    :param dfc_data: Frame Counter
    :type dfc_data: np.ma.array
    :param dfc_freq: Frequency of the frame counter, default is 1/4Hz
    :type dfc_freq: Float
    :returns: 1Hz Slices where a jump occurs
    :rtype: list of slices
    """
    rate = 1.0 / dfc_freq
    #TODO: Convert to Numpy array manipulation!
    dfc_slices = []
    # Now read each line in turn
    previous_dfc = dfc_data[0]
    start_index = 0
    for index, value in enumerate(dfc_data[1:]):
        index += 1 # for missing first item
        step = value - previous_dfc
        previous_dfc = value
        if step == 1 or step == -4094: #TODO: This won't work for Hercules jumps which are much larger (24bits)
            # expected increment
            continue
        elif step == 0:
            # flat line - shouldn't happen but we'll ignore, so just log
            logging.warning("DFC failed to increment, ignoring")
            continue
        else:
            # store
            dfc_slices.append(slice(start_index*rate, index*rate))
            start_index = index  #TODO: test case for avoiding overlaps...
    else:
        # append the final slice
        dfc_slices.append(slice(start_index*rate, len(dfc_data)*rate))
    return dfc_slices


def _split_by_flight_data(airspeed, offset, engine_list=None):
    """ 
    Identify flights by those above AIRSPEED_FOR_FLIGHT (kts) where being
    airborne is most likely.
    
    TODO: Split by engine_list
    
    :param airspeed: Numpy masked array
    :type airspeed: np.ma.array
    :param offset: Index to offset into the overall system - added to all slices
    :type offset: integet
    """
    if engine_list:
        raise NotImplementedError("Splitting with Engines is not implemented")

    #TODO: replace notmasked_contiguous with clump_unmasked as its return type is more consistent
    speedy_slices = np.ma.notmasked_contiguous(airspeed)
    if not speedy_slices or isinstance(speedy_slices, slice) or len(speedy_slices) <= 1 or speedy_slices == (len(airspeed), [0, -1]):
        # nothing to split (no fast bits) or only one slice returned not in a list or only one flight or no mask on array due to mask being "False"
        # NOTE: flatnotmasked_contiguous returns (a.size, [0, -1]) if no mask!
        return [slice(offset, offset+len(airspeed))]
    
    segment_slices = []
    start_index = offset
    prev_slice = speedy_slices[0]
    for speedy_slice in speedy_slices[1:]:
        #TODO: Include engine_list information to more accurately split
        stop_index = (prev_slice.stop+1 + speedy_slice.start) // 2 + offset
        segment_slices.append(slice(start_index, stop_index))
        prev_slice = speedy_slice
        start_index = stop_index
    segment_slices.append(slice(start_index, offset+len(airspeed)))
        
    return segment_slices
        
        
def _identify_segment_type(airspeed):
    """
    Identify the type of segment based on airspeed trace.
    
    :param airspeed: Airspeed after invalid data has been masked
    :type airspeed: np.ma.array
    """
    # Find the first and last valid airspeed samples
    try:
        start, stop = np.ma.flatnotmasked_edges(airspeed)
    except TypeError:
        # NoneType object is not iterable as no valid airspeed
        return 'GROUND_ONLY'

    # and if these are both lower than airspeed_threshold, we must have a sensible start and end;
    # the alternative being that we start (or end) in mid-flight.
    first_airspeed = airspeed[start]
    last_airspeed = airspeed[stop]
    
    if len(np.ma.where(airspeed > settings.AIRSPEED_THRESHOLD)[0]) < 30:
        # failed to go fast or go fast for any length of time
        return 'GROUND_ONLY'
    elif first_airspeed > settings.AIRSPEED_THRESHOLD \
       and last_airspeed > settings.AIRSPEED_THRESHOLD:
        # snippet of MID-FLIGHT data
        return 'MID_FLIGHT'
    elif first_airspeed > settings.AIRSPEED_THRESHOLD:
        # starts too fast
        logging.warning('STOP_ONLY. Airspeed initial: %s final: %s.', 
                        first_airspeed, last_airspeed)
        return 'STOP_ONLY'
    elif last_airspeed > settings.AIRSPEED_THRESHOLD:
        # ends too fast
        logging.warning('START_ONLY. Airspeed initial: %s final: %s.', 
                        first_airspeed, last_airspeed)
        return 'START_ONLY'
    else:
        # starts and ends at reasonable speeds and went fast between!
        return 'START_AND_STOP'
    
    
