import numpy as np
import logging

from analysis import settings
from analysis.library import hysteresis
from analysis.recordtype import recordtype

Segment = recordtype('Segment', 'slice type part duration path', default=None)

def split_segments(airspeed, dfc=None):
    """
    Splits data looking for dfc jumps (if dfc provided) and changes in airspeed.
    
    :param airspeed: 1Hz airspeed data in Knots
    :type airspeed: np.ma.array
    :param dfc: 1Hz Data frame counter signal
    :type dfc: Numpy.array
    :returns: Segments of flight-like data
    :rtype: list of slices
    
    TODO: Currently requires 1Hz Airspeed Data - make multi-hertz friendly
    Q: should we not perfrom a split using DFC if Airspeed is still high to avoid cutting mid-flight?
    """
    # Split by frame count is optional
    if dfc is not None:
        # split hdf where frame counter is reset
        data_slices = _split_by_frame_counter(dfc)
    else:
        data_slices = [slice(0, len(airspeed))]

    # mask where airspeed drops below min airspeed, using hysteresis
    hysteresisless_spd = hysteresis(airspeed, settings.HYSTERESIS_FPIAS)
    hyst_mask_below_min_airspeed = np.ma.masked_less(hysteresisless_spd, settings.AIRSPEED_THRESHOLD)

    segment_slices = []
    for data_slice in data_slices: ## or [slice(0,hdf.size)]: # whole data is a single segment
        # split based on airspeed for fixed wing / rotorspeed for heli
        data_seg_slices = _split_by_flight_data(hyst_mask_below_min_airspeed[data_slice], data_slice.start)
        segment_slices.extend(data_seg_slices)
        
    segments = []
    # build information about each slice
    for part, segment_slice in enumerate(segment_slices):
        # identify subsection of airspeed, using original array with mask
        # removing all invalid data (ARINC etc)
        segment_type = _identify_segment_type(airspeed[segment_slice])
        duration = segment_slice.stop - segment_slice.start
        segment = Segment(segment_slice, segment_type, part + 1, duration)
        segments.append(segment)
        
    return segments


def _split_by_frame_counter(dfc):
    """
    Q: Return chunks of data rather than slices
    """
    #TODO: Convert to Numpy array manipulation!
    dfc_slices = []
    # Now read each line in turn
    previous_dfc = dfc[0]
    start_index = 0
    for index, value in enumerate(dfc[1:]):
        index += 1 # for missing first item
        step = value - previous_dfc
        previous_dfc = value
        if step == 1 or step == -4095:
            # expected increment
            continue
        elif step == 0:
            # flat line - shouldn't happen but we'll ignore, so just log
            logging.warning("DFC failed to increment, ignoring")
            continue
        else:
            # store
            dfc_slices.append(slice(start_index, index))
            start_index = index  #TODO: test case for avoiding overlaps...
    else:
        # append the final slice
        dfc_slices.append(slice(start_index, len(dfc)))
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
    As this is run after _split_by_flight_data, we know that the data has 
    exceeded the minium AIRSPEED_FOR_FLIGHT
    """

    #WARNING: This incorrectly currently assumes that the airspeed went fast!
    
    
    
    # Find the first and last valid airspeed samples
    try:
        start, stop = np.ma.flatnotmasked_edges(airspeed)
    except TypeError:
        # NoneType object is not iterable as no valid airspeed above 80kts
        return 'GROUND-RUN'

    # and if these are both lower than airspeed_threshold, we must have a sensible start and end;
    # the alternative being that we start (or end) in mid-flight.
    first_airspeed = airspeed[start]
    last_airspeed = airspeed[stop]
    
    if first_airspeed > settings.AIRSPEED_THRESHOLD \
       and last_airspeed > settings.AIRSPEED_THRESHOLD:
        # snippet of MID-FLIGHT data
        return 'MID-FLIGHT'
    elif first_airspeed > settings.AIRSPEED_THRESHOLD:
        # starts too fast
        logging.warning('STOP-ONLY. Airspeed initial: %s final: %s.', 
                        first_airspeed, last_airspeed)
        return 'STOP-ONLY'
    elif last_airspeed > settings.AIRSPEED_THRESHOLD:
        # ends too fast
        logging.warning('START-ONLY. Airspeed initial: %s final: %s.', 
                        first_airspeed, last_airspeed)
        return 'START-ONLY'
    else:
        # starts and ends at reasonable speeds and went fast between!
        return 'START-AND-STOP'
    
    
def subslice(orig, new):
    """
    a = slice(2,10,2)
    b = slice(2,2)
    c = subslice(a, b)
    assert range(100)[c] == range(100)[a][b]
    
    See tests for capabilities.
    """
    step = (orig.step or 1) * (new.step or 1)
    start = (orig.start or 0) + (new.start or orig.start or 0) * (orig.step or 1)
    stop = (orig.start or 0) + (new.stop or orig.stop or 0) * (orig.step or 1) # the bit after "+" isn't quite right!!
    return slice(start, stop, None if step == 1 else step)
    
