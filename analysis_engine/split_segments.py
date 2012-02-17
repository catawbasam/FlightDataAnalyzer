import numpy as np
import logging
from datetime import datetime, timedelta

from analysis_engine import settings
from analysis_engine.library import (align, calculate_timebase, hash_array,
                                     hysteresis, max_abs_value, normalise,
                                     repair_mask, rate_of_change,
                                     straighten_headings,
                                     vstack_params)
from analysis_engine.node import P
from hdfaccess.file import hdf_file
from utilities.filesystem_tools import sha_hash_file

from datastructures import Segment

def _segment_type_and_slice(airspeed, frequency, start, stop):
    airspeed_start = start * frequency
    airspeed_stop = stop * frequency    
    unmasked_start, unmasked_stop = np.ma.flatnotmasked_edges(airspeed[airspeed_start:airspeed_stop])
    unmasked_start += airspeed_start
    unmasked_stop += airspeed_start
    
    slow_start = airspeed[unmasked_start] < settings.AIRSPEED_THRESHOLD
    slow_stop = airspeed[unmasked_stop] < settings.AIRSPEED_THRESHOLD
    
    threshold_exceedance = np.ma.sum(airspeed[airspeed_start:airspeed_stop] > \
                                     settings.AIRSPEED_THRESHOLD) * frequency
    if threshold_exceedance < 30: # Q: What is a sensible value?
        logging.info("Airspeed was below threshold.")
        segment_type = 'GROUND_ONLY'
    elif slow_start and slow_stop:
        logging.info("Airspeed started below threshold, rose above and stopped "
                     "below.")
        segment_type = 'START_AND_STOP'
    elif slow_start:
        logging.info("Airspeed started below threshold and stopped above.")
        segment_type = 'START_ONLY'
    elif slow_stop:
        logging.info("Airspeed started above threshold and stopped below.")
        segment_type = 'STOP_ONLY'
    else:
        logging.info("Airspeed started and stopped above threshold.")
        segment_type = 'MID_FLIGHT'
    logging.info("Segment type is '%s' between '%s' and '%s'.",
                 segment_type, start, stop)
    return segment_type, slice(start, stop)


def _get_normalised_split_params(hdf):
    '''
    Get split parameters (currently only engine power) from hdf, normalise
    them on a scale from 0-1.0 and return the minimum.
    
    :param hdf: hdf_file object.
    :type hdf: hdfaccess.file.hdf_file
    :returns: Minimum of normalised split parameters along with its frequency. Will return None, None if no split parameters are available.
    :rtype: (None, None) or (np.ma.masked_array, float)
    '''
    params = []
    first_split_param = None
    for param_name in ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',
                       'Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',
                       'Eng (1) NP', 'Eng (2) NP', 'Eng (3) NP', 'Eng (4) NP'):
        try:
            param = hdf[param_name]
        except KeyError:
            continue
        if first_split_param:
            # Align all other parameters to first available.
            param.array = align(param, first_split_param)
        else:
            first_split_param = param
            params.append(param)
    
    if not first_split_param:
        return None, None
    # If there is at least one split parameter available.
    # normalise the parameters we'll use for splitting the data
    stacked_params = vstack_params(*params)
    normalised_params = normalise(stacked_params, scale_max=100)
    split_params_min = np.ma.min(normalised_params, axis=0)
    return split_params_min, first_split_param.frequency

def _rate_of_turn(heading):
    '''
    Create rate of turn from heading.
    
    :param heading: Heading parameter.
    :type heading: Parameter
    '''
    heading.array = repair_mask(straighten_headings(heading.array),
                                repair_duration=None)
    rate_of_turn = np.ma.abs(rate_of_change(heading, 1))
    rate_of_turn_masked = np.ma.masked_greater(rate_of_turn,
                                               settings.RATE_OF_TURN_SPLITTING_THRESHOLD)    
    return rate_of_turn_masked


def split_segments(hdf):
    '''
    TODO: DJ suggested not to use decaying engine oil temperature.
    
    Notes:
     * We do not want to split on masked superframe data if mid-flight (e.g. short section of corrupt data) - repair_mask without defining repair_duration should fix that.
     * Use turning alongside engine parameters to ensure there is no movement?
     Q: Beware of pre-masked minimums to ensure we don't split on padded superframes     
    
    TODO: Use L3UQAR num power ups for difficult cases?
    '''
    airspeed = hdf['Airspeed']
    airspeed_array = repair_mask(airspeed.array, repair_duration=None)
    # mask where airspeed drops below min airspeed, using hysteresis
    airspeed_array = hysteresis(airspeed_array, settings.HYSTERESIS_FPIAS)
    airspeed_secs = len(airspeed_array) / airspeed.frequency
    slow_array = np.ma.masked_less_equal(airspeed_array,
                                         settings.AIRSPEED_THRESHOLD)
    
    speedy_slices = np.ma.clump_unmasked(slow_array)
    if len(speedy_slices) <= 1:
        logging.info("There are '%s' sections of data where airspeed is "
                     "above the splitting threshold. Therefore there can only "
                     "be at maximum one flights worth of data. Creating a single "
                     "segment comprising all data.", len(speedy_slices))
        # Use the first and last available unmasked values to determine segment
        # type.
        return [_segment_type_and_slice(airspeed_array, airspeed.frequency, 0,
                                        airspeed_secs)]
    slow_slices = np.ma.clump_masked(slow_array)
    
    heading = hdf['Heading']
    rate_of_turn = _rate_of_turn(heading)    
    
    split_params_min, \
    split_params_frequency = _get_normalised_split_params(hdf)
    
    if hdf.reliable_frame_counter:
        dfc = hdf['Frame Counter']
        dfc_diff = np.ma.diff(dfc.array)
        # Mask 'Frame Counter' incrementing by 1.
        dfc_diff = np.ma.masked_equal(dfc_diff, 1)
        # Mask 'Frame Counter' overflow where the Frame Counter transitions from
        # 4095 to 0. Q: This used to be 4094, are there some Frame Counters which increment from 1 rather than 0 or something else?
        dfc_diff = np.ma.masked_equal(dfc_diff, -4095)
        # Gap between difference values.
        dfc_half_period = (1 / dfc.frequency) / 2
    else:
        logging.info("'Frame Counter' will not be used for splitting since "
                     "'reliable_frame_counter' is False.")
        dfc = None
    
    segments = []
    start = 0
    for slow_slice in slow_slices:
        if slow_slice.start == 0:
            # Do not split if slow_slice is at the beginning of the data.
            # Since we are working with masked slices, masked padded superframe
            # data will be included within the first slow_slice.
            continue
        if slow_slice.stop == len(airspeed_array):
            # After the loop we will add the remaining data to a segment.
            break
        
        # Get start and stop at 1Hz.
        slice_start_secs = slow_slice.start / airspeed.frequency
        slice_stop_secs = slow_slice.stop / airspeed.frequency
        
        slow_duration = slice_stop_secs - slice_start_secs
        if slow_duration < settings.MINIMUM_SPLIT_DURATION:
            logging.info("Disregarding period of airspeed below '%s' "
                          "since '%s' is shorter than MINIMUM_SPLIT_DURATION "
                          "('%s').", settings.AIRSPEED_THRESHOLD, slow_duration,
                          settings.MINIMUM_SPLIT_DURATION)
            continue
        
        # Split using 'Frame Counter'.
        if dfc is not None:
            dfc_slice = slice(slice_start_secs * dfc.frequency,
                              slice_stop_secs * dfc.frequency)
            unmasked_edges = np.ma.flatnotmasked_edges(dfc_diff[dfc_slice])
            if unmasked_edges is not None:
                # Split on the first DFC jump.
                dfc_index = unmasked_edges[0]
                split_index = round((dfc_index / dfc.frequency) + slice_start_secs + \
                    dfc_half_period)
                logging.info("'Frame Counter' jumped within slow_slice '%s' "
                             "at index '%s'.", slow_slice, split_index)
                segments.append(_segment_type_and_slice(airspeed_array,
                                                        airspeed.frequency,
                                                        start, split_index))
                start = split_index
                continue
            else:
                logging.info("'Frame Counter' did not jump within slow_slice "
                             "'%s'.", slow_slice)
        
        # Split using engine parameters.        
        if split_params_min is not None:
            split_params_slice = slice(slice_start_secs * split_params_frequency,
                                       slice_stop_secs * split_params_frequency)
            split_params_masked = np.ma.masked_greater(split_params_min[split_params_slice],
                                                       settings.MINIMUM_SPLIT_PARAM_VALUE)
            try:
                below_min_slice = np.ma.clump_unmasked(split_params_masked)[0]
            except IndexError:
                logging.info("Average of normalised split parameters did not drop "
                             "below MINIMUM_SPLIT_PARAM_VALUE ('%s') within slow_slice '%s'.",
                             settings.MINIMUM_SPLIT_PARAM_VALUE,
                             split_params_slice)
            else:
                below_min_duration = below_min_slice.stop - below_min_slice.start
                param_split_index = split_params_slice.start + \
                    below_min_slice.start + (below_min_duration / 2)
                split_index = round(param_split_index / split_params_frequency)
                logging.info("Average of normalised split parameters value was "
                             "below MINIMUM_SPLIT_PARAM_VALUE ('%s') within "
                             "slow_slice '%s' at index '%s'.",
                             settings.MINIMUM_SPLIT_PARAM_VALUE,
                             slow_slice, split_index)    
                segments.append(_segment_type_and_slice(airspeed_array,
                                                        airspeed.frequency,
                                                        start, split_index))
                start = split_index
                continue
        
        # Split using rate of turn. Q: Should this be considered in other
        # splitting methods.
        rot_slice = slice(slice_start_secs * heading.frequency,
                          slice_stop_secs * heading.frequency)
        stopped_slices = np.ma.clump_unmasked(rate_of_turn[rot_slice])
        try:
            first_stop = stopped_slices[0]
        except IndexError:
            # The aircraft did not stop turning.
            logging.info("Aircraft did not stop turning during slow_slice "
                         "'%s'. Therefore a split will not be made.",
                         slow_slice)
        else:
            # Split half-way within the stop slice.
            stop_duration = first_stop.stop - first_stop.start
            rot_split_index = rot_slice.start + first_stop.start + (stop_duration / 2)
            # Get the absolute split index at 1Hz.
            split_index = round(rot_split_index / heading.frequency)
            segments.append(_segment_type_and_slice(airspeed_array, airspeed.frequency,
                                                    start, split_index))
            start = split_index
            continue
        
        logging.info("Splitting methods failed to split within slow_slice "
                     "'%s'.", slow_slice)
    
    # Add remaining data to a segment.
    segments.append(_segment_type_and_slice(airspeed_array, airspeed.frequency,
                                            start, airspeed_secs))
    return segments
        
        
# Pseudo-code version, still useful to reference!
#def split_segments_pseudocode(hdf):
    #"""
    #TODO: Ensure nan is masked!
    
    #"""
    ## TODO: Apply hook to hdf params.
    #airspeed = hdf['Airspeed']
    
    #dfc = hdf['Frame Counter'] if hdf.reliable_frame_counter else None
    
    ## I do not like splitting on speedy segments based on airspeed which may have superframe padded masks within mid-flight - i.e. we don't want to split at that point!
    ## repair mask first?
    #speedy_slices = np.ma.notmasked_contiguous(mask_slow_airspeed(airspeed.array))
    ## be clever about splitting between speedy slices
    #if len(speedy_slices) <= 1:
        #return [slice(0, len(airspeed))]
    

    ## normalise the parameters we'll use for splitting the data
    #params = vstack_params(
        #hdf.get('Eng (1) N1'), hdf.get('Eng (1) Oil Temp'),
        #hdf.get('Eng (2) N1'), hdf.get('Eng (2) Oil Temp'),
        #hdf.get('Eng (3) N1'), hdf.get('Eng (3) Oil Temp'),
        #hdf.get('Eng (4) N1'), hdf.get('Eng (4) Oil Temp'), 
        ##TODO: Add "Turning" to ensure we're staying still (rate of turn)
    #)
    #norm_split_params = normalise(params)
    
    ## more than one speedy section
    #dfc_diff = np.ma.diff(dfc.array)
    #dfc_mask_one = np.ma.masked_equal(dfc_diff, 1)
    #dfc_mask_4094 = np.ma.masked_equal(dfc_mask_one, -4094)
    #rate = 1.0 / dfc.frequency
    #segment_slices = []
    #origin = 0
    #start = speedy_slices[0].stop
    #for speedy_slice in speedy_slices[1:]:
        #stop = speedy_slice.start
        

        ##TODO: If stop - start * rate < settings.MINIMUM_SLOW_SPEED_FOR_FLIGHT_SPLIT seconds,
        ## then continue
        ##TODO: TEST
        #if stop - start < settings.ASBELOW * rate:
            #start = speedy_slice.stop
            #continue
        
        
        
        #'''
        #Q: How many splits shall we make if the DFC jumps more than once? e.g. engine runups?
        #Q: Shall we allow multiple splits if we don't use DFC between flights, e.g. params.
        #Q: Be ware of pre-masked minimums to ensure we don't split on padded superframes
        #'''
        
        
        
        
        
        ## find DFC split within speedy sections

        ## take the biggest jump (not that it means anything, but only one jump is enough!
        #index, value = max_abs_value(dfc_mask_4094, slice(start*dfc.frequency,
                                                          #stop*dfc.frequency))
        #cut_index = index * (1.0/dfc.frequency) # align index again
        
        ## if masked (no jump in dfc to split on) or the cut_index is the same
        ## as the start (again, no jump so it returns the first index)
        #if np.ma.is_masked(value) or cut_index == start:
            ## NO DFC JUMP - so use parameters to cut most accurately.
            ##TODO: implement and test below:
            #'''
            #from analysis_engine.library import vstack_params, min_value
            ##TODO: Improve by ensuring we were sat still for the longest period
            ##TODO: ensure all at same freq or align!
            
            ## NOTE: This will only allow for a single split!
            #cut_index, value = min_value(norm_params, slice(start, stop)) 
            
            ##Q: Use a threshold for value? if not met, then cut halfway
            ## between the two. To improve effectiveness of threshold, you
            ## could mask the Engine values when they are above a "turning off"
            ## state so that you're sure the minimum is when they were nearly
            ## off, otherwise you'll have a masked value and you can use that
            ## to cut upon.
            
            ## e.g. if value < .5 (50%) then it's likely that they didn't really turn off anything! 
            ## so split by lack of rate of turn and groundspeed or just half way!?
            
            #'''
            ## no jump, take half way between values
            #cut_index = (start + stop) / 2.0
            
        #if origin == cut_index:
            ## zero-length slice
            #continue
        #segment_slices.append(slice(origin, cut_index))
        ## keep track of slices
        #origin = cut_index
        #start = speedy_slice.stop
    #else:
        ## end slice
        #segment_slices.append(slice(origin, None))    
    #return segment_slices
        
#def split_segments_oldest(airspeed, dfc=None):
    #"""
    #Splits data looking for dfc jumps (if dfc provided) and changes in airspeed.
    
    #:param airspeed: Airspeed data in Knots
    #:type airspeed: Parameter
    #:param dfc: Data frame counter signal
    #:type dfc: Parameter
    #:returns: Segments of flight-like data
    #:rtype: list of slices
    
    #TODO: Currently requires 1Hz Airspeed Data - make multi-hertz friendly
    #Q: should we not perfrom a split using DFC if Airspeed is still high to avoid cutting mid-flight?
    #"""
    ## Split by frame count is optional
    #if dfc is not None:
        ## split hdf where frame counter is reset
        #data_slices = _split_by_frame_counter(dfc.array, dfc.frequency)
    #else:
        #data_slices = [slice(0, len(airspeed.array))]

    #hyst_mask_below_min_airspeed = mask_slow_airspeed(airspeed.array)
    #segment_slices = []
    #for data_slice in data_slices: ## or [slice(0,hdf.size)]: # whole data is a single segment
        ## split based on airspeed for fixed wing / rotorspeed for heli
        #data_seg_slices = _split_by_flight_data(hyst_mask_below_min_airspeed[data_slice], data_slice.start)
        #segment_slices.extend(data_seg_slices)
    #return segment_slices
        
def append_segment_info(hdf_segment_path, segment_type, segment_slice, part):
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
        # align required parameters to 1Hz
        onehz = P(frequency = 1)
        dt_arrays = []
        for name in ('Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'):
            dt_arrays.append(align(hdf.get(name), onehz, signaltype='Multi-State'))
        
        # establish timebase for start of data
        try:
            #TODO: use hdf.get('Year', [])[segment.slice] to provide empty slices.
            start_datetime = calculate_timebase(*dt_arrays)
                ##hdf['Year'].array, hdf['Month'].array, hdf['Day'].array,
                ##hdf['Hour'].array, hdf['Minute'].array, hdf['Second'].array)
        except (KeyError, ValueError):
            logging.exception("Unable to calculate timebase, using epoch 1.1.1970!")
            start_datetime = datetime.fromtimestamp(0)
        stop_datetime = start_datetime + timedelta(seconds=duration)
        
        # hdf.starttime
        # hdf.endtime
            
    #end with

    # identify subsection of airspeed, using original array with mask
    # removing all invalid data (ARINC etc)
    ##segment_type = _identify_segment_type(airspeed)
    ##logging.info("Segment type: %s", segment_type)
    
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
        
#def _split_by_frame_counter(dfc_data, dfc_freq=0.25):
    #"""
    #Creates 1Hz slices by creating slices at 4 times the stop and start
    #indexes of the 0.25Hz DFC.
    
    #:param dfc_data: Frame Counter
    #:type dfc_data: np.ma.array
    #:param dfc_freq: Frequency of the frame counter, default is 1/4Hz
    #:type dfc_freq: Float
    #:returns: 1Hz Slices where a jump occurs
    #:rtype: list of slices
    #"""
    #rate = 1.0 / dfc_freq
    ##TODO: Convert to Numpy array manipulation!
    #dfc_slices = []
    ## Now read each line in turn
    #previous_dfc = dfc_data[0]
    #start_index = 0
    #for index, value in enumerate(dfc_data[1:]):
        #index += 1 # for missing first item
        #step = value - previous_dfc
        #previous_dfc = value
        #if step == 1 or step == -4094: #TODO: This won't work for Hercules jumps which are much larger (24bits)
            ## expected increment
            #continue
        #elif step == 0:
            ## flat line - shouldn't happen but we'll ignore, so just log
            #logging.warning("DFC failed to increment, ignoring")
            #continue
        #else:
            ## store
            #dfc_slices.append(slice(start_index*rate, index*rate))
            #start_index = index  #TODO: test case for avoiding overlaps...
    #else:
        ## append the final slice
        #dfc_slices.append(slice(start_index*rate, len(dfc_data)*rate))
    #return dfc_slices


#def _split_by_flight_data(airspeed, offset, engine_list=None):
    #""" 
    #Identify flights by those above AIRSPEED_FOR_FLIGHT (kts) where being
    #airborne is most likely.
    
    #TODO: Split by engine_list
    
    #:param airspeed: Numpy masked array
    #:type airspeed: np.ma.array
    #:param offset: Index to offset into the overall system - added to all slices
    #:type offset: integet
    #"""
    #if engine_list:
        #raise NotImplementedError("Splitting with Engines is not implemented")

    ##TODO: replace notmasked_contiguous with clump_unmasked as its return type is more consistent
    #speedy_slices = np.ma.notmasked_contiguous(airspeed)
    #if not speedy_slices or isinstance(speedy_slices, slice) or len(speedy_slices) <= 1 or speedy_slices == (len(airspeed), [0, -1]):
        ## nothing to split (no fast bits) or only one slice returned not in a list or only one flight or no mask on array due to mask being "False"
        ## NOTE: flatnotmasked_contiguous returns (a.size, [0, -1]) if no mask!
        #return [slice(offset, offset+len(airspeed))]
    
    #segment_slices = []
    #start_index = offset
    #prev_slice = speedy_slices[0]
    #for speedy_slice in speedy_slices[1:]:
        ##TODO: Include engine_list information to more accurately split
        #stop_index = (prev_slice.stop+1 + speedy_slice.start) // 2 + offset
        #segment_slices.append(slice(start_index, stop_index))
        #prev_slice = speedy_slice
        #start_index = stop_index
    #segment_slices.append(slice(start_index, offset+len(airspeed)))
        
    #return segment_slices


        
#def _identify_segment_type_old(airspeed):
    #"""
    #Identify the type of segment based on airspeed trace.
    
    #:param airspeed: Airspeed after invalid data has been masked
    #:type airspeed: np.ma.array
    #"""
    ## Find the first and last valid airspeed samples
    #try:
        #start, stop = np.ma.flatnotmasked_edges(airspeed)
    #except TypeError:
        ## NoneType object is not iterable as no valid airspeed
        #return 'GROUND_ONLY'

    ## and if these are both lower than airspeed_threshold, we must have a sensible start and end;
    ## the alternative being that we start (or end) in mid-flight.
    #first_airspeed = airspeed[start]
    #last_airspeed = airspeed[stop]
    
    #if np.ma.sum(airspeed > settings.AIRSPEED_THRESHOLD) < 30:
        ## failed to go fast or go fast for any length of time
        #return 'GROUND_ONLY'
    #elif first_airspeed > settings.AIRSPEED_THRESHOLD \
       #and last_airspeed > settings.AIRSPEED_THRESHOLD:
        ## snippet of MID-FLIGHT data
        #return 'MID_FLIGHT'
    #elif first_airspeed > settings.AIRSPEED_THRESHOLD:
        ## starts too fast
        #logging.warning('STOP_ONLY. Airspeed initial: %s final: %s.', 
                        #first_airspeed, last_airspeed)
        #return 'STOP_ONLY'
    #elif last_airspeed > settings.AIRSPEED_THRESHOLD:
        ## ends too fast
        #logging.warning('START_ONLY. Airspeed initial: %s final: %s.', 
                        #first_airspeed, last_airspeed)
        #return 'START_ONLY'
    #else:
        ## starts and ends at reasonable speeds and went fast between!
        #return 'START_AND_STOP'
    