import numpy as np
import logging

AIRSPEED_FOR_FLIGHT = 80

def split_segments(airspeed, dfc=None):
    """
    Splits data looking for dfc jumps (if dfc provided) and changes in airspeed.
    
    :param airspeed: 1Hz airspeed data in Knots
    :type airspeed: Numpy.array
    :param dfc: 1Hz Data frame counter signal
    :type dfc: Numpy.array
    :returns: Segments of flight-like data
    :rtype: list of slices
    
    TODO: Currently requires 1Hz Airspeed Data - make multi-hertz friendly
    Q: should we not perfrom a split using DFC if Airspeed is still high to avoid cutting mid-flight?
    """
    
    # mask fast sections
    # split = False
    # if use dfc, split slow sections on DFC
    #    # find biggest dfc jump, split = True
    #    # if no jump, split = False
    
    # if not split:
         # if engines, use these
    #    # else: split half-way between jumps
    # 
    
    # TODO: Don't split too eagerly - hysteresis
    
    
    # Split by frame count is optional
    if dfc:
        # split hdf where frame counter is reset
        data_slices = _split_by_frame_counter(dfc)
    else:
        data_slices = [slice(0, len(airspeed))]
    
    segment_slices = []
    for data_slice in data_slices: ## or [slice(0,hdf.size)]: # whole data is a single segment
        # split based on airspeed for fixed wing / rotorspeed for heli
        data_seg_slices = _split_by_flight_data(airspeed[data_slice])
        segment_slices.extend(data_seg_slices)
        
    return segment_slices


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


def _split_by_flight_data(airspeed, engine_list=None):
    """ 
    Purpose: To identify flights by thoseAbove AIRSPEED_FOR_FLIGHT (kts) we may get airborne.
    TODO: Split by engine_list
    """
    if engine_list:
        raise NotImplementedError("Splitting with Engines is not implemented")

    #hysteresisless_spd = hysteresis(airspeed)
    
    mask_below_min_aispeed = np.ma.masked_less(hysteresisless_spd, AIRSPEED_FOR_FLIGHT)
    speedy_slices = np.ma.flatnotmasked_contiguous(mask_below_min_aispeed)
    if not speedy_slices or len(speedy_slices) <= 1 or speedy_slices == (len(airspeed), [0, -1]):
        # nothing to split (no fast bits) or only one flight or no mask on array due to mask being "False"
        # NOTE: flatnotmasked_contiguous returns (a.size, [0, -1]) if no mask!
        return [slice(0, len(airspeed))]
    
    segment_slices = []
    start_index = 0
    prev_slice = speedy_slices[0]
    for speedy_slice in speedy_slices[1:]:
        #TODO: Include engine_list information to more accurately split
        stop_index = (prev_slice.stop+1 + speedy_slice.start) // 2
        segment_slices.append(slice(start_index, stop_index))
        prev_slice = speedy_slice
        start_index = stop_index
    segment_slices.append(slice(start_index, len(airspeed)))
        
    return segment_slices
        

def subslice(orig, new):
    """
    a = slice(2,10,2)
    b = slice(2,2)
    c = subslice(a, b)
    assert range(100)[c] == range(100)[a][b]
    
    See tests for capabilities.
    """
    step = (orig.step or 1) * (new.step or 1)
    start = (orig.start or 0) + new.start * (orig.step or 1)
    stop = (orig.start or 0) + new.stop * (orig.step or 1)  ### or 0?
    return slice(start, stop, None if step == 1 else step)
    
    
    
    
    


    #### Note: The initial scan is repeated in compute flight phases, but it's tidier to do just this scan twice.

    ### go_fast returns a list of slices, like this:
    ### [slice(85, 5816, None), slice(6089, 11820, None)]
    ### The "None" refers to the list step which is always omitted by this Numpy function.

    ### How many chunks of data look like flights?
    ##chunks = len(go_fast)
    ### Provision for more advanced segmentation later where we
    ### could skip periods of engines not running on the ground between "flights"
    
    ##if chunks == 0:
        ##logging.info('No data over {} knots'.format(AIRSPEED_FOR_FLIGHT))
        ##return []
    ##elif chunks == 1:
        ##logging.info('1 flight to analyse')
        ##return [dict(slice=slice(start, end,None))]
    ##else:
        ##pass #continue
    
    ### Split the data midway between the flight chunks.
    ###---------------------
    ###TODO: Improve slices by looking for engine shutdown etc.
    ####engine_list
            
    ##segment_slices = []
    ##for i in range(chunks): #TODO: Replace range()? Rename 'i' to new variable
        ##if i == 0:
            ##sect = slice(start, start+(go_fast[0].stop + go_fast[1].start)/2)
        ##elif i == chunks-1:
            ##sect = slice(start+(go_fast[i-1].stop + go_fast[i].start)/2, end)
        ##else:
            ##sect = slice(start+(go_fast[i-1].stop + go_fast[i].start)/2,
                         ##start+(go_fast[i].stop + go_fast[i+1].start)/2)
        ##segment_slices.append(dict(slice=sect))
    ##return segment_slices






        
                
##def calculate_timebase():            
    ##new_datetime = myclock.checktime(line)
    
    
    ###for dfc, hrs, mins, secs in zip(rec_dfc, rec_hrs, rec_mins, rec_secs):
    ### Find date time - but only new ones are of interest. Old stuff we ignore with 'None' returned.
    ##new_datetime = myclock.checktime(line)
        
        
        
    ### Initialisation of local variables.
    ##segment = 0
    ##start_slope_datetime = None
    ##segment_details = []
    ### Set up a dictionary to collect the "histogram" buckets;
    ### Non-empty initialisation allows for cases where the recorded data doesn't vary.
    ##clock_variation = {0:0}  
    ### Instance of the clock watcher which will look for changes in recorded time.
    ##myclock = ClockWatcher()
    ### The recorded time in the first subframe will probably be missing or rubbish so don't bother to look.
    ##subf_from_start = 1
    ##subf_this_segment = 1
    #### Should read data from the beginning and ignore this problem.

    ##########################################################################################
        ##if step == 0 or step == 1 or step == -4095:
            ### This is normal data, so just look for a new date recording.
            ##if new_datetime is not None:
                ### The datetime was valid and fresh, so work out the time error...
                ##if start_slope_datetime is None:
                    ### This is the first time reading on this segment, so record the start point of the 
                    ### slope where we will monitor the time coordinates.
                    ##start_slope_datetime = new_datetime
                    ##start_slope_subf = subf_this_segment
                    #### recording = recording + [[1, segment, subf_from_start, dfc, recorded_datetime]]
                ##else:
                    ### For all other time values, record the time variations from 1 sec per subframe.
                    ##delta_time = int(datetime.timedelta.total_seconds(new_datetime - start_slope_datetime))
                    ##delta_slope = (subf_this_segment - start_slope_subf)
                    ##step = delta_time - delta_slope  # * 4 if the test data is in frames not subframes 
                    
                    #### ToDo - rename as we have two step parameters !!!                    
                    
                    ### step = int(datetime.timedelta.total_seconds(recorded_datetime - start_slice_datetime)) - (subf_this_slice - start_subf)
    
                    ### Store the variation of clock errors in a histogram.
                    ##clock_variation = clock_reading (clock_variation, step)
                    ### print delta_time, delta_slope
    
        ##else:
            ### Oh gosh, we've met a jump. 
    
            ### This tricky little line finds the maximum dictionary value and returns the key.
            ### In our case, returns the time delta for the most common time offset
            ##clock_delta = max(clock_variation, key=clock_variation.get)
    
            ### but we only started with a reference time after the beginning of the data
            ### so the time starts a little further back.
            ##clock_delta -= start_slope_subf # * 4 if the test data is in frames not subframes	    
    
            ### Convert this to a time delta and offset the first recorded time
            ##segment_start_datetime = start_slope_datetime + datetime.timedelta(seconds = clock_delta)
    
            ##segment_details.append({'start':subf_from_start-subf_this_segment,\
                                    ##'length':subf_this_segment,\
                                    ##'end':subf_from_start,\
                                    ##'start_datetime':segment_start_datetime})
    
            ### Non-empty initialisation allows for cases where the recorded data doesn't vary.
            ##clock_variation = {0:0}  
    
            ### Now prepare to scan the new slice.
            ##subf_this_segment = 0
            ##segment += 1
            ##start_slope_datetime = None
    
        ### No matter how we got here, this should have been called once per 1 second subframe.
        ##subf_from_start += 1
        ##subf_this_segment += 1
    
        ##last_dfc = dfc
        
        ### Untidy way to fill up the data arrays, but convenient for CSV files.
        
        ##self.get_flight_data(line, fp) # Gather the flight parameters we need
        
    ##### When we have read all the raw data we know how long the masked arrays need to be:
    ####for each_param in RawParameter:
        ####each_param.data=np.ma.empty_like(each_param.raw)
    
    
    ### Save details of the last segment at the end of the data file.
    ### (OK I know it could be done more tidily :o)
    ##clock_delta = max(clock_variation, key=clock_variation.get)
    ##clock_delta -= start_slope_subf # * 4 if the test data is in frames not subframes	    
    ##segment_start_datetime = start_slope_datetime + datetime.timedelta(seconds = clock_delta)
    ##segment_details.append({'start':subf_from_start-subf_this_segment,\
                            ##'length':subf_this_segment,\
                            ##'end':subf_from_start,\
                            ##'start_datetime':segment_start_datetime})
    
    ##return segment_details