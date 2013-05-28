import os
import logging
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from analysis_engine import hooks, settings
from analysis_engine.datastructures import Segment
from analysis_engine.node import P
from analysis_engine.library import (align, calculate_timebase, hash_array,
                                     min_value, normalise, repair_mask,
                                     rate_of_change, runs_of_ones, 
                                     straighten_headings, vstack_params)

from hdfaccess.file import hdf_file
from hdfaccess.utils import write_segment

from flightdatautilities.filesystem_tools import sha_hash_file


logger = logging.getLogger(name=__name__)


class AircraftMismatch(ValueError):
    pass

class TimebaseError(ValueError):
    pass


def validate_aircraft(aircraft_info, hdf):
    """
    """
    #if 'Aircraft Ident' in hdf and so on:
    # TODO: Implement validate_aircraft.
    logger.warning("Validate Aircraft not implemented")
    if True:
        return True
    else:
        raise AircraftMismatch("Tail does not match identification %s" % \
                               aircraft_info['Tail Number'])


def _segment_type_and_slice(airspeed, frequency, start, stop):
    """
    segment_type is one of:
    * 'GROUND_ONLY' (didn't go fast)
    * 'START_AND_STOP'
    * 'START_ONLY'
    * 'STOP_ONLY'
    * 'MID_FLIGHT'
    """
    airspeed_start = start * frequency
    airspeed_stop = stop * frequency
    try:
        unmasked_start, unmasked_stop = \
            np.ma.flatnotmasked_edges(airspeed[airspeed_start:airspeed_stop])
    except TypeError:
        # Raised when flatnotmasked_edges returns None because all data is
        # masked.
        segment_type = 'GROUND_ONLY'
        logger.debug("Airspeed data was entirely masked. Assuming '%s' between"
                     "'%s' and '%s'." % (segment_type, start, stop))
        return segment_type, slice(start, stop)
    unmasked_start += airspeed_start
    unmasked_stop += airspeed_start
    
    slow_start = airspeed[unmasked_start] < settings.AIRSPEED_THRESHOLD
    slow_stop = airspeed[unmasked_stop] < settings.AIRSPEED_THRESHOLD
    
    threshold_exceedance = np.ma.sum(airspeed[airspeed_start:airspeed_stop] > \
                                     settings.AIRSPEED_THRESHOLD) * frequency
    if threshold_exceedance < 30: # Q: What is a sensible value?
        logger.debug("Airspeed was below threshold.")
        segment_type = 'GROUND_ONLY'
    elif slow_start and slow_stop:
        logger.debug("Airspeed started below threshold, rose above and stopped "
                     "below.")
        segment_type = 'START_AND_STOP'
    elif slow_start:
        logger.debug("Airspeed started below threshold and stopped above.")
        segment_type = 'START_ONLY'
    elif slow_stop:
        logger.debug("Airspeed started above threshold and stopped below.")
        segment_type = 'STOP_ONLY'
    else:
        logger.debug("Airspeed started and stopped above threshold.")
        segment_type = 'MID_FLIGHT'
    logger.info("Segment type is '%s' between '%s' and '%s'.",
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
                       'Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'):
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
    rate_of_turn = np.ma.abs(rate_of_change(heading, 2))
    rate_of_turn_masked = \
        np.ma.masked_greater(rate_of_turn,
                             settings.RATE_OF_TURN_SPLITTING_THRESHOLD)    
    return rate_of_turn_masked


def split_segments(hdf):
    '''
    TODO: DJ suggested not to use decaying engine oil temperature.
    
    Notes:
     * We do not want to split on masked superframe data if mid-flight (e.g. short section of corrupt data) - repair_mask without defining repair_duration should fix that.
     * Use turning alongside engine parameters to ensure there is no movement?
     XXX: Beware of pre-masked minimums to ensure we don't split on padded superframes
    
    TODO: Use L3UQAR num power ups for difficult cases?
    '''
    airspeed = hdf['Airspeed']
    try:
        airspeed_array = repair_mask(airspeed.array, repair_duration=None,
                                     repair_above=settings.AIRSPEED_THRESHOLD)
    except ValueError:
        # Airspeed array is masked, most likely under min threshold so it did 
        # not go fast.
        return [('GROUND_ONLY', slice(0, hdf.duration))]
    
    airspeed_secs = len(airspeed_array) / airspeed.frequency
    slow_array = np.ma.masked_less_equal(airspeed_array,
                                         settings.AIRSPEED_THRESHOLD)
    
    speedy_slices = np.ma.clump_unmasked(slow_array)
    if len(speedy_slices) <= 1:
        logger.info("There are '%d' sections of data where airspeed is "
                    "above the splitting threshold. Therefore there can only "
                    "be at maximum one flights worth of data. Creating a "
                    "single segment comprising all data.", len(speedy_slices))
        # Use the first and last available unmasked values to determine segment
        # type.
        return [_segment_type_and_slice(airspeed_array, airspeed.frequency, 0,
                                        airspeed_secs)]
    
    slow_slices = np.ma.clump_masked(slow_array)
    
    try:
        # Fetch Heading if available
        heading = hdf.get_param('Heading', valid_only=True)
    except KeyError:
        # try Heading True, otherwise fall die with KeyError
        heading = hdf.get_param('Heading True', valid_only=True)
    
    rate_of_turn = _rate_of_turn(heading)
    
    split_params_min, \
    split_params_frequency = _get_normalised_split_params(hdf)
    
    if hdf.reliable_frame_counter:
        dfc = hdf['Frame Counter']
        dfc_diff = np.ma.diff(dfc.array)
        # Mask 'Frame Counter' incrementing by 1.
        dfc_diff = np.ma.masked_equal(dfc_diff, 1)
        # Mask 'Frame Counter' overflow where the Frame Counter transitions from
        # 4095 to 0.
        # Q: This used to be 4094, are there some Frame Counters which increment
        # from 1 rather than 0 or something else?
        dfc_diff = np.ma.masked_equal(dfc_diff, -4095)
        # Gap between difference values.
        dfc_half_period = (1 / dfc.frequency) / 2
    else:
        logger.info("'Frame Counter' will not be used for splitting since "
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
            logger.info("Disregarding period of airspeed below '%s' "
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
                dfc_jump = unmasked_edges[0]
                dfc_index = round((dfc_jump / dfc.frequency) +
                                  slice_start_secs + dfc_half_period)
                # account for rounding of dfc index exceeding slow slice
                if dfc_index > slice_stop_secs:
                    split_index = slice_stop_secs
                elif dfc_index < slice_start_secs:
                    split_index = slice_start_secs
                else:
                    split_index = dfc_index
                    logger.info("'Frame Counter' jumped within slow_slice '%s' "
                                "at index '%d'.", slow_slice, split_index)
                segments.append(_segment_type_and_slice(airspeed_array,
                                                        airspeed.frequency,
                                                        start, split_index))
                start = split_index
                continue
            else:
                logger.info("'Frame Counter' did not jump within slow_slice "
                             "'%s'.", slow_slice)
        
        # Split using engine parameters.        
        if split_params_min is not None:
            split_params_slice = \
                slice(slice_start_secs * split_params_frequency,
                      slice_stop_secs * split_params_frequency)
            split_index, split_value = min_value(split_params_min,
                                                 _slice=split_params_slice)
            split_index = round(split_index / split_params_frequency)
            if split_value is not None and \
               split_value < settings.MINIMUM_SPLIT_PARAM_VALUE:
                logger.info("Minimum of normalised split parameters ('%s') was "
                            "below MINIMUM_SPLIT_PARAM_VALUE ('%s') within "
                            "slow_slice '%s' at index '%d'.",
                            split_value, settings.MINIMUM_SPLIT_PARAM_VALUE,
                            slow_slice, split_index)
                segments.append(_segment_type_and_slice(airspeed_array,
                                                        airspeed.frequency,
                                                        start, split_index))
                start = split_index
                continue
            else:
                logger.info("Minimum of normalised split parameters ('%s') was "
                            "not below MINIMUM_SPLIT_PARAM_VALUE ('%s') within "
                            "slow_slice '%s' at index '%d'.",
                            split_value, settings.MINIMUM_SPLIT_PARAM_VALUE,
                            slow_slice, split_index)
        
        # Split using rate of turn. Q: Should this be considered in other
        # splitting methods.
        if rate_of_turn is None:
            continue
        rot_slice = slice(slice_start_secs * heading.frequency,
                          slice_stop_secs * heading.frequency)
        stopped_slices = np.ma.clump_unmasked(rate_of_turn[rot_slice])
        try:
            first_stop = stopped_slices[0]
        except IndexError:
            # The aircraft did not stop turning.
            logger.info("Aircraft did not stop turning during slow_slice "
                        "'%s'. Therefore a split will not be made.",
                        slow_slice)
        else:
            # Split half-way within the stop slice.
            stop_duration = first_stop.stop - first_stop.start
            rot_split_index = \
                rot_slice.start + first_stop.start + (stop_duration / 2)
            # Get the absolute split index at 1Hz.
            split_index = round(rot_split_index / heading.frequency)
            segments.append(_segment_type_and_slice(airspeed_array,
                                                    airspeed.frequency,
                                                    start, split_index))
            start = split_index
            continue

        #Q: Raise error here?
        logger.warning("Splitting methods failed to split within slow_slice "
                       "'%s'.", slow_slice)

    # Add remaining data to a segment.
    segments.append(_segment_type_and_slice(airspeed_array, airspeed.frequency,
                                            start, airspeed_secs))
    return segments


def _calculate_start_datetime(hdf, fallback_dt=None):
    """
    Calculate start datetime.
    
    :param hdf: Flight data HDF file 
    :type hdf: hdf_access object
    :param fallback_dt: Used to replace elements of datetimes which are not available in the hdf file (e.g. YEAR not being recorded)
    :type fallback_dt: datetime
    
    HDF params used:
    :Year: Optional (defaults to 1970)
    :Month: Optional (defaults to 1)
    :Day: Optional (defaults to 1)
    :Hour: Required
    :Minute: Required
    :Second: Required

    If required parameters are not available and fallback_dt is not provided,
    a TimebaseError is raised
    """
    now = datetime.now()
    if fallback_dt is not None:
        assert fallback_dt < now, \
               ("Fallback time '%s' in the future is not allowed. Current time "
                "is '%s'." % (fallback_dt, now))
    # align required parameters to 1Hz
    onehz = P(frequency = 1)
    dt_arrays = []
    for name in ('Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'):
        param = hdf.get(name)
        if param:
            # do not interpolate date/time parameters to avoid rollover issues
            array = align(param, onehz, interpolate=False)
            if len(array) == 0 or np.ma.count(array) == 0:
                logger.warning("No valid values returned for %s", name)
            else:
                # values returned, continue
                dt_arrays.append(array)
                continue
        if fallback_dt:
            array = [getattr(fallback_dt, name.lower())]
            logger.warning("%s not available, using %d from fallback_dt %s", 
                         name, array[0], fallback_dt)
            dt_arrays.append(array)
            continue
        else:
            raise TimebaseError("Required parameter '%s' not available" % name)
        
    length = max([len(array) for array in dt_arrays])
    if length > 1:
        # ensure all arrays are the same length
        for n, arr in enumerate(dt_arrays):
            if len(arr) == 1:
                # repeat to the correct size
                arr = np.repeat(arr, length)
                dt_arrays[n] = arr
            elif len(arr) != length:
                raise ValueError("After align, all arrays should be the same "
                                 "length")
            else:
                pass
        
    # establish timebase for start of data
    try:
        timebase = calculate_timebase(*dt_arrays)
    except (KeyError, ValueError) as err:
        raise TimebaseError("Error with timestamp values: %s" % err)
    
    if timebase > now:
        # Flight Data Analysis in the future is a challenge, lets see if we
        # can correct this first...
        if 'Day' not in hdf:
            # unlikely to have year, month or day.            
            # Scenario: that fallback_dt is of the current day but recorded
            # time is in the future of the fallback time, therefore resulting
            # in a futuristic date.
            a_day_before = timebase - relativedelta(days=1)
            if a_day_before < now:
                logger.info("Timebase was in the future, using a day before satisfies requirements")
                return a_day_before
            # continue to take away a Year
        if 'Year' not in hdf:
            # remove a year from the timebase
            a_year_before = timebase - relativedelta(years=1)
            if a_year_before < now:
                logger.info("Timebase was in the future, using a day before satisfies requirements")
                return a_year_before

        raise TimebaseError("Timebase '%s' is in the future.", timebase)
    
    if settings.MAX_TIMEBASE_AGE and \
       timebase < (now - timedelta(days=settings.MAX_TIMEBASE_AGE)):
        # Only allow recent timebases.
        error_msg = "Timebase '%s' older than the allowed '%d' days." % (
            timebase, settings.MAX_TIMEBASE_AGE)
        raise TimebaseError(error_msg)
    
    
    return timebase
        

def append_segment_info(hdf_segment_path, segment_type, segment_slice, part,
                        fallback_dt=None):
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
    :param fallback_dt: Used to replace elements of datetimes which are not available in the hdf file (e.g. YEAR not being recorded)
    :type fallback_dt: datetime
    :returns: Segment named tuple
    :rtype: Segment
    """
    # build information about a slice
    with hdf_file(hdf_segment_path) as hdf:
        airspeed = hdf['Airspeed'].array
        duration = hdf.duration
        # For now, raise TimebaseError up rather than using EPOCH
        # TODO: Review whether to revert to epoch again.
        ##try:
        start_datetime = _calculate_start_datetime(hdf, fallback_dt)
        ##except TimebaseError:
            ##logger.warning("Unable to calculate timebase, using epoch "
                           ##"1.1.1970!")
            ##start_datetime = datetime.fromtimestamp(0)
        stop_datetime = start_datetime + timedelta(seconds=duration)
        hdf.start_datetime = start_datetime
    
    if segment_type != 'GROUND_ONLY':
        # we went fast, so get the index
        spd_above_threshold = \
            np.ma.where(airspeed > settings.AIRSPEED_THRESHOLD)
        go_fast_index = spd_above_threshold[0][0]
        go_fast_datetime = \
            start_datetime + timedelta(seconds=int(go_fast_index))
        # Identification of raw data airspeed hash
        airspeed_hash_sections = runs_of_ones(airspeed.data > settings.AIRSPEED_THRESHOLD)
        airspeed_hash = hash_array(airspeed.data,airspeed_hash_sections,
                                   settings.AIRSPEED_HASH_MIN_SAMPLES)
    else:
        go_fast_index = None
        go_fast_datetime = None
        # if not go_fast, create hash from entire file
        airspeed_hash = sha_hash_file(hdf_segment_path)
    #                ('slice         type          part  path              hash           start_dt        go_fast_dt        stop_dt')
    segment = Segment(segment_slice, segment_type, part, hdf_segment_path, airspeed_hash, start_datetime, go_fast_datetime, stop_datetime)
    return segment


def split_hdf_to_segments(hdf_path, aircraft_info, fallback_dt=None,
                          draw=False):
    """
    Main method - analyses an HDF file for flight segments and splits each
    flight into a new segment appropriately.
    
    :param hdf_path: path to HDF file
    :type hdf_path: string
    :param aircraft_info: Information which identify the aircraft, specfically with the keys 'Tail Number', 'MSN'...
    :type aircraft_info: Dict
    :param fallback_dt: A datetime which is as close to the end of the data file as possible. Used to replace elements of datetimes which are not available in the hdf file (e.g. YEAR not being recorded)
    :type fallback_dt: datetime
    :param draw: Whether to use matplotlib to plot the flight
    :type draw: Boolean
    :returns: List of Segments
    :rtype: List of Segment recordtypes ('slice type part duration path hash')
    """
    logger.info("Processing file: %s", hdf_path)
    if draw:
        from analysis_engine.plot_flight import plot_essential
        plot_essential(hdf_path)
        
    with hdf_file(hdf_path) as hdf:
        superframe_present = hdf.superframe_present

        # Confirm aircraft tail for the entire datafile
        logger.info("Validating aircraft matches that recorded in data")
        validate_aircraft(aircraft_info, hdf)

        # now we know the Aircraft is correct, go and do the PRE FILE ANALYSIS
        if hooks.PRE_FILE_ANALYSIS:
            logger.info("Performing PRE_FILE_ANALYSIS analysis: %s", 
                         hooks.PRE_FILE_ANALYSIS.func_name)
            hooks.PRE_FILE_ANALYSIS(hdf, aircraft_info)
        else:
            logger.info("No PRE_FILE_ANALYSIS actions to perform")
        
        segment_tuples = split_segments(hdf)
        if fallback_dt:
            # fallback_dt is relative to the end of the data; remove the data
            # duration to make it relative to the start of the data
            secs = seconds=hdf.duration
            fallback_dt -= timedelta(seconds=secs)
            logger.info("Reduced fallback_dt by %ddays %dhr %dmin to %s",
               secs//86400, secs%86400//3600, secs%86400%3600//60, fallback_dt)
        
    # process each segment (into a new file) having closed original hdf_path
    segments = []
    previous_stop_dt = None
    for part, segment_tuple in enumerate(segment_tuples, start=1):
        segment_type, segment_slice = segment_tuple
        # write segment to new split file (.001)
        dest_path = os.path.splitext(hdf_path)[0] + '.%03d.hdf5' % part
        logger.debug("Writing segment %d: %s", part, dest_path)
        write_segment(hdf_path, segment_slice, dest_path, supf_boundary=superframe_present)
        segment = append_segment_info(dest_path, segment_type, segment_slice,
                                      part, fallback_dt=fallback_dt)
        
        if previous_stop_dt and segment.start_dt < previous_stop_dt:
            # In theory, this should not happen - but be warned of superframe padding?
            logger.warning("Segment start_dt '%s' comes before the previous segment ended '%s'")
        previous_stop_dt = segment.stop_dt
        
        if fallback_dt:
            # move the fallback_dt on to be relative to start of next segment
            fallback_dt += segment.stop_dt - segment.start_dt
        segments.append(segment)
        if draw:
            plot_essential(dest_path)
            
    if draw:
        # show all figures together
        from matplotlib.pyplot import show
        show()
        #close('all') # closes all figures
         
    return segments


def main():
    print 'FlightDataSplitter (c) Copyright 2013 Flight Data Services, Ltd.'
    print '  - Powered by POLARIS'
    print '  - http://www.flightdatacommunity.com'
    print ''
    import argparse
    import pprint
    import tempfile
    from flightdatautilities.filesystem_tools import copy_file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    
    parser = argparse.ArgumentParser(description="Process a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-tail', dest='tail_number', type=str, default='G-FDSL',
                        help='Aircraft Tail Number for processing.')
    args = parser.parse_args()    
    hdf_copy = copy_file(args.file, dest_dir=tempfile.gettempdir(), postfix='_split')
    logger.info("Working on copy: %s", hdf_copy)
    segs = split_hdf_to_segments(hdf_copy,
                                 {'Tail Number': args.tail_number,},
                                 fallback_dt=datetime(2012,12,12,12,12,12),
                                 draw=False)    
    pprint.pprint(segs)

      
if __name__ == '__main__':
    main()
