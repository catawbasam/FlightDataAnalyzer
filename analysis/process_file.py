import os.path

# requires IOLA FDS repository
from iolafds.chart.hdf import process_flight_csv

from split_segments import split_segments

# Minimum valid data percentage
VALIDITY_THRESHOLD = 50  # (%)
# Minimum duration of flight in seconds
DURATION_THRESHOLD = 60  # (sec)

def identify_segment_type(airspeed):
    """
    As this is run after _split_by_flight_data, we know that the data has 
    exceeded the minium AIRSPEED_FOR_FLIGHT
    """

    # Check it's a "Commercial" flight:
    
    # Find the first and last valid airspeed samples
    start, stop = np.ma.flatnotmasked_edges(airspeed)

    # and if these are both lower than airspeed_threshold, we must have a sensible start and end;
    # the alternative being that we start (or end) in mid-flight.
    first_airspeed = airspeed[start]
    last_airspeed = airspeed[stop]
    
    if first_airspeed > AIRSPEED_THRESHOLD and last_airspeed > AIRSPEED_THRESHOLD:
        # snippet of MID-FLIGHT data
        return 'MID-FLIGHT'
    elif first_airspeed > AIRSPEED_THRESHOLD:
        # starts too fast
        logging.warn ('\tIncomplete. Initial airspeed {} and final airspeed {}.'.format(first_airspeed, last_airspeed))
        return 'STOP-ONLY'
    elif last_airspeed > AIRSPEED_THRESHOLD:
        # ends too fast
        logging.warn ('\tIncomplete. Initial airspeed {} and final airspeed {}.'.format(first_airspeed, last_airspeed))
        return 'START-ONLY'
    else:
        # starts and ends at reasonable speeds and went fast between!
        return 'START-AND-STOP'
    
def join_files(first_part, second_part):
    """
    Flight Joining
    """
    hdf_path = concat_hdf(first_part, second_part, dest=first_part) # accepts multiple parts
    #first_part.append(second_part)!
    pass

def deidentify_file(file_path):
    """
    Removes any specific meta-data.
    Removes timebase / amends.
    Removes parameters.
    """
    pass

def parse_raw_data(file_path, lfl):
    lfl_parser = dfp.FrameFormatParser(file_name=lfl_file)
    parameter_list = lfl_parser.build_parameters(param_set=options.param_set)
    hdf_path = create_hdf(file_path, parameter_list)
    return hdf_path

def store_segment(hdf_path, hdf_part_path, segment_type, duration):
    """
    Stores segment information to persistent storage.
    """
    # connect to DB / REST / XML-RPC
    # make response
    logging.info("Storing segment: %s", '|'.join((hdf_path, hdf_part_path, 
                                                  segment_type, duration)))
    pass
    
def process_file(file_path):
    
    #=============================
    # store raw -> eng. units data into HDF5 file, including offsets
    try:
        hdf_path = parse_raw_data(file_path, logical_frame_layout) #Q: raise error if LFL does not match? but this requires Validity checking!
    except SyncError:
        # catch error when sync words do not align (data is at a greater wps than the LFL)
        raise
    # close?
    
    #=============================
    # open HDF5 for reading
    with hdf_file(hdf_path) as hdf:
        # validate essential params for data analysis	
        #TODO: Create parameter objects!
        validity = validate_operating_limits([altitude, airspeed, heading])  #Q: Include DFC,UTC etc?
        if validity < VALIDITY_THRESHOLD:
            raise ValidityError, "Validity %.2f is below threshold %.2f" % (validity, VALIDITY_THRESHOLD)
        ##else:
            ### validate all remaining parameters as we'll want to assume from this point all all work has been done
            ##validate_operating_limits(all_raw_params)

    
        # uses flight phases and DFC if aircraft determines to do so
        # segments is a list of dicts
        segments = split_segments(use_dfc=True)
        part = 0
        for segment in segments: # segment isinstance slice()
            # We analyse everything we're given!
            ### Find out if it is worth analysing this block of data
            ##duration = segment.stop - segment.start
            ##if duration < DURATION_THRESHOLD:
                ##logging.warning('Data with only %d seconds of data ignored', '%.f'%duration)
                ##continue
            
            segment_type = identify_segment_type(segment, hdf.data.airspeed[segment])

                
            # store in DB for decision whether to process for flights or flight join
            # also write to new split file (.001)
            dest_path = hdf_path + '%03d' % part
            hdf_part_path = write_segment(hdf, segment, dest=dest_path)
            store_segment(hdf_path, hdf_part_path, segment_type, duration)
            part += 1
    ##os.remove(hdf_path) # delete original hdf file?!
            
    return part # number of segments created, 0 if none found
            
      
if __name__ == '__main__':
    file_path = os.path.join('.', 'file.csv')
    process_file(file_path)
