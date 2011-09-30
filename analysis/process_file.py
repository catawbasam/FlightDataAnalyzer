import os.path

# requires IOLA FDS repository
from iolafds.chart.hdf import process_flight_csv


# TODO: Move out to another module!
def split_flights(hdf, flight_phases, use_dfc=True):
    """
    hdf: supplied in Frames (4 second time increments)
    Returns list of segments (which has a slice)
    """
    #TODO: by frame count is optional
    dfc_segments = None
    if use_dfc:
        dfc_segments = split_flights_by_frame_counter(hdf)
    
    for seg in dfc_segments or hdf: # whole data is a single segment
        segment_slices = split_by_landings(flight_phases) #Q: Chuck incompletes? Or analyse incase of engine runup events?
    return segment_slices


stuck_with_ags = True # shucks!

def process_file(file_path):
    
    
    #=============================
    # store raw -> eng. units data into HDF5 file, including offsets
    if stuck_with_ags:
        hdf = process_flight_csv(file_path) #Q: Return hdf file object or numpy array?
    else:
        hdf = parse_raw_data(file_path, logical_frame_layout) #Q: raise error if LFL does not match? but this requires Validity checking!
    # close?
    
    #=============================
    # open HDF5 for reading
    
    # validate essential params for data analysis	
    #TODO: Create parameter objects!
    validity = validate_operating_limits([altitude, airspeed, heading])  #Q: Include DFC,UTC etc?
    if validity < VALIDITY_THRESHOLD:
        raise ValidityError, "Validity %.2f is below threshold %.2f" % (validity, VALIDITY_THRESHOLD)
    else:
        # validate all remaining parameters as we'll want to assume from this point all all work has been done
        validate_operating_limits(all_raw_params)
        
    flight_phase_map1 = flight_phases(altitude, airspeed, heading) # inc. runway turn on / off KPTs
    hdf.append_table(flight_phase_map1, table='flight_phase') # add to HDF
    
    # uses flight phases and DFC if aircraft determines to do so
    segments = split_flights(flight_phase_map1, use_dfc=True)
    
    
    for segment in segments:
        timebase = calculate_timebase(segment['data'])
        hdf.append_table(timebase, table='parameter')
        segment['data_start'] = timebase.start
        # not required for analysis, used for partial flight matching.
        segment['data_end'] = timebase.start + timedelta(seconds=(segment['slice'].stop - segment['slice'].start) * 4)
            
        if not force_analysis:
            # ensure the aircraft's the same one as we're told it is
            aircraft_found = validate_aircraft(aircraft, segment['aircraft_ident']) # raises error? or just returns aircraft?
            segment['aircraft'] = aircraft #TODO: DO SOMETHING CLEVER!!!
            
        # store in DB for decision whether to process for flights or flight join
        store_segment(segment)
        
        '''
        if segment['type'] == 'flight':
            # store in DB for processing
            pass
        elif segment['type'] == 'takeoff':
            # search for landing whose data starts around segment['data_end']
            pass
        elif segment['type'] == 'landing':
            pass
        else:
            #TODO: search for other sectors to possibly join_flights
            # store in DB and reject further processing
            pass
        '''
    
    #close_hdf_file
#================================
if __name__ == '__main__':
    file_path = os.path.join('.', 'file.csv')
    process_file(file_path)
