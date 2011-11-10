import os
import logging
import numpy as np
import time

from analysis import settings
from analysis.hdf_access import concat_hdf, hdf_file, write_segment
from analysis.recordtype import recordtype
from analysis.split_segments import split_segments

Segment = recordtype('Segment', 'slice type part duration path', default=None)

def identify_segment_type(airspeed):
    """
    As this is run after _split_by_flight_data, we know that the data has 
    exceeded the minium AIRSPEED_FOR_FLIGHT
    """
    # Find the first and last valid airspeed samples
    start, stop = np.ma.flatnotmasked_edges(airspeed)

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
    
def join_files(first_part, second_part):
    """
    Flight Joining
    """
    hdf_path = concat_hdf([first_part, second_part], dest=first_part) 
    return hdf_path

def deidentify_file(file_path):
    """
    Removes any specific meta-data.
    Removes timebase / amends.
    Removes parameters.
    """
    pass



def store_segment(hdf_path, segment):
    """
    Stores segment information to persistent storage.
    
    :param hdf_path: 
    :type hdf_path: String
    :param segment: Details about a segment of flight data.
    :type segment: Segment
    """
    # connect to DB / REST / XML-RPC
    # make response
    logging.info("Storing segment: %s", '|'.join(
        (hdf_path, segment.path, segment.type, segment.duration)))
    return



def split_hdf_to_segments(hdf_path): #aircraft):
    use_dfc = True # TODO: Determine by aircraft.split_using_dfc?

    with hdf_file(hdf_path) as hdf:
        if settings.PRE_FILE_ANALYSIS:
            settings.PRE_FILE_ANALYSIS(hdf)
        
        # uses flight phases and DFC if aircraft determines to do so
        dfc = hdf['Frame Counter'] if use_dfc else None
        segment_slices = split_segments(hdf['Indicated Airspeed'], dfc)
        segments = []
        for part, segment_slice in enumerate(segment_slices):
            # build information about each slice
            segment_type = identify_segment_type(hdf.data.airspeed[segment_slice])
            duration = segment.stop - segment.start
            segment = Segment(segment_slice, segment_type, part + 1, duration)
            segments.append(segment)
            
    # process each segment (into a new file) having closed original hdf_path
    for segment in segments:
        # write segment to new split file (.001)
        dest_path = hdf_path.rstrip('.hdf5') + '.%03d' % segment.part + '.hdf5'
        segment.path = write_segment(hdf_path, segment.slice, dest_path)
        # store in DB for decision whether to process for flights or flight join
        store_segment(hdf_path, segment)
        
    return segments
            
      
if __name__ == '__main__':
    import sys
    import pprint
    hdf_path = sys.argv[1]
    segs = split_hdf_to_segments(hdf_path)    
    pprint.pprint(segs)
    ##os.remove(file_path) # delete original raw data file?
    ##os.remove(hdf_path) # delete original hdf file?
    