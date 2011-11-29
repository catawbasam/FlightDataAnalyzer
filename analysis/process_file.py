import os
import logging
import numpy as np
import time

from analysis import settings
from analysis.hdf_access import concat_hdf, hdf_file, write_segment
from analysis.plot_flight import plot_essential
from analysis.split_segments import split_segments



    
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
        (segment.path, segment.type, str(segment.duration))))
    return



def split_hdf_to_segments(hdf_path, draw=False): #aircraft):
    logging.info("Processing file: %s", hdf_path)
    if draw:
        plot_essential(hdf_path)
        
    with hdf_file(hdf_path) as hdf:
        if settings.PRE_FILE_ANALYSIS:
            logging.debug("Performing pre-file analysis: %s", settings.PRE_FILE_ANALYSIS.func_name)
            settings.PRE_FILE_ANALYSIS(hdf)
        
        # uses flight phases and DFC if aircraft determines to do so
        airspeed = hdf['Airspeed']
        
        if settings.POST_LFL_PARAM_PROCESS:
            # perform post lfl retrieval steps
            _airspeed = settings.POST_LFL_PARAM_PROCESS(hdf, airspeed)            
            if _airspeed:
                hdf.set_param(_airspeed)
                airspeed = _airspeed
        # split large dataset into segments
        logging.debug("Splitting segments. Data length: %s", len(airspeed.array))
        if hdf.reliable_frame_counter:
            dfc = hdf['Frame Counter']
            if settings.POST_LFL_PARAM_PROCESS:
                # perform post lfl retrieval steps
                _dfc = settings.POST_LFL_PARAM_PROCESS(hdf, dfc)
                if _dfc:
                    hdf.set_param(_dfc)
                    dfc = _dfc
            dfc_stretched = np.ma.repeat(dfc.array.data, airspeed.frequency/dfc.frequency)
        else:
            dfc_stretched = None
        segments = split_segments(airspeed.array, dfc=dfc_stretched)
            
    # process each segment (into a new file) having closed original hdf_path
    for segment in segments:
        # write segment to new split file (.001)
        dest_path = hdf_path.rstrip('.hdf5') + '.%03d' % segment.part + '.hdf5'
        logging.debug("Writing segment %d (%s): %s", segment.part, segment.duration, dest_path)
        segment.path = write_segment(hdf_path, segment.slice, dest_path)
        # store in DB for decision whether to process for flights or flight join
        store_segment(hdf_path, segment)
        if draw:
            plot_essential(dest_path)
    if draw:
        # show all figures together
        from matplotlib.pyplot import show
        show()
        #close('all') # closes all figures
         
    return segments
            
      
if __name__ == '__main__':
    import sys
    import pprint
    hdf_path = sys.argv[1]
    segs = split_hdf_to_segments(hdf_path, draw=True)    
    pprint.pprint(segs)
    ##os.remove(file_path) # delete original raw data file?
    ##os.remove(hdf_path) # delete original hdf file?
    