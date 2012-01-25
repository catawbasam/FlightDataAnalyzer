import logging

from analysis_engine.process_file import split_hdf_file_into_segments
from compass.data_frame_parser import FrameFormatParser
from compass.hdf import create_hdf


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
stream.setFormatter(formatter)

logger.addHandler(stream)



def parse_raw_data(file_path, lfl_path):
    with open(lfl_file_path, 'rb') as lfl:
        lfl_parser = FrameFormatParser(lfl)
    parameter_list = lfl_parser.parse_parameters()
    hdf_path = os.path.splitext(file_path)[0] + '.hdf5'
    start_time = time.time()
    superframe_count = create_hdf(file_path, hdf_path,
                                  lfl_parser.frame, parameter_list,
                                  superframes_in_memory=10)
    duration = time.time() - start_time
    logging.info("Processed '%s' in %s secs with %d superframes",
                 dest_path, duration, superframe_count)
    
    ##lfl_parser = dfp.FrameFormatParser(file_name=lfl_path)
    ##parameter_list = lfl_parser.build_parameters(param_set=options.param_set)
    ##hdf_path = create_hdf(file_path, parameter_list)
    return hdf_path

def process_file(file_path, lfl_path): 
    """
    :param file_path: Path to Byte aligned Raw Data file to process.
    :type file_path: String
    :param lfl_path: Path to the require Logical Frame Layout
    :type lfl_path: String
    """
    # store raw -> eng. units data into HDF5 file, including offsets
    logging.debug("Processing raw data file: %s using LFL: %s", file_path, lfl_path)
    try:
        hdf_path = parse_raw_data(file_path, lfl_path) #Q: raise error if LFL does not match? but this requires Validity checking!
    except SyncError:
        # catch error when sync words do not align (data is at a greater wps than the LFL)
        logging.exception("SyncError when analysing file: %s", file_path)
        raise
    return hdf_path


if __name__ == '__main__':
    import sys
    import pprint
    #file_path = os.path.join('.', 'file.dat')
    file_path = sys.argv[1]
    hdf_path = process_file(file_path)
    segs = split_hdf_to_segments(hdf_path)
    pprint.pprint(segs)
    
    for seg in segs:
        aircraft = None
        process_flight(seg, aircraft)
        
