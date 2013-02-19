import os

from analysis_engine.split_hdf_file_into_segments import \
        split_hdf_file_into_segments
from analysis_engine.process_flight import process_flight

def main():
    file_path = os.path.join('.', 'file.csv')
    segments = split_hdf_file_into_segments(file_path, param_group='FFD',
                                            split=False)
    # process one?
    
    process_flight(segments[0])
    graph_flight(segments[0])
    
    
if __name__ == '__main__':
    main()
    
