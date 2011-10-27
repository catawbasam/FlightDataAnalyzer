import h5py
import numpy as np

from pprint import pprint

def playing_about():
    storage_frequency = flight["table"].attrs["frequency"] # max freq?
    # 8.0
            
    print "Params:", hdf_flight['series'].keys()
    param_series = hdf_flight['series'][parameter]
    data = param_series['data']
    
    pprint(hdf.attrs.items())
    [(u'tailmark', ''),
     (u'from', ''),
     (u'to', ''),
     (u'timestamp', ''),
     (u'starttime', 0),
     (u'endtime', 16459000),
     (u'version', 1)]
             
 
class hdf_file(object):    # rare case of lower case?! # rename to hdf_file
    """ with hdf_file('path/file.hdf5') as hdf:
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf = None
                
    def __enter__(self):
        self.open()
        return self.hdf

    def __exit__(self, a_type, value, traceback):
        self.close()
        
    def open(self):
        self.hdf = h5py.File(self.file_path, 'r')
    
    def close(self):
        self.hdf.flush() # required?
        self.hdf.close()

    def get_param_data(self, param_name):
        """ param_name e.g. "Head Mag"
        """
        return self.hdf['series'][param_name]['data']
    
    ##def get_table_range(self):
        ##return self.hdf['table'][1:2]
        
        
def main():
    path = '/home/chris/src/nelson/Daves_Python_Spikes/resources/data/hdf5/flight_1626325.hdf5'
    with hdf_file(path) as flight_data:
            
        heading = flight_data.get_param_data('Head Mag')
        
        straight_head = straight_headings(heading)
    
                
if __name__ == '__main__':
    #main()
    test_straight_headings()
    
    