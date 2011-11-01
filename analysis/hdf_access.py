import h5py #TODO: or pytables for masked array support?
             
 
class hdf_file(object):    # rare case of lower case?!
    """ usage example:
    with hdf_file('path/to/file.hdf5') as hdf:
        print hdf['Altitude AAL'][:20]
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
        
    def __getattr__(self, key):
        return self.get_param_data(key)
        
    
    def set_param_data(self, name, param_data):
        """
        reshape data as required and store.
        """
        raise NotImplementedError()
    
    ##def get_table_range(self):
        ##return self.hdf['table'][1:2]
        
        
def print_hdf_info(hdf):
    # Glen to fill this gap!
    pass



if __name__ == '__main__':
    file_path = 'resources/data/hdf5/flight_1626325.hdf5'    
    with hdf_file(file_path) as hdf:
        print_hdf_info(hdf)
    
    