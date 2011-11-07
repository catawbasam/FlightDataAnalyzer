import h5py #TODO: or pytables for masked array support?
import numpy as np
import simplejson as json


class hdf_file(object):    # rare case of lower case?!
    """ usage example:
    with hdf_file('path/to/file.hdf5') as hdf:
        print hdf['Altitude AAL']['data'][:20]
        
    # bits of interest
    hdf['series']['Altitude AAL']['levels']['1'] (Float array)
    hdf['series']['Altitude AAL']['data'] (Float array)
    hdf['series']['Altitude AAL']['mask'] (Bool list)
    hdf['series']['Altitude AAL'].attrs['limits'] (json)
    """
    def __repr__(self):
        # TODO: Glen to put a nice pretty representation of this object
        return ''
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf = None
                
    def __enter__(self):
        self.open()
        return self.hdf

    def __exit__(self, a_type, value, traceback):
        self.close()
        
    def __getitem__(self, key):
        """ Allows for: hdf['Altitude AAL'][:30]
        """
        return self.get_param_data(key)
        
    def __setitem__(self, key, value):
        """ Allows for: hdf['Altitude AAL'] = np.ma.array()
        """
        return self.set_param_data(key, value)
        
    def open(self):
        self.hdf = h5py.File(self.file_path, 'r+')
    
    def close(self):
        self.hdf.flush() # required?
        self.hdf.close()
        
    def get_param_list(self):
        """ List of parameters stored in hdf file
        """
        return self.hdf['series'].keys()
    
    def get_params(self, params=None):
        """ Returns params that are available, `ignores` those that aren't
        """
        if not params:
            params = self.get_param_list()
        d = {}
        for name in params:
            try:
                d[name] = self.get_param_data(name)
            except KeyError:
                pass # ignore parameters that aren't available
        return d

    def get_param_data(self, name):
        """ name e.g. "Head Mag"
        Returns masked array. If mask has been stored, returns it - otherwise returns False
        
        Uses np.ma.getmaskarray() to ensure the mask is fully fledged to the
        length of the data (opposite of shrink_mask())
        """
        ##return np.ma.array(self.hdf['series'][name]['data'], 
                           ##mask=self.hdf['series'][name].get('mask', False))

        data = self.hdf['series'][name]['data']
        mask = np.ma.getmaskarray(self.hdf['series'][name].get('mask', data))
        return np.ma.array(data, mask=mask)

    def set_param_data(self, name, array):
        """
        reshape data as required and store.
        """
        self.hdf['series'][name]['data'] = array.data
        if hasattr(array, 'mask'):
            # store the shrunk array to save space    #TODO: How much does this really save?
            self.hdf['series'][name]['mask'] = array.shrink_mask().mask
            
        #TODO: Possible to store validity percentage upon name.attrs
        
    
    def set_param_limits(self, name, limits):
        """
        Stores limits for a parameter in JSON format
        :param name: Parameter name
        :type name: string
        :param limits: Operating limits storage
        :type limits: dict
        """
        self.hdf['series'][name].attrs['limits'] = json.dumps(limits)
        
    def get_param_limits(self, name):
        """
        """
        limits = self.hdf['series'][name].attrs['limits']
        if limits:
            json.loads(limits)
        else:
            return None


def print_hdf_info(hdf):
    # Glen to fill this gap!
    #pprint some interesting stuff
    pass

if __name__ == '__main__':
    file_path = 'AnalysisEngine/resources/data/hdf5/flight_1626325.hdf5'    
    with hdf_file(file_path) as hdf:
        print_hdf_info(hdf)
        
    hdf = h5py.File(
        'AnalysisEngine/resources/data/hdf5/flight_1626326.hdf5', 'w')
    hdf['series']['Altitude AAL'].attrs['limits'] = {'min':0,  'max':50000}