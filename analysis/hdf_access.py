import os
import h5py #TODO: or pytables for masked array support?
import numpy as np
import simplejson as json

class Parameter(object):
    def __init__(self, array, frequency, offset):
        '''
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        '''
        self.array = array
        self.frequency = frequency
        self.offset = offset


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
    # HDF file settings should be consistent, therefore hardcoding defaults.
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 3}
    
    def __repr__(self):
        '''
        TODO: Glen to put a nice pretty representation of this object
        Q: What else should be displayed?
        '''
        filename = os.path.dirname(self.hdf.filename)
        return '%s (%d parameters)' % (filename, len(self))
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf = h5py.File(self.file_path, 'r+')
                
    def __enter__(self):
        '''
        HDF file is opened on __init__.
        '''
        pass

    def __exit__(self, a_type, value, traceback):
        self.close()
        
    def __getitem__(self, key):
        '''
        Allows for dictionary access: hdf['Altitude AAL'][:30]
        '''
        return self.get_param_data(key)
        
    def __setitem__(self, key, value):
        """ Allows for: hdf['Altitude AAL'] = np.ma.array()
        """
        return self.set_param_data(key, value)
    
    def __len__(self):
        '''
        Number of parameter groups within the series group.
        
        :returns: Number of parameters.
        :rtype: int
        '''
        return len(self.hdf['series'])
    
    def keys(self):
        '''
        Parameter group names within the series group.
        
        :returns: List of parameter names.
        :rtype: list of str
        '''
        return self.hdf['series'].keys()
    get_param_list = keys
    
    def close(self):
        self.hdf.flush() # Q: required?
        self.hdf.close()
    
    def get_params(self, param_names=None):
        '''
        Returns params that are available, `ignores` those that aren't.
        
        :param param_names:
        :type param_names: list of str or None
        :returns:
        :rtype: dict
        '''
        if not param_names:
            param_names = self.keys()
        param_name_to_obj = {}
        for name in param_names:
            try:
                param_name_to_obj[name] = self[name]
            except KeyError:
                pass # ignore parameters that aren't available
        return param_name_to_obj

    def get_param_data(self, name):
        '''
        name e.g. "Head Mag"
        Returns a masked_array. If 'mask' is stored it will be the mask of the
        returned masked_array, otherwise it will be False.
        
        :param name: Name of parameter with 'series'.
        :type name: str
        :returns: Parameter object containing HDF data and attrs.
        :rtype: Parameter
        '''
        param_group = self.hdf['series'][name]
        data = param_group['data']
        mask = param_group.get('mask', False)
        array = np.ma.masked_array(data, mask=mask)
        frequency = param_group.attrs['frequency']
        # Differing terms: latency is known internally as frame offset.
        offset = param_group.attrs['latency']
        return Parameter(array, frequency, offset)

    def set_param_data(self, name, array):
        '''
        reshape data as required and store.        
        
        :param name: Name of parameter. Forward slashes are not allowed within an HDF identifier as it supports filesystem-style indexing, e.g. '/series/CAS'.
        :type name: str
        :param array: Array containing data and potentially a mask for the data.
        :type array: np.array or np.ma.masked_array
        '''
        # Allow both arrays and masked_arrays.
        if not hasattr(array, 'mask'):
            array = np.ma.masked_array(array, mask=False)
        # Either get or create parameter.
        series = self.hdf['series']
        if name in series:
            param_group = series[name]
            # Dataset must be deleted before recreation.
            del param_group['data']
        else:
            param_group = series.create_group(name)
        dataset = param_group.create_dataset('data', data=array.data, 
                                             **self.DATASET_KWARGS)
        if 'mask' in param_group:
            # Existing mask will no longer reflect the new data.
            del param_group['mask']
        # Q: Should we create a mask of False by default?
        mask_dataset = param_group.create_dataset('mask', data=array.mask,
                                                  **self.DATASET_KWARGS)
        #TODO: Possible to store validity percentage upon name.attrs
    
    def set_param_limits(self, name, limits):
        '''
        Stores limits for a parameter in JSON format.
        
        :param name: Parameter name
        :type name: string
        :param limits: Operating limits storage
        :type limits: dict
        '''
        self.hdf['series'][name].attrs['limits'] = json.dumps(limits)
        
    def get_param_limits(self, name):
        '''
        
        '''
        limits = self.hdf['series'][name].attrs['limits']
        if limits:
            json.loads(limits)
        else:
            return None


def print_hdf_info(hdf_file):
    hdf_file = hdf_file.hdf
    series = hdf_file['series']
    # IOLA
    # 8.0
    if 'Time' in series:
        print 'Tailmark:', hdf_file.attrs['tailmark']
        print 'Start Time:', hdf_file.attrs['starttime']
        print 'End Time:', hdf_file.attrs['endtime']
    
    for group_name, group in series.iteritems():
        print '[%s]' % group_name
        print 'Frequency:', group.attrs['frequency']
        print group.attrs.items()
        # IOLA's latency is our frame offset.
        print 'Offset:', group.attrs['latency']
        print 'External Data Type:', group.attrs['external_datatype']
        print 'External Data Format:', group.attrs['external_dataformat']
        print 'Number of recorded values:', len(group['data'])
    #param_series = hdf_file['series'][parameter]
    #data = param_series['data']

if __name__ == '__main__':
    import sys
    print_hdf_info(hdf_file(sys.argv[1]))
    sys.exit()
    file_path = 'AnalysisEngine/resources/data/hdf5/flight_1626325.hdf5'    
    with hdf_file(file_path) as hdf:
        print_hdf_info(hdf)
        
    hdf = h5py.File(
        'AnalysisEngine/resources/data/hdf5/flight_1626326.hdf5', 'w')
    hdf['series']['Altitude AAL'].attrs['limits'] = {'min':0,  'max':50000}