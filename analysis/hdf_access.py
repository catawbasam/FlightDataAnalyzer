import h5py
import numpy as np
import shutil
try:
    import json
except ImportError:
    import simplejson as json
    
    
#TODO: This is a DUPLICATE of the function in compass.hdf, move to own HDF Repo
HDF_DEFAULTS = {'dataset': {'compression': 'gzip',
                            'compression_opts': 3}}
DEFAULT_MAX_SIZE = 8 * 10 * 24 * 60 * 60 # Ten days of an 8Hz parameter.
def _write_to_dataset(hdf_group, array, dtype='f', max_size=DEFAULT_MAX_SIZE):
    """Default max_size is very large to ensure that it is not reached during
    processing. We would have to know more about the file before creating
    datasets for parameters if we want to be more specific."""
    # 1) Create dataset within group if it does not already exist.
    if not 'data' in hdf_group:
        return hdf_group.create_dataset("data", data=array,
                                        maxshape=(max_size,),
                                        **HDF_DEFAULTS['dataset'])
    dataset = hdf_group['data']
    start_index = len(dataset)
    dataset.resize((start_index + len(array),))
    # 2) Write array to dataset.
    dataset[start_index:start_index+len(array)] = array
    return dataset


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


def concat_hdf(hdf_paths, dest=None):
    '''
    Takes in a list of HDF file paths and concatenates the parameter
    datasets which match the path 'series/<Param Name>/data'. The first file
    in the list of paths is the template for the output file, with only the
    'series/<Param Name>/data' datasets being replaced with the concatenated
    versions.
    
    :param hdf_paths: File paths.
    :type hdf_paths: list of strings
    :param dest: optional destination path, which will be the first path in
                 'paths'
    :type dest: dict
    :return: path to concatenated hdf file.
    :rtype: str
    '''
    param_name_to_arrays = {}
    for hdf_path in hdf_paths:
        with h5py.File(hdf_path, 'r') as hdf_file:
            for param_name, param_group in hdf_file['series'].iteritems():
                try:
                    param_name_to_arrays[param_name].append(param_group['data'][:])
                except KeyError:
                    param_name_to_arrays[param_name] = [param_group['data'][:]]
    if dest:
        # Copy first file in hdf_paths so that the concatenated file includes
        # non-series data. XXX: Is there a simple way to do this with h5py?
        shutil.copy(hdf_paths[0], dest)
        
    else:
        dest = hdf_paths[0]
    with h5py.File(dest, 'r+') as dest_hdf_file:
        for param_name, array_list in param_name_to_arrays.iteritems():
            concat_array = np.concatenate(array_list)
            param_group = dest_hdf_file['series'][param_name]
            del param_group['data']
            dataset = _write_to_dataset(param_group, concat_array,
                                        max_size=len(concat_array))
    return dest

    
def write_segment(hdf_path, segment, dest):
    '''
    Writes a segment of the HDF file stored in hdf_path to dest defined by 
    segments, a slice in seconds.
    
    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param segment: segment of flight to write in seconds. step is disregarded.
    :type segment: slice
    :param dest: destination path for output file containing segment.
    :type dest: str
    :return: path to output hdf file containing specified segment.
    :rtype: str
    '''
    # Q: Is there a better way to clone the contents of an hdf file?
    shutil.copy(hdf_path, dest)
    param_name_to_array = {}
    with h5py.File(hdf_path, 'r') as hdf_file:
        for param_name, param_group in hdf_file['series'].iteritems():
            frequency = param_group.attrs['frequency']
            start_index = segment.start * frequency if segment.start else None
            stop_index = segment.stop * frequency if segment.stop else None
            param_segment = param_group['data'][start_index:stop_index]
            param_name_to_array[param_name] = param_segment
    with h5py.File(dest, 'r+') as hdf_file:
        for param_name, array in param_name_to_array.iteritems():
            param_group = hdf_file['series'][param_name]
            del param_group['data']
            _write_to_dataset(param_group, array, max_size=len(array))
    return dest


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
    