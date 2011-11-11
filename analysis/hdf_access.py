import os
import h5py
import numpy as np
import shutil
try:
    import json
except ImportError:
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
        HDF file is opened by __init__.
        '''
        return self

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
    
    def close(self):
        self.hdf.flush() # Q: required?
        self.hdf.close()
        
    def get_param_list(self):
        '''
        List of parameters stored in hdf file
        '''
        return self.hdf['series'].keys()
    
    def get_params(self, params=None):
        '''
        Returns params that are available, `ignores` those that aren't
        '''
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
        Returns a masked_array. If 'mask' is stored it will be the mask of the
        returned masked_array, otherwise it will be False.
        """
        param = self.hdf['series'][name]
        data = param['data']
        mask = param.get('mask', False)
        # Using np.ma.getmaskarray() to ensure the mask is fully fledged to the
        # length of the data (opposite of shrink_mask()) should be unnecessary.
        #mask = np.ma.getmaskarray(self.hdf['series'][name].get('mask', data).value)
        return np.ma.array(data, mask=mask)
    
    def keys(self):
        '''
        Parameter group names within the series group.
        
        :returns: List of parameter names.
        :rtype: list of str
        '''
        return self.hdf['series'].keys()
    
    def __len__(self):
        '''
        Number of parameter groups within the series group.
        
        :returns: Number of parameters.
        :rtype: int
        '''
        return len(self.hdf['series'])

    def set_param_data(self, name, array):
        """
        reshape data as required and store.        
        
        :param name: Name of parameter. Forward slashes are not allowed within an HDF identifier as it supports filesystem-style indexing, e.g. '/series/CAS'.
        :type name: str
        :param array: Array containing data and potentially a mask for the data.
        :type array: np.array or np.ma.masked_array
        """
        # Allow both arrays and masked_arrays.
        if hasattr(array, 'mask'):
            data = array.data
            # store the shrunk mask to save space #Q: How much does this really save?
            mask = array.shrink_mask().mask
        else:
            data = array
            mask = None
        # Either get or create parameter.
        series = self.hdf['series']
        if name in series:
            param = series[name]
            # Dataset must be deleted before recreation.
            del param['data']
        else:
            param = series.create_group(name)
        dataset = param.create_dataset('data', data=array, 
                                       **self.DATASET_KWARGS)
        if 'mask' in param:
            # Existing mask will no longer reflect the new data.
            del param['mask']
        # Q: Should we create a mask of False by default?
        if mask is not None: # Testing is None as shrunk mask may simply be False.
            # Dataset compression options can only be provided for an
            # array-like object and not a single value.
            kwargs = {} if isinstance(mask, np.bool_) else self.DATASET_KWARGS
            mask_dataset = param.create_dataset('mask', data=mask, **kwargs)
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
            param_group.create_dataset("data", data=array, maxshape=(len(array),))
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
            param_group.create_dataset("data", data=array, maxshape=(len(array),))
    return dest


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
    