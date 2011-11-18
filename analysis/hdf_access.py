import os
import h5py
import numpy as np
import shutil
try:
    import json
except ImportError:
    import simplejson as json

from analysis.node import Parameter
from utilities.filesystem_tools import pretty_size

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
        size = pretty_size(os.path.getsize(self.hdf.filename))
        return '%s %s (%d parameters)' % (self.hdf.filename, size, len(self))
    
    def __str__(self):
        return self.__repr__()
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf = h5py.File(self.file_path, 'r+')
        self.duration = self.hdf.attrs.get('duration')
                
    def __enter__(self):
        '''
        HDF file is opened on __init__.
        '''
        return self

    def __exit__(self, a_type, value, traceback):
        self.close()
        
    def __getitem__(self, key):
        '''
        Allows for dictionary access: hdf['Altitude AAL'][:30]
        '''
        return self.get_param(key)
        
    def __setitem__(self, key, param):
        """ Allows for: hdf['Altitude AAL'] = np.ma.array()
        """
        assert key == param.name
        return self.set_param(param)
    
    def __contains__(self, key):
        """check if the key exists"""
        return key in self.keys()
    
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
    
        :param param_names: Parameters to return, if None returns all parameters
        :type param_names: list of str or None
        :returns: Param name to Param object dict
        :rtype: dict
        '''
        if param_names is None:
            param_names = self.keys()
        param_name_to_obj = {}
        for name in param_names:
            try:
                param_name_to_obj[name] = self[name]
            except KeyError:
                pass # ignore parameters that aren't available
        return param_name_to_obj

    def get_param(self, name):
        '''
        name e.g. "Head Mag"
        Returns a masked_array. If 'mask' is stored it will be the mask of the
        returned masked_array, otherwise it will be False.
        
        :param name: Name of parameter with 'series'.
        :type name: str
        :returns: Parameter object containing HDF data and attrs.
        :rtype: Parameter
        '''
        if name not in self:
            # catch exception otherwise HDF will crash and close
            raise KeyError("%s" % name)
        param_group = self.hdf['series'][name]
        data = param_group['data']
        mask = param_group.get('mask', False)
        array = np.ma.masked_array(data, mask=mask)
        frequency = param_group.attrs.get('frequency', 1) # default=1Hz for old CSV files #TODO: Remove .get
        # Differing terms: latency is known internally as frame offset.
        offset = param_group.attrs.get('latency', 0) # default=0sec for old CSV files #TODO: Remove .get
        return Parameter(name, array, frequency, offset)
    
    def get_or_create(self, param_name):
        # Either get or create parameter.
        if param_name in self.hdf['series']:
            param_group = self.hdf['series'][param_name]
        else:
            param_group = self.hdf['series'].create_group(param_name)
        return param_group

    def set_param(self, param):
        '''
        Store parameter and associated attributes on the HDF file.
        
        Parameter.name canot contain forward slashes as they are used as an
        HDF identifier which supports filesystem-style indexing, e.g.
        '/series/CAS'.
        
        :param param: Parameter like object with attributes name (must not contain forward slashes), array. 
        
        :type name: str
        :param array: Array containing data and potentially a mask for the data.
        :type array: np.array or np.ma.masked_array
        '''
        # Allow both arrays and masked_arrays.
        if hasattr(param.array, 'mask'):
            array = param.array
        else:
            array = np.ma.masked_array(param.array, mask=False)
            
        param_group = self.get_or_create(param.name)
        if 'data' in param_group:
             # Dataset must be deleted before recreation.
            del param_group['data']
        dataset = param_group.create_dataset('data', data=array.data, 
                                             **self.DATASET_KWARGS)
        if 'mask' in param_group:
            # Existing mask will no longer reflect the new data.
            del param_group['mask']
        mask = np.ma.getmaskarray(array)
        mask_dataset = param_group.create_dataset('mask', data=mask,
                                                  **self.DATASET_KWARGS)
        # Set parameter attributes
        param_group.attrs['latency'] = param.offset
        param_group.attrs['frequency'] = param.frequency
        #TODO: param_group.attrs['available_dependencies'] = param.available_dependencies
        #TODO: Possible to store validity percentage upon name.attrs
    
    def set_param_limits(self, name, limits):
        '''
        Stores limits for a parameter in JSON format.
        
        :param name: Parameter name
        :type name: string
        :param limits: Operating limits storage
        :type limits: dict
        '''
        param_group = self.get_or_create(name)
        param_group.attrs['limits'] = json.dumps(limits)
        
    def get_param_limits(self, name):
        '''
        '''
        if name not in self:
            # catch exception otherwise HDF will crash and close
            raise KeyError("%s" % name)
        limits = self.hdf['series'][name].attrs.get('limits')
        if limits:
            return json.loads(limits)
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
            param_group.create_dataset("data", data=concat_array, maxshape=(len(concat_array),))
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
    
    TODO: Support segmenting parameter masks
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
        # duration taken from last parameter
        duration = len(param_segment) / frequency
    with h5py.File(dest, 'r+') as hdf_file:
        for param_name, array in param_name_to_array.iteritems():
            param_group = hdf_file['series'][param_name]
            del param_group['data']
            param_group.create_dataset("data", data=array, maxshape=(len(array),))
        hdf_file.attrs['duration'] = duration
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
    