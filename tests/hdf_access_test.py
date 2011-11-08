try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import os
import random
import errno
import h5py
import numpy as np 
from analysis.hdf_access import hdf_file


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')

class TestHdfFile(unittest.TestCase):
    def setUp(self):
        self.hdf_path = os.path.join(TEST_DATA_DIR, 'test_hdf_access.hdf')
        hdf = h5py.File(self.hdf_path, 'w')
        series = hdf.create_group('series')
        self.param_name = 'TEST_PARAM10'
        param_group = series.create_group(self.param_name)
        self.param_frequency = 2
        self.param_latency = 1.5
        param_group.attrs['frequency'] = self.param_frequency
        param_group.attrs['latency'] = self.param_latency
        self.param_data = np.arange(100)
        dataset = param_group.create_dataset('data', data=self.param_data)
        self.masked_param_name = 'TEST_PARAM11'
        masked_param_group = series.create_group(self.masked_param_name)
        self.masked_param_frequency = 4
        self.masked_param_latency = 2.5
        masked_param_group.attrs['frequency'] = self.masked_param_frequency
        masked_param_group.attrs['latency'] = self.masked_param_latency
        self.param_mask = [bool(random.randint(0, 1)) for x in range(len(self.param_data))]
        dataset = masked_param_group.create_dataset('data', data=self.param_data)
        mask_dataset = masked_param_group.create_dataset('mask', data=self.param_mask)
        hdf.close()
        self.hdf_file = hdf_file(self.hdf_path)
    
    def tearDown(self):
        self.hdf_file.close()
        os.remove(self.hdf_path)
    
    def test_open_and_close_and_full_masks(self):
        with hdf_file('blah') as hdf:
            self.assertFalse(hdf.id is None)
            hdf['sample'] = np.array(range(10))
            self.assertEqual(list(hdf['sample'].data), range(10))
            self.assertTrue(hasattr(hdf['sample'], 'mask'))
            
            hdf['masked sample'] = np.ma.array(range(10))
            self.assertEqual(list(hdf['masked sample'].data), range(10))
            # check masks are returned in full (not just a single False)
            self.assertEqual(hdf['masked sample'].mask, [False]*10)
        self.assertTrue(hdf.id is None)
        
    def test_limit_storage(self):
        pass
    
    def test_mask_storage(self):
        pass
    
    def test_get_params(self):
        hdf_file = self.hdf_file
        # Test retrieving all parameters.
        params = hdf_file.get_params()
        self.assertTrue(len(params) == 2)
        param = params['TEST_PARAM10']
        self.assertEqual(param.frequency, self.param_frequency)
        param = params['TEST_PARAM11']
        self.assertEqual(param.offset, self.masked_param_latency)
        # Test retrieving single specified parameter.
        params = hdf_file.get_params(param_names=['TEST_PARAM10'])
        self.assertTrue(len(params) == 1)
        param = params['TEST_PARAM10']
        self.assertEqual(param.frequency, self.param_frequency)
        
    
    def test_get_attr(self):
        pass
    
    def test_set_param_data(self):
        self.__test_set_param_data(self.hdf_file.set_param_data)
    
    def test___set_item__(self):
        self.__test_set_param_data(self.hdf_file.__setitem__)
        
    def __test_set_param_data(self, set_param_data):
        '''
        :param set_param_data: Allows passing of either hdf_file.set_param_data or __getitem__.
        :type set_param_data: method
        '''
        hdf_file = self.hdf_file
        # Create new parameter with np.array.
        name1 = 'TEST_PARAM1'
        array = np.arange(100)
        set_param_data(name1, array)
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        # Create new parameter with np.ma.masked_array.
        name2 = 'TEST_PARAM2'
        mask = [False] * len(array)
        masked_array = np.ma.masked_array(data=array, mask=mask)
        set_param_data(name2, masked_array)
        self.assertTrue(np.all(hdf_file.hdf['series'][name2]['data'].value == array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name2]['mask'].value == mask))
        # Set existing parameter's data with np.array.
        array = np.arange(200)
        set_param_data(name1, array)
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        # Set existing parameter's data with np.ma.masked_array.
        mask = [bool(random.randint(0, 1)) for x in range(len(array))]
        masked_array = np.ma.masked_array(data=array, mask=mask)
        set_param_data(name1, masked_array)
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['mask'].value == mask))
    
    def test_get_param_data(self):
        self.__test_get_param_data(self.hdf_file.get_param_data)
    
    def test___get_item__(self):
        self.__test_get_param_data(self.hdf_file.__getitem__)
    
    def __test_get_param_data(self, get_param_data):
        '''
        :param set_param_data: Allows passing of either hdf_file.get_param_data or __getitem__.
        :type get_param_data: method
        '''
        # Create new parameter with np.array.
        param = get_param_data(self.param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertEqual(self.param_frequency, param.frequency)
        self.assertEqual(self.param_latency, param.offset)
        # Create new parameter with np.array.
        param = get_param_data(self.masked_param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertTrue(np.all(self.param_mask == param.array.mask))
        self.assertEqual(self.masked_param_frequency, param.frequency)
        self.assertEqual(self.masked_param_latency, param.offset)
    
    def test_len(self):
        '''
        Depends upon HDF creation in self.setUp().
        '''
        self.assertEqual(len(self.hdf_file), 2)
    
    def test_keys(self):
        '''
        Depends upon HDF creation in self.setUp().
        '''
        self.assertEqual(sorted(self.hdf_file.keys()),
                         sorted([self.param_name, self.masked_param_name]))
    
    