import errno
import h5py
import numpy as np
import os
import random

try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.hdf_access import concat_hdf, hdf_file, write_segment, Parameter

TEST_DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
TEMP_DIR_PATH = os.path.join(TEST_DATA_DIR_PATH, 'temp')
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
        if self.hdf_file.hdf.id:
            self.hdf_file.close()
        os.remove(self.hdf_path)
    
    def test_open_and_close_and_full_masks(self):
        self.hdf_file.close()
        with hdf_file(self.hdf_path) as hdf:
            # check it's open
            self.assertFalse(hdf.hdf.id is None)
            hdf['sample'] = Parameter('sample', np.array(range(10)))
            self.assertEqual(list(hdf['sample'].array.data), range(10))
            self.assertTrue(hasattr(hdf['sample'].array, 'mask'))
            
            hdf['masked sample'] = Parameter('masked sample', np.ma.array(range(10)))
            self.assertEqual(list(hdf['masked sample'].array.data), range(10))
            # check masks are returned in full (not just a single False)
            self.assertEqual(list(hdf['masked sample'].array.mask), [False]*10)
        # check it's closed
        self.assertEqual(hdf.hdf.__repr__(), '<Closed HDF5 file>')
        self.assertEqual(hdf.duration, None)
        
    def test_limit_storage(self):
        self.assertTrue(False)
    
    def test_mask_storage(self):
        self.assertTrue(False)
    
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
        # test appending a limit (arinc first, then adding others)
        self.assertTrue(False)
    
    ##def test_set_param_data(self):
        ##self.__test_set_param_data(self.hdf_file.set_param)
    
    ##def test___set_item__(self):
        ##self.__test_set_param_data(self.hdf_file.__setitem__)
            
    def test___set_item__(self):
        '''
        set_item uses set_param
        '''
        set_param_data = self.hdf_file.__setitem__
        hdf_file = self.hdf_file
        # Create new parameter with np.array.
        name1 = 'TEST_PARAM1'
        array = np.arange(100)
        set_param_data(name1, Parameter(name1, array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        # Create new parameter with np.ma.masked_array.
        name2 = 'TEST_PARAM2'
        mask = [False] * len(array)
        masked_array = np.ma.masked_array(data=array, mask=mask)
        set_param_data(name2, Parameter(name2, masked_array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name2]['data'].value == array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name2]['mask'].value == mask))
        # Set existing parameter's data with np.array.
        array = np.arange(200)
        set_param_data(name1, Parameter(name1, array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        # Set existing parameter's data with np.ma.masked_array.
        mask = [bool(random.randint(0, 1)) for x in range(len(array))]
        masked_array = np.ma.masked_array(data=array, mask=mask)
        set_param_data(name1, Parameter(name1, masked_array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['data'].value == array))
        self.assertTrue(np.all(hdf_file.hdf['series'][name1]['mask'].value == mask))
    
    def test_get_param_data(self):
        self.__test_get_param_data(self.hdf_file.get_param)
    
    def test___get_item__(self):
        self.__test_get_param_data(self.hdf_file.__getitem__)
    
    def __test_get_param_data(self, get_param):
        '''
        :param set_param_data: Allows passing of either hdf_file.get_param or __getitem__.
        :type get_param_data: method
        '''
        # Create new parameter with np.array.
        param = get_param(self.param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertEqual(self.param_frequency, param.frequency)
        self.assertEqual(self.param_latency, param.offset)
        # Create new parameter with np.array.
        param = get_param(self.masked_param_name)
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
    

class TestConcatHDF(unittest.TestCase):
    def setUp(self):
        try:
            os.makedirs(TEMP_DIR_PATH)
        except OSError, err:
            if err.errno == errno.EEXIST:
                pass
        
        self.hdf_path_1 = os.path.join(TEMP_DIR_PATH,
                                       'concat_hdf_1.hdf5')
        self.hdf_data_1 = np.arange(100, dtype=np.dtype(np.float))
        self.hdf_path_2 = os.path.join(TEMP_DIR_PATH,
                                       'concat_hdf_2.hdf5')
        self.hdf_data_2 = np.arange(200, dtype=np.dtype(np.float))
        
        # Create test hdf files.
        with h5py.File(self.hdf_path_1, 'w') as hdf_file_1:
            hdf_file_1.attrs['tailmark'] = 'G-DEM'
            series = hdf_file_1.create_group('series')
            series.attrs['frame_type'] = '737-3C'
            group = series.create_group('PARAM')
            group.create_dataset('data', data=self.hdf_data_1)
            group.attrs['frequency'] = 8
            group.create_dataset('other', data=self.hdf_data_1)
        with h5py.File(self.hdf_path_2, 'w') as hdf_file_2:
            series = hdf_file_2.create_group('series')
            group = series.create_group('PARAM')
            group.create_dataset('data', data=self.hdf_data_2)
            group.create_dataset('other', data=self.hdf_data_2)
            
        self.hdf_out_path = os.path.join(TEMP_DIR_PATH,
                                         'concat_out.dat')
    
    def test_concat_hdf__without_dest(self):
        self.__test_concat_hdf()
    
    def test_concat_hdf__with_dest(self):
        self.__test_concat_hdf(dest=self.hdf_out_path)
    
    def __test_concat_hdf(self, dest=None):
        '''
        Tests that the dataset within the path matching
        'series/<Param Name>/data' is concatenated, while other datasets and
        attributes are unaffected.
        '''
        out_path = concat_hdf((self.hdf_path_1, self.hdf_path_2),
                                  dest=dest)
        if dest:
            self.assertEqual(dest, out_path)
        with h5py.File(out_path, 'r') as hdf_out_file:
            self.assertEqual(hdf_out_file.attrs['tailmark'], 'G-DEM')
            series = hdf_out_file['series']
            self.assertEqual(series.attrs['frame_type'], '737-3C') 
            param = series['PARAM']
            self.assertEqual(param.attrs['frequency'], 8)
            data_result = param['data'][:]
            data_expected_result = np.concatenate((self.hdf_data_1, self.hdf_data_2))
            # Cannot test numpy array equality with simply == operator.
            self.assertTrue(all(data_result == data_expected_result))
            # Ensure 'other' dataset has not been concatenated.
            other_result = param['other'][:]
            other_expected_result = self.hdf_data_1
            self.assertTrue(all(other_result == other_expected_result))
    
    def tearDown(self):
        for file_path in (self.hdf_path_1, self.hdf_path_2, self.hdf_out_path):
            try:
                os.remove(file_path)
            except OSError, err:
                if err.errno != errno.ENOENT:
                    raise


class TestWriteSegment(unittest.TestCase):
    def setUp(self):
        self.hdf_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_for_write_segment.hdf5')
        self.data_secs = 100
        
        with h5py.File(self.hdf_path, 'w') as hdf_file:
            series = hdf_file.create_group('series')
            # 'IVV' - 1Hz parameter.
            ivv_group = series.create_group('IVV')
            self.ivv_frequency = 1
            ivv_group.attrs['frequency'] = self.ivv_frequency
            self.ivv_latency = 2.1
            ivv_group.attrs['latency'] = self.ivv_latency
            self.ivv_data = np.arange(self.data_secs * self.ivv_frequency,
                                      dtype=np.dtype(np.float))
            ivv_group.create_dataset('data', data=self.ivv_data)
            # 'WOW' - 4Hz parameter.
            wow_group = series.create_group('WOW')
            self.wow_frequency = 4
            wow_group.attrs['frequency'] = self.wow_frequency
            self.wow_data = np.arange(self.data_secs * self.wow_frequency, 
                                      dtype=np.dtype(np.float))
            wow_group.create_dataset('data', data=self.wow_data)
            # 'DME' - 0.15Hz parameter.
            dme_group = series.create_group('DME')
            self.dme_frequency = 0.15
            dme_group.attrs['frequency'] = self.dme_frequency
            self.dme_data = np.arange(self.data_secs * self.dme_frequency, 
                                      dtype=np.dtype(np.float))
            dme_group.create_dataset('data', data=self.dme_data)
        
        self.out_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_segment.hdf5')
    
    def test_write_segment__start_and_stop(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(10,20)
        dest = write_segment(self.hdf_path, segment, self.out_path)
        self.assertEqual(dest, self.out_path)
        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['series']['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['latency'],
                             self.ivv_latency)
            ivv_result = ivv_group['data'][:]
            ivv_expected_result = np.arange(segment.start * self.ivv_frequency,
                                            segment.stop * self.ivv_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(ivv_result == ivv_expected_result))
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['series']['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(segment.start * self.wow_frequency,
                                            segment.stop * self.wow_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(wow_result == wow_expected_result))
            # 'DME' - 0.15Hz parameter.
            dme_group = hdf_file['series']['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.array([1, 2], dtype=np.dtype(np.float))
            self.assertTrue(all(dme_result == dme_expected_result))
            self.assertEqual(hdf_file.attrs['duration'], 10)
    
    def test_write_segment__start_only(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(50,None)
        dest = write_segment(self.hdf_path, segment, self.out_path)
        self.assertEqual(dest, self.out_path)
        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['series']['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['latency'],
                             self.ivv_latency)
            ivv_result = ivv_group['data'][:]
            ivv_expected_result = np.arange(segment.start * self.ivv_frequency,
                                            self.data_secs * self.ivv_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(ivv_result == ivv_expected_result))
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['series']['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(segment.start * self.wow_frequency,
                                            self.data_secs * self.wow_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(wow_result == wow_expected_result))
            # 'DME' - 0.15Hz parameter.
            dme_group = hdf_file['series']['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.arange(7, 15, dtype=np.dtype(np.float))
            self.assertTrue(all(dme_result == dme_expected_result))
            self.assertEqual(hdf_file.attrs['duration'], 50)
        
    def test_write_segment__stop_only(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(None, 70)
        dest = write_segment(self.hdf_path, segment, self.out_path)
        self.assertEqual(dest, self.out_path)
        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['series']['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['latency'],
                             self.ivv_latency)
            ivv_result = ivv_group['data'][:]
            ivv_expected_result = np.arange(0,
                                            segment.stop * self.ivv_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(ivv_result == ivv_expected_result))
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['series']['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(0,
                                            segment.stop * self.wow_frequency,
                                            dtype=np.dtype(np.float))
            self.assertTrue(all(wow_result == wow_expected_result))
            # 'DME' - 0.15Hz parameter.
            dme_group = hdf_file['series']['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.arange(0, 10, dtype=np.dtype(np.float))
            self.assertTrue(all(dme_result == dme_expected_result))
            self.assertEqual(hdf_file.attrs['duration'], 70)
    
    def test_write_segment__all_data(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(None)
        dest = write_segment(self.hdf_path, segment, self.out_path)
        self.assertEqual(dest, self.out_path)
        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['series']['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['latency'],
                             self.ivv_latency)
            ivv_result = ivv_group['data'][:]
            self.assertTrue(all(ivv_result == self.ivv_data))
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['series']['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            self.assertTrue(all(wow_result == self.wow_data))
            # 'DME' - 0.15Hz parameter.
            dme_group = hdf_file['series']['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            self.assertTrue(all(dme_result == self.dme_data))
            self.assertEqual(hdf_file.attrs['duration'], 100)
    
    def tearDown(self):
        try:
            os.remove(self.hdf_path)
        except OSError, err:
            if err.errno != errno.ENOENT:
                raise
        try:
            os.remove(self.out_path)
        except OSError, err:
            if err.errno != errno.ENOENT:
                raise


if __name__ == "__main__":
    unittest.main()