try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np 
from analysis.hdf_access import hdf_file


class TestHdfFile(unittest.TestCase):
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
        pass
    
    def test_get_attr(self):
        pass
    