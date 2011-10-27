try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from analysis.validate import Limit, check_for_change, validate
    
class TestMasks(unittest.TestCase):
    def test_arinc_mask(self):
        self.assertFalse(True)
        
    def test_min_max_limits(self):
        self.assertFalse(True)
        
        
class TestValidateParam(unittest.TestCase):
    def test_validate(self):
        limit = Limit()
        data = [49,48,42,54,61,60,57]
        ma = np.ma.array(data)
        result = validate(ma, limit , False)
        self.assertEqual(result, np.ma.array(data))
        
    def test_check_for_change(self):
        # Check result is false for all same and zero
        data = np.ma.array([0,0,0,0,0,0,0,0])
        self.assertFalse(check_for_change(data))
        
        # Check result is false for all same and non-zero
        data = np.ma.array([3,3,3,3,3,3,3,3])
        self.assertFalse(check_for_change(data))

        # Check result is true for one different value
        data[3]=0.1
        self.assertTrue(check_for_change(data))
    
    def test_check_for_change_using_mask(self):
        masked_data = np.ma.array([5] * 10)
        #masked_data[5] = np.ma.masked
        self.assertFalse(check_for_change(masked_data))
        
    def test_for_change_with_mask_applied(self):
        masked_array = np.ma.array([4,4,6,4,4,4,4])
        self.assertTrue(check_for_change(masked_array))
        masked_array[2] = np.ma.masked
        self.assertFalse(check_for_change(masked_array))