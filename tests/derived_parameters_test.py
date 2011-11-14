try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import utilities.masked_array_testutils as ma_test

from analysis.derived_parameters import RateOfTurn

class TestRateOfTurn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Straight Heading',)]
        opts = RateOfTurn.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_turn(self):
        params = {'Straight Heading':np.ma.array(range(10))}
        rot = RateOfTurn()
        res = rot.derive(params)
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(res, answer)
        
    def test_rate_of_turn_phase_stability(self):
        params = {'Straight Heading':np.ma.array([0,0,0,1,0,0,0], dtype=float)}
        rot = RateOfTurn()
        res = rot.derive(params)
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        ma_test.assert_masked_array_approx_equal(res, answer)
        
        