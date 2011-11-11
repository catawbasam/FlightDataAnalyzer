try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

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
        np.testing.assert_array_equal(res.filled(np.nan), np.ma.array([1]*10))
        
    def test_rate_of_turn_phase_stability(self):
        params = {'Straight Heading':np.ma.array([0,0,0,1,0,0,0])}
        rot = RateOfTurn()
        res = rot.derive(params)
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        answer[0]=np.ma.masked
        answer[-1] = np.ma.masked
        np.testing.assert_array_equal(res.filled(np.nan), answer)
        
        