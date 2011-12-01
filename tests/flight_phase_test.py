import unittest

from hdfaccess.parameter import Parameter
from analysis import flight_phase

class TestTurning(unittest.TestCase):
    def test_derive(self):
        rate_of_turn_data = np.arange(-2, 2, 0.2)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        turning = flight_phase.Turning()
        turning.derive()
        
