import unittest
import numpy as np

from hdfaccess.parameter import Parameter
from analysis.node import Section, KeyTimeInstance
from analysis.key_time_instances import (TopOfClimbTopOfDescent
                                         )


class TestTopOfClimbTopOfDescent(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD','Climb Cruise Descent')]
        opts = TopOfClimbTopOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_climb_and_descent_basic(self):
        alt_data = np.ma.array(range(0,400,50)+
                               [400]*5+
                               range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        tcd = TopOfClimbTopOfDescent()
        in_air = [slice(0,len(alt.array))]
        tcd.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, state='Top Of Climb'), 
                    KeyTimeInstance(index=13, state='Top Of Descent')]
        self.assertEqual(tcd._kti_list, expected)

    def test_top_of_climb_and_descent_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        tcd = TopOfClimbTopOfDescent()
        in_air = [slice(0,len(alt.array))]
        tcd.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, state='Top Of Descent')]
        self.assertEqual(tcd._kti_list, expected)
        self.assertEqual(len(tcd._kti_list),1)

    def test_top_of_climb_and_descent_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        tcd = TopOfClimbTopOfDescent()
        in_air = [slice(0,len(alt.array))]
        tcd.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, state='Top Of Climb')]
        self.assertEqual(tcd._kti_list, expected)
        self.assertEqual(len(tcd._kti_list),1)
