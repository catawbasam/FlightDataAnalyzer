import unittest
import numpy as np

from hdfaccess.parameter import Parameter
from analysis import flight_phase
from analysis.node import Section
from analysis.flight_phase import (Fast,
                                   LevelFlight,
                                   Turning
                                   )

class TestFast(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = Fast.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_fast_phase_basic(self):
        slow_and_fast_data = np.ma.concatenate([np.ma.arange(60,120,10),
                                                np.ma.arange(120,50,-10)])
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = flight_phase.Fast()
        phase_fast.derive(ias)
        result = [Section(name='Fast',slice=slice(2,11,None))]
        self.assertEqual(phase_fast._sections, result)
        
    def test_fast_phase_with_mask(self):
        slow_and_fast_data = np.ma.concatenate([np.ma.arange(60,120,10),
                                                np.ma.arange(120,50,-10)])
        slow_and_fast_data[5:8] = np.ma.masked
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = flight_phase.Fast()
        phase_fast.derive(ias)
        result = [Section(name='Fast',slice=slice(2,5,None)),
                  Section(name='Fast',slice=slice(8,11,None))]
        self.assertEqual(phase_fast._sections, result)
        
class TestTurning(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn',)]
        opts = Turning.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_turning_phase_basic(self):
        rate_of_turn_data = np.arange(-2, 2.2, 0.2)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        turning = flight_phase.Turning()
        turning.derive(rate_of_turn)
        result = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning._sections, result)
        
    def test_turning_phase_basic_masked_not_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = flight_phase.Turning()
        turning.derive(rate_of_turn)
        result = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning._sections, result)
        
    def test_turning_phase_basic_masked_while_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[1] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = flight_phase.Turning()
        turning.derive(rate_of_turn)
        result = [Section(name='Turning', slice=slice(0, 1, None)),
                  Section(name='Turning', slice=slice(2, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]

        self.assertEqual(turning._sections, result)
        
class TestLevelFlight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb',)]
        opts = LevelFlight.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_level_flight_phase_basic(self):
        rate_of_climb_data = np.ma.concatenate([np.ma.arange(0,400,50),
                                                np.ma.arange(400,-450,-50),
                                                np.ma.arange(-450,50,50)])
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array(rate_of_climb_data))
        level = flight_phase.LevelFlight()
        level.derive(rate_of_climb)
        result = [Section(name='Level Flight', slice=slice(0, 7, None)),
                  Section(name='Level Flight', slice=slice(10, 23, None)), 
                  Section(name='Level Flight', slice=slice(28, 35, None))]
        self.assertEqual(level._sections, result)
        
    def test_turning_phase_basic_masked_not_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = flight_phase.Turning()
        turning.derive(rate_of_turn)
        result = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning._sections, result)
        
