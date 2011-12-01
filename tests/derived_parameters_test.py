try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import mock

import utilities.masked_array_testutils as ma_test
from utilities.struct import Struct
#from utilities.parameter_test import parameter_test
from hdfaccess.parameter import P, Parameter

from analysis.derived_parameters import (AccelerationVertical,
                                         AltitudeRadio,
                                         AltitudeTail,
                                         FlightPhaseRateOfClimb,
                                         HeadContinuous,
                                         Pitch,
                                         RateOfClimb, RateOfTurn)


class TestAltitudeRadio(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_radio(self):
        alt_rad = AltitudeRadio()
        alt_rad.aircraft = Struct({'model':{'geometry':{'main_gear_to_rad_alt':10.0}}})
        alt_rad.derive(Parameter('Pitch', (np.ma.array(range(10))-2)*5, 1,0.0),
                    Parameter('Altitude Radio Sensor', 
                              np.ma.ones(10)*10, 1,0.0))
        result = alt_rad.array

        #ralt = AltitudeRadio()
        #ralt.derive(P('Pitch',(np.ma.array(range(10))-2)*5, 1,),
        #            P('Altitude Radio Sensor',np.ma.ones(10)*10, 1,),
        #            10)

        answer = np.ma.array(data=[11.7364817767,
                                   10.8715574275,
                                   10.0,
                                   9.12844257252,
                                   8.26351822333,
                                   7.41180954897,
                                   6.57979856674,
                                   5.77381738259,
                                   5.0,
                                   4.26423563649],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(alt_rad.array, answer)

class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        ##params = {'Pitch':
                  ##Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                  ##'Altitude Radio':
                  ##Parameter('Altitude Radio', np.ma.ones(10)*10, 1,0.0)
                  ##}
        talt = AltitudeTail()
        talt.aircraft = Struct({'model':{'dist_gear_to_tail': 35.0}})
        talt.derive(Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                    Parameter('Altitude Radio', np.ma.ones(10)*10, 1,0.0))
        result = talt.array
        # At 35ft and 18deg nose up, the tail just scrapes the runway with 10ft
        # clearance at the mainwheels...
        answer = np.ma.array(data=[10.0,
                                   8.77851761541,
                                   7.55852341896,
                                   6.34150378563,
                                   5.1289414664,
                                   3.92231378166,
                                   2.72309082138,
                                   1.53273365401,
                                   0.352692546405,
                                   -0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)

class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal', 'Acceleration Lateral', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)
                        
        acc_vert.derive(
            acc_norm=Parameter('Acceleration Normal',np.ma.ones(8),8),
            acc_lat=Parameter('Acceleration Lateral',np.ma.zeros(4),4),
            acc_long=Parameter('Acceleration Longitudinal',np.ma.zeros(4),4),
            pitch=Parameter('Pitch',np.ma.zeros(2),2),
            roll=Parameter('Roll',np.ma.zeros(2),2))
        
        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))

     
class TestFlightPhaseRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = FlightPhaseRateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_flight_phase_rate_of_climb(self):
        params = {'Altitude STD':Parameter('', np.ma.array(range(10))+100)}
        roc = FlightPhaseRateOfClimb()
        roc.derive(P('Altitude STD', np.ma.array(range(10))+100))
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, answer)

    def test_flight_phase_rate_of_climb_check_hysteresis(self):
        return NotImplemented
        
        
class TestHeadContinuous(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Magnetic',)]
        opts = HeadContinuous.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_continuous(self):
        f = HeadContinuous()
        f.derive(P('Heading Magnetic',np.ma.remainder(
            np.ma.array(range(10))+355,360.0)))
        
        answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   361.0, 362.0, 363.0, 364.0],
                             dtype=np.float, mask=False)

        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(f.array.data, answer.data)
        
        
class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_pitch_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5)), 1,0.1),
                   P('Pitch (2)', np.ma.array(range(5))+10, 1,0.6)
                  )
        answer = np.ma.array(data=[0,10,1,11,2,12,3,13,4,14],
                             dtype=np.float, mask=False)
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_reverse_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5))+1, 1,0.75),
                   P('Pitch (2)', np.ma.array(range(5))+10, 1,0.25)
                  )
        answer = np.ma.array(data=[10,1,11,2,12,3,13,4,14,5],
                             dtype=np.float, mask=False)
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_error_different_rates(self):
        pch = Pitch()
        self.assertRaises(ValueError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5)), 2,0.1),
                          P('Pitch (2)', np.ma.array(range(10))+10, 4,0.6))
        
    def test_pitch_error_different_offsets(self):
        pch = Pitch()
        self.assertRaises(ValueError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5)), 1,0.11),
                          P('Pitch (2)', np.ma.array(range(5)), 1,0.6))
        
class TestRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)] #'Altitude Radio')]
        opts = RateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb(self):
        roc = RateOfClimb()
        roc.derive(P('Altitude STD', np.ma.array(range(10))+100))
                   #P('Altitude Radio', np.ma.array(range(10))))
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        np.testing.assert_array_equal(roc.array, answer)
        
class TestRateOfTurn(unittest.TestCase):
   def test_can_operate(self):
       expected = [('Head Continuous',)]
       opts = RateOfTurn.get_operational_combinations()
       self.assertEqual(opts, expected)
       
   def test_rate_of_turn(self):
       rot = RateOfTurn()
       rot.derive(P('Head Continuous', np.ma.array(range(10))))
       answer = np.ma.array(data=[1]*10, dtype=np.float)
       np.testing.assert_array_equal(rot.array, answer) # Tests data only; NOT mask
       
   def test_rate_of_turn_phase_stability(self):
        params = {'Head Continuous':Parameter('', np.ma.array([0,0,0,1,0,0,0], 
                                                               dtype=float))}
        rot = RateOfTurn()
        rot.derive(P('Head Continuous', np.ma.array([0,0,0,1,0,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)