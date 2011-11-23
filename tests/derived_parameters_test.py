try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

import utilities.masked_array_testutils as ma_test
#from utilities.parameter_test import parameter_test

from analysis.derived_parameters import (AccelerationVertical,
                                         AltitudeRadio,
                                         AltitudeTail,
                                         FlightPhaseRateOfClimb,
                                         HeadContinuous,
                                         Pitch,
                                         RateOfClimb, RateOfTurn)
from analysis.node import Parameter

class TestAltitudeRadio(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_radio(self):
        params = {'Pitch':
                  Parameter('Pitch', (np.ma.array(range(10))-2)*5, 1,0.0),
                  'Altitude Radio Sensor':
                  Parameter('Altitude Radio Sensor', np.ma.ones(10)*10, 1,0.0)
                  }
        ralt = AltitudeRadio(params)
        ralt.derive(params, 10.0)
        result = params['Altitude Radio'].array
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
        np.testing.assert_array_almost_equal(result.data, answer.data)

class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        params = {'Pitch':
                  Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                  'Altitude Radio':
                  Parameter('Altitude Radio', np.ma.ones(10)*10, 1,0.0)
                  }
        talt = AltitudeTail(params)
        talt.derive(params, 35.0)
        result = params['Altitude Tail'].array
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
        params = {'Acceleration Normal':Parameter('Acceleration Normal',
                                                  np.ma.ones(8),8,0.0),
                  'Acceleration Lateral':Parameter('Acceleration Lateral',
                                                  np.ma.zeros(4),4,0.0),
                  'Acceleration Longitudinal':Parameter('Acceleration Longitudinal',
                                                  np.ma.zeros(4),4,0.0),
                  'Pitch':Parameter('Pitch',
                                                  np.ma.zeros(2),2,0.0),
                  'Roll':Parameter('Roll',
                                                  np.ma.zeros(2),2,0.0)}
        # Invoke the class object
        acc_vert = AccelerationVertical(params)
        # Run the derive method
        acc_vert.derive(params)
        # Compare the result to the expected answer
        result = params['Acceleration Vertical']
        answer = Parameter('Acceleration Vertical',
                           np.ma.array(data=[1]*8, dtype=np.float,mask=False),
                           8.0,0.0)
        # These four checks will be repeated so there may be a means to
        # reduce repetition here, but I think the unittest framework makes this tricky.
        self.assertEqual(result.name, answer.name)
        self.assertEqual(result.hz, answer.hz)
        self.assertEqual(result.offset, answer.offset)
        ma_test.assert_masked_array_approx_equal(result.array, answer.array)
        
    def test_acceleration_vertical_pitch_up(self):
        params = {'Acceleration Normal':Parameter('Acceleration Normal',
                                                  np.ma.ones(8)*0.8660254,8,0.0),
                  'Acceleration Lateral':Parameter('Acceleration Lateral',
                                                  np.ma.zeros(4),4,0.0),
                  'Acceleration Longitudinal':Parameter('Acceleration Longitudinal',
                                                  np.ma.ones(4)*0.5,4,0.0),
                  'Pitch':Parameter('Pitch',
                                                  np.ma.ones(2)*30.0,2,0.0),
                  'Roll':Parameter('Roll',
                                                  np.ma.zeros(2),2,0.0)}
        # Invoke the class object
        acc_vert = AccelerationVertical(params)
        # Run the derive method
        acc_vert.derive(params)
        # Compare the result to the expected answer
        result = params['Acceleration Vertical']
        answer = Parameter('Acceleration Vertical',
                           np.ma.array(data=[1]*8, dtype=np.float,mask=False),
                           8.0,0.0)
        # These four checks will be repeated so there may be a means to
        # reduce repetition here, but I think the unittest framework makes this tricky.
        self.assertEqual(result.name, answer.name)
        self.assertEqual(result.hz, answer.hz)
        self.assertEqual(result.offset, answer.offset)
        ma_test.assert_masked_array_approx_equal(result.array, answer.array)

    def test_acceleration_vertical_roll_right(self):
        params = {'Acceleration Normal':Parameter('Acceleration Normal',
                                                  np.ma.ones(8)*0.7071068,8,0.0),
                  'Acceleration Lateral':Parameter('Acceleration Lateral',
                                                  np.ma.ones(4)*(-0.7071068),4,0.0),
                  'Acceleration Longitudinal':Parameter('Acceleration Longitudinal',
                                                  np.ma.zeros(4),4,0.0),
                  'Pitch':Parameter('Pitch',
                                                  np.ma.zeros(2),2,0.0),
                  'Roll':Parameter('Roll',
                                                  np.ma.ones(2)*45.0,2,0.0)}
        # Invoke the class object
        acc_vert = AccelerationVertical(params)
        # Run the derive method
        acc_vert.derive(params)
        # Compare the result to the expected answer
        result = params['Acceleration Vertical']
        answer = Parameter('Acceleration Vertical',
                           np.ma.array(data=[1]*8, dtype=np.float,mask=False),
                           8.0,0.0)
        # These four checks will be repeated so there may be a means to
        # reduce repetition here, but I think the unittest framework makes this tricky.
        self.assertEqual(result.name, answer.name)
        self.assertEqual(result.hz, answer.hz)
        self.assertEqual(result.offset, answer.offset)
        ma_test.assert_masked_array_approx_equal(result.array, answer.array)
        
     
class TestFlightPhaseRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = FlightPhaseRateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_flight_phase_rate_of_climb(self):
        params = {'Altitude STD':Parameter('', np.ma.array(range(10))+100)}
        roc = FlightPhaseRateOfClimb(params)
        roc.derive(params)
        # !!! I dont know why this does not return a new parameter !!!
        result = params['Flight Phase Rate Of Climb'].array
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(result, answer)
        
        
class TestHeadContinuous(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Magnetic',)]
        opts = HeadContinuous.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_continuous(self):
        params = {'Heading Magnetic':Parameter('Heading Magnetic',
                                               np.ma.remainder(
                                                   np.ma.array(range(10))+355,360.0))}
        f = HeadContinuous(params)
        f.derive(params)
        answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   361.0, 362.0, 363.0, 364.0], dtype=np.float, mask=False)
        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(f.array.data, answer.data)
        
        
class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_pitch_combination(self):
        params = {'Pitch (1)':
                  Parameter('Pitch (1)', np.ma.array(range(5)), 1,0.1),
                  'Pitch (2)':
                  Parameter('Pitch (2)', np.ma.array(range(5))+10, 1,0.6)
                  }
        pch = Pitch(params)
        pch.derive(params)
        result = params['Pitch'].array
        answer = np.ma.array(data=[0,10,1,11,2,12,3,13,4,14],
                             dtype=np.float, mask=False)
        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(result.data, answer.data)

    def test_pitch_reverse_combination(self):
        params = {'Pitch (1)':
                  Parameter('Pitch (1)', np.ma.array(range(5))+1, 1,0.75),
                  'Pitch (2)':
                  Parameter('Pitch (2)', np.ma.array(range(5))+10, 1,0.25)
                  }
        pch = Pitch(params)
        pch.derive(params)
        result = params['Pitch'].array
        answer = np.ma.array(data=[10,1,11,2,12,3,13,4,14,5],
                             dtype=np.float, mask=False)
        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(result.data, answer.data)

    def test_pitch_error_different_rates(self):
        params = {'Pitch (1)':
                  Parameter('Pitch (1)', np.ma.array(range(5)), 2,0.1),
                  'Pitch (2)':
                  Parameter('Pitch (2)', np.ma.array(range(10))+10, 4,0.6)
                  }
        pch = Pitch(params)
        self.assertRaises(ValueError, pch.derive, params)
        

    def test_pitch_error_different_offsets(self):
        params = {'Pitch (1)':
                  Parameter('Pitch (1)', np.ma.array(range(5)), 1,0.11),
                  'Pitch (2)':
                  Parameter('Pitch (2)', np.ma.array(range(5)), 1,0.6)
                  }
        pch = Pitch(params)
        self.assertRaises(ValueError, pch.derive, params)
        

class TestRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD', 'Altitude Radio')]
        opts = RateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb(self):
        params = {'Altitude STD':Parameter('', np.ma.array(range(10))+100),
                  'Altitude Radio':Parameter('', np.ma.array(range(10)))}
        roc = RateOfClimb(params)
        roc.derive(params)
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(roc.array.data, answer.data)
        
class TestRateOfTurn(unittest.TestCase):
   def test_can_operate(self):
       expected = [('Head Continuous',)]
       opts = RateOfTurn.get_operational_combinations()
       self.assertEqual(opts, expected)
       
   def test_rate_of_turn(self):
       params = {'Head Continuous':Parameter('', np.ma.array(range(10)))}
       rot = RateOfTurn(params)
       rot.derive(params)
       answer = np.ma.array(data=[1]*10, dtype=np.float,
                            mask=False)
       #ma_test.assert_masked_array_approx_equal(res, answer)
       np.testing.assert_array_equal(rot.array.data, answer.data)
       
   def test_rate_of_turn_phase_stability(self):
        params = {'Head Continuous':Parameter('', np.ma.array([0,0,0,1,0,0,0], 
                                                               dtype=float))}
        rot = RateOfTurn(params)
        res = rot.derive(params)
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        #ma_test.assert_masked_array_approx_equal(res, answer)