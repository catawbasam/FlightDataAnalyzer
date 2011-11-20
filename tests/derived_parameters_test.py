try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

import utilities.masked_array_testutils as ma_test
#from utilities.parameter_test import parameter_test

from analysis.derived_parameters import (AccelerationVertical,
                                         RateOfClimb, RateOfTurn)
from analysis.node import Parameter

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