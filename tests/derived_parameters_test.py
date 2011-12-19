try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import mock

import utilities.masked_array_testutils as ma_test
from utilities.struct import Struct
from analysis.settings import GRAVITY
from analysis.node import Attribute, A, KPV, KTI, Parameter, P, Section, S
from analysis.flight_phase import Fast

from analysis.derived_parameters import (AccelerationVertical,
                                         AirspeedForFlightPhases,
                                         AltitudeAALForFlightPhases,
                                         AltitudeForFlightPhases,
                                         AltitudeRadio,
                                         AltitudeRadioForFlightPhases,
                                         AltitudeTail,
                                         ClimbForFlightPhases,
                                         HeadingContinuous,
                                         Pitch,
                                         RateOfClimb,
                                         RateOfClimbForFlightPhases,
                                         RateOfTurn)



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


class TestAccelerationForwardsForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Longitudinal', 'Airspeed')]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_accelearation_forwards_for_phases_basic(self):
        # If acceleration data is available, this is used without change.
        acc = np.ma.array(range(0,0.5,0.1))
        accel_fwd = AccelerationForwardsForFlightPhases()
        accel_fwd.derive(Parameter('Airspeed', fast_and_slow))
        ma_test.assert_masked_array_approx_equal(accel_fwd.array, acc)

    def test_accelearation_forwards_for_phases_mask_repair(self):
        # Show that the mask is repaired in case of minor corruption.
        acc = np.ma.array(range(0,0.5,0.1))
        acc[1:4] = np.ma.masked
        accel_fwd = AccelerationForwardsForFlightPhases()
        accel_fwd.derive(Parameter('Airspeed', fast_and_slow))
        ma_test.assert_masked_array_approx_equal(accel_fwd.array, acc)

    
class TestAirspeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_airspeed_for_phases_basic(self):
        fast_and_slow = np.ma.array([40,200,190,180,170])
        speed = AirspeedForFlightPhases()
        speed.derive(Parameter('Airspeed', fast_and_slow))
        expected = np.ma.array([40,195,195,185,175])
        ma_test.assert_masked_array_approx_equal(speed.array, expected)

   



class TestAltitudeAALForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Fast')]
        opts = AltitudeAALForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_AAL_for_flight_phases_basic(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        up_and_down_data = slow_and_fast_data * 10
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0, 0, 0, 100, 200, 300, 
                                500, 400, 300, 200, 100, 0, 0])
        ma_test.assert_masked_array_approx_equal(alt_4_ph.array, expected)

    def test_altitude_AAL_for_flight_phases_masked_at_lift(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        up_and_down_data = slow_and_fast_data * 10
        up_and_down_data[1:4] = np.ma.masked
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0, 0, 0, 100, 200, 300, 
                                500, 400, 300, 200, 100, 0, 0])
        ma_test.assert_masked_array_approx_equal(alt_4_ph.array, expected)

class TestAltitudeRadio(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch',
                     'Main Gear To Altitude Radio')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_radio(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(
            Parameter('Altitude Radio Sensor', np.ma.ones(10)*10, 1,0.0),
            Parameter('Pitch', (np.ma.array(range(10))-2)*5, 1,0.0),
            Attribute('Main Gear To Altitude Radio', 10.0)
        )
        result = alt_rad.array
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

class TestAltitudeForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = AltitudeForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_phases_repair(self):
        alt_4_ph = AltitudeForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude STD', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)
        
    def test_altitude_for_phases_hysteresis(self):
        alt_4_ph = AltitudeForFlightPhases()
        testwave = np.sin(np.arange(0,6,0.1))*200
        alt_4_ph.derive(Parameter('Altitude STD', np.ma.array(testwave), 1,0.0))

        answer = np.ma.array(data = [0.,0.,0.,0.,0.,0.,12.92849468,28.84353745,
                                     43.47121818,56.66538193,68.29419696,
                                     78.24147201,86.40781719,92.71163708,
                                     97.089946,99.49899732,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     88.32517131,68.45086117,48.89177959,
                                     29.84335446,11.49591134,-5.96722818,
                                     -22.37157819,-37.55323184,-51.36049906,
                                     -63.65542221,-74.31515448,-83.23318735,
                                     -90.32041478,-95.50602353,-98.73820073,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151],mask = False)
        np.testing.assert_array_almost_equal(alt_4_ph.array, answer)


class TestAltitudeRadioForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio',)]
        opts = AltitudeRadioForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_radio_phases_repair(self):
        alt_4_ph = AltitudeRadioForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude Radio', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)


class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch','Dist Gear To Tail')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.ones(10)*10, 1,0.0),
                    Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                    Attribute('Dist Gear To Tail', 35.0)
                    )
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


class TestClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Fast')]
        opts = ClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_climb_for_flight_phases_basic(self):
        up_and_down_data = np.ma.array([0,2,5,3,2,5,6,8])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([100]*8)))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0,2,5,0,0,3,4,6])
        ma_test.assert_masked_array_approx_equal(climb.array, expected)

   
'''
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
'''

        
class TestHeadContinuous(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Magnetic',)]
        opts = HeadingContinuous.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_continuous(self):
        head = HeadingContinuous()
        head.derive(P('Heading Magnetic',np.ma.remainder(
            np.ma.array(range(10))+355,360.0)))
        
        answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   361.0, 362.0, 363.0, 364.0],
                             dtype=np.float, mask=False)

        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(head.array.data, answer.data)
        
        
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
        expected = [('Acceleration Vertical',
                     'Altitude STD',
                     'Altitude Radio')]
        opts = RateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb_basic(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10))
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)

    def test_rate_of_climb_bump(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10,dtype=float))
        az.array[2:4] = 1.1
        # (Low acceleration for this test as the sample rate is only 1Hz).
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[0, 0, 88.432295, 250.230226, 295.804651,
                                     244.545721, 201.267830, 164.741556, 
                                     133.926642, 107.942898],
                               mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)

    def test_rate_of_climb_combined_signals(self):
        # ======================================================================
        # NOTE: The results of this test are dependent upon the settings
        # parameters GRAVITY = 32.2, RATE_OF_CLIMB_LAG_TC = 6.0,
        # AZ_WASHOUT_TC = 60.0. Changes in any of these will result in a test
        # failure and recomputation of the result array will be necessary.
        # ======================================================================
        
        # Initialise to 1g
        az = P('Acceleration Vertical', np.ma.array([1]*30,dtype=float))
        # After 2 seconds, increment by 1 ft/s^2
        az.array[2:] += 1/GRAVITY
        
        # This will give a linearly increasing rate of climb 0>28 ft/sec...
        # which integrated (cumcum) gives a parabolic theoretical solution.
        parabola = (np.cumsum(np.arange(0.0,28.0,1)))

        # The pressure altitude datum could be anything. Set 99ft for fun.
        alt_std = P('Altitude STD', np.ma.array([99]*30,dtype=float))
        # and add the increasing parabola 
        alt_std.array[2:] += parabola 
        alt_rad = P('Altitude Radio', np.ma.array([0]*30,dtype=float))
        # parabola *= 1.1 # Allows you to make the values different for debug.
        alt_rad.array[2:] += parabola
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[-3.47482E-11,
                                     -3.47482E-11,
                                     27.5061571,
                                     87.0628767,
                                     145.786014,
                                     203.818717,
                                     261.281866,
                                     318.277502,
                                     374.891728,
                                     431.197166,
                                     487.255037,
                                     543.116913,
                                     598.826213,
                                     654.419455,
                                     709.927322,
                                     765.375567,
                                     820.785771,
                                     876.175989,
                                     931.561299,
                                     986.954258,
                                     1042.3653,
                                     1097.80305,
                                     1153.27464,
                                     1208.7859,
                                     1264.34159,
                                     1319.94555,
                                     1375.60086,
                                     1431.30995,
                                     1487.0747,
                                     1542.89655], mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)


class TestRateOfClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = RateOfClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb_for_flight_phases_basic(self):
        alt_std = P('Altitude STD', np.ma.arange(10))
        roc = RateOfClimbForFlightPhases()
        roc.derive(alt_std)
        expected = np.ma.array(data=[60]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(roc.array, expected)

    def test_rate_of_climb_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        roc = RateOfClimbForFlightPhases()
        roc.derive(alt_std)
        expected = np.ma.array(data=[0]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(roc.array, expected)

        
class TestRateOfTurn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = RateOfTurn.get_operational_combinations()
        self.assertEqual(opts, expected)
       
    def test_rate_of_turn(self):
        rot = RateOfTurn()
        rot.derive(P('Heading Continuous', np.ma.array(range(10))))
        answer = np.ma.array(data=[1]*10, dtype=np.float)
        np.testing.assert_array_equal(rot.array, answer) # Tests data only; NOT mask
       
    def test_rate_of_turn_phase_stability(self):
        params = {'Heading Continuous':Parameter('', np.ma.array([0,0,0,1,0,0,0], 
                                                               dtype=float))}
        rot = RateOfTurn()
        rot.derive(P('Heading Continuous', np.ma.array([0,0,0,1,0,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)