try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import mock

import utilities.masked_array_testutils as ma_test
from utilities.struct import Struct
from analysis.settings import GRAVITY, HYSTERESIS_FPIAS
from analysis.node import Attribute, A, KPV, KTI, Parameter, P, Section, S
from analysis.flight_phase import Fast

from analysis.derived_parameters import (AccelerationVertical,
                                         AccelerationForwards,
                                         AccelerationSideways,
                                         AirspeedForFlightPhases,
                                         AltitudeAALForFlightPhases,
                                         AltitudeForFlightPhases,
                                         AltitudeRadio,
                                         AltitudeRadioForFlightPhases,
                                         AltitudeSTD,
                                         AltitudeTail,
                                         ClimbForFlightPhases,
                                         Eng_N1Avg,
                                         Eng_N1Min,
                                         Eng_N2Avg,
                                         FuelQty,
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
            acc_norm=Parameter('Acceleration Normal', np.ma.ones(8),8),
            acc_lat=Parameter('Acceleration Lateral', np.ma.zeros(4),4),
            acc_long=Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            pitch=Parameter('Pitch', np.ma.zeros(2),2),
            roll=Parameter('Roll', np.ma.zeros(2),2))
        
        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([1]*8))
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([1]*8))

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([1]*8))


class TestAccelerationForwards(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal',
                    'Acceleration Longitudinal', 'Pitch')]
        opts = AccelerationForwards.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_forward_level_on_gound(self):
        # Invoke the class object
        acc_fwd = AccelerationForwards(frequency=4)
                        
        acc_fwd.derive(
            acc_norm=Parameter('Acceleration Normal', np.ma.ones(8),8),
            acc_long=Parameter('Acceleration Longitudinal', np.ma.ones(4)*0.1,4),
            pitch=Parameter('Pitch', np.ma.zeros(2),2))
        
        ma_test.assert_masked_array_approx_equal(acc_fwd.array,
                                                 np.ma.array([0.1]*8))
        
    def test_acceleration_forward_pitch_up(self):
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2))

        ma_test.assert_masked_array_approx_equal(acc_fwd.array,
                                                 np.ma.array([0]*8))


class TestAccelerationSideways(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal', 'Acceleration Lateral', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationSideways.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_sideways_level_on_gound(self):
        # Invoke the class object
        acc_lat = AccelerationSideways(frequency=8)
                        
        acc_lat.derive(
            acc_norm=Parameter('Acceleration Normal', np.ma.ones(8),8),
            acc_lat=Parameter('Acceleration Lateral', np.ma.ones(4)*0.05,4),
            acc_long=Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            pitch=Parameter('Pitch', np.ma.zeros(2),2),
            roll=Parameter('Roll', np.ma.zeros(2),2))
        
        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0.05]*8))
        
    def test_acceleration_sideways_pitch_up(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2))

        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0]*8))

    def test_acceleration_sideways_roll_right(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2))

        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0]*8))

"""
-------------------------------------------------------------------------------
Superceded by Truck and Trailer analysis of airspeed during takeoff and landing
-------------------------------------------------------------------------------

class TestAccelerationForwardsForFlightPhases(unittest.TestCase):
    def test_can_operate_only_airspeed(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_can_operate_only_acceleration(self):
        expected = [('Airspeed',),
                    ('Acceleration Longitudinal',),
                    ('Acceleration Longitudinal','Airspeed')]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_accelearation_forwards_for_phases_using_acceleration(self):
        # If acceleration data is available, this is used without change.
        acc = np.ma.arange(0,0.5,0.1)
        accel_fwd = AccelerationForwardsForFlightPhases()
        accel_fwd.derive(Parameter('Acceleration Longitudinal', acc), None)
        ma_test.assert_masked_array_approx_equal(accel_fwd.array, acc)

    def test_acceleration_forwards_for_phases_using_airspeed(self):
        # If only airspeed data is available, it needs differentiating.
        speed = np.ma.arange(0,150,10)
        speed[3:5] = np.ma.masked
        accel_fwd = AccelerationForwardsForFlightPhases()
        accel_fwd.derive(None, Parameter('Airspeed', speed))
        expected = np.ma.array([0.5241646]*15)
        ma_test.assert_masked_array_approx_equal(accel_fwd.array, expected)

    def test_acceleration_forwards_for_phases_mask_repair(self):
        # Show that the mask is repaired in case of minor corruption.
        acc = np.ma.arange(0,0.5,0.1)
        acc[1:4] = np.ma.masked
        accel_fwd = AccelerationForwardsForFlightPhases()
        accel_fwd.derive(Parameter('Acceleration Longitudinal', acc), None)
        ma_test.assert_masked_array_approx_equal(accel_fwd.array, acc)
-------------------------------------------------------------------------------
Superceded by Truck and Trailer analysis of airspeed during takeoff and landing
-------------------------------------------------------------------------------
"""

class TestAirspeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    @mock.patch('analysis.derived_parameters.hysteresis')
    def test_airspeed_for_phases_basic(self, hysteresis):
        # Avoiding testing hysteresis.
        param = mock.Mock()
        param.array = mock.Mock()
        hysteresis.return_value = mock.Mock()
        speed = AirspeedForFlightPhases()
        speed.derive(param)
        self.assertEqual(hysteresis.call_args,
                         ((param.array, HYSTERESIS_FPIAS), {}))
        self.assertEqual(speed.array, hysteresis.return_value)


class TestAltitudeAALForFlightPhases(unittest.TestCase):
    """
    THIS TEST FAILS IF THE AIRSPEED_THRESHOLD = 70  # kts IS SET
    THIS LOW VALUE WAS USED FOR HERCULES TESTING BUT SHOULD BE RESET TO 80 KTS
    WHEN WE GET THINGS WORKING
   
    DON'T PANIC !!!   
    
    """
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

    def test_altitude_AAL_for_flight_phases_to_ends_of_array(self):
        slow_and_fast_data = np.ma.array(range(90,160,10)+range(160,90,-10))
        up_and_down_data = slow_and_fast_data * 10
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = slow_and_fast_data #  Result unchanged, and running to the ends of the array.
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


class TestAltitudeSTD(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeSTD.get_operational_combinations(),
          [('Altitude STD High', 'Altitude STD Low'),
           ('Altitude STD Rough', 'Inertial Vertical Speed'),
           ('Altitude STD High', 'Altitude STD Low', 'Altitude STD Rough'),
           ('Altitude STD High', 'Altitude STD Low', 'Inertial Vertical Speed'),
           ('Altitude STD High', 'Altitude STD Rough',
            'Inertial Vertical Speed'),
           ('Altitude STD Low', 'Altitude STD Rough',
            'Inertial Vertical Speed'),
           ('Altitude STD High', 'Altitude STD Low', 'Altitude STD Rough',
            'Inertial Vertical Speed')])
    
    def test__high_and_low(self):
        high_values = np.ma.array([15000, 16000, 17000, 18000, 19000, 20000,
                                   19000, 18000, 17000, 16000],
                                  mask=[False] * 9 + [True])
        low_values = np.ma.array([15500, 16500, 17500, 17800, 17800, 17800,
                                  17800, 17800, 17500, 16500],
                                 mask=[False] * 8 + [True] + [False])
        alt_std_high = Parameter('Altitude STD High', high_values)
        alt_std_low = Parameter('Altitude STD Low', low_values)
        alt_std = AltitudeSTD()
        result = alt_std._high_and_low(alt_std_high, alt_std_low)
        ma_test.assert_equal(result,
                             np.ma.masked_array([15500, 16500, 17375, 17980, 19000,
                                                 20000, 19000, 17980, 17375, 16500],
                                                mask=[False] * 8 + 2 * [True]))
    
    @mock.patch('analysis.derived_parameters.first_order_lag')
    def test__rough_and_ivv(self, first_order_lag):
        alt_std = AltitudeSTD()
        alt_std_rough = Parameter('Altitude STD Rough',
                                  np.ma.array([60, 61, 62, 63, 64, 65],
                                              mask=[False] * 5 + [True]))
        first_order_lag.side_effect = lambda arg1, arg2, arg3: arg1
        ivv = Parameter('Inertial Vertical Speed',
                        np.ma.array([60, 120, 180, 240, 300, 360],
                                    mask=[False] * 4 + [True] + [False]))
        result = alt_std._rough_and_ivv(alt_std_rough, ivv)
        ma_test.assert_equal(result,
                             np.ma.masked_array([61, 63, 65, 67, 0, 0],
                                                mask=[False] * 4 + [True] * 2))
    
    def test_derive(self):
        alt_std = AltitudeSTD()
        # alt_std_high and alt_std_low passed in.
        alt_std._high_and_low = mock.Mock()
        high_and_low_array = 3
        alt_std._high_and_low.return_value = high_and_low_array
        alt_std_high = 1
        alt_std_low = 2
        alt_std.derive(alt_std_high, alt_std_low, None, None)
        self.assertEqual(alt_std._high_and_low.call_args, ((alt_std_high, 
                                                            alt_std_low), {}))
        self.assertEqual(alt_std.array, high_and_low_array)
        # alt_std_rough and ivv passed in.
        rough_and_ivv_array = 6
        alt_std._rough_and_ivv = mock.Mock()
        alt_std._rough_and_ivv.return_value = rough_and_ivv_array
        alt_std_rough = 4        
        ivv = 5
        alt_std.derive(None, None, alt_std_rough, ivv)
        self.assertEqual(alt_std._rough_and_ivv.call_args,
                         ((alt_std_rough, ivv), {}))
        self.assertEqual(alt_std.array, rough_and_ivv_array)
        # All parameters passed in (improbable).
        alt_std.derive(alt_std_high, alt_std_low, alt_std_rough, ivv)
        self.assertEqual(alt_std.array, high_and_low_array)


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
   

class TestEng_N1Avg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N1Avg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N1',))
        self.assertEqual(opts[-1], ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
        
    
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N1Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )

class TestEng_N1Max(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N1Max.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N1',))
        self.assertEqual(opts[-1], ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
  
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,10])
        )
        
        
class TestEng_N1Min(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N1Min.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N1',))
        self.assertEqual(opts[-1], ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
  
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )
        
        
class TestEng_N2Avg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N2Avg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N2',))
        self.assertEqual(opts[-1], ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
        
    
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N2Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestFuelQty(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FuelQty.get_operational_combinations(),
          [('Fuel Qty (1)',), ('Fuel Qty (2)',), ('Fuel Qty (3)',),
           ('Fuel Qty (1)', 'Fuel Qty (2)'), ('Fuel Qty (1)', 'Fuel Qty (3)'),
           ('Fuel Qty (2)', 'Fuel Qty (3)'), ('Fuel Qty (1)', 'Fuel Qty (2)',
                                              'Fuel Qty (3)')])
    
    def test_derive(self):
        fuel_qty1 = P('Fuel Qty (1)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (2)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18]))
        # Works without all parameters.
        fuel_qty_node.derive(fuel_qty1, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))
                         
        
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
        self.assertEqual(RateOfClimb.get_operational_combinations(),
                         [('Altitude STD',),
                          ('Acceleration Vertical', 'Altitude STD'),
                          ('Altitude STD', 'Altitude Radio'),
                          ('Acceleration Vertical', 'Altitude STD',
                           'Altitude Radio')])
        
    def test_rate_of_climb_basic(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10))
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)

    def test_rate_of_climb_alt_std_only(self):
        az = None
        alt_std = P('Altitude STD', np.ma.arange(100,200,10))
        alt_rad = None
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[600]*10, dtype=np.float,
                             mask=False) #  10 ft/sec = 600 fpm
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
        # ----------------------------------------------------------------------
        # NOTE: The results of this test are dependent upon the settings
        # parameters GRAVITY = 32.2, RATE_OF_CLIMB_LAG_TC = 6.0,
        # AZ_WASHOUT_TC = 60.0. Changes in any of these will result in a test
        # failure and recomputation of the result array will be necessary.
        # ----------------------------------------------------------------------
        
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
        parabola *= 1.0 #  Allows you to make the values different for debug.
        alt_rad.array[2:] += parabola
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[-3.47482043e-11, -3.47482043e-11, 
                                2.74634456e+01, 8.69420194e+01, 1.45600433e+02,
                                2.03579748e+02, 2.60999077e+02, 3.17958965e+02, 
                                3.74544254e+02, 4.30826495e+02, 4.86866004e+02,
                                5.42713590e+02, 5.98412024e+02, 6.53997276e+02,
                                7.09499568e+02, 7.64944261e+02, 8.20352606e+02, 
                                8.75742379e+02, 9.31128421e+02, 9.86523090e+02, 
                                1.04193665e+03, 1.09737759e+03, 1.15285292e+03, 
                                1.20836836e+03, 1.26392858e+03, 1.31953737e+03,
                                1.37519774e+03, 1.43091206e+03, 1.48668218e+03, 
                                1.54250948e+03], mask=False)
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