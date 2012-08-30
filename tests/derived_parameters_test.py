import unittest
import sys
from datetime import datetime, timedelta
import mock
import numpy as np
import os

import utilities.masked_array_testutils as ma_test
from analysis_engine.settings import GRAVITY_IMPERIAL, METRES_TO_FEET
from analysis_engine.node import Attribute, A, KPV, KeyTimeInstance, KTI, Parameter, P, Section, S
from analysis_engine.flight_phase import Approach, Fast
from analysis_engine.library import np_ma_masked_zeros_like
from flight_phase_test import buildsection, buildsections
from analysis_engine.plot_flight import plot_parameter
from hdfaccess.file import hdf_file

from analysis_engine.derived_parameters import (
    AccelerationVertical,
    AccelerationForwards,
    AccelerationSideways,
    AccelerationAlongTrack,
    AccelerationAcrossTrack,
    AirspeedForFlightPhases,
    AirspeedReference,
    AirspeedRelative,
    AirspeedTrue,
    AltitudeAAL,
    AltitudeAALForFlightPhases,
    #AltitudeForFlightPhases,
    AltitudeRadio,
    #AltitudeRadioForFlightPhases,
    #AltitudeSTD,
    AltitudeTail,
    ClimbForFlightPhases,
    Configuration,
    ControlColumn,
    ControlColumnForce,
    ControlColumnForceCapt,
    ControlColumnForceFO,
    ControlWheel,
    DescendForFlightPhases,
    DistanceTravelled,
    DistanceToLanding,
    Eng_N1Avg,
    Eng_N1Max,
    Eng_N1Min,
    Eng_N2Avg,
    Eng_N2Max,
    Eng_N2Min,
    Eng_N3Avg,
    Eng_N3Max,
    Eng_N3Min,
    Flap,
    FuelQty,
    GrossWeightSmoothed,
    GroundspeedAlongTrack,
    HeadingContinuous,
    HeadingIncreasing,
    HeadingTrue,
    Headwind,
    ILSFrequency,
    LatitudePrepared,
    LongitudePrepared,
    Mach,
    Pitch,
    VerticalSpeed,
    VerticalSpeedForFlightPhases,
    RateOfTurn,
    V2,
    WindAcrossLandingRunway,
)

debug = sys.gettrace() is not None

class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal', 'Acceleration Lateral', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)
                        
        acc_vert.get_derived([
            Parameter('Acceleration Normal', np.ma.ones(8),8),
            Parameter('Acceleration Lateral', np.ma.zeros(4),4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            Parameter('Pitch', np.ma.zeros(2),2),
            Parameter('Roll', np.ma.zeros(2),2)
        ])
        
        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([1]*8))
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2)
        ])

        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([1]*8))

    def test_acceleration_vertical_pitch_up_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.8,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.2),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.3,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.ones(2)*20,2)])

        ma_test.assert_masked_array_approx_equal(acc_vert.array,
                                                 np.ma.array([0.86027777]*8))

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2)
        ])

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
                        
        acc_fwd.get_derived([
            Parameter('Acceleration Normal', np.ma.ones(8),8),
            Parameter('Acceleration Longitudinal', np.ma.ones(4)*0.1,4),
            Parameter('Pitch', np.ma.zeros(2),2)
        ])
        
        ma_test.assert_masked_array_approx_equal(acc_fwd.array,
                                                 np.ma.array([0.1]*8))
        
    def test_acceleration_forward_pitch_up(self):
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2)
        ])

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
                        
        acc_lat.get_derived([
            Parameter('Acceleration Normal', np.ma.ones(8),8),
            Parameter('Acceleration Lateral', np.ma.ones(4)*0.05,4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            Parameter('Pitch', np.ma.zeros(2),2),
            Parameter('Roll', np.ma.zeros(2),2)
        ])
        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0.05]*8))
        
    def test_acceleration_sideways_pitch_up(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2)
        ])
        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0]*8))

    def test_acceleration_sideways_roll_right(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2)
        ])
        ma_test.assert_masked_array_approx_equal(acc_lat.array,
                                                 np.ma.array([0]*8))

        
class TestAccelerationAcrossTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAcrossTrack.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_across_side_only(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8), 8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1, 4),
            Parameter('Drift', np.ma.zeros(2), 2)])
        ma_test.assert_masked_array_approx_equal(acc_across.array,
                                                 np.ma.array([0.1]*8))
        
    def test_acceleration_across_resolved(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.5,4),
            P('Drift',np.ma.ones(2)*30.0,2)])

        ma_test.assert_masked_array_approx_equal(acc_across.array,
                                                 np.ma.array([0]*8))


class TestAccelerationAlongTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_along_forward_only(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8)*0.2,8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1,4),
            Parameter('Drift', np.ma.zeros(2),2)])
        
        ma_test.assert_masked_array_approx_equal(acc_along.array,
                                                 np.ma.array([0.2]*8))
        
    def test_acceleration_along_resolved(self):
        acc_across = AccelerationAlongTrack()
        acc_across.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.1,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.2,4),
            P('Drift',np.ma.ones(2)*10.0,2)])

        ma_test.assert_masked_array_approx_equal(acc_across.array,
                                                 np.ma.array([0.13321041]*8))


class TestAirspeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)


class TestAirspeedReference(unittest.TestCase):
    def setUp(self):
        self.approach_slice = slice(105, 120)
        apps = S('Approach', items=(Section(name='Approach',
                                            slice=self.approach_slice,
                                            start_edge=104.5,
                                            stop_edge=119.5),))
        self.default_kwargs = {'spd':False,
                               'gw':None,
                               'flap':None,
                               'conf':None,
                               'vapp':None,
                               'vref':None,
                               'fdr_vapp':None,
                               'fdr_vref':None,
                               'apps':apps,
                               'series':None,
                               'family':None}


    def test_can_operate(self):
        # TODO: test expected combinations are in get_operational_combinations
        expected = [('Airspeed', 'FDR Vref')]
        opts = AirspeedReference.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_airspeed_reference__fdr_vapp(self):

        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['fdr_vapp'] = A('FDR Vapp', value=120)

        param = AirspeedReference()
        param.derive(**kwargs)
        expected = np.ma.zeros(128)
        expected.mask = True
        expected[self.approach_slice] = 120
        np.testing.assert_array_equal(param.array, expected)

    def test_airspeed_reference__fdr_vref(self):
        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['fdr_vref'] = A('FDR Vref', value=120)

        param = AirspeedReference()
        param.derive(**kwargs)
        expected = np.ma.zeros(128)
        expected.mask = True
        expected[self.approach_slice] = 120
        np.testing.assert_array_equal(param.array, expected)


    def test_airspeed_reference__recorded_vapp(self):
        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['vapp'] = P('Vapp', np.ma.array([120]*128))

        param = AirspeedReference()
        param.derive(**kwargs)

        expected=np.array([120]*128)
        np.testing.assert_array_equal(param.array, expected)

    def test_airspeed_reference__recorded_vref(self):
        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['vref'] = P('Vref', np.ma.array([120]*128))

        param = AirspeedReference()
        param.derive(**kwargs)

        expected=np.array([120]*128)
        np.testing.assert_array_equal(param.array, expected)

    def test_airspeed_reference__boeing_lookup(self):
        with hdf_file('test_data/airspeed_reference.hdf5') as hdf:
            approaches = (Section(name='Approach', slice=slice(3346, 3540), start_edge=3345.5, stop_edge=3539.5),
                          Section(name='Approach', slice=slice(5502, 5795), start_edge=5501.5, stop_edge=5794.5))
            args = [
                P(**hdf['Airspeed'].__dict__),
                P(**hdf['Gross Weight Smoothed'].__dict__),
                P(**hdf['Flap'].__dict__),
                None,
                None,
                None,
                None,
                None,
                S('Approach', items=approaches),
                A('Series', value='B737-300'),
                A('Family', value='B737 Classic'),
            ]
            param = AirspeedReference()
            param.get_derived(args)
            expected = np.ma.load('test_data/boeing_reference_speed.ma')
            np.testing.assert_array_equal(param.array, expected.array)

    def test_airspeed_reference__airbus_lookup(self):
        #with hdf_file('test_data/airspeed_reference.hdf5') as hdf:
            #approaches = (Section(name='Approach', slice=slice(3346, 3540, None), start_edge=3345.5, stop_edge=3539.5),
                          #Section(name='Approach', slice=slice(5502, 5795, None), start_edge=5501.5, stop_edge=5794.5))
            #args = [
                #P(**hdf['Airspeed'].__dict__),
                #P(**hdf['Gross Weight Smoothed'].__dict__),
                #P(**hdf['Flap'].__dict__),
                #None,
                #None,
                #None,
                #None,
                #None,
                #S('Approach', items=approaches),
                #A('Series', value='B737-300'),
                #A('Family', value='B737 Classic'),
            #]
            #param = AirspeedReference()
            #param.get_derived(args)
            #expected = np.ma.load('test_data/boeing_reference_speed.ma')
            #np.testing.assert_array_equal(param.array, expected.array)
        self.assertTrue(False, msg='Test Not implemented')


class TestAirspeedRelative(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','FDR Vref')]
        opts = AirspeedRelative.get_operational_combinations()
        self.assertEqual(opts, expected)
        
        # ???????????????????????????????????????????????????????????????
        # THIS MAY NEED TO BE ALTERED SO THAT Vref IS VARIABLE AND NOT FIXED
        # NEED A DIFFERENT Vref FOR EACH APPROACH ??? DISCUSS WITH DEREK AND
        # DAVE BEFORE CHANGING
    
    def test_airspeed_for_phases_basic(self):
        speed=P('Airspeed',np.ma.array([200]*128),frequency=1)
        ref = A('Vref',value=120)
        # Offset is frame-related, not superframe based, so is to some extent
        # meaningless.
        param = AirspeedRelative()
        param.get_derived([speed, ref])
        expected=np.array([80]*128)
        np.testing.assert_array_equal(param.array, expected)

class TestAirspeedTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AirspeedTrue.get_operational_combinations(), [
            ('Airspeed', 'Altitude STD'),
            ('Airspeed', 'Altitude STD', 'TAT'),
            ('Airspeed', 'Altitude STD', 'Takeoff'),
            ('Airspeed', 'Altitude STD', 'Landing'),
            ('Airspeed', 'Altitude STD', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 'Groundspeed', 'Acceleration Forwards')
        ])
        
    def test_tas_basic(self):
        cas = P('Airspeed', np.ma.array([100,200,300]))
        alt = P('Altitude STD', np.ma.array([0,20000,40000]))
        tat = P('TAT', np.ma.array([20,-10,-16.2442]))
        tas = AirspeedTrue()
        tas.derive(cas,alt,tat)
        # Answers with compressibility are:
        result = [100.6341, 273.0303, 552.8481]
        self.assertLess(abs(tas.array.data[0]-result[0]),0.1)
        self.assertLess(abs(tas.array.data[1]-result[1]),0.7)
        self.assertLess(abs(tas.array.data[2]-result[2]),6.0)
        
    def test_tas_masks(self):
        cas = P('Airspeed', np.ma.array([100,200,300]))
        alt = P('Altitude STD', np.ma.array([0,20000,40000]))
        tat = P('TAT', np.ma.array([20,-10,-40]))
        tas = AirspeedTrue()
        cas.array[0]=np.ma.masked
        alt.array[1]=np.ma.masked
        tat.array[2]=np.ma.masked
        tas.derive(cas,alt,tat)
        np.testing.assert_array_equal(tas.array.mask,[True]*3)
        
    def test_tas_no_tat(self):
        cas = P('Airspeed', np.ma.array([100,200,300]))
        alt = P('Altitude STD', np.ma.array([0,10000,20000]))
        tas = AirspeedTrue()
        tas.derive(cas,alt,None)
        result = [100.000, 231.575, 400.097]
        self.assertLess(abs(tas.array.data[0]-result[0]),0.01)
        self.assertLess(abs(tas.array.data[1]-result[1]),0.01)
        self.assertLess(abs(tas.array.data[2]-result[2]),0.01)
        

class TestAltitudeAAL(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeAAL.get_operational_combinations()
        self.assertTrue(('Altitude STD','Fast') in opts)
        self.assertTrue(('Altitude Radio','Altitude STD','Fast') in opts)
        
    def test_alt_AAL_basic(self):
        data = np.ma.array([-3,0,30,80,150,260,120,70,20,-5])
        alt_std = P(array=data+300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100]*10)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, phase_fast)
        expected = np.ma.array([0,0,30,80,150,260,120,70,20,0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)

    def test_alt_AAL_bounce_rejection(self):
        data = np.ma.array([-3,0,30,80,150,260,120,70,20,-5,2,5,2,-3,-3])
        alt_std = P(array=data+300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100]*15)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, phase_fast)
        expected = np.ma.array([0,0,30,80,150,260,120,70,20,0,0,0,0,0,0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)
    
    def test_alt_AAL_no_ralt(self):
        data = np.ma.array([-3,0,30,80,150,280,120,70,20,-5])
        alt_std = P(array=data+300)
        slow_and_fast_data = np.ma.array([70]+[85]*7+[75,70])
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, phase_fast)
        expected = np.ma.array([0,0,30,80,150,210,50,0,0,0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)
    
    def test_alt_aal_complex(self):
        testwave = np.ma.cos(np.arange(0,3.14*2*5,0.1))*(-3000)+\
            np.ma.cos(np.arange(0,3.14*2,0.02))*(-5000)+7996
        # plot_parameter (testwave)
        rad_wave = np.copy(testwave)
        rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        # plot_parameter (rad_data)
        phase_fast = buildsection('Fast',0,len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', rad_data),
                       P('Altitude STD', testwave),
                       phase_fast)
        # plot_parameter (alt_aal.array)
        
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34],7013,decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60],3308,decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124],217,decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191],8965,decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254],3288,decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[313],17,decimal=0)
        
    
class TestAltitudeAALForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL',)]
        opts = AltitudeAALForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_AAL_for_flight_phases_basic(self):
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude AAL', 
                                  np.ma.array(data=[0,100,200,100,0],
                                              mask=[0,0,1,1,0])))
        expected = np.ma.array(data=[0,100,66,33,0],mask=False)
        # ...because data interpolates across the masked values and integer
        # values are rounded.
        ma_test.assert_array_equal(alt_4_ph.array, expected)



'''
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
        answer = np.ma.array(data=[50.0]*3+
                             list(testwave[3:6])+
                             [np.ma.max(testwave)-100.0]*21+
                             list(testwave[27:39])+
                             [testwave[-1]-50.0]*21,
                             mask = False)
        np.testing.assert_array_almost_equal(alt_4_ph.array, answer)
        '''


class TestAltitudeRadio(unittest.TestCase):
    """
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch',
                     'Main Gear To Altitude Radio')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
    """
    
    def test_altitude_radio_737_3C(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Attribute('Frame','737-3C'), 
                       None,
                       Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5,  0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.25, 1.0),
                       Parameter('Altitude Radio (C)',
                                 np.ma.array([30.0,30.0,30.0,30.0,30.3]), 0.25, 3.0),
                       None
                       )
        result = alt_rad.array
        answer = np.ma.array(data=[25.0]*7+[25.05,25.175,25.25])
        ma_test.assert_array_almost_equal(alt_rad.array, answer)
        self.assertEqual(alt_rad.offset,1.0)
        self.assertEqual(alt_rad.frequency,0.5)

    def test_altitude_radio_737_5_EFIS(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Attribute('Frame','737-5'), 
                       Attribute('Frame Qualifier','Altitude_Radio_EFIS'),
                       Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       )
        result = alt_rad.array
        answer = np.ma.array(data=[15.0]*7+[15.025,15.1,15.15])
        ma_test.assert_array_almost_equal(alt_rad.array, answer)
        self.assertEqual(alt_rad.offset,0.0)
        self.assertEqual(alt_rad.frequency,1.0)

    def test_altitude_radio_737_5_Analogue(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Attribute('Frame','737-5'), 
                       Attribute('Frame Qualifier','Altitude_Radio_ARINC_552'),
                       Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       )
        result = alt_rad.array
        answer = np.ma.array(data=[15.0]*7+[15.025,15.1,15.15])
        ma_test.assert_array_almost_equal(alt_rad.array, answer)
        self.assertEqual(alt_rad.offset,0.0)
        self.assertEqual(alt_rad.frequency,1.0)

'''
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
        '''


"""
class TestAltitudeQNH(unittest.TestCase):
    # Needs airport database entries simulated. TODO.

"""    
    
'''
class TestAltitudeSTD(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeSTD.get_operational_combinations(),
          [('Altitude STD Coarse', 'Altitude STD Fine'),
           ('Altitude STD Coarse', 'Vertical Speed')])
    
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
    
    @mock.patch('analysis_engine.derived_parameters.first_order_lag')
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
        alt_std._high_and_low.assert_called_once_with(alt_std_high, alt_std_low)
        self.assertEqual(alt_std.array, high_and_low_array)
        # alt_std_rough and ivv passed in.
        rough_and_ivv_array = 6
        alt_std._rough_and_ivv = mock.Mock()
        alt_std._rough_and_ivv.return_value = rough_and_ivv_array
        alt_std_rough = 4        
        ivv = 5
        alt_std.derive(None, None, alt_std_rough, ivv)
        alt_std._rough_and_ivv.assert_called_once_with(alt_std_rough, ivv)
        self.assertEqual(alt_std.array, rough_and_ivv_array)
        # All parameters passed in (improbable).
        alt_std.derive(alt_std_high, alt_std_low, alt_std_rough, ivv)
        self.assertEqual(alt_std.array, high_and_low_array)
        '''


class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch',
                     'Ground To Lowest Point Of Tail',
                     'Main Gear To Lowest Point Of Tail')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.zeros(10), 1,0.0),
                    Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                    Attribute('Ground To Lowest Point Of Tail', 10.0/METRES_TO_FEET),
                    Attribute('Main Gear To Lowest Point Of Tail', 35.0/METRES_TO_FEET))
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

    def test_altitude_tail_after_lift(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.array([5])),
                    Parameter('Pitch', np.ma.array([18])),
                    Attribute('Ground To Lowest Point Of Tail', 10.0/METRES_TO_FEET),
                    Attribute('Main Gear To Lowest Point Of Tail', 35.0/METRES_TO_FEET))
        result = talt.array
        # Lift 5ft
        answer = np.ma.array(data=[5.0-0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)

class TestClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Fast')]
        opts = ClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_climb_for_flight_phases_basic(self):
        up_and_down_data = np.ma.array([0,0,2,5,3,2,5,6,8,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0,0,2,5,0,0,3,4,6,0])
        ma_test.assert_masked_array_approx_equal(climb.array, expected)
   
   

class TestConfiguration(unittest.TestCase):
    
    def setUp(self):
        # last state is invalid
        s = np.ma.array([0]*2 + [16]*4 + [20]*4 + [23]*6 + [16])
        self.slat = P('Slat', np.tile(s, 10000)) # 23 long
        f = np.ma.array([0]*4 + [8]*4 + [14]*4 + [22]*2 + [32]*2 + [14])
        self.flap = P('Flap', np.tile(f, 10000))
        a = np.ma.array([0]*4 + [5]*2 + [10]*10 + [10])
        self.ails = P('Aileron', np.tile(a, 10000))
        
    def test_can_operate(self):
        expected = [('Flap','Slat', 'Series', 'Family'),
                    ('Flap','Slat', 'Aileron', 'Series', 'Family')]
        opts = Configuration.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_conf_for_a330(self):
        # last state is invalid
        conf = Configuration()
        conf.derive(self.flap, self.slat, self.ails, 
                      A('','A330-301'), A('','A330'))
        self.assertEqual(list(np.ma.filled(conf.array[:17], fill_value=-999)),
                         [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,-999]
                         )
        
    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_conf_for_a330)
        time = min(timer.repeat(1, 1))
        print "Time taken %s secs" % time
        self.assertLess(time, 0.1, msg="Took too long")


class TestControlColumn(unittest.TestCase):

    def setUp(self):
        ccc = np.ma.array(data=[])
        self.ccc = P('Control Column (Capt)', ccc)
        ccf = np.ma.array(data=[])
        self.ccf = P('Control Column (FO)', ccf)

    def test_can_operate(self):
        expected = [('Control Column (Capt)', 'Control Column (FO)')]
        opts = ControlColumn.get_operational_combinations()
        self.assertEqual(opts, expected)

    @mock.patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_column(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cc = ControlColumn()
        cc.derive(self.ccc, self.ccf)
        blend_two_parameters.assert_called_once_with(self.ccc, self.ccf)


class TestControlColumnForce(unittest.TestCase):

    def setUp(self):
        ccff = np.ma.arange(1, 4)
        self.ccff = P('Control Column Force (Capt)', ccff)
        ccfl = np.ma.arange(1, 4)
        ccfl[-1:] = np.ma.masked
        self.ccfl = P('Control Column Force (FO)', ccfl)

    def test_can_operate(self):
        expected = [('Control Column Force (Capt)',
                     'Control Column Force (FO)')]
        opts = ControlColumnForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force(self):
        ccf = ControlColumnForce()
        ccf.derive(self.ccff, self.ccfl)
        result = ccf.array
        answer = np.ma.array(data=[2, 4, 6], mask=[False, False, True])
        np.testing.assert_array_almost_equal(result, answer)


class TestControlColumnForceCapt(unittest.TestCase):

    def setUp(self):
        ccfl = np.ma.arange(0, 16)
        self.ccfl = P('Control Column Force (Local)', ccfl)
        ccff = ccfl[-1::-1]
        self.ccff = P('Control Column Force (Foreign)', ccff)
        fcc = np.repeat(np.ma.arange(0, 4), 4)
        self.fcc = P('FCC Local Limited Master', fcc)

    def test_can_operate(self):
        expected = [('Control Column Force (Local)',
                     'Control Column Force (Foreign)',
                     'FCC Local Limited Master')]
        opts = ControlColumnForceCapt.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force_capt(self):
        ccfc = ControlColumnForceCapt()
        ccfc.derive(self.ccfl, self.ccff, self.fcc)
        result = ccfc.array
        answer = self.ccfl.array
        answer[4:8] = self.ccff.array[4:8]
        np.testing.assert_array_almost_equal(result, answer)


class TestControlColumnForceFO(unittest.TestCase):

    def setUp(self):
        ccfl = np.ma.arange(0, 16)
        self.ccfl = P('Control Column Force (Local)', ccfl)
        ccff = ccfl[-1::-1]
        self.ccff = P('Control Column Force (Foreign)', ccff)
        fcc = np.repeat(np.ma.arange(0, 4), 4)
        self.fcc = P('FCC Local Limited Master', fcc)

    def test_can_operate(self):
        expected = [('Control Column Force (Local)',
                     'Control Column Force (Foreign)',
                     'FCC Local Limited Master')]
        opts = ControlColumnForceFO.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force_fo(self):
        ccff = ControlColumnForceFO()
        ccff.derive(self.ccfl, self.ccff, self.fcc)
        result = ccff.array
        answer = self.ccff.array
        answer[4:8] = self.ccfl.array[4:8]
        np.testing.assert_array_almost_equal(result, answer)


class TestControlWheel(unittest.TestCase):

    def setUp(self):
        cwc = np.ma.array(data=[])
        self.cwc = P('Control Wheel (Capt)', cwc)
        cwf = np.ma.array(data=[])
        self.cwf = P('Control Wheel (FO)', cwf)

    def test_can_operate(self):
        expected = [('Control Wheel (Capt)', 'Control Wheel (FO)')]
        opts = ControlWheel.get_operational_combinations()
        self.assertEqual(opts, expected)

    @mock.patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_wheel(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cw = ControlWheel()
        cw.derive(self.cwc, self.cwf)
        blend_two_parameters.assert_called_once_with(self.cwc, self.cwf)


class TestDescendForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Airborne')]
        opts = DescendForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_descend_for_flight_phases_basic(self):
        down_and_up_data = np.ma.array([0,0,12,5,3,12,15,10,7,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        descend = DescendForFlightPhases()
        descend.derive(Parameter('Altitude STD', down_and_up_data ), phase_fast)
        expected = np.ma.array([0,0,0,-7,-9,0,0,-5,-8,0])
        ma_test.assert_masked_array_approx_equal(descend.array, expected)

        
class TestDistanceToLanding(unittest.TestCase):
    
    def test_can_operate(self):
        expected = [('Distance Travelled', 'Touchdown')]
        opts = DistanceToLanding.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_derive(self):
        distance_travelled = P('Distance Travelled', array=np.ma.arange(0, 100))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown'),
                                        KeyTimeInstance(95, 'Touchdown')])
        
        expected_result = np.ma.concatenate((np.ma.arange(95, 0, -1),np.ma.arange(0, 5, 1)))
        dtl = DistanceToLanding()
        dtl.derive(distance_travelled, tdwns)
        ma_test.assert_array_equal(dtl.array, expected_result)
        
           
class TestDistanceTravelled(unittest.TestCase):
    
    def test_can_operate(self):
        expected = [('Groundspeed',)]
        opts = DistanceTravelled.get_operational_combinations()
        self.assertEqual(opts, expected)

    @mock.patch('analysis_engine.derived_parameters.integrate')
    def test_derive(self, integrate_patch):
        
        gndspeed = mock.Mock()
        gndspeed.array = mock.Mock()
        gndspeed.frequency = mock.Mock()
        DistanceTravelled().derive(gndspeed)
        integrate_patch.assert_called_once_with(gndspeed.array, gndspeed.frequency, scale=1.0)
     


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
                      11,12,13,14,15,16,17,18,9])
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

class TestEng_N2Max(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N2Max.get_operational_combinations()
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
        eng = Eng_N2Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )
        
        
class TestEng_N2Min(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N2Min.get_operational_combinations()
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
        eng = Eng_N2Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )
        
        
class TestEng_N3Avg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N3Avg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N3',))
        self.assertEqual(opts[-1], ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
        
    
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N3Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )

class TestEng_N3Max(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N3Max.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N3',))
        self.assertEqual(opts[-1], ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
  
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Max()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )
        
        
class TestEng_N3Min(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N3Min.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) N3',))
        self.assertEqual(opts[-1], ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!
  
    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and 
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Min()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )
        
        
class TestFlap(unittest.TestCase):
    def test_can_operate(self):
        opts = Flap.get_operational_combinations()
        self.assertEqual(opts, [('Flap Surface', 'Series', 'Family'),
                                ])
        
    def test_flap_stepped_nearest_5(self):
        flap = P('Flap Surface', np.ma.array(range(50)))
        fstep = Flap()
        fstep.derive(flap, A('Series', None), A('Family', None))
        self.assertEqual(list(fstep.array[:15]), 
                         [0,0,0,5,5,5,5,5,10,10,10,10,10,15,15])
        self.assertEqual(list(fstep.array[-7:]), [45]*5 + [50]*2)

        # test with mask
        flap = P('Flap Surface', np.ma.array(range(20), mask=[True]*10 + [False]*10))
        fstep.derive(flap, A('Series', None), A('Family', None))
        self.assertEqual(list(np.ma.filled(fstep.array, fill_value=-1)),
                         [-1]*10 + [10,10,10,15,15,15,15,15,20,20])
        
    def test_flap_using_md82_settings(self):
        # MD82 has flaps (0, 11, 15, 28, 40)
        flap = P('Flap Surface', np.ma.array(range(50) + range(-5,0) + [13.1,1.3,10,10]))
        flap.array[1] = np.ma.masked
        flap.array[57] = np.ma.masked
        flap.array[58] = np.ma.masked
        fstep = Flap()
        fstep.derive(flap, A('Series', None), A('Family', 'MD80'))
        self.assertEqual(len(fstep.array), 59)
        self.assertEqual(
            list(np.ma.filled(fstep.array, fill_value=-999)), 
            [0,-999,0,0,0,0, # 0 -> 5.5
             11,11,11,11,11,11,11,11, # 6 -> 13.5
             15,15,15,15,15,15,15,15, # 14 -> 21
             28,28,28,28,28,28,28,28,28,28,28,28,28, # 22.5 -> 34
             40,40,40,40,40,40,40,40,40,40,40,40,40,40,40, # 35 -> 49
             0,0,0,0,0, # -5 -> -1
             15,0, # odd float values
             -999,-999 # masked values
             ])
        self.assertTrue(np.ma.is_masked(fstep.array[1]))
        self.assertTrue(np.ma.is_masked(fstep.array[57]))
        self.assertTrue(np.ma.is_masked(fstep.array[58]))
    
    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_flap_using_md82_settings)
        time = min(timer.repeat(2, 100))
        print "Time taken %s secs" % time
        self.assertLess(time, 1.0, msg="Took too long")
        
        
        
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


class TestGrossWeightSmoothed(unittest.TestCase):
    def test_gw_formula(self):
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*256,dtype=float),offset=0.0,frequency=1.0)
        climb = buildsection('Climbing',None,None)
        descend = buildsection('Descending',None,None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(result.array[0], 292.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_with_many_samples(self):
        weight = P('Gross Weight',np.ma.array(data=range(56400,50000,-64), 
                                              mask=False, dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*64*100,dtype=float),offset=0.0,frequency=1.0)
        climb = buildsection('Climbing',None,None)
        descend = buildsection('Descending',None,None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(result.array[1], 56400-1)
        
    def test_gw_formula_with_good_data(self):
        weight = P('Gross Weight',np.ma.array(data=[484,420,356,292,228,164,100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*64*7,dtype=float),
                      offset=0.0,frequency=1.0)
        climb = buildsection('Climbing',None,None)
        descend = buildsection('Descending',None,None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_climbing(self):
        weight = P('Gross Weight',np.ma.array(data=[484,420,356,292,228,164,100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*64*7,dtype=float),
                      offset=0.0,frequency=1.0)
        climb = buildsection('Climbing',1,4)
        descend = buildsection('Descending',None,None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_descending(self):
        weight = P('Gross Weight',np.ma.array(data=[484,420,356,292,228,164,100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*64*7,dtype=float),
                      offset=0.0,frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = buildsection('Climbing',None,None)
        descend = buildsection('Descending',3,5)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_one_masked_data_point(self):
        weight = P('Gross Weight',np.ma.array(data=[0],
                                              mask=[1],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([0]*64,dtype=float),
                      offset=0.0,frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = buildsection('Climbing',None,None)
        descend = buildsection('Descending',None,None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend])
        self.assertEqual(len(gws.array),64)
        self.assertEqual(gws.frequency, fuel_flow.frequency)
        self.assertEqual(gws.offset, fuel_flow.offset)
        


class TestGroundspeedAlongTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Groundspeed','Acceleration Along Track', 'Altitude AAL',
                     'ILS Glideslope')]
        opts = GroundspeedAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_groundspeed_along_track_basic(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed',np.ma.array(data=[100]*2+[120]*18), frequency=1)
        accel = P('Acceleration Along Track',np.ma.zeros(20), frequency=1)
        gat.derive(gspd, accel)
        # A first order lag of 6 sec time constant rising from 100 to 120
        # will pass through 110 knots between 13 & 14 seconds after the step
        # rise.
        self.assertLess(gat.array[5],56.5)
        self.assertGreater(gat.array[6],56.5)
        
    def test_groundspeed_along_track_accel_term(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed',np.ma.array(data=[100]*200), frequency=1)
        accel = P('Acceleration Along Track',np.ma.ones(200)*.1, frequency=1)
        accel.array[0]=0.0
        gat.derive(gspd, accel)
        # The resulting waveform takes time to start going...
        self.assertLess(gat.array[4],55.0)
        # ...then rises under the influence of the lag...
        self.assertGreater(gat.array[16],56.0)
        # ...to a peak...
        self.assertGreater(np.ma.max(gat.array.data),16)
        # ...and finally decays as the longer washout time constant takes effect.
        self.assertLess(gat.array[199],52.0)
        
        
class TestHeadContinuous(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading',)]
        opts = HeadingContinuous.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_continuous(self):
        head = HeadingContinuous()
        head.derive(P('Heading',np.ma.remainder(
            np.ma.array(range(10))+355,360.0)))
        
        answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   361.0, 362.0, 363.0, 364.0],
                             dtype=np.float, mask=False)

        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(head.array.data, answer.data)


class TestHeadingIncreasing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = HeadingIncreasing.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_heading_increasing(self):
        head = P('Heading Continuous', array=np.ma.array([0.0,1.0,-2.0]),
                 frequency=0.5)
        head_inc=HeadingIncreasing()
        head_inc.derive(head)
        expected = np.ma.array([0.0, 1.0, 5.0])
        ma_test.assert_array_equal(head_inc.array, expected)
        
        
class TestLatitudeAndLongitudePrepared(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LatitudePrepared.get_operational_combinations(),
                         [('Latitude','Longitude')])

    def test_latitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,0,0,0,0,0.001],dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        self.assertGreater(smoother.array[3],0.01)
        self.assertLess(smoother.array[3],0.013)
        
    def test_latitude_smoothing_masks_static_data(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(7,dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        self.assertEqual(np.ma.count(smoother.array),0) # No non-masked values.
        
    def test_latitude_smoothing_short_array(self):
        lat = P('Latitude',np.ma.array([0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(2,dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        
    def test_longitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,-2,-4,-2,0,0],dtype=float))
        smoother = LongitudePrepared()
        smoother.get_derived([lat,lon])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        self.assertGreater(smoother.array[3],0.011)
        self.assertLess(smoother.array[3],0.012)


class TestHeadingTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(HeadingTrue.get_operational_combinations(),
            [('Heading Continuous', 'Magnetic Variation')])
        
    def test_basic(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        true = HeadingTrue()
        true.derive(head, var)
        expected = P('HeadingTrue', np.ma.array([2.0, 8.0, 358.0, 348.0, 5.0]))
        ma_test.assert_array_equal(true.array, expected.array)
                 

class TestILSFrequency(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS-VOR (1) Frequency','ILS-VOR (2) Frequency','Frame')]
        opts = ILSFrequency.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_ils_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_ils_frequency_matched(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([108.10]*3+[111.95]*3),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([108.10,111.95]*3),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2])
        expected_array = np.ma.array(
            data=[108.10,99,108.10,111.95,99,111.95], 
             mask=[False,True,False,False,True,False])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
        
class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_pitch_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float), 1,0.1),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float)+10, 1,0.6)
                  )
        answer = np.ma.array(data=([5.0,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.0]))
        combo = P('Pitch',answer,frequency=2,offset=0.1)
        ma_test.assert_array_equal(pch.array, combo.array)
        self.assertEqual(pch.frequency, combo.frequency)
        self.assertEqual(pch.offset, combo.offset)

    def test_pitch_reverse_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float)+1, 1,0.95),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float)+10, 1,0.45)
                  )
        answer = np.ma.array(data=(range(10)),mask=([1]+[0]*9))/2.0+5.0
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_error_different_rates(self):
        pch = Pitch()
        self.assertRaises(AssertionError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5),dtype=float), 2,0.1),
                          P('Pitch (2)', np.ma.array(range(10),dtype=float)+10, 4,0.6))
        
    def test_pitch_different_offsets(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5),dtype=float), 1,0.11),
                   P('Pitch (2)', np.ma.array(range(5),dtype=float), 1,0.6))
        # This originally produced an error, but with amended merge processes
        # this is not necessary. Simply check the result is the right length.
        self.assertEqual(len(pch.array),10)
        

class TestVerticalSpeed(unittest.TestCase):
    def test_can_operate(self):
        opts = VerticalSpeed.get_operational_combinations()
        self.assertEqual(len(opts), 8)
        self.assertEqual(opts[0], ('Altitude STD',))
        self.assertTrue(('Altitude STD', 'Altitude Radio') in opts)
        self.assertEqual(opts[-1],
                         ('Acceleration Vertical', 'Altitude STD', 
                          'Altitude Radio', 'Airspeed'))
                         
    def test_vertical_speed_basic(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10))
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)

    def test_vertical_speed_masked(self):
        # The blocks of masked values have to exceed the repair_mask
        # threshold of 10 samples, hence the large arrays.
        az = P('Acceleration Vertical', np.ma.array([1]*100, dtype=np.float))
        az.array[5:20]=np.ma.masked
        alt_std = P('Altitude STD', np.ma.array([100]*100, dtype=np.float))
        alt_std.array[35:50]=np.ma.masked
        alt_rad = P('Altitude Radio', np.ma.array([0]*100, dtype=np.float))
        alt_rad.array[65:80]=np.ma.masked
        vert_spd = VerticalSpeed()
        vert_spd.derive(az, alt_std, alt_rad, None)
        expected = np.ma.array(data=[0]*100, dtype=np.float,
                             mask=[[False]*5+[True]*15+[False]*15+
                                   [True]*15+[False]*15+
                                   [False]*15+[False]*15+[False]*5])
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)

    def test_vertical_speed_alt_std_only(self):
        az = None
        alt_std = P('Altitude STD', np.ma.arange(100,200,10))
        alt_rad = None
        vert_spd = VerticalSpeed()
        vert_spd.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[600]*10, dtype=np.float,
                             mask=False) #  10 ft/sec = 600 fpm
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)

    def test_vertical_speed_bump(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10,dtype=float))
        az.array[2:4] = 1.1
        # (Low acceleration for this test as the sample rate is only 1Hz).
        alt_std = P('Altitude STD', np.ma.array([100]*10,dtype=float))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10,dtype=float))
        vert_spd = VerticalSpeed()
        vert_spd.derive(az, alt_std, alt_rad)
        expected = np.ma.array(data=[0, 0, 82.11570, 221.52819, 236.30071,
                                     163.44645,	111.49595, 74.47526, 48.11727,
                                     29.37410],  mask=False)
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)

    def test_vertical_speed_combined_signals(self):
        # ----------------------------------------------------------------------
        # NOTE: The results of this test are dependent upon the settings
        # parameters GRAVITY = 32.2, VERTICAL_SPEED_LAG_TC = 6.0,
        # AZ_WASHOUT_TC = 60.0. Changes in any of these will result in a test
        # failure and recomputation of the result array will be necessary.
        # ----------------------------------------------------------------------
        
        # Initialise to 1g
        az = P('Acceleration Vertical', np.ma.array([1]*30,dtype=float))
        # After 2 seconds, increment by 1 ft/s^2
        az.array[2:] += 1/GRAVITY_IMPERIAL
        
        # This will give a linearly increasing vertical speed 0>28 ft/sec...
        # which integrated (cumcum) gives a parabolic theoretical solution.
        parabola = (np.cumsum(np.arange(0.0,28.0,1)))

        # The pressure altitude datum could be anything. Set 99ft for fun.
        alt_std = P('Altitude STD', np.ma.array([99]*30,dtype=float))
        # and add the increasing parabola 
        alt_std.array[2:] += parabola 
        alt_rad = P('Altitude Radio', np.ma.array([0]*30,dtype=float))
        parabola *= 1.0 #  Allows you to make the values different for debug.
        alt_rad.array[2:] += parabola
        
        vert_spd = VerticalSpeed()
        vert_spd.derive(az, alt_std, alt_rad)
        self.assertEqual(np.argmax(vert_spd.array), 29)
        self.assertGreater(vert_spd.array[29],1589)
        self.assertLess(vert_spd.array[29],1590)


class TestVerticalSpeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = VerticalSpeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_vertical_speed_for_flight_phases_basic(self):
        alt_std = P('Altitude STD', np.ma.arange(10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        expected = np.ma.array(data=[60]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(vert_spd.array, expected)

    def test_vertical_speed_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        expected = np.ma.array(data=[0]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(vert_spd.array, expected)

        
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
        
        
class TestMach(unittest.TestCase):
    def test_can_operate(self):
        opts = Mach.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed', 'Altitude STD')])
        
    def test_all_cases(self):
        cas = P('Airspeed', np.ma.array(data=[0, 100, 200, 200, 200, 500, 200],
                                        mask=[0,0,0,0,1,0,0], dtype=float))
        alt = P('Altitude STD', np.ma.array(data=[0, 10000, 20000, 30000, 30000, 45000, 20000],
                                        mask=[0,0,0,0,0,0,1], dtype=float))
        mach = Mach()
        mach.derive(cas, alt)
        expected = np.ma.array(data=[0, 0.182, 0.4402, 0.5407, 0.5407, 1.6825, 45000],
                                        mask=[0,0,0,0,1,1,1], dtype=float)
        ma_test.assert_masked_array_approx_equal(mach.array, expected, decimal=2)
        
class TestV2(unittest.TestCase):
    def setUp(self):
        self.default_kwargs = {'spd':False,
                               'flap':None,
                               'conf':None,
                               'fdr_v2':None,
                               'weight_liftoff':None,
                               'series':None,
                               'family':None}

    def test_can_operate(self):
        # TODO: test expected combinations are in get_operational_combinations
        expected = [('Airspeed', 'FDR Vref')]
        opts = V2.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_v2__fdr_v2(self):

        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['fdr_v2'] = A('FDR V2', value=120)

        param = V2()
        param.derive(**kwargs)
        expected = np.array([120]*128)
        np.testing.assert_array_equal(param.array, expected)

    def test_v2__boeing_lookup(self):
        gw = KPV('Gross Weight At Liftoff')
        gw.create_kpv(451, 54192.06)
        with hdf_file('test_data/airspeed_reference.hdf5') as hdf:
            args = [
                P(**hdf['Airspeed'].__dict__),
                P(**hdf['Flap'].__dict__),
                None,
                None,
                gw,
                A('Series', value='B737-300'),
                A('Family', value='B737 Classic'),
            ]
            param = V2()
            param.get_derived(args)
            expected = np.array([144.868884]*5888)
            np.testing.assert_array_equal(param.array, expected)

    def test_v2__airbus_lookup(self):
        # TODO: create airbus lookup test and add conf to test hdf file

        #with hdf_file('test_data/airspeed_reference.hdf5') as hdf:
            #approaches = (Section(name='Approach', slice=slice(3346, 3540, None), start_edge=3345.5, stop_edge=3539.5),
                          #Section(name='Approach', slice=slice(5502, 5795, None), start_edge=5501.5, stop_edge=5794.5))
            #args = [
                #P(**hdf['Airspeed'].__dict__),
                #P(**hdf['Flap'].__dict__),
                #None,
                #None,
                #KPV('Gross Weight At Liftoff'),
                #A('Series', value='B737-300'),
                #A('Family', value='B737 Classic'),
            #]
            #param = V2()
            #param.get_derived(args)
            #expected = np.ma.load('test_data/boeing_reference_speed.ma')
            #np.testing.assert_array_equal(param.array, expected.array)
        self.assertTrue(False, msg='Test Not implemented')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestConfiguration('test_time_taken2'))
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestHeadwind(unittest.TestCase):
    def test_can_operate(self):
        opts=Headwind.get_operational_combinations()
        self.assertEqual(opts, [('Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous')])
    
    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        head=P('Heading True Continuous', np.ma.array([30]))
        hw = Headwind()
        hw.derive(ws,wd,head)
        expected = np.ma.array([52.8629128481863])
        self.assertAlmostEqual(hw.array.data, expected.data)
        
    def test_odd_angles(self):
        ws = P('Wind Speed', np.ma.array([20.0]*8))
        wd = P('Wind Direction Continuous', np.ma.array([0, 90, 180, -180, -90, 360, 23, -23], dtype=float))
        head=P('Heading True Continuous', np.ma.array([-180, -90, 0, 180, 270, 360*15, 361*23, 359*23], dtype=float))
        hw = Headwind()
        hw.derive(ws,wd,head)
        expected = np.ma.array([-20]*3+[20]*5)
        ma_test.assert_almost_equal(hw.array, expected)
        


class TestWindAcrossLandingRunway(unittest.TestCase):
    def test_can_operate(self):
        opts=WindAcrossLandingRunway.get_operational_combinations()
        self.assertEqual(opts, [('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway')])
    
    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {'start': {'latitude': 60.18499999999998,
                                    'longitude': 11.073744}, 
                          'end': {'latitude': 60.216066999999995,
                                  'longitude': 11.091663999999993}}
        
        walr = WindAcrossLandingRunway()
        walr.derive(ws,wd,land_rwy)
        expected = np.ma.array([50.55619778])
        self.assertAlmostEqual(walr.array.data, expected.data)
        
