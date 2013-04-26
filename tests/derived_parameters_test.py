import numpy as np

import os
import sys
import shutil
import unittest
import datetime
import tempfile

from mock import Mock, call, patch

from hdfaccess.file import hdf_file
from flightdatautilities import masked_array_testutils as ma_test
from flightdatautilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import Fast, Mobile
from analysis_engine.library import (align, max_value, np_ma_masked_zeros_like)
from analysis_engine.node import (
    Attribute, A, App, ApproachItem, KPV, KeyTimeInstance, KTI, load, M,
    Parameter, P, Section, S)
from analysis_engine.process_flight import process_flight
from analysis_engine.settings import GRAVITY_IMPERIAL, METRES_TO_FEET

from flight_phase_test import buildsection

from analysis_engine.derived_parameters import (
    APEngaged,
    #ATEngaged,
    AccelerationVertical,
    AccelerationForwards,
    AccelerationSideways,
    AccelerationAlongTrack,
    AccelerationAcrossTrack,
    Aileron,
    AimingPointRange,
    AirspeedForFlightPhases,
    AirspeedMinusV2For3Sec,
    AirspeedReference,
    AirspeedRelative,
    AirspeedRelativeFor3Sec,
    AirspeedTrue,
    AltitudeAAL,
    AltitudeAALForFlightPhases,
    #AltitudeForFlightPhases,
    AltitudeQNH,
    AltitudeRadio,
    #AltitudeRadioForFlightPhases,
    #AltitudeSTD,
    AltitudeTail,
    ApproachRange,
    ClimbForFlightPhases,
    Configuration,
    ControlColumn,
    ControlColumnForce,
    ControlWheel,
    CoordinatesSmoothed,
    Daylight,
    DescendForFlightPhases,
    DistanceTravelled,
    DistanceToLanding,
    Elevator,
    Eng_Fire,
    Eng_N1Avg,
    Eng_N1Max,
    Eng_N1Min,
    Eng_N1MinFor5Sec,
    Eng_N2Avg,
    Eng_N2Max,
    Eng_N2Min,
    Eng_N3Avg,
    Eng_N3Max,
    Eng_N3Min,
    Eng_VibN1Max,
    Eng_VibN2Max,
    Eng_VibN3Max,
    Eng_1_Fire,
    Eng_2_Fire,
    Eng_3_Fire,
    Eng_4_Fire,
    Eng_1_FuelBurn,
    Eng_2_FuelBurn,
    Eng_3_FuelBurn,
    Eng_4_FuelBurn,
    Flap,
    FlapSurface,
    FuelQty,
    GearDownSelected,
    GearOnGround,
    GearUpSelected,
    GrossWeightSmoothed,
    #GroundspeedAlongTrack,
    Heading,
    HeadingContinuous,
    HeadingIncreasing,
    HeadingTrue,
    Headwind,
    ILSFrequency,
    #ILSLocalizerRange,
    LatitudePrepared,
    LatitudeSmoothed,
    LongitudePrepared,
    LongitudeSmoothed,
    Mach,
    MagneticVariation,
    MasterWarning,
    Pitch,
    SpeedbrakeSelected,
    StableApproach,
    VerticalSpeed,
    VerticalSpeedForFlightPhases,
    RateOfTurn,
    ThrustReversers,
    TAWSAlert,
    TrackDeviationFromRunway,
    TrackTrue,
    TurbulenceRMSG,
    V2,
    VerticalSpeedInertial,
    WindAcrossLandingRunway,
)

debug = sys.gettrace() is not None

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

def assert_array_within_tolerance(actual, desired, tolerance=1, similarity=100):
    '''
    Check that the actual array within tolerance of the desired array is
    at least similarity percent.
    
    :param tolerance: relative difference between the two array values
    :param similarity: percentage that must pass the tolerance test
    '''
    within_tolerance = abs(actual -  desired) <= tolerance
    percent_similar = sum(within_tolerance) / float(len(within_tolerance)) * 100
    if percent_similar <= similarity:
        raise AssertionError(
            'actual array tolerance only is %.2f%% similar to desired array.'
            'tolerance %.2f minimum similarity required %.2f%%' % (
                percent_similar, tolerance, similarity))


class TemporaryFileTest(object):
    '''
    Test using a temporary copy of a predefined file.
    '''
    def setUp(self):
        if getattr(self, 'source_file_path', None):
            self.make_test_copy()

    def tearDown(self):
        if self.test_file_path:
            os.unlink(self.test_file_path)
            self.test_file_path = None

    def make_test_copy(self):
        '''
        Copy the test file to temporary location, used by setUp().
        '''
        # Create the temporary file in the most secure way
        f = tempfile.NamedTemporaryFile(delete=False)
        self.test_file_path = f.name
        f.close()
        shutil.copy2(self.source_file_path, self.test_file_path)


class NodeTest(object):
    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            self.assertEqual(
                self.node_class.get_operational_combinations(),
                self.operational_combinations,
            )


##############################################################################
# Automated Systems


class TestAPEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngaged
        self.operational_combinations = [
            ('AP (1) Engaged',),
            ('AP (2) Engaged',),
            ('AP (3) Engaged',),
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]

    def test_single_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        eng = APEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.1)        
        ma_test.assert_array_equal(expected.array.data, eng.array.data)

    def test_dual_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')        
        ap3 = None
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,1,2,1,0]),
                   values_mapping={0: '-', 1: 'Engaged', 2: 'Duplex'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.1)        
        
        ma_test.assert_array_equal(expected.array.data, eng.array.data)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged', 
                   frequency=1, 
                   offset=0.1)        
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged', 
                   frequency=1, 
                   offset=0.2)        
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged', 
                   frequency=1, 
                   offset=0.4)        
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,1,2,3,2,1]),
                   values_mapping={0: '-', 1: 'Engaged', 2: 'Duplex', 3: 'Triplex'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.25)        
        
        ma_test.assert_array_equal(expected.array.data, eng.array.data)

        

##### FIXME: Re-enable when 'AT Engaged' has been implemented.
####class TestATEngaged(unittest.TestCase, NodeTest):
####
####    def setUp(self):
####        self.node_class = ATEngaged
####        self.operational_combinations = [
####            ('AT (1) Engaged',),
####            ('AT (2) Engaged',),
####            ('AT (3) Engaged',),
####            ('AT (1) Engaged', 'AT (2) Engaged'),
####            ('AT (1) Engaged', 'AT (3) Engaged'),
####            ('AT (2) Engaged', 'AT (3) Engaged'),
####            ('AT (1) Engaged', 'AT (2) Engaged', 'AT (3) Engaged'),
####        ]
####
####    @unittest.skip('Test Not Implemented')
####    def test_derive(self):
####        self.assertTrue(False, msg='Test not implemented.')


##############################################################################

class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                     'Acceleration Lateral Offset Removed',
                     'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)
                        
        acc_vert.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.zeros(4), 4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4), 4),
            Parameter('Pitch', np.ma.zeros(2), 2),
            Parameter('Roll', np.ma.zeros(2), 2),
        ])
        
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8) * 0.8660254,8),
            P('Acceleration Lateral Offset Removed',np.ma.zeros(4), 4),
            P('Acceleration Longitudinal',np.ma.ones(4) * 0.5,4),
            P('Pitch',np.ma.ones(2) * 30.0,2),
            P('Roll',np.ma.zeros(2), 2)
        ])

        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)

    def test_acceleration_vertical_pitch_up_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * (-0.2), 4),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.3, 4),
            P('Pitch',np.ma.ones(2) * 30.0, 2),
            P('Roll',np.ma.ones(2) * 20, 2)])
        
        expected = np.ma.array([0.86027777] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.7071068, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * -0.7071068, 4),
            P('Acceleration Longitudinal', np.ma.zeros(4), 4),
            P('Pitch', np.ma.zeros(2), 2),
            P('Roll', np.ma.ones(2) * 45, 2),
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)


class TestAccelerationForwards(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Longitudinal', 'Pitch')]
        opts = AccelerationForwards.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_forward_level_on_gound(self):
        # Invoke the class object
        acc_fwd = AccelerationForwards(frequency=4)
                        
        acc_fwd.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Longitudinal', np.ma.ones(4) * 0.1,4),
            Parameter('Pitch', np.ma.zeros(2), 2)
        ])
        expected = np.ma.array([0.1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_fwd.array, expected)
        
    def test_acceleration_forward_pitch_up(self):
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8660254, 8),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.5, 4),
            P('Pitch', np.ma.ones(2) * 30.0, 2)
        ])

        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_fwd.array, expected)


class TestAccelerationSideways(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Lateral Offset Removed', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationSideways.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_sideways_level_on_gound(self):
        # Invoke the class object
        acc_lat = AccelerationSideways(frequency=8)
                        
        acc_lat.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8),8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.ones(4)*0.05,4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            Parameter('Pitch', np.ma.zeros(2),2),
            Parameter('Roll', np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0.05] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)
        
    def test_acceleration_sideways_pitch_up(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral Offset Removed',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)

    def test_acceleration_sideways_roll_right(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral Offset Removed',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2)
        ])
        #                                     x   interp  x  pitch/roll masked
        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_lat.array, expected)

        
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
        expected = np.ma.array([0.1] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_across.array, expected)
        
    def test_acceleration_across_resolved(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.5,4),
            P('Drift',np.ma.ones(2)*30.0,2)])

        expected = np.ma.array([0] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_across.array, expected)

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
        
        expected = np.ma.array([0.2] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_along.array, expected)
        
    def test_acceleration_along_resolved(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.1,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.2,4),
            P('Drift',np.ma.ones(2)*10.0,2)])
        expected = np.ma.array([0.13321041] * 8, mask=[0, 0, 0, 0, 0,   1, 1, 1])
        ma_test.assert_masked_array_approx_equal(acc_along.array, expected)


class TestAirspeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)


class TestAirspeedMinusV2(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedReference(unittest.TestCase):
    def setUp(self):
        self.approach_slice = slice(105, 120)
        apps = App('Approach', items=[ApproachItem('LANDING',
                                                   self.approach_slice)])
        self.default_kwargs = {'spd':False,
                               'gw':None,
                               'flap':None,
                               'conf':None,
                               'vapp':None,
                               'vref':None,
                               'afr_vapp':None,
                               'afr_vref':None,
                               'apps':apps,
                               'series':None,
                               'family':None}


    def test_can_operate(self):
        expected = [('Vapp',),
                    ('Vref',),
                    ('AFR Vapp',),
                    ('AFR Vref',),
                    ('Airspeed', 'Gross Weight Smoothed', 'Series',
                     'Family', 'Approach', 'Flap',),
                    ('Airspeed', 'Gross Weight Smoothed', 'Series',
                     'Family', 'Approach', 'Configuration',)]
        opts = AirspeedReference.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])

    def test_airspeed_reference__fdr_vapp(self):
        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['afr_vapp'] = A('AFR Vapp', value=120)

        param = AirspeedReference()
        param.derive(**kwargs)
        expected = np.ma.zeros(128)
        expected.mask = True
        expected[self.approach_slice] = 120
        np.testing.assert_array_equal(param.array, expected)

    def test_airspeed_reference__fdr_vref(self):
        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['afr_vref'] = A('AFR Vref', value=120)

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

    @patch('analysis_engine.derived_parameters.get_vspeed_map')
    def test_airspeed_reference__boeing_lookup(self, vspeed_map):
        vspeed_table = Mock
        vspeed_table.vref = Mock(side_effect = [135, 130])
        vspeed_table.reference_settings = [15, 20, 30]
        vspeed_map.return_value = vspeed_table
        test_hdf = copy_file(os.path.join(test_data_path, 'airspeed_reference.hdf5'))
        with hdf_file(test_hdf) as hdf:
            approaches = [ApproachItem('TOUCH_AND_GO', slice(3346, 3540)),
                          ApproachItem('LANDING', slice(5502, 5795))]
            args = [
                P(**hdf['Flap'].__dict__),
                P(**hdf['Airspeed'].__dict__),
                P(**hdf['Gross Weight Smoothed'].__dict__),
                None,
                None,
                None,
                None,
                None,
                App('Approach Information', items=approaches),
                KTI('Touchdown', items=[KeyTimeInstance(3450, 'Touchdown'),
                                        KeyTimeInstance(5700, 'Touchdown')]),
                A('Series', value='B737-300'),
                A('Family', value='B737 Classic'),
                None,
            ]
            param = AirspeedReference()
            param.get_derived(args)
            expected = np_ma_masked_zeros_like(hdf['Airspeed'].array)
            expected[slice(3346, 3540)] = 135
            expected[slice(5502, 5795)] = 130
            np.testing.assert_array_equal(param.array, expected)
        if os.path.isfile(test_hdf):
            os.remove(test_hdf)

    @unittest.skip('Airbus Reference Lookup not Implemented')
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
                #None,
            #]
            #param = AirspeedReference()
            #param.get_derived(args)
            #expected = np.ma.load('test_data/boeing_reference_speed.ma')
            #np.testing.assert_array_equal(param.array, expected.array)
        self.assertTrue(False, msg='Test Not implemented')


class TestAirspeedRelative(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed', 'Airspeed Reference')]
        opts = AirspeedRelative.get_operational_combinations()
        self.assertEqual(opts, expected)
        
        # ???????????????????????????????????????????????????????????????
        # THIS MAY NEED TO BE ALTERED SO THAT Vref IS VARIABLE AND NOT FIXED
        # NEED A DIFFERENT Vref FOR EACH APPROACH ??? DISCUSS WITH DEREK AND
        # DAVE BEFORE CHANGING
    
    def test_airspeed_for_phases_basic(self):
        speed=P('Airspeed', np.ma.array([200] * 128))
        ref = P('Airspeed Relative', np.ma.array([120] * 128))
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
            ('Airspeed', 'Altitude STD', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 
             'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'TAT', 'Takeoff', 'Landing', 
             'Groundspeed', 'Acceleration Forwards')
        ])
        
    def test_tas_basic(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 20000, 40000]))
        tat = P('TAT', np.ma.array([20, -10, -16.2442]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, tat)
        # Answers with compressibility are:
        result = [100.6341, 273.0303, 552.8481]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.1)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.7)
        self.assertLess(abs(tas.array.data[2] - result[2]), 6.0)
        
    def test_tas_masks(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 20000, 40000]))
        tat = P('TAT', np.ma.array([20, -10, -40]))
        tas = AirspeedTrue()
        cas.array[0] = np.ma.masked
        alt.array[1] = np.ma.masked
        tat.array[2] = np.ma.masked
        tas.derive(cas, alt, tat)
        np.testing.assert_array_equal(tas.array.mask, [True] * 3)
        
    def test_tas_no_tat(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 10000, 20000]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, None)
        result = [100.000, 231.575, 400.097]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)
        

class TestAltitudeAAL(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeAAL.get_operational_combinations()
        self.assertTrue(('Altitude STD Smoothed', 'Fast') in opts)
        self.assertTrue(('Altitude Radio', 'Altitude STD Smoothed', 'Fast') in opts)
        
    def test_alt_aal_basic(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100] * 10)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad,alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 220, 70, 20, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_bounce_rejection(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5, 2, 5, 2,
                            -3, -3])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.array([100] * 15)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 220, 70, 20, 0, 0, 0, 0,
                                0, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)
    
    def test_alt_aal_no_ralt(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 580, 220, 70, 20, -5])
        alt_std = P(array=data + 300)
        slow_and_fast_data = np.ma.array([70] + [85] * 8 + [70])
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 510, 150, 0, 0, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)
    
    def test_alt_aal_complex(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        # plot_parameter (testwave)
        rad_wave = np.copy(testwave)
        rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        # plot_parameter (rad_data)
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', rad_data),
                       P('Altitude STD', testwave),
                       phase_fast)
        # plot_parameter (alt_aal.array)

        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 7013, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 3308, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 217, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8965, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 3288, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[313], 0, decimal=0)

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_faulty_alt_rad(self):
        '''
        When 'Altitude Radio' does not reach 0 after touchdown due to an arinc
        signal being recorded, 'Altitude AAL' did not fill the second half of
        its array. Since the array is initialised as zeroes
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_faulty_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')
    
    @unittest.skip('Test Not Implemented')
    def test_alt_aal_without_alt_rad(self):
        '''
        When 'Altitude Radio' is not available, 'Altitude AAL' is created from
        'Altitude STD' using the cycle_finder and peak_curvature algorithms.
        Currently, cycle_finder is accurately locating the index where the
        aircraft begins to climb. This section of data is passed into 
        peak_curvature, which is designed to find the first curve in a piece of
        data. The problem is that data from before the first curve, where the 
        aircraft starts climbing, is not included, and peak_curvature detects
        the second curve at approximately 120 feet.
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_without_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')

    def test_alt_aal_training_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        peak_detect = np.ma.masked_where(alt_aal.array < 500, alt_aal.array)
        peaks = np.ma.clump_unmasked(peak_detect)
        # Check to test that all 6 altitude sections are inculded in alt aal
        self.assertEqual(len(peaks), 6)

    def test_alt_aal_goaround_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        difs = np.diff(alt_aal.array)
        index, value = max_value(np.abs(difs))
        # Check to test that the step occurs during cruse and not the go-around
        self.assertTrue(index in range(1290, 1850))
    


class TestAimingPointRange(unittest.TestCase):
    def test_basic_scaling(self):
        approaches = App(items=[ApproachItem(
            'Landing', slice(3, 8),
            runway={'end': 
                    {'elevation': 3294,
                     'latitude': 31.497511,
                     'longitude': 65.833933},
                    'start': 
                    {'elevation': 3320,
                     'latitude': 31.513997,
                     'longitude': 65.861714}})])
        app_rng=P('Approach Range',
                  array=np.ma.arange(10000.0, -2000.0, -1000.0))
        apr = AimingPointRange()
        apr.derive(app_rng, approaches)
        # convoluted way to check masked outside slice !
        self.assertEqual(apr.array[0].mask, np.ma.masked.mask)
        self.assertAlmostEqual(apr.array[4], 1.67, places=2)
        
        
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


class TestAltitudeQNH(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeQNH
        self.operational_combinations = [
            ('Altitude AAL', 'Altitude Peak'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Runway', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
            ('Altitude AAL', 'Altitude Peak', 'FDR Landing Airport', 'FDR Landing Runway', 'FDR Takeoff Airport', 'FDR Takeoff Runway'),
        ]
        data = [np.ma.arange(0, 1000, step=30)]
        data.append(data[0][::-1] + 50)
        self.alt_aal = P(name='Altitude AAL', array=np.ma.concatenate(data))
        self.alt_peak = KTI(name='Altitude Peak', items=[KeyTimeInstance(name='Altitude Peak', index=len(self.alt_aal.array) / 2)])
        self.land_fdr_apt = A(name='FDR Landing Airport', value={'id': 10, 'elevation': 100})
        self.land_fdr_rwy = A(name='FDR Landing Runway', value={'ident': '27L', 'start': {'elevation': 90}, 'end': {'elevation': 110}})
        self.toff_fdr_apt = A(name='FDR Takeoff Airport', value={'id': 20, 'elevation': 50})
        self.toff_fdr_rwy = A(name='FDR Takeoff Runway', value={'ident': '09R', 'start': {'elevation': 40}, 'end': {'elevation': 60}})

        self.expected = []
        peak = self.alt_peak[0].index

        # Ensure that we have a sensible drop at the splitting point...
        self.alt_aal.array[peak + 1] += 30
        self.alt_aal.array[peak] -= 30

        # 1. Data same as Altitude AAL, no mask applied:
        data = np.ma.copy(self.alt_aal.array)
        self.expected.append(data)
        # 2. None masked, data Altitude AAL, +50 ft t/o, +100 ft ldg:
        data = np.ma.array([50, 80, 110, 140, 170, 200, 230, 260, 290, 320,
            350, 351, 352, 354, 355, 357, 358, 360, 361, 363, 364, 366, 367,
            368, 370, 371, 373, 374, 376, 377, 379, 380, 382, 383, 385, 386,
            387, 389, 390, 392, 393, 395, 396, 398, 399, 401, 402, 403, 405,
            406, 408, 409, 411, 412, 414, 415, 417, 418, 420, 390, 360, 330,
            300, 270, 240, 210, 180, 150])
        data.mask = False
        self.expected.append(data)
        # 3. Data Altitude AAL, +50 ft t/o; ldg assumes t/o elevation:
        data = np.ma.copy(self.alt_aal.array)
        data += 50
        self.expected.append(data)
        # 4. Data Altitude AAL, +100 ft ldg; t/o assumes ldg elevation:
        data = np.ma.copy(self.alt_aal.array)
        data += 100
        self.expected.append(data)

    def test_derive__function_calls(self):
        alt_qnh = self.node_class()
        alt_qnh._calc_apt_elev = Mock(return_value=0)
        alt_qnh._calc_rwy_elev = Mock(return_value=0)
        # Check no airport/runway information results in a fully masked copy of Altitude AAL:
        alt_qnh.derive(self.alt_aal, self.alt_peak)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        self.assertFalse(alt_qnh._calc_rwy_elev.called, 'method should not have been called')
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with runway details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, self.land_fdr_rwy, None, self.toff_fdr_rwy)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        alt_qnh._calc_rwy_elev.assert_has_calls([
            call(self.toff_fdr_rwy.value),
            call(self.land_fdr_rwy.value),
        ])
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, None, self.toff_fdr_apt, None)
        alt_qnh._calc_apt_elev.assert_has_calls([
            call(self.toff_fdr_apt.value),
            call(self.land_fdr_apt.value),
        ])
        self.assertFalse(alt_qnh._calc_rwy_elev.called, 'method should not have been called')
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()
        # Check everything works calling with runway and airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, self.toff_fdr_apt, self.toff_fdr_rwy)
        self.assertFalse(alt_qnh._calc_apt_elev.called, 'method should not have been called')
        alt_qnh._calc_rwy_elev.assert_has_calls([
            call(self.toff_fdr_rwy.value),
            call(self.land_fdr_rwy.value),
        ])
        alt_qnh._calc_apt_elev.reset_mock()
        alt_qnh._calc_rwy_elev.reset_mock()

    def test_derive__output(self):
        alt_qnh = self.node_class()
        # Check no airport/runway information results in a fully masked copy of Altitude AAL:
        alt_qnh.derive(self.alt_aal, self.alt_peak)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[0])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with runway details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, self.land_fdr_rwy, None, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, None, self.toff_fdr_apt, None)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check everything works calling with runway and airport details:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, self.toff_fdr_apt, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check second half masked when no elevation at landing:
        alt_qnh.derive(self.alt_aal, self.alt_peak, None, None, self.toff_fdr_apt, self.toff_fdr_rwy)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[2])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)
        # Check first half masked when no elevation at takeoff:
        alt_qnh.derive(self.alt_aal, self.alt_peak, self.land_fdr_apt, self.land_fdr_rwy, None, None)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[3])
        self.assertEqual(alt_qnh.offset, self.alt_aal.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal.frequency)


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
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]*2), 0.5,  0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.25, 1.0),
                       Parameter('Altitude Radio (C)',
                                 np.ma.array([30.0,30.0,30.0,30.0,30.3]), 0.25, 3.0),
                       None, None, None
                       )
        answer = np.ma.array(data=[17.5]*20)
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=0)
        self.assertEqual(alt_rad.offset,0.0)
        self.assertEqual(alt_rad.frequency,1.0)

    def test_altitude_radio_737_5_EFIS(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None)
        answer = np.ma.array(data=[ 15.0, 14.9, 14.9, 15.0, 15.0, 14.9, 14.9, 15.0, 15.0, 15.2])
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset,0.0)
        self.assertEqual(alt_rad.frequency,1.0)

    def test_altitude_radio_737_5_Analogue(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)', 
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None)
        answer = np.ma.array(data=[ 15.0, 14.9, 14.9, 15.0, 15.0, 14.9, 14.9, 15.0, 15.0, 15.2])
        ma_test.assert_array_almost_equal(alt_rad.array, answer, decimal=1)
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
          [('Altitude STD (Coarse)', 'Altitude STD (Fine)'),
           ('Altitude STD (Coarse)', 'Vertical Speed')])
    
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
    
    @patch('analysis_engine.derived_parameters.first_order_lag')
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
        alt_std._high_and_low = Mock()
        high_and_low_array = 3
        alt_std._high_and_low.return_value = high_and_low_array
        alt_std_high = 1
        alt_std_low = 2
        alt_std.derive(alt_std_high, alt_std_low, None, None)
        alt_std._high_and_low.assert_called_once_with(alt_std_high, alt_std_low)
        self.assertEqual(alt_std.array, high_and_low_array)
        # alt_std_rough and ivv passed in.
        rough_and_ivv_array = 6
        alt_std._rough_and_ivv = Mock()
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
        talt.derive(Parameter('Altitude Radio', np.ma.array([0, 5])),
                    Parameter('Pitch', np.ma.array([0, 18])),
                    Attribute('Ground To Lowest Point Of Tail', 10.0/METRES_TO_FEET),
                    Attribute('Main Gear To Lowest Point Of Tail', 35.0/METRES_TO_FEET))
        result = talt.array
        # Lift 5ft
        answer = np.ma.array(data=[10, 5 - 0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)

class TestClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Fast')]
        opts = ClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_climb_for_flight_phases_basic(self):
        up_and_down_data = np.ma.array([0,0,2,5,3,2,5,6,8,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD Smoothed', up_and_down_data), phase_fast)
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
        self.assertLess(time, 0.2, msg="Took too long")


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

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_column(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cc = ControlColumn()
        cc.derive(self.ccc, self.ccf)
        blend_two_parameters.assert_called_once_with(self.ccc, self.ccf)


class TestControlColumnForce(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Control Column Force (Capt)',
                     'Control Column Force (FO)')]
        opts = ControlColumnForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force(self):
        ccf = ControlColumnForce()
        ccf.derive(
            ControlColumnForce('Control Column Force (Capt)', np.ma.arange(8)),
            ControlColumnForce('Control Column Force (FO)', np.ma.arange(8)))
        np.testing.assert_array_almost_equal(ccf.array, np.ma.arange(0, 16, 2))


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

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_wheel(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cw = ControlWheel()
        cw.derive(self.cwc, self.cwf)
        blend_two_parameters.assert_called_once_with(self.cwc, self.cwf)


class TestDaylight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Latitude Smoothed', 'Longitude Smoothed', 
                     'Start Datetime', 'HDF Duration')]
        opts = Daylight.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_daylight_aligning(self):
        lat = P('Latitude', np.ma.array([51.1789]*128), offset=0.1)
        lon = P('Longitude', np.ma.array([-1.8264]*128))
        start_dt = A('Start Datetime', datetime.datetime(2012,6,20, 20,25))
        dur = A('HDF Duration', 128)
        
        don = Daylight()
        don.get_derived((lat, lon, start_dt, dur))
        self.assertEqual(list(don.array), [np.ma.masked, 'Day'])
        self.assertEqual(don.frequency, 1/64.0)
        self.assertEqual(don.offset, 0)

    def test_father_christmas(self):
        # Starting on the far side of the world, he flies all round
        # delivering parcels mostly by night (in the northern lands).
        lat=P('Latitude', np.ma.arange(60,64,1/64.0))
        lon=P('Longitude', np.ma.arange(-180,180,90/64.0))
        start_dt = A('Start Datetime', datetime.datetime(2012,12,25,01,00))
        dur = A('HDF Duration', 256)
        
        don = Daylight()
        don.get_derived((lat, lon, start_dt, dur))
        expected = ['Day', 'Night', 'Night', 'Night']
        np.testing.assert_array_equal(don.array, expected)


class TestDescendForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed', 'Fast')]
        opts = DescendForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_descend_for_flight_phases_basic(self):
        down_and_up_data = np.ma.array([0,0,12,5,3,12,15,10,7,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*8+[0])))
        descend = DescendForFlightPhases()
        descend.derive(Parameter('Altitude STD Smoothed', down_and_up_data ), phase_fast)
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

    @patch('analysis_engine.derived_parameters.integrate')
    def test_derive(self, integrate):
        gndspeed = Mock()
        gndspeed.array = Mock()
        gndspeed.frequency = Mock()
        DistanceTravelled().derive(gndspeed)
        integrate.assert_called_once_with(gndspeed.array, gndspeed.frequency,
                                          scale=1.0 / 3600)


class TestEng_EPRMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


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
        
    def test_derive_two_engines_offset(self):
        # this tests that average is performed on data sampled alternately.
        a = np.ma.array(range(50, 55))
        b = np.ma.array(range(54, 49, -1)) + 0.2
        eng = Eng_N1Max()
        eng.derive(P('Eng (1)',a,offset=0.25), P('Eng (2)',b, offset=0.75), None, None)
        ma_test.assert_array_equal(eng.array,np.ma.array([54.2, 53.2, 52.2, 53, 54]))
        self.assertEqual(eng.offset, 0)
        
        
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


class TestEng_N1MinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1MinFor5Sec
        self.operational_combinations = [('Eng (*) N1 Min',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


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
        fstep.derive(flap, A('Series', None), A('Family', 'DC-9'))
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
           ('Fuel Qty (Aux)',), ('Fuel Qty (1)', 'Fuel Qty (2)'),
           ('Fuel Qty (1)', 'Fuel Qty (3)'), ('Fuel Qty (1)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (2)', 'Fuel Qty (3)'), ('Fuel Qty (2)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (3)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (1)', 'Fuel Qty (2)', 'Fuel Qty (3)'),
           ('Fuel Qty (1)', 'Fuel Qty (2)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (1)', 'Fuel Qty (3)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (2)', 'Fuel Qty (3)', 'Fuel Qty (Aux)'),
           ('Fuel Qty (1)', 'Fuel Qty (2)', 'Fuel Qty (3)', 'Fuel Qty (Aux)')])
    
    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (1)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (2)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18]))
        # Works without all parameters.
        fuel_qty_node.derive(fuel_qty1, None, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))

    def test_four_tanks(self):
        fuel_qty1 = P('Fuel Qty (1)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (2)', 
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_a = P('Fuel Qty (Aux)',
                      array=np.ma.array([11,12,13], mask=[False, False, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, fuel_qty_a)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([17, 24, 31]))
    
    def test_masked_tank(self):
        fuel_qty1 = P('Fuel Qty (1)', 
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (2)', 
                      array=np.ma.array([2,4,6], mask=[True, True, True]))
        # Mask will be interpolated by repair_mask.
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, None, None)
        np.testing.assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))    


class TestGrossWeightSmoothed(unittest.TestCase):
    def test_gw_real_data_1(self):
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_gw.nod'))
        gw_orig = gw.array.copy()
        climbs = load(os.path.join(test_data_path,
                                   'gross_weight_smoothed_1_climbs.nod'))        
        descends = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_1_descends.nod'))
        fast = load(os.path.join(test_data_path,
                                 'gross_weight_smoothed_1_fast.nod'))
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[640] - gw_orig[640]) < 30)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1150] - gw_orig[1150]) < 260)
        # End is similar.
        self.assertTrue(abs(gws.array[2500] - gw_orig[2500]) < 30)
        
    def test_gw_real_data_2(self): 
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_gw.nod'))
        gw_orig = gw.array.copy()
        climbs = load(os.path.join(test_data_path,
                                   'gross_weight_smoothed_2_climbs.nod'))        
        descends = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_2_descends.nod'))
        fast = load(os.path.join(test_data_path,
                                 'gross_weight_smoothed_2_fast.nod'))
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[600] - gw_orig[600]) < 35)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1500] - gw_orig[1500]) < 180)
        # Descending diverges.
        self.assertTrue(abs(gws.array[5800] - gw_orig[5800]) < 120)
    
    def test_gw_masked(self): 
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*256,dtype=float),offset=0.0,frequency=1.0)
        weight_aligned = align(weight, fuel_flow)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', None, None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])  
        ma_test.assert_equal(result.array, weight_aligned)
    
    def test_gw_formula(self):
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([3600]*256,dtype=float),offset=0.0,frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 292.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_with_many_samples(self):
        weight = P('Gross Weight',np.ma.array(data=range(56400, 50000, -64), 
                                              mask=False, dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.array([3600] * 64 * 100,
                                                       dtype=float),
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 50, 60)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[1], 56400-1)
        
    def test_gw_formula_with_good_data(self):
        weight = P('Gross Weight', np.ma.array(data=[484, 420, 356, 292, 228, 164, 100],
                                               mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.array([3600] * 64 * 7, dtype=float),
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 60, 70)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_formula_climbing(self):
        weight = P('Gross Weight',np.ma.array(data=[484,420,356,292,228,164,100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',
                      np.ma.array([3600] * 64 * 7, dtype=float))
        climb = buildsection('Climbing', 1, 4)
        descend = buildsection('Descending', 20, 30)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_descending(self):
        weight = P('Gross Weight',np.ma.array(
            data=[484, 420, 356, 292, 228, 164, 100],
            mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow',
                      np.ma.array([3600] * 64 * 7, dtype=float),
                      offset=0.0, frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = buildsection('Descending', 3, 5)
        fast = buildsection('Fast', 50, 450)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)
        
    def test_gw_one_masked_data_point(self):
        weight = P('Gross Weight',np.ma.array(data=[0],
                                              mask=[1],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow',np.ma.array([0]*64,dtype=float),
                      offset=0.0,frequency=1.0)
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = S('Descending')
        fast = buildsection('Fast', 0, 1)
        gws = GrossWeightSmoothed()
        gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(len(gws.array),64)
        self.assertEqual(gws.frequency, fuel_flow.frequency)
        self.assertEqual(gws.offset, fuel_flow.offset)


class TestGroundspeedAlongTrack(unittest.TestCase):

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_can_operate(self):
        expected = [('Groundspeed','Acceleration Along Track', 'Altitude AAL',
                     'ILS Glideslope')]
        opts = GroundspeedAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    @unittest.skip('Commented out until new computation of sliding motion')
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
        
    @unittest.skip('Commented out until new computation of sliding motion')
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
        
        
class TestHeadingContinuous(unittest.TestCase):
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


class TestTrackDeviationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            TrackDeviationFromRunway.get_operational_combinations(),
            [('Track True', 'FDR Takeoff Runway'),
             ('Track True', 'Approach Information'),
             ('Track', 'FDR Takeoff Runway'),
             ('Track', 'Approach Information'),
             ('Track True', 'Track', 'FDR Takeoff Runway'),
             ('Track True', 'Track', 'Approach Information'),
             ('Track True', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True', 'Takeoff', 'Approach Information'),
             ('Track True', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track', 'Takeoff', 'Approach Information'),
             ('Track', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True', 'Track', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True', 'Track', 'Takeoff', 'Approach Information'),
             ('Track True', 'Track', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True', 'Track', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information')]
        )
        
    def test_deviation(self):
        apps = App(items=[ApproachItem(
            'LANDING', slice(8763, 9037),
            airport={'code': {'iata': 'FRA', 'icao': 'EDDF'},
                     'distance': 2.2981699358981746,
                     'id': 2289,
                     'latitude': 50.0264,
                     'location': {'city': 'Frankfurt-Am-Main',
                                  'country': 'Germany'},
                     'longitude': 8.54313,
                     'magnetic_variation': 'E000459 0106',
                     'name': 'Frankfurt Am Main'},
            runway={'end': {'latitude': 50.027542, 'longitude': 8.534175},
                    'glideslope': {'angle': 3.0,
                                   'latitude': 50.037992,
                                   'longitude': 8.582733,
                                   'threshold_distance': 1098},
                    'id': 4992,
                    'identifier': '25L',
                    'localizer': {'beam_width': 4.5,
                                  'frequency': 110700.0,
                                  'heading': 249,
                                  'latitude': 50.026722,
                                  'longitude': 8.53075},
                    'magnetic_heading': 248.0,
                    'start': {'latitude': 50.040053, 'longitude': 8.586531},
                    'strip': {'id': 2496,
                              'length': 13123,
                              'surface': 'CON',
                              'width': 147}},
            turnoff=8998.2717013888887)])
        heading_track = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_heading_track.nod'))
        to_runway = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_runway.nod'))
        takeoff = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_takeoff.nod'))

        deviation = TrackDeviationFromRunway()
        deviation.get_derived((heading_track, None, takeoff, to_runway, apps))
        # check average stays close to 0
        self.assertAlmostEqual(np.ma.average(deviation.array[8775:8975]), 1.5, places = 1)
        self.assertAlmostEqual(np.ma.min(deviation.array[8775:8975]), -10.5, places = 1)
        self.assertAlmostEqual(np.ma.max(deviation.array[8775:8975]), 12.3, places = 1)


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
        combinations = LatitudePrepared.get_operational_combinations()
        # Longitude should be the same list
        self.assertEqual(combinations, LongitudePrepared.get_operational_combinations())
        # only lat long
        self.assertTrue(('Latitude','Longitude') in combinations)
        # with lat long and all the rest
        self.assertTrue(('Latitude',
                         'Longitude',
                         'Heading True',
                         'Airspeed True',
                         'Latitude At Liftoff',
                         'Longitude At Liftoff',
                         'Latitude At Touchdown',
                         'Longitude At Touchdown') in combinations)
        
        # without lat long
        self.assertTrue(('Heading True',
                         'Airspeed True',
                         'Latitude At Liftoff',
                         'Longitude At Liftoff',
                         'Latitude At Touchdown',
                         'Longitude At Touchdown') in combinations)
        
    def test_latitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,0,0,0,0,0.001],dtype=float))
        smoother = LatitudePrepared()
        smoother.get_derived([lat,lon])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        expected = [0.0, 0.0, 0.00088, 0.00088, 0.00088, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)

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
        expected = [0.0, 0.0, -0.00176, -0.00176, -0.00176, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)



class TestHeadingTrueTrack(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            TrackTrue.get_operational_combinations(),
            [('Heading True Continuous', 'Drift')])

    def test_heading_track(self):
        hdg = load(os.path.join(test_data_path, 'HeadingTrack_Heading_True.nod'))
        dft = load(os.path.join(test_data_path, 'HeadingTrack_Drift.nod'))
        head_track = TrackTrue()
        head_track.derive(heading=hdg, drift=dft)
        
        # compare IRU Track Angle True (recorded) against the derived
        track_rec = load(os.path.join(test_data_path, 'HeadingTrack_IRU_Track_Angle_Recorded.nod'))
        assert_array_within_tolerance(head_track.array, track_rec.array, 10, 98)


class TestHeading(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Heading.get_operational_combinations(),
            [('Heading True Continuous', 'Magnetic Variation')])
        
    def test_basic(self):
        true = P('Heading True Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        head = Heading()
        head.derive(true, var)
        expected = P('Heading True', np.ma.array([358.0, 2.0, 14.0, 2.0, 347.0]))
        ma_test.assert_array_equal(head.array, expected.array)


class TestHeadingTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(HeadingTrue.get_operational_combinations(),
            [('Heading Continuous', 'Magnetic Variation')])
        
    def test_basic(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        true = HeadingTrue()
        true.derive(head, var)
        expected = P('Heading True', np.ma.array([2.0, 8.0, 358.0, 348.0, 5.0]))
        ma_test.assert_array_equal(true.array, expected.array)


class TestILSFrequency(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS (1) Frequency', 'ILS (2) Frequency',),
                    ('ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',),
                    ('ILS (1) Frequency', 'ILS (2) Frequency',
                     'ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',)]
        opts = ILSFrequency.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])
        
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


class TestILSLocalizerRange(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


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
        self.assertEqual(VerticalSpeed.get_operational_combinations(),
                         [('Altitude STD Smoothed',),
                           ('Altitude STD Smoothed', 'Frame')])
                         
    def test_vertical_speed_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.array([100]*10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)
    
    def test_vertical_speed_alt_std_only(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(100, 200, 10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.array(data=[600] * 10, dtype=np.float,
                               mask=False) #  10 ft/sec = 600 fpm
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)


class TestVerticalSpeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed',)]
        opts = VerticalSpeedForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_vertical_speed_for_flight_phases_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        expected = np.ma.array(data=[60]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(vert_spd.array, expected)

    def test_vertical_speed_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD Smoothed', np.ma.array([100]*10))
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
        rot = RateOfTurn()
        rot.derive(P('Heading Continuous', np.ma.array([0,0,2,4,2,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,1.95,0.5,0,-0.5,-1.95,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)
        
    def test_sample_long_gentle_turn(self):
        # Sample taken from a long circling hold pattern
        head_cont = P(array=np.ma.array(
            np.load(os.path.join(test_data_path, 'heading_continuous_in_hold.npy'))), frequency=2)
        rot = RateOfTurn()
        rot.get_derived((head_cont,))
        np.testing.assert_allclose(rot.array[50:1150],
                                   np.ones(1100, dtype=float)*2.1, rtol=0.1)
        
        
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
                               'afr_v2':None,
                               'weight_liftoff':None,
                               'series':None,
                               'family':None}

    def test_can_operate(self):
        # TODO: test expected combinations are in get_operational_combinations
        expected = [('AFR V2',),
                    ('Airspeed', 'Gross Weight At Liftoff', 'Series', 'Family',
                     'Configuration',),
                    ('Airspeed', 'Gross Weight At Liftoff', 'Series', 'Family',
                     'Flap',),]
        opts = V2.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])

    def test_v2__fdr_v2(self):

        kwargs = self.default_kwargs.copy()
        kwargs['spd'] = P('Airspeed', np.ma.array([200]*128), frequency=1)
        kwargs['afr_v2'] = A('AFR V2', value=120)

        param = V2()
        param.derive(**kwargs)
        expected = np.array([120]*128)
        np.testing.assert_array_equal(param.array, expected)

    def test_v2__boeing_lookup(self):
        gw = KPV('Gross Weight At Liftoff')
        gw.create_kpv(451, 54192.06)
        test_hdf = copy_file(os.path.join(test_data_path, 'airspeed_reference.hdf5'))
        with hdf_file(test_hdf) as hdf:
            args = [
                P(**hdf['Flap'].__dict__),
                P(**hdf['Airspeed'].__dict__),
                None,
                None,
                None,
                None,
                gw,
                A('Series', value='B737-300'),
                A('Family', value='B737 Classic'),
                None,
                None,
            ]
            param = V2()
            param.get_derived(args)
            expected = np.ma.array([151.70729599999999]*5888)
            np.testing.assert_array_equal(param.array, expected)
        if os.path.isfile(test_hdf):
            os.remove(test_hdf)

    @unittest.skip('Airbus V2 not Implemented')
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
                #None,
            #]
            #param = V2()
            #param.get_derived(args)
            #expected = np.ma.load('test_data/boeing_reference_speed.ma')
            #np.testing.assert_array_equal(param.array, expected.array)
        self.assertTrue(False, msg='Test Not implemented')


class TestHeadwind(unittest.TestCase):
    def test_can_operate(self):
        opts=Headwind.get_operational_combinations()
        self.assertTrue(('Wind Speed', 'Wind Direction Continuous', 'Heading True Continuous') in opts)
    
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
        opts = WindAcrossLandingRunway.get_operational_combinations()
        expected = [('Wind Speed', 'Wind Direction True Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction True Continuous', 'Wind Direction Continuous', 'FDR Landing Runway', 'Heading During Landing')]
        self.assertEqual(opts, expected)
    
    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {'start': {'latitude': 60.18499999999998,
                                    'longitude': 11.073744}, 
                          'end': {'latitude': 60.216066999999995,
                                  'longitude': 11.091663999999993}}
        
        walr = WindAcrossLandingRunway()
        walr.derive(ws,wd,None,land_rwy,None)
        expected = np.ma.array([50.55619778])
        self.assertAlmostEqual(walr.array.data, expected.data)
        
    def test_error_cases(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction True Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {}
        walr = WindAcrossLandingRunway()

        walr.derive(ws,wd,None,land_rwy,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)
        
        walr.derive(ws,wd,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)


class TestAOA(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalOffsetRemoved(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAileron(unittest.TestCase):
    
    def test_can_operate(self):
        opts = Aileron.get_operational_combinations()
        self.assertTrue(('Aileron (L)',) in opts)
        self.assertTrue(('Aileron (R)',) in opts)
        self.assertTrue(('Aileron (L) Outboard',) in opts)
        self.assertTrue(('Aileron (R) Outboard',) in opts)
        self.assertTrue(('Aileron (L)', 'Aileron (R)', 'Aileron (L) Outboard',
                         'Aileron (R) Outboard') in opts)

    def test_normal_two_sensors(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right = P('Aileron (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        aileron = Aileron()
        aileron.derive(left, right, None, None)
        expected_data = np.ma.array([1.5]*3+[1.75]*2+[1.5]*3)
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 1.0)
        self.assertEqual(aileron.offset, 0.1)

    def test_left_only(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.derive(left, None, None, None)
        expected_data = left.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)
        left_outboard = P('Aileron (L) Outboard', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.derive(None, None, left_outboard, None)
        expected_data = left_outboard.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_right_only(self):
        right = P('Aileron (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        aileron = Aileron()
        aileron.derive(None, right, None, None)
        expected_data = right.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 2.0)
        self.assertEqual(aileron.offset, 0.3)
        right_outboard = P('Aileron (R) Outboard', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.derive(None, None, right_outboard, None)
        expected_data = right_outboard.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)        

    def test_outboard_two_sensors(self):
        left_outboard = P('Aileron (L) Outboard', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right_outboard = P('Aileron (R) Outboard', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        aileron = Aileron()
        aileron.derive(None, None, left_outboard, right_outboard)
        expected_data = np.ma.array([1.5]*3+[1.75]*2+[1.5]*3)
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 1.0)
        self.assertEqual(aileron.offset, 0.1)


class TestAileronTrim(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedMinusV2For3Sec(unittest.TestCase):
    def test_can_operate(self):
        opts = AirspeedMinusV2For3Sec.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed Minus V2',)])
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedRelativeFor3Sec(unittest.TestCase):
    def test_can_operate(self):
        opts = AirspeedRelativeFor3Sec.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed Relative',)])
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeSTD(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestElevator(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_normal_two_sensors(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset = 0.1)
        right = P('Elevator (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset = 1.1)
        elevator = Elevator()
        elevator.derive(left, right)
        expected_data = np.ma.array([1.5]*3+[1.75]*2+[1.5]*3)
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 1.0)
        self.assertEqual(elevator.offset, 0.1)

    def test_left_only(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset = 0.1)
        elevator = Elevator()
        elevator.derive(left, None)
        expected_data = left.array
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 0.5)
        self.assertEqual(elevator.offset, 0.1)
    
    def test_right_only(self):
        right = P('Elevator (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        elevator = Elevator()
        elevator.derive(None, right)
        expected_data = right.array
        np.testing.assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 2.0)
        self.assertEqual(elevator.offset, 0.3)


class TestEng_EPRAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_Fire(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_FuelFlow(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_1_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_Fire
        self.operational_combinations = [('Eng (1) Fire On Ground', 'Eng (1) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_2_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_Fire
        self.operational_combinations = [('Eng (2) Fire On Ground', 'Eng (2) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_3_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_Fire
        self.operational_combinations = [('Eng (3) Fire On Ground', 'Eng (3) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_4_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_Fire
        self.operational_combinations = [('Eng (4) Fire On Ground', 'Eng (4) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_1_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_FuelBurn
        self.operational_combinations = [('Eng (1) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_2_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_FuelBurn
        self.operational_combinations = [('Eng (2) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_3_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_FuelBurn
        self.operational_combinations = [('Eng (3) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_4_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_FuelBurn
        self.operational_combinations = [('Eng (4) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_FuelBurn(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN1Max
        self.operational_combination_length = 255
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN2Max
        self.operational_combination_length = 255
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN3Max
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapLeverDetent(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapSurface(unittest.TestCase):
    def test_can_operate(self):
        combinations = FlapSurface.get_operational_combinations()
        #self.assertTrue(all('Longitude Prepared' in c for c in combinations))
        self.assertTrue(('Flap (L)', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (R)', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (L) Inboard', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (R) Inboard', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (L)', 'Flap (R)', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (L) Inboard', 'Flap (R) Inboard', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (L) Inboard', 'Flap (R) Inboard', 'Altitude AAL') in combinations)
        self.assertTrue(('Flap (L)', 'Flap (R)', 'Flap (L) Inboard', 
                         'Flap (R) Inboard', 'Frame', 'Approach', 
                         'Altitude AAL') in combinations)
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearDown(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearDownSelected(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_gear_down_selected_basic(self):
        gdn = M(array=np.ma.array(data=[0,0,0,1,1,1]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        dn_sel=GearDownSelected()
        dn_sel.derive(gdn, None, None, None)
        np.testing.assert_array_equal(dn_sel.array, [0,0,0,1,1,1])
        self.assertEqual(dn_sel.frequency, 1.0)
        self.assertAlmostEqual(dn_sel.offset, 0.1)

    def test_gear_down_selected_with_warnings(self):
        gdn = M(array=np.ma.array(data=[0,0,0,1,1,1]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        red = M(array=np.ma.array(data=[0,1,1,1,0,0]),
                values_mapping={0:'-',1:'Warning'},
                name='Gear (*) Red Warning', 
                frequency=1, 
                offset=0.6)
        dn_sel=GearDownSelected()
        dn_sel.derive(gdn, red, red, red)
        np.testing.assert_array_equal(dn_sel.array.raw, [0,1,1,1,1,1])
        self.assertEqual(dn_sel.frequency, 1.0)
        self.assertAlmostEqual(dn_sel.offset, 0.1)



class TestGearOnGround(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_gear_on_ground_basic(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.6)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1,1,1,1,1])
        self.assertEqual(wow.frequency, 2.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_common_word(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,1,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_left_only(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, None)
        np.testing.assert_array_equal(wow.array, [0,0,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_right_only(self):
        p_right = M(array=np.ma.array(data=[0,0,0,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.7)
        wow=GearOnGround()
        wow.derive(None, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.7)



class TestGearUpSelected(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_gear_up_selected_basic(self):
        gdn = M(array=np.ma.array(data=[1,1,1,0,0,0]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        up_sel=GearUpSelected()
        up_sel.derive(gdn, None, None, None)
        np.testing.assert_array_equal(up_sel.array, [0,0,0,1,1,1])
        self.assertEqual(up_sel.frequency, 1.0)
        self.assertAlmostEqual(up_sel.offset, 0.1)

    def test_gear_up_selected_with_warnings(self):
        gdn = M(array=np.ma.array(data=[1,1,1,0,0,0]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        red = M(array=np.ma.array(data=[0,1,1,1,0,0]),
                values_mapping={0:'-',1:'Warning'},
                name='Gear (*) Red Warning', 
                frequency=1, 
                offset=0.6)
        up_sel=GearUpSelected()
        up_sel.derive(gdn, red, red, red)
        np.testing.assert_array_equal(up_sel.array, [0,1,1,1,1,1])
        self.assertEqual(up_sel.frequency, 1.0)
        self.assertAlmostEqual(up_sel.offset, 0.1)


class TestHeadingTrueContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestILSGlideslope(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestILSLocalizer(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudePrepared(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LatitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Latitude Prepared' in c for c in combinations))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudePrepared(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LongitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Longitude Prepared' in c for c in combinations))
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariation(unittest.TestCase):
    def test_can_operate(self):
        combinations = MagneticVariation.get_operational_combinations()
        self.assertTrue(
            ('Latitude', 'Longitude', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude (Coarse)', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude', 'Latitude (Coarse)', 'Longitude', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)        
        
    def test_derive(self):
        mag_var = MagneticVariation()
        lat = P('Latitude', array=np.ma.array(
            [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            mask=[False, False, False, True, False, False]))
        lon = P('Longitude', array=np.ma.array(
            [-10.0, -10.1, -10.2, -10.3, -10.4, -10.5],
            mask=[False, False, True, True, False, False]))
        alt_aal = P('Altitude AAL', array=np.ma.array(
            [20000, 20100, 20200, 20300, 20400, 20500],
            mask=[False, False, False, False, True, False]))
        start_datetime = A('Start Datetime',
                           value=datetime.datetime(2013, 3, 23))
        mag_var.derive(lat, None, lon, None, alt_aal, start_datetime)
        expected_result = np.ma.array(
            [-6.06444546099, -6.07639239453, 0, 0, 0, -6.12614056456],
            mask=[False, False, True, True, True, False])
        ma_test.assert_almost_equal(mag_var.array, expected_result)
        # Test with Coarse parameters.
        mag_var.derive(None, lat, None, lon, alt_aal, start_datetime)
        expected_result = np.ma.array(
            [-6.06444546099, -6.07639239453, 0, 0, 0, -6.12614056456],
            mask=[False, False, True, True, True, False])        


class TestPackValvesOpen(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRelief(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollRate(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlat(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlopeToLanding(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrake(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakeSelected(unittest.TestCase):

    def test_can_operate(self):
        opts = SpeedbrakeSelected.get_operational_combinations()
        self.assertTrue(('Speedbrake Deployed',) in opts)
        self.assertTrue(('Speedbrake', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Speedbrake', 'Family') in opts)
        
    def test_derive(self):
        # test with deployed
        spd_sel = SpeedbrakeSelected()
        spd_sel.derive(
            deployed=M(array=np.ma.array(
                [0, 0, 0, 1, 1, 0]), values_mapping={1:'Deployed'}),
            armed=M(array=np.ma.array(
                [0, 0, 1, 1, 0, 0]), values_mapping={1:'Armed'})
        )
        self.assertEqual(list(spd_sel.array),
            ['Stowed', 'Stowed', 'Armed/Cmd Dn', 'Deployed/Cmd Up', 'Deployed/Cmd Up', 'Stowed'])
        
    def test_b737_speedbrake(self):
        self.maxDiff = None
        spd_sel = SpeedbrakeSelected()
        spdbrk = P(array=np.ma.array([0]*10 + [1.3]*20 + [0.2]*10))
        handle = P(array=np.ma.arange(40))
        # Follow the spdbrk only
        res = spd_sel.b737_speedbrake(spdbrk, None)
        self.assertEqual(list(res),
                        ['Stowed']*10 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)
        # Follow the handle only
        res = spd_sel.b737_speedbrake(None, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*32 + ['Deployed/Cmd Up']*5)
        # Follow the combination
        res = spd_sel.b737_speedbrake(spdbrk, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*7 + ['Deployed/Cmd Up']*20 + ['Armed/Cmd Dn']*5 + ['Deployed/Cmd Up']*5)
        

class TestStickShaker(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAT(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSAlert(unittest.TestCase):
    def test_can_operate(self):
        parameters = ['TAWS Caution Terrain',
                       'TAWS Caution',
                       'TAWS Dont Sink',
                       'TAWS Glideslope'
                       'TAWS Predictive Windshear',
                       'TAWS Pull Up',
                       'TAWS Sink Rate',
                       'TAWS Terrain',
                       'TAWS Terrain Warning Amber',
                       'TAWS Terrain Pull Up',
                       'TAWS Terrain Warning Red',
                       'TAWS Too Low Flap',
                       'TAWS Too Low Gear',
                       'TAWS Too Low Terrain',
                       'TAWS Windshear Warning',
                       ]
        for p in parameters:
            self.assertTrue(TAWSAlert.can_operate(p))

    def setUp(self):
        terrain_array = [ 1,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        pull_up_array = [ 0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0]

        self.airs = S(name='Airborne')
        self.airs.create_section(slice(5,15))
        self.terrain = M(name='TAWS Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        self.pull_up = M(name='TAWS Pull Up', array=np.ma.array(pull_up_array), values_mapping={1:'Warning'})
        self.taws_alert = TAWSAlert()

    def test_derive(self):
        result =        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_masked_values(self):
        result =        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]
        self.terrain.array[8] = np.ma.masked
        self.terrain.array[10] = np.ma.masked

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_zeros(self):
        result = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]
        
        terrain_array = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        
        caution = M(name='TAWS Caution Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        caution.array.mask = True

        self.taws_alert.get_derived((self.airs,
                                caution,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

class TestTailwind(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrottleLevers(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustReversers(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def setUp(self):
        eng_1_unlocked_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_1_deployed_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_2_unlocked_array = [ 1,  1,  1,  1,  0,  1,  0,  0,  0,  0]
        eng_2_deployed_array = [ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0]

        self.eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        self.eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})
        self.thrust_reversers = ThrustReversers()

    def test_derive(self):
        result = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_masked_value(self):
        self.eng_1_unlocked.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.eng_1_deployed.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.eng_2_unlocked.array.mask = [ 0,  0,  0,  1,  0,  0,  0,  1,  1,  0]
        self.eng_2_deployed.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 2,  2,  2,  2,  1,  2,  0,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result_array)
        np.testing.assert_equal(self.thrust_reversers.array.mask, result_mask)

    def test_derive_in_transit_avaliable(self):
        result = [ 2,  2,  1,  1,  1,  1,  0,  0,  0,  0]
        transit_array = [ 0,  0,  1,  1,  1,  1,  0,  0,  0,  0]
        eng_1_in_transit = M(name='Eng (1) Thrust Reverser In Transit', array=np.ma.array(transit_array), values_mapping={1:'In Transit'})
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                eng_1_in_transit,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)


class TestTurbulence(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        accel = np.ma.array([1]*40+[2]+[1]*40)
        turb = TurbulenceRMSG()
        turb.derive(P('Acceleration Vertical', accel, frequency=8))
        expected = np.array([0]*20+[0.156173762]*41+[0]*20)
        np.testing.assert_array_almost_equal(expected, turb.array.data)


class TestVOR1Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVOR2Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVerticalSpeedInertial(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        time = np.arange(100)
        zero = np.array([0]*10)
        acc_values = np.concatenate([zero, np.cos(time*np.pi*0.02), zero])
        vel_values = np.concatenate([zero, np.sin(time*np.pi*0.02), zero])
        ht_values = np.concatenate([zero, 1.0-np.cos(time*np.pi*0.02), zero])
        
        # For a 0-400ft leap, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL
        
        '''
        import matplotlib.pyplot as plt
        plt.plot(acc_values)
        plt.plot(vel_values)
        plt.plot(ht_values)
        plt.show()
        '''
        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 5, len(acc_values)-5)

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast)
        
        expected = vel_values

        '''
        import matplotlib.pyplot as plt
        plt.plot(expected)
        plt.plot(vsi.array)
        plt.show()
        '''
        # Just check the graphs are similar in shape - there will always be
        # errors because of the integration technique used.
        np.testing.assert_almost_equal(vsi.array, expected, decimal=-2)


class TestWindDirectionContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestCoordinatesSmoothed(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App('Approach Information',
            items=[ApproachItem('GO_AROUND', slice(3198.0, 3422.0),
                            ils_freq=108.55,
                            gs_est=slice(3200, 3390),
                            loc_est=slice(3199, 3445),
                            airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                                     'distance': 2.483270162497824,
                                     'elevation': 3301,
                                     'id': 3279,
                                     'latitude': 31.5058,
                                     'location': {'country': 'Afghanistan'},
                                     'longitude': 65.8478,
                                     'magnetic_variation': 'E001590 0506',
                                     'name': 'Kandahar'},
                            runway={'end': {'elevation': 3294,
                                            'latitude': 31.497511,
                                            'longitude': 65.833933},
                                    'id': 44,
                                    'identifier': '23',
                                    'magnetic_heading': 232.9,
                                    'start': {'elevation': 3320,
                                              'latitude': 31.513997,
                                              'longitude': 65.861714},
                                    'strip': {'id': 22,
                                              'length': 10532,
                                              'surface': 'ASP',
                                              'width': 147}}),
                   ApproachItem('LANDING', slice(12928.0, 13440.0),
                            ils_freq=111.3,
                            gs_est=slice(13034, 13262),
                            loc_est=slice(12929, 13347),
                            turnoff=13362.455208333333,
                            airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                                     'distance': 1.6842014290716794,
                                     'id': 3302,
                                     'latitude': 25.2528,
                                     'location': {'city': 'Dubai',
                                                  'country': 'United Arab Emirates'},
                                     'longitude': 55.3644,
                                     'magnetic_variation': 'E001315 0706',
                                     'name': 'Dubai Intl'},
                            runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                                    'glideslope': {'angle': 3.0,
                                                   'latitude': 25.246333,
                                                   'longitude': 55.378417,
                                                   'threshold_distance': 1508},
                                    'id': 22,
                                    'identifier': '30L',
                                    'localizer': {'beam_width': 4.5,
                                                  'frequency': 111300.0,
                                                  'heading': 300,
                                                  'latitude': 25.263139,
                                                  'longitude': 55.345722},
                                    'magnetic_heading': 299.7,
                                    'start': {'latitude': 25.243322, 'longitude': 55.381519},
                                    'strip': {'id': 11,
                                              'length': 13124,
                                              'surface': 'ASP',
                                              'width': 150}})])
        
        self.toff = [Section(name='Takeoff', 
                             slice=slice(372, 414, None), 
                             start_edge=371.32242063492066, 
                             stop_edge=413.12204760355382)]
        
        self.toff_rwy = A(name = 'FDR Takeoff Runway',
                          value = {'end': {'elevation': 4843, 
                                           'latitude': 34.957972, 
                                           'longitude': 69.272944},
                                   'id': 41,
                                   'identifier': '03',
                                   'magnetic_heading': 26.0,
                                   'start': {'elevation': 4862, 
                                             'latitude': 34.934306, 
                                             'longitude': 69.257},
                                   'strip': {'id': 21, 
                                             'length': 9852, 
                                             'surface': 'CON', 
                                             'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestCoordinatesSmoothed, self).setUp()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_precise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = True)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        
        cs = CoordinatesSmoothed()    
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),3)
        self.assertEqual(chunks,[slice(44, 372, None), 
                                 slice(3200, 3445, None), 
                                 slice(12930, 13424, None)])
        
    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_imprecise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)
        
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        cs = CoordinatesSmoothed()    
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])
        

        #import matplotlib.pyplot as plt
        #plt.plot(lat_new, lon_new)
        #plt.show()
        #plt.plot(lon.array, lat.array)
        #plt.show()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_visual(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        
        self.approaches.value[0].pop('ILS localizer established')
        self.approaches.value[1].pop('ILS localizer established')
        # Don't need to pop the glideslopes as these won't be looked for.
        cs = CoordinatesSmoothed()
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            self.toff, self.toff_rwy, self.approaches, mobile, precision)
        
        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])


class TestApproachRange(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App(items=[
            ApproachItem('GO_AROUND', slice(3198, 3422),
                     ils_freq=108.55,
                     gs_est=slice(3200, 3390),
                     loc_est=slice(3199, 3445),
                     airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                              'distance': 2.483270162497824,
                              'elevation': 3301,
                              'id': 3279,
                              'latitude': 31.5058,
                              'location': {'country': 'Afghanistan'},
                              'longitude': 65.8478,
                              'magnetic_variation': 'E001590 0506',
                              'name': 'Kandahar'},
                     runway={'end': {'elevation': 3294,
                                     'latitude': 31.497511,
                                     'longitude': 65.833933},
                             'id': 44,
                             'identifier': '23',
                             'magnetic_heading': 232.9,
                             'start': {'elevation': 3320,
                                       'latitude': 31.513997,
                                       'longitude': 65.861714},
                             'strip': {'id': 22,
                                       'length': 10532,
                                       'surface': 'ASP',
                                       'width': 147}}),
            ApproachItem('LANDING', slice(12928, 13440),
                     ils_freq=111.3,
                     gs_est=slice(13034, 13262),
                     loc_est=slice(12929, 13347),
                     turnoff=13362.455208333333,
                     airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                              'distance': 1.6842014290716794,
                              'id': 3302,
                              'latitude': 25.2528,
                              'location': {'city': 'Dubai',
                                           'country': 'United Arab Emirates'},
                              'longitude': 55.3644,
                              'magnetic_variation': 'E001315 0706',
                              'name': 'Dubai Intl'},
                     runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                             'glideslope': {'angle': 3.0,
                                            'latitude': 25.246333,
                                            'longitude': 55.378417,
                                            'threshold_distance': 1508},
                             'id': 22,
                             'identifier': '30L',
                             'localizer': {'beam_width': 4.5,
                                           'frequency': 111300.0,
                                           'heading': 300,
                                           'latitude': 25.263139,
                                           'longitude': 55.345722},
                             'magnetic_heading': 299.7,
                             'start': {'latitude': 25.243322, 'longitude': 55.381519},
                             'strip': {'id': 11,
                                       'length': 13124,
                                       'surface': 'ASP',
                                       'width': 150}})])
        
        self.toff = Section(name='Takeoff', 
                       slice=slice(372, 414, None), 
                       start_edge=371.32242063492066, 
                       stop_edge=413.12204760355382)
        
        self.toff_rwy = A(name='FDR Takeoff Runway',
                          value={'end': {'elevation': 4843, 
                                         'latitude': 34.957972, 
                                         'longitude': 69.272944},
                                 'id': 41,
                                 'identifier': '03',
                                 'magnetic_heading': 26.0,
                                 'start': {'elevation': 4862,
                                           'latitude': 34.934306,
                                           'longitude': 69.257},
                                 'strip': {'id': 21,
                                           'length': 9852,
                                           'surface': 'CON',
                                           'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestApproachRange, self).setUp()

    def test_range_basic(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Heading True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']
        
        ar = ApproachRange()    
        ar.derive(None, glide, None, None, None, hdg, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(3198, 3422, None), 
                                 slice(12928, 13423, None)])
        
    def test_range_full_param_set(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Track True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']
            gspd = hdf['Groundspeed']
        
        ar = ApproachRange()    
        ar.derive(gspd, glide, None, None, hdg, None, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(3198, 3422, None), 
                                 slice(12928, 13423, None)])
        
        
class TestStableApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            StableApproach.get_operational_combinations(),
            [('Approach', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min', 'Altitude AAL'),
             ('Approach', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min', 'Altitude AAL'),
                ])
        
    def test_stable_approach(self):
        stable = StableApproach()
        
        # Arrays will be 20 seconds long, index 4, 13,14,15 are stable
        #0. first and last values are not in approach slice
        apps = S()
        apps.create_section(slice(1,20))
        #1. gear up for index 0-2
        g = [ 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        gear = M(array=np.ma.array(g), values_mapping={1:'Down'})
        #2. landing flap invalid index 0, 5
        f = [ 5, 15, 15, 15, 15,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        flap = P(array=np.ma.array(f))
        #3. Heading stays within limits except for index 11-12, although we weren't on the heading sample 15 (masked out)
        h = [20, 20,  2,  3,  4,  8,  0,  0,  0,  0,  2, 20, 20,  8,  2,  0,  1,  1,  1,  1,  1]
        hm= [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        head = P(array=np.ma.array(h, mask=hm))
        #4. airspeed relative within limits for periods except 0-3
        a = [30, 30, 30, 26,  9,  8,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        aspd = P(array=np.ma.array(a))
        #5. glideslope deviation is out for index 9-11, last 4 values ignored due to alt cutoff
        g = [ 6,  6,  6,  6,  0, .5, .5,-.5,  0,1.1,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm= [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide = P(array=np.ma.array(g, mask=gm))
        #6. localizer deviation is out for index 7-10, last 4 values ignored due to alt cutoff
        l = [ 0,  0,  0,  0,  0,  0,  0,  2,  2,  2, -3,  0,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        loc = P(array=np.ma.array(l))
        #7. Vertical Speed too great at index 8, but change is smoothed out and at 17 (59ft)
        v = [-500] * 20
        v[6] = -2000
        v[18:19] = [-2000]*1
        vert_spd = P(array=np.ma.array(v))
        
        #TODO: engine cycling at index 12?
        
        #8. Engine power too low at index 5-12
        e = [80, 80, 80, 80, 80, 30, 20, 30, 20, 30, 20, 30, 44, 40, 80, 80, 80, 50, 50, 50, 50]
        eng = P(array=np.ma.array(e))
        
        # Altitude for cutoff heights, last 4 values are velow 100ft last 2 below 50ft
        al= range(2000,199,-200) + range(199,18, -20)
        # == [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 200, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19]
        alt = P(array=np.ma.array(al))
        # DERIVE
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide, loc, eng, alt)
        
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 6, 5, 5, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])
        self.assertEqual(list(stable.array.mask),
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
        #========== NO GLIDESLOPE ==========
        # Test without the use of Glideslope (not on it at 1000ft) therefore
        # instability for index 7-10 is now due to low Engine Power
        glide2 = P(array=np.ma.array([3.5]*20))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide2, loc, eng, alt)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 8, 8, 8, 8, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])
        
        #========== VERTICAL SPEED ==========
        # Test with a lot of vertical speed (rather than just gusts above)
        v = [-1200] * 20
        vert_spd = P(array=np.ma.array(v))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide2, loc, eng, alt)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 7, 7, 7, 7, 7, 3, 3, 7, 7, 7, 7, 9, 9, 9, 0])


class TestMasterWarning(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarning
        self.operational_combinations = [
            ('Master Warning (Capt)',),
            ('Master Warning (FO)',),
            ('Master Warning (Capt)', 'Master Warning (FO)'),
        ]

    def test_derive(self):
        warn_capt = M(
            name='Master Warning (Capt)',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Warning (FO)',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestConfiguration('test_time_taken2'))
    unittest.TextTestRunner(verbosity=2).run(suite)
