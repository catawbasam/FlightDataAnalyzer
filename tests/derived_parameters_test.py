import numpy as np

import datetime
import os
import shutil
import sys
import tempfile
import unittest

from mock import Mock, call, patch

from hdfaccess.file import hdf_file
from flightdatautilities import masked_array_testutils as ma_test
from flightdatautilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import Fast, Mobile
from analysis_engine.library import (align,
                                     max_value,
                                     np_ma_masked_zeros,
                                     np_ma_masked_zeros_like,
                                     np_ma_ones_like)

from analysis_engine.node import (Attribute,
                                  A,
                                  App,
                                  ApproachItem,
                                  KeyPointValue,
                                  KPV,
                                  KeyTimeInstance,
                                  KTI,
                                  load,
                                  M,
                                  Parameter,
                                  P,
                                  Section,
                                  S)
from analysis_engine.process_flight import process_flight
from analysis_engine.settings import GRAVITY_IMPERIAL, METRES_TO_FEET

from flight_phase_test import buildsection

from analysis_engine.derived_parameters import (
    #ATEngaged,
    AccelerationVertical,
    AccelerationForwards,
    AccelerationSideways,
    AccelerationAlongTrack,
    AccelerationAcrossTrack,
    Aileron,
    AimingPointRange,
    AirspeedForFlightPhases,
    AirspeedMinusV2,
    AirspeedMinusV2For3Sec,
    AirspeedReference,
    AirspeedReferenceLookup,
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
    ControlColumn,
    ControlColumnForce,
    ControlWheel,
    CoordinatesSmoothed,
    DescendForFlightPhases,
    DistanceTravelled,
    DistanceToLanding,
    Elevator,
    ElevatorLeft,
    ElevatorRight,
    Eng_EPRAvg,
    Eng_EPRMax,
    Eng_EPRMin,
    Eng_EPRMinFor5Sec,
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
    Eng_NpAvg,
    Eng_NpMax,
    Eng_NpMin,
    Eng_VibBroadbandMax,
    Eng_VibN1Max,
    Eng_VibN2Max,
    Eng_VibN3Max,
    Eng_1_FuelBurn,
    Eng_2_FuelBurn,
    Eng_3_FuelBurn,
    Eng_4_FuelBurn,
    Flaperon,
    FlapAngle,
    FuelQty,
    GrossWeightSmoothed,
    #GroundspeedAlongTrack,
    Heading,
    HeadingContinuous,
    HeadingIncreasing,
    HeadingTrue,
    Headwind,
    ILSFrequency,
    #ILSGlideslope,
    #ILSLocalizer,
    #ILSLocalizerRange,
    LatitudePrepared,
    LatitudeSmoothed,
    LongitudePrepared,
    LongitudeSmoothed,
    Mach,
    MagneticVariation,
    MagneticVariationFromRunway,
    Pitch,
    RollRate,
    SlatSurface,
    Speedbrake,
    VerticalSpeed,
    VerticalSpeedForFlightPhases,
    RateOfTurn,
    TrackDeviationFromRunway,
    Track,
    TrackContinuous,
    TrackTrue,
    TrackTrueContinuous,
    TurbulenceRMSG,
    V2,
    V2Lookup,
    VerticalSpeedInertial,
    WheelSpeed,
    WheelSpeedLeft,
    WheelSpeedRight,
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
    percent_similar = np.ma.sum(within_tolerance) / float(len(within_tolerance)) * 100
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
            combinations = map(set, self.node_class.get_operational_combinations())
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)




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
    def test_can_operate(self):
        expected = [('Airspeed', 'V2'),
                    ('Airspeed', 'V2 Lookup'),
                    ('Airspeed', 'V2', 'V2 Lookup')]
        opts = AirspeedMinusV2.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_recorded_v2(self):
        air_spd = P('Airspeed', np.ma.array([102] * 6))
        v2 = P('V2', np.ma.arange(90,120,5))
        amv2 = AirspeedMinusV2()
        result = amv2.get_derived([air_spd, v2, None])
        expected = np.ma.arange(12,-18,-5)
        ma_test.assert_array_equal(result.array, expected)
        
    def test_lookup_v2(self):
        air_spd = P('Airspeed', np.ma.array([102] * 6))
        v2_lu = P('V2 Lookup', np.ma.arange(90,120,5))
        amv2 = AirspeedMinusV2()
        result = amv2.get_derived([air_spd, None, v2_lu])
        expected = np.ma.arange(12,-18,-5)
        ma_test.assert_array_equal(result.array, expected)
        
    def test_recorded_preferred_to_lookup_v2(self):
        # If both forms are available, the recorded version is used in preference to the lookup tables.
        air_spd = P('Airspeed', np.ma.array([102] * 6))
        v2 = P('V2', np.ma.arange(90,120,5))
        v2_lu = P('V2 Lookup', np.ma.arange(80,110,5))
        amv2 = AirspeedMinusV2()
        result = amv2.get_derived([air_spd, v2, v2_lu])
        expected = np.ma.arange(12,-18,-5)
        ma_test.assert_array_equal(result.array, expected)
        

class TestAirspeedReference(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedReference
        self.operational_combinations = [
            ('Vapp',),
            ('Vref',),
            ('Airspeed', 'AFR Vapp'),
            ('Airspeed', 'AFR Vref'),
        ]

        self.air_spd = P('Airspeed', np.ma.array([200] * 128))
        self.afr_vapp = A('AFR Vapp', value=120)
        self.afr_vref = A('AFR Vref', value=120)
        self.vapp = P('Vapp', np.ma.array([120] * 128))
        self.vref = P('Vref', np.ma.array([120] * 128))
        self.approaches = App('Approach', items=[
            ApproachItem('LANDING', slice(105, 120)),
        ])
        self.expected = np_ma_masked_zeros(128)
        self.expected[self.approaches.get_last().slice] = 120

    def test_derive__afr_vapp(self):
        args = [self.air_spd, None, None, self.afr_vapp, None, self.approaches]
        node = self.node_class()
        node.get_derived(args)
        np.testing.assert_array_equal(node.array, self.expected)

    def test_derive__afr_vref(self):
        args = [self.air_spd, None, None, None, self.afr_vref, self.approaches]
        node = self.node_class()
        node.get_derived(args)
        np.testing.assert_array_equal(node.array, self.expected)

    def test_derive__vapp(self):
        args = [self.air_spd, self.vapp, None, None, None, self.approaches]
        node = self.node_class()
        node.get_derived(args)
        np.testing.assert_array_equal(node.array, self.vapp.array)

    def test_derive__vref(self):
        args = [self.air_spd, None, self.vref, None, None, self.approaches]
        node = self.node_class()
        node.get_derived(args)
        np.testing.assert_array_equal(node.array, self.vref.array)


class TestAirspeedReferenceLookup(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedReferenceLookup
        self.operational_combinations = [
            # Airbus:
            ('Airspeed', 'Series', 'Family', 'Approach And Landing', 'Touchdown', 'Gross Weight Smoothed', 'Configuration'),
            # Boeing:
            ('Airspeed', 'Series', 'Family', 'Approach And Landing', 'Touchdown', 'Gross Weight Smoothed', 'Flap'),
            ##### Propeller:
            ####('Airspeed', 'Series', 'Family', 'Approach And Landing', 'Touchdown', 'Eng (*) Np Avg'),
        ]

    def test_can_operate(self):
        self.assertTrue(AirspeedReferenceLookup.can_operate(
            ('Airspeed', 'Configuration', 'Approach And Landing', 'Touchdown', 'Gross Weight Smoothed'),
            series=Attribute('Series', 'A320-200'),
            family=Attribute('Family', 'A320'),
            engine=Attribute('Engine Series', 'CFM56-5B'),
            engine_type=Attribute('Engine Type', 'CFM56-5B5/P'),
        ))
        self.assertTrue(AirspeedReferenceLookup.can_operate(
            ('Airspeed', 'Flap', 'Approach And Landing', 'Touchdown', 'Gross Weight Smoothed'),
            series=Attribute('Series', 'B737-300'),
            family=Attribute('Family', 'B737'),
        ))
        self.assertFalse(AirspeedReferenceLookup.can_operate(
            ('Airspeed', 'Approach And Landing', 'Touchdown'),
            series=Attribute('Series', 'B737-300'),
            family=Attribute('Family', 'B737'),
        ))

    # TODO: Remove mock patch - our tables should be correct.
    @patch('analysis_engine.derived_parameters.get_vspeed_map')
    def test_airspeed_reference__boeing_lookup(self, vspeed_map):
        vspeed_table = Mock
        vspeed_table.vref = Mock(side_effect=[135, 130])
        vspeed_table.vref_settings = [15, 20, 30]
        vspeed_map.return_value = vspeed_table

        series = A('Series', value='B737-300')
        family = A('Family', value='B737 Classic')
        approaches = App('Approach Information', items=[
            ApproachItem('TOUCH_AND_GO', slice(3346, 3540)),
            ApproachItem('LANDING', slice(5502, 5795)),
        ])
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3450, 'Touchdown'),
            KeyTimeInstance(5700, 'Touchdown'),
        ])

        hdf_path = os.path.join(test_data_path, 'airspeed_reference.hdf5')
        hdf_copy = copy_file(hdf_path)
        with hdf_file(hdf_copy) as hdf:

            flap = P(**hdf['Flap'].__dict__)
            air_spd = P(**hdf['Airspeed'].__dict__)
            gw = P(**hdf['Gross Weight Smoothed'].__dict__)

            expected = np_ma_masked_zeros_like(hdf['Airspeed'].array)
            expected[approaches[0].slice] = 135
            expected[approaches[1].slice] = 130

            args = [flap, None, air_spd, gw, approaches, touchdowns, series, family, None, None, None]
            node = self.node_class()
            node.get_derived(args)
            np.testing.assert_array_equal(node.array, expected)

        if os.path.isfile(hdf_copy):
            os.remove(hdf_copy)

    @unittest.skip('Test Not Implemented')
    def test_derive__airbus(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive__beechcraft(self):
        air_spd = P('Airspeed', np.ma.array([0] * 120))
        series = A('Series', value='1900D')
        family = A('Family', value='1900')
        approaches = App('Approach Information', items=[
            ApproachItem('LANDING', slice(105, 120)),
        ])
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3450, 'Touchdown'),
            KeyTimeInstance(5700, 'Touchdown'),
        ])

        for detent, vref in ((35, 97), ):
            flap = M('Flap', np.ma.array([detent] * 120),
                     values_mapping={detent: str(detent)})
            args = [flap, None, air_spd, None, approaches, touchdowns, 
                    series, family, None, None, None]
            node = self.node_class()
            node.get_derived(args)
            expected = np.ma.array([vref] * 120)
            np.testing.assert_array_equal(node.array, expected)


# TODO: Check whether this needs altering so that vref is variable, not fixed.
#       Need a different vref for each approach? Discuss before changing...
class TestAirspeedRelative(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelative
        self.operational_combinations = [
            ('Airspeed', 'Airspeed Reference'),
            ('Airspeed', 'Airspeed Reference Lookup'),
            ('Airspeed', 'Airspeed Reference', 'Airspeed Reference Lookup'),
        ]

    def test_airspeed_for_phases_basic(self):
        # Note: Offset is frame-related, not superframe based, so is to some
        #       extent meaningless.
        air_spd=P('Airspeed', np.ma.array([200] * 128))
        air_spd_ref = P('Airspeed Reference', np.ma.array([120] * 128))
        node = AirspeedRelative()
        node.get_derived([air_spd, air_spd_ref])
        np.testing.assert_array_equal(node.array, np.ma.array([80] * 128))


class TestAirspeedTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AirspeedTrue.get_operational_combinations(), [
            ('Airspeed', 'Altitude STD'),
            ('Airspeed', 'Altitude STD', 'SAT'),
            ('Airspeed', 'Altitude STD', 'Takeoff'),
            ('Airspeed', 'Altitude STD', 'Landing'),
            ('Airspeed', 'Altitude STD', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Landing'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Landing'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Landing', 
             'Groundspeed'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Landing', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'Takeoff', 'Landing', 'Groundspeed', 
             'Acceleration Forwards'),
            ('Airspeed', 'Altitude STD', 'SAT', 'Takeoff', 'Landing', 
             'Groundspeed', 'Acceleration Forwards')
        ])
        
    def test_tas_basic(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD', np.ma.array([0, 20000, 40000]))
        sat = P('SAT', np.ma.array([20, -10, -55]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, sat)
        result = [100.864, 278.375, 555.595]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)
        
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
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5, 2, 2, 2,
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
        data = np.ma.array([-3, 0, 30, 80, 250, 580, 220, 70, 20, 25])
        alt_std = P(array=data + 300)
        slow_and_fast_data = np.ma.array([70] + [85] * 7 + [70]*2)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 200, 50, 0, 0])
        np.testing.assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_complex(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        rad_wave = np.copy(testwave)
        rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', rad_data),
                       P('Altitude STD', testwave),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(rad_data)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 7013, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 3308, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 217, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8965, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 3288, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[313], 17, decimal=0)

    def test_alt_aal_complex_with_mask(self):
        #testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            #np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        
        std_wave = np.ma.concatenate([np.ma.zeros(50), np.ma.arange(0,5000, 200), np.ma.zeros(50)+5000, np.ma.arange(5000,5500, 200), np.ma.zeros(50)+5500, np.ma.arange(5500, 0, -500), np.ma.zeros(50)])
        rad_wave = np.copy(std_wave) - 8
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 35, len(std_wave))
        std_wave += 1000
        rad_data[42:48] = np.ma.masked
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', np.ma.copy(rad_data)),
                       P('Altitude STD', np.ma.copy(std_wave)),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(std_wave, '-b')
        plt.plot(rad_data, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        #  Check alt aal does not try to jump to alt std in masked period of
        #  alt rad
        self.assertEqual(alt_aal.array[45].mask, True)

    def test_alt_aal_complex_doubled(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 5500
        rad_wave = np.copy(testwave)-500
        #rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        double_test = np.ma.concatenate((testwave, testwave))
        double_rad = np.ma.concatenate((rad_data, rad_data))
        phase_fast = buildsection('Fast', 0, 2*len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', double_rad),
                       P('Altitude STD', double_test),
                       phase_fast)
        
        
        '''import matplotlib.pyplot as plt
        plt.plot(double_test, '-b')
        plt.plot(double_rad, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()'''
        self.assertNotEqual(alt_aal.array[200], 0.0)
        np.testing.assert_equal(alt_aal.array[0], 0.0)


    def test_alt_aal_complex_no_rad_alt(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        testwave[255:]=testwave[254]
        testwave[:5]=500.0
        phase_fast = buildsection('Fast', 0, 254)
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, 
                       P('Altitude STD', testwave),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 6620, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 2915, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 0, decimal=0)

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

    @unittest.skip('Test Not Implemented')
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

    @unittest.skip('Test Not Implemented')
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
        expected = ('Control Wheel (Capt)', 
                    'Control Wheel (FO)', 
                    'Control Wheel Potentiometer', 
                    'Control Wheel Synchro')
        opts = ControlWheel.get_operational_combinations()
        self.assertEqual(opts[-1], expected)
        self.assertEqual(len(opts),6)

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_wheel(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cw = ControlWheel()
        cw.derive(self.cwc, self.cwf)
        blend_two_parameters.assert_called_once_with(self.cwc, self.cwf)


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


class TestEng_EPRAvg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRAvg
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMax
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMin
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMinFor5Sec
        self.operational_combinations = [('Eng (*) EPR Min',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_N1Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Avg
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

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


class TestEng_N1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Max
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

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
        
        
class TestEng_N1Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Min
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

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


class TestEng_N2Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Avg
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

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


class TestEng_N2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Max
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

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


class TestEng_N2Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Min
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

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


class TestEng_N3Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Avg
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

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


class TestEng_N3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Max
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

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


class TestEng_N3Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Min
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

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


class TestEng_NpAvg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpAvg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!


    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_NpAvg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_NpMax(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMax.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMax()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_NpMin(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMin.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.array(range(0, 10))
        b = np.ma.array(range(10,20))
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMin()
        eng.derive(P('a',a), P('b',b), None, None)
        ma_test.assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


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


class TestTrack(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Track
        self.operational_combinations = [('Track Continuous',)]

    def test_derive_basic(self):
        track = Parameter('Track Continuous', array=np.ma.arange(0, 1000, 100))
        node = self.node_class()
        node.derive(track)
        expected = [0, 100, 200, 300, 40, 140, 240, 340, 80, 180]
        ma_test.assert_equal(node.array, expected)


class TestTrackContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackContinuous
        self.operational_combinations = [('Heading Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = Parameter('Heading Continuous', array=np.ma.arange(0, 100, 10))
        drift = Parameter('Drift', array=np.ma.arange(0, 1, 0.1))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [0, 10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9]
        ma_test.assert_equal(node.array, expected)


class TestTrackTrue(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrue
        self.operational_combinations = [('Track True Continuous',)]

    def test_derive_basic(self):
        track = Parameter('Track True Continuous', array=np.ma.arange(0, 1000, 100))
        node = self.node_class()
        node.derive(track)
        expected = [0, 100, 200, 300, 40, 140, 240, 340, 80, 180]
        ma_test.assert_equal(node.array, expected)


class TestTrackTrueContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrueContinuous
        self.operational_combinations = [('Heading True Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = Parameter('Heading True Continuous', array=np.ma.arange(0, 100, 10))
        drift = Parameter('Drift', array=np.ma.arange(0, 1, 0.1))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [0, 10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9]
        ma_test.assert_equal(node.array, expected)

    def test_derive_extra(self):
        # Compare IRU Track Angle True (recorded) against the derived:
        heading = load(os.path.join(test_data_path, 'HeadingTrack_Heading_True.nod'))
        drift = load(os.path.join(test_data_path, 'HeadingTrack_Drift.nod'))
        node = self.node_class()
        node.derive(heading, drift)
        expected = load(os.path.join(test_data_path, 'HeadingTrack_IRU_Track_Angle_Recorded.nod'))
        assert_array_within_tolerance(node.array % 360, expected.array, 10, 98)


class TestTrackDeviationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            TrackDeviationFromRunway.get_operational_combinations(),
            [('Track True Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Takeoff', 'Approach Information'),
             ('Track True Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Takeoff', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff', 'FDR Takeoff Runway', 'Approach Information')]
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
            [('Heading Continuous', 'Magnetic Variation From Runway'),
             ('Heading Continuous', 'Magnetic Variation'),
             ('Heading Continuous', 'Magnetic Variation From Runway', 'Magnetic Variation')])
        
    def test_basic_magnetic(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        true = HeadingTrue()
        true.derive(head, None, var)
        expected = P('Heading True', np.ma.array([2.0, 8.0, 358.0, 348.0, 5.0]))
        ma_test.assert_array_equal(true.array, expected.array)
        
    def test_from_runway_used_in_preference(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        mag_var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        rwy_var = P('Magnetic Variation From Runway',np.ma.array([0,1,2,3,4]))
        true = HeadingTrue()
        true.derive(head, rwy_var, mag_var)
        expected = P('Heading True', np.ma.array([0, 6, 8, 358, 0]))
        ma_test.assert_array_equal(true.array, expected.array)


class TestILSFrequency(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS (1) Frequency', 'ILS (2) Frequency',),
                    ('ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',),
                    ('ILS (1) Frequency', 'ILS (2) Frequency',
                     'ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',)]
        opts = ILSFrequency.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])
        
    def test_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_single_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency', 
               np.ma.array(data=[1,2,108.10,108.15,111.95,112.00],
                           mask=[True,True,False,False,False,True]),
               offset = 0.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, None])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00], 
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)
        
    def test_ils_frequency_in_range(self):
        f1 = P('ILS (1) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS (2) Frequency', 
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2, None, None])
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


class TestV2(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = V2
        self.operational_combinations = [
            ('Airspeed', 'AFR V2'),
            ('Airspeed', 'AFR V2', 'Auto Speed Control'),
            ('Airspeed', 'AFR V2', 'Selected Speed'),
            ('Airspeed', 'Auto Speed Control', 'Selected Speed'),
            ('Airspeed', 'AFR V2', 'Auto Speed Control', 'Selected Speed'),
        ]

        self.air_spd = P('Airspeed', np.ma.array([200] * 128))
        self.afr_v2 = A('AFR V2', value=120)

    def test_derive__afr_v2(self):
        node = self.node_class()
        node.get_derived([self.air_spd, self.afr_v2, None, None])
        np.testing.assert_array_equal(node.array, np.array([120] * 128))

    def test_derive__spd_sel(self):
        spd_ctl = P('Auto Speed Control', np.ma.array([1] * 64 + [0] * 64))
        spd_sel = P('Selected Speed', np.ma.array([120] * 128))
        node = self.node_class()
        node.get_derived([self.air_spd, None, spd_ctl, spd_sel])
        expected = np.ma.array(data=[120] * 128, mask=[False] * 64 + [True] * 64)
        np.testing.assert_array_equal(node.array, expected)


class TestV2Lookup(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(V2Lookup.can_operate(
            ('Airspeed', 'Configuration', 'Liftoff', 'Gross Weight At Liftoff'),
            series=Attribute('Series', 'A320-200'),
            family=Attribute('Family', 'A320'),
            engine=Attribute('Engine Series', 'CFM56-5B'),
            engine_type=Attribute('Engine Type', 'CFM56-5B5/P'),
        ))
        self.assertTrue(V2Lookup.can_operate(
            ('Airspeed', 'Flap', 'Liftoff', 'Gross Weight At Liftoff'),
            series=Attribute('Series', 'B737-300'),
            family=Attribute('Family', 'B737'),
        ))
        self.assertFalse(V2Lookup.can_operate(
            ('Airspeed', 'Flap', 'Liftoff'),
            series=Attribute('Series', 'B737-300'),
            family=Attribute('Family', 'B737'),
        ))
    
    def test_derive__boeing(self):
        series = A('Series', value='B737-300')
        family = A('Family', value='B737 Classic')
        gw = KPV(name='Gross Weight At Liftoff', items=[
            KeyPointValue(index=451, value=54192.06),
        ])

        hdf_path = os.path.join(test_data_path, 'airspeed_reference.hdf5')
        hdf_copy = copy_file(hdf_path)
        with hdf_file(hdf_copy) as hdf:
            hdf_flap = hdf['Flap']
            hdf_flap.__dict__['values_mapping'] = {f: str(f) for f in np.ma.unique(hdf_flap.array)}
            flap = M(**hdf_flap.__dict__)
            air_spd = P(**hdf['Airspeed'].__dict__)

            args = [flap, None, air_spd, gw, series, family, None, None, None, None]

            node = V2Lookup()
            node.get_derived(args)
            expected = np.ma.array([150.868884] * 5888)
            np.testing.assert_array_equal(node.array, expected)

        if os.path.isfile(hdf_copy):
            os.remove(hdf_copy)

    @unittest.skip('Test Not Implemented')
    def test_derive__airbus(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive__beechcraft(self):
        air_spd = P('Airspeed', np.ma.array([0] * 20))
        series = A('Series', value='1900D')
        family = A('Family', value='1900')
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=5)])

        for detent, v2 in ((0, 125), (17.5, 114)):
            flap = M('Flap', np.ma.array([detent] * 20),
                     values_mapping={detent: str(detent)})
            args = [flap, None, air_spd, None, series, family, None, None, None, liftoffs]
            node = V2Lookup()
            node.get_derived(args)
            expected = np.ma.array([v2] * 20)
            np.testing.assert_array_equal(node.array, expected)


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

# Also Accelerations Lateral and Longitudinal are not tested yet.

class TestAileron(unittest.TestCase):
    
    def test_can_operate(self):
        opts = Aileron.get_operational_combinations()
        self.assertTrue(opts,
                        [('Aileron (L)',),
                         ('Aileron (R)',),
                         ('Aileron (L)', 'Aileron (R)'),
                        ])

    def test_normal_two_sensors(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right = P('Aileron (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        aileron = Aileron()
        aileron.get_derived([left, right])
        expected_data = np.ma.array([np.ma.masked, 1.5, 1.75, 1.5])
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_left_only(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.get_derived([left, None])
        expected_data = left.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_right_only(self):
        right = P('Aileron (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        aileron = Aileron()
        aileron.get_derived([None, right])
        expected_data = right.array
        np.testing.assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 2.0)
        self.assertEqual(aileron.offset, 0.3)    
        
    def test_aileron_with_flaperon(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        ail = Aileron()
        ail.derive(al, ar)
        # this section is averaging 4.833 degrees on the way in
        self.assertAlmostEqual(np.ma.average(ail.array[160:600]), 0.04, 1)
        # this section is averaging 9.106 degrees, ensure it gets moved to 0
        #self.assertAlmostEqual(np.ma.average(ail.array[800:1000]), 0.2, 1)
        assert_array_within_tolerance(ail.array[800:1000], 0, 4, 90)


class TestAileronTrim(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
        

class TestFlaperon(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Flaperon.can_operate(
            ('Aileron (L)', 'Aileron (R)'),
            series=Attribute('Series', 'A330-200'),
            family=Attribute('Family', 'A330')))
        
    def test_derive(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        series = A('Series', 'A330-200')
        family = A('Family', 'A330')
        flaperon = Flaperon()
        flaperon.derive(al, ar, series, family)
        def unique_values(array):
            y = np.bincount(array)
            ii = np.nonzero(y)[0]
            return zip(ii,y[ii])
        
        # ensure values are grouped into aileron settings accordingly
        self.assertEqual(unique_values(flaperon.array.astype(int)),
                         [(0, 22123), (5, 278), (10, 1151)])
        
        

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
    def test_can_operate(self):
        self.assertEqual(Elevator.get_operational_combinations(),
                         [('Elevator (L)',),
                          ('Elevator (R)',),
                          ('Elevator (L)', 'Elevator (R)'),
                          ])
        
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

class TestElevatorLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorLeft.get_operational_combinations()
        self.assertEqual(opts, [('Elevator (L) Potentiometer',),
                                ('Elevator (L) Synchro',),
                                ('Elevator (L) Potentiometer','Elevator (L) Synchro'),
                                ])
        
    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(None, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, None)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
class TestElevatorRight(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorRight.get_operational_combinations()
        self.assertEqual(opts, [('Elevator (R) Potentiometer',),
                                ('Elevator (R) Synchro',),
                                ('Elevator (R) Potentiometer','Elevator (R) Synchro'),
                                ])
        
    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(None, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, None)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)
              
    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, pot.array)
              
    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn)
        ma_test.assert_array_equal(elevator.array, syn.array)


class TestEng_FuelFlow(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')




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


class TestEng_VibBroadbandMax(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = Eng_VibBroadbandMax
        self.operational_combinations = [
            ('Eng (1) Vib Broadband',),
            ('Eng (1) Vib Broadband Accel A',),
            ('Eng (1) Vib Broadband Accel B',),
            ('Eng (1) Vib Broadband', 'Eng (2) Vib Broadband', 'Eng (3) Vib Broadband', 'Eng (4) Vib Broadband'),
            ('Eng (1) Vib Broadband Accel A', 'Eng (2) Vib Broadband Accel A', 'Eng (3) Vib Broadband Accel A', 'Eng (4) Vib Broadband Accel A'),
            ('Eng (1) Vib Broadband Accel B', 'Eng (2) Vib Broadband Accel B', 'Eng (3) Vib Broadband Accel B', 'Eng (4) Vib Broadband Accel B',),
        ]
    
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


class TestFlapAngle(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAngle
        self.operational_combinations = [
            ('Flap Angle (L)',),
            ('Flap Angle (R)',),
            ('Flap Angle (L) Inboard',),
            ('Flap Angle (R) Inboard',),
            ('Flap Angle (L)', 'Flap Angle (R)'),
            ('Flap Angle (L) Inboard', 'Flap Angle (R) Inboard'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard', 'Frame'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard', 'Slat Surface', 'Frame'),
        ]
    
    def test_derive_787(self):
        flap_angle_l = load(os.path.join(test_data_path,
                                         '787_flap_angle_l.nod'))
        flap_angle_r = load(os.path.join(test_data_path,
                                         '787_flap_angle_r.nod'))
        slat_l = load(os.path.join(test_data_path, '787_slat_l.nod'))
        slat_r = load(os.path.join(test_data_path, '787_slat_r.nod'))
        slat = SlatSurface()
        slat.derive(slat_l, slat_r)
        family = A('Family', 'B787')
        f = FlapAngle()
        f.derive(flap_angle_l, flap_angle_r, None, None, slat, None, family)
        # Include transitions.
        self.assertEqual(f.array[18635], 0.70000000000000007)
        self.assertEqual(f.array[18650], 1.0)
        self.assertEqual(f.array[18900], 5.0)
        # The original Flap data does not always record exact Flap settings.
        self.assertEqual(f.array[19070], 19.945)
        self.assertEqual(f.array[19125], 24.945)
        self.assertEqual(f.array[19125], 24.945)
        self.assertEqual(f.array[19250], 30.0)
    
    def test__combine_flap_set_basic(self):
        conf_map = {
            0:    (0, 0),
            1:    (50, 0),
            5:    (50, 5),
            15:   (50, 15),
            20:   (50, 20),
            25:   (100, 20),
            30:   (100, 30),
        }
        slat_array = np.ma.array([0, 50, 50, 50, 50, 100, 100])
        flap_array = np.ma.array([0, 0, 5, 15, 20, 20, 30])
        flap_slat = FlapAngle._combine_flap_slat(slat_array, flap_array,
                                                 conf_map)
        self.assertEqual(flap_slat.tolist(),
                         [0.0, 1.0, 5.0, 15.0, 20.0, 25.0, 30.0])

    def test_hercules(self):
        f = FlapAngle()
        f.derive(P(array=np.ma.array(range(0, 5000, 100) + range(5000, 0, -200))),
                 None, None, None, A('Frame', 'L382-Hercules'))


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
        lat = P('Latitude', array=np.ma.arange(10, 14, 0.01))
        lat.array[3] = np.ma.masked
        lon = P('Longitude', array=np.ma.arange(-10, -14, -0.01))
        lon.array[2:4] = np.ma.masked
        alt_aal = P('Altitude AAL', array=np.ma.arange(20000, 24000, 10))
        alt_aal.array[4] = np.ma.masked
        start_datetime = A('Start Datetime',
                           value=datetime.datetime(2013, 3, 23))
        mag_var.derive(lat, None, lon, None, alt_aal, start_datetime)
        ma_test.assert_almost_equal(
            mag_var.array[0:10],
            [-6.064445460989708, -6.065693019716132, -6.066940578442557,
             -6.068188137168981, -6.069435695895405, -6.070683254621829,
             -6.071930813348254, -6.073178372074678, -6.074425930801103,
             -6.075673489527527])
        # Test with Coarse parameters.
        mag_var.derive(None, lat, None, lon, alt_aal, start_datetime)
        ma_test.assert_almost_equal(
            mag_var.array[300:310],
            [-6.506129083442324, -6.507848687633959, -6.509568291825593,
             -6.511287896017228, -6.513007500208863, -6.514727104400498,
             -6.516446708592133, -6.518166312783767, -6.519885916975402,
             -6.521605521167037])

class TestMagneticVariationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        opts = MagneticVariationFromRunway.get_operational_combinations()
        self.assertEqual(opts,
                    [('HDF Duration',
                     'Heading During Takeoff',
                     'Heading During Landing',
                     'FDR Takeoff Runway',
                     'FDR Landing Runway',
                     )])
        
    def test_derive_both_runways(self):
        toff_rwy = {'end': {'elevation': 10,
                            'latitude': 52.7100630002283,
                            'longitude': -8.907803520515461},
                    'start': {'elevation': 43,
                              'latitude': 52.69327604095164,
                              'longitude': -8.943465355819775},
                    'strip': {'id': 2014, 'length': 10495, 
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {'end': {'elevation': 374,
                            'latitude': 49.024719,
                            'longitude': 2.524892},
                    'start': {'elevation': 377,
                              'latitude': 49.026694,
                              'longitude': 2.561689},
                    'strip': {'id': 2322, 'length': 8858,
                              'surface': 'ASP', 'width': 197}}
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            A('HDF Duration', 14272),
            KPV([KeyPointValue(index=62.143, value=58.014, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=213.869, value=266.5128, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -5.84060605)
        self.assertAlmostEqual(mag_var_rwy.array[62], -5.84060605)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[213], -1.20610555)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -1.20610555)
        
    def test_derive_only_takeoff_available(self):
        toff_rwy = {'end': {'elevation': 10,
                            'latitude': 52.7100630002283,
                            'longitude': -8.907803520515461},
                    'start': {'elevation': 43,
                              'latitude': 52.69327604095164,
                              'longitude': -8.943465355819775},
                    'strip': {'id': 2014, 'length': 10495, 
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {# MISSING VITAL LAT/LONG INFORMATION
                    'strip': {'id': 2322, 'length': 8858,
                              'surface': 'ASP', 'width': 197}}
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            A('HDF Duration', 14272),
            KPV([KeyPointValue(index=62.143, value=58.014, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=213.869, value=266.5128, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -5.84060605)
        self.assertAlmostEqual(mag_var_rwy.array[62], -5.84060605)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[213], -5.84060605)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -5.84060605)
        

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
    def test_can_operate(self):
        opts = RollRate.get_operational_combinations()
        self.assertTrue(('Roll',) in opts)
        
    def test_derive(self):
        roll = P(array=[0,2,4,6,8,10,12], name='Roll', frequency=2.0)
        rr = RollRate()
        rr.derive(roll)
        expected=np_ma_ones_like(roll.array)*4.0
        ma_test.assert_array_equal(expected[2:4], rr.array[2:4]) # Differential process blurs ends of the array, so just test the core part.

class TestSlat(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlatSurface(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            SlatSurface.get_operational_combinations(),
            [('Slat (L)',), ('Slat (R)',), ('Slat (L)', 'Slat (R)')])
    
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
    def test_can_operate(self):
        opts = Speedbrake.get_operational_combinations()
        self.assertTrue(('Spoiler (2)', 'Spoiler (7)', 'Frame') in opts)
        self.assertTrue(('Spoiler (7)', 'Spoiler (1)', 'Frame') in opts)
        self.assertTrue(('Spoiler (4)', 'Spoiler (9)', 'Frame') in opts)
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
     

class TestSAT(unittest.TestCase):
    # Note: the core function machtat2sat is tested by the library test.
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
        zero = np.array([0]*50)
        acc_values = np.concatenate([zero, np.cos(time*np.pi*0.02), zero])
        vel_values = np.concatenate([zero, np.sin(time*np.pi*0.02), zero])
        ht_values = np.concatenate([zero, 1.0-np.cos(time*np.pi*0.02), zero])
        
        # For a 0-400ft leap over 100 seconds, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0 # fpm
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL # g
        
        #import wx
        #import matplotlib.pyplot as plt
        #plt.plot(acc_values,'k')
        #plt.plot(vel_values,'b')
        #plt.plot(ht_values,'r')
        #plt.show()
        
        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 10, len(acc_values)-10)

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast)
        
        expected = vel_values

        # Just check the graphs are similar in shape - there will always be
        # errors because of the integration technique used.
        np.testing.assert_almost_equal(vsi.array, expected, decimal=-2)


class TestWheelSpeed(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeed.get_operational_combinations()
        self.assertEqual(opts, 
                         [('Wheel Speed (L)', 'Wheel Speed (R)'),
                          #('Wheel Speed (L)', 'Wheel Speed (C)', 'Wheel Speed (R)'),
                          ])
         
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedLeft.get_operational_combinations()
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)'), opts)
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)', 'Wheel Speed (L) (3)', 'Wheel Speed (L) (4)'), opts)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedRight(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedRight.get_operational_combinations()
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)'), opts)
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)', 'Wheel Speed (R) (3)', 'Wheel Speed (R) (4)'), opts)
         
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


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
                                 slice(12928, 13440, None)])
        
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
                                 slice(12928, 13440, None)])
        
        


if __name__ == '__main__':
    ##suite = unittest.TestSuite()
    ##suite.addTest(TestConfiguration('test_time_taken2'))
    ##unittest.TextTestRunner(verbosity=2).run(suite)
    pass
