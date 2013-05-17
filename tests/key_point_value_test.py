import operator
import os
import numpy as np
import sys
import unittest

from mock import Mock, call, patch

from flightdatautilities.geometry import midpoint

from analysis_engine.derived_parameters import Flap, StableApproach
from analysis_engine.library import align
from analysis_engine.node import (
    A, KPV, KTI, M, P, KeyPointValue,
    KeyTimeInstance, Section, S
)

from analysis_engine.key_point_values import (
    AccelerationLateralAtTouchdown,
    AccelerationLateralDuringLandingMax,
    AccelerationLateralMax,
    AccelerationLateralDuringTakeoffMax,
    AccelerationLateralWhileTaxiingStraightMax,
    AccelerationLateralWhileTaxiingTurnMax,
    AccelerationLateralOffset,
    AccelerationLongitudinalDuringTakeoffMax,
    AccelerationLongitudinalDuringLandingMax,
    AccelerationNormal20FtToFlareMax,
    AccelerationNormalWithFlapDownWhileAirborneMax,
    AccelerationNormalWithFlapDownWhileAirborneMin,
    AccelerationNormalWithFlapUpWhileAirborneMax,
    AccelerationNormalWithFlapUpWhileAirborneMin,
    AccelerationNormalAtLiftoff,
    AccelerationNormalAtTouchdown,
    AccelerationNormalLiftoffTo35FtMax,
    AccelerationNormalMax,
    AccelerationNormalOffset,
    Airspeed10000To8000FtMax,
    AirspeedBelow10000FtDuringDescentMax,
    Airspeed1000To500FtMax,
    Airspeed1000To500FtMin,
    Airspeed1000To8000FtMax,
    Airspeed3000To1000FtMax,
    Airspeed35To1000FtMax,
    Airspeed35To1000FtMin,
    Airspeed5000To3000FtMax,
    Airspeed500To20FtMax,
    Airspeed500To20FtMin,
    Airspeed8000To10000FtMax,
    Airspeed8000To5000FtMax,
    AirspeedWhileGearExtendingMax,
    AirspeedWhileGearRetractingMax,
    AirspeedAt8000Ft,
    AirspeedAt35FtDuringTakeoff,
    AirspeedAtGearDownSelection,
    AirspeedAtGearUpSelection,
    AirspeedAtLiftoff,
    AirspeedAtTouchdown,
    AirspeedDuringCruiseMax,
    AirspeedDuringCruiseMin,
    AirspeedGustsDuringFinalApproach,
    AirspeedDuringLevelFlightMax,
    AirspeedMax,
    AirspeedMinusV235To1000FtMax,
    AirspeedMinusV235To1000FtMin,
    AirspeedMinusV2For3Sec35To1000FtMax,
    AirspeedMinusV2For3Sec35To1000FtMin,
    AirspeedMinusV2At35FtDuringTakeoff,
    AirspeedMinusV2AtLiftoff,
    AirspeedDuringRejectedTakeoffMax,
    AirspeedRelative1000To500FtMax,
    AirspeedRelative1000To500FtMin,
    AirspeedRelative20FtToTouchdownMax,
    AirspeedRelative20FtToTouchdownMin,
    AirspeedRelative500To20FtMax,
    AirspeedRelative500To20FtMin,
    AirspeedRelativeAtTouchdown,
    AirspeedRelativeFor3Sec1000To500FtMax,
    AirspeedRelativeFor3Sec1000To500FtMin,
    AirspeedRelativeFor3Sec20FtToTouchdownMax,
    AirspeedRelativeFor3Sec20FtToTouchdownMin,
    AirspeedRelativeFor3Sec500To20FtMax,
    AirspeedRelativeFor3Sec500To20FtMin,
    AirspeedRelativeWithFlapDuringDescentMin,
    AirspeedTopOfDescentTo10000FtMax,
    AirspeedWithThrustReversersDeployedMin,
    AirspeedAtThrustReversersSelection,
    AirspeedTrueAtTouchdown,
    AirspeedVacatingRunway,
    AirspeedWithFlapDuringClimbMax,
    AirspeedWithFlapDuringClimbMin,
    AirspeedWithFlapDuringDescentMax,
    AirspeedWithFlapDuringDescentMin,
    AirspeedWithFlapMax,
    AirspeedWithFlapMin,
    AirspeedWithGearDownMax,
    AirspeedWithSpoilerDeployedMax,
    AltitudeAtFirstFlapChangeAfterLiftoff,
    AltitudeAtGearUpSelectionDuringGoAround,
    AltitudeDuringGoAroundMin,
    AltitudeAtLastFlapChangeBeforeTouchdown,
    AltitudeAtMachMax,
    AltitudeOvershootAtSuspectedLevelBust,
    AltitudeAtGearDownSelection,
    AltitudeAtGearUpSelection,
    AltitudeAtAPDisengagedSelection,
    AltitudeAtAPEngagedSelection,
    AltitudeAtATDisengagedSelection,
    AltitudeAtATEngagedSelection,
    AltitudeFirstStableDuringApproachBeforeGoAround,
    AltitudeFirstStableDuringLastApproach,
    AltitudeAtFlapExtension,
    AltitudeAtFirstFlapRetractionDuringGoAround,
    AltitudeLastUnstableDuringApproachBeforeGoAround,
    AltitudeLastUnstableDuringLastApproach,
    AltitudeSTDAtTouchdown,
    AltitudeSTDAtLiftoff,
    AltitudeQNHAtTouchdown,
    AltitudeQNHAtLiftoff,
    AltitudeMax,
    AltitudeWithFlapMax,
    AltitudeWithGearDownMax,
    AOADuringGoAroundMax,
    AOAWithFlapMax,
    APDisengagedDuringCruiseDuration,
    BrakePressureInTakeoffRollMax,
    ControlColumnStiffness,
    DecelerationFromTouchdownToStopOnRunway,
    DelayedBrakingAfterTouchdown,
    EngBleedValvesAtLiftoff,
    EngEPRDuringTaxiMax,
    EngEPRDuringTakeoff5MinRatingMax,
    EngEPRDuringGoAround5MinRatingMax,
    EngEPRDuringMaximumContinuousPowerMax,
    EngEPR500To50FtMax,
    EngEPR500To50FtMin,
    EngFireWarningDuration,
    EngGasTempDuringTakeoff5MinRatingMax,
    EngGasTempDuringGoAround5MinRatingMax,
    EngGasTempDuringMaximumContinuousPowerMax,
    EngGasTempDuringMaximumContinuousPowerForXMinMax,
    EngGasTempDuringEngStartMax,
    EngGasTempDuringEngStartForXSecMax,
    EngGasTempDuringFlightMin,
    EngN1AtTOGADuringTakeoff,
    EngN1DuringTaxiMax,
    EngN1DuringApproachMax,
    EngN1DuringTakeoff5MinRatingMax,
    EngN1DuringGoAround5MinRatingMax,
    EngN1DuringMaximumContinuousPowerMax,
    EngN1CyclesDuringFinalApproach,
    EngN1500To50FtMax,
    EngN1500To50FtMin,
    EngN1For5Sec1000To500FtMin,
    EngN1For5Sec500To50FtMin,
    EngN1WithThrustReversersInTransitMax,
    EngN1Below60PercentAfterTouchdownDuration,
    EngN2DuringTaxiMax,
    EngN2DuringTakeoff5MinRatingMax,
    EngN2DuringGoAround5MinRatingMax,
    EngN2DuringMaximumContinuousPowerMax,
    EngN2CyclesDuringFinalApproach,
    EngN3DuringTaxiMax,
    EngN3DuringTakeoff5MinRatingMax,
    EngN3DuringGoAround5MinRatingMax,
    EngN3DuringMaximumContinuousPowerMax,
    EngOilPressMax,
    EngOilPressMin,
    EngOilQtyMax,
    EngOilQtyMin,
    EngOilTempMax,
    EngOilTempForXMinMax,
    EngShutdownDuringFlightDuration,
    EngTorqueDuringTaxiMax,
    EngTorqueDuringTakeoff5MinRatingMax,
    EngTorqueDuringGoAround5MinRatingMax,
    EngTorqueDuringMaximumContinuousPowerMax,
    EngTorque500To50FtMax,
    EngTorque500To50FtMin,
    EngVibN1Max,
    EngVibN2Max,
    EngVibN3Max,
    FlapAtGearDownSelection,
    FlapAtLiftoff,
    FlapAtTouchdown,
    FlapWithGearUpMax,
    FlapWithSpeedbrakeDeployedMax,
    FlareDistance20FtToTouchdown,
    FlareDuration20FtToTouchdown,
    FuelQtyAtLiftoff,
    FuelQtyAtTouchdown,
    FuelQtyLowWarningDuration,
    GroundspeedAtTOGA,
    GroundspeedAtTouchdown,
    GroundspeedMax,
    GroundspeedDuringRejectedTakeoffMax,
    GroundspeedWhileTaxiingStraightMax,
    GroundspeedWhileTaxiingTurnMax,
    GroundspeedWithThrustReversersDeployedMin,
    GroundspeedVacatingRunway,
    GrossWeightAtLiftoff,
    GrossWeightAtTouchdown,
    HeadingDuringLanding,
    HeadingTrueDuringLanding,
    HeadingAtLowestAltitudeDuringApproach,
    HeadingDuringTakeoff,
    HeadingTrueDuringTakeoff,
    HeadingDeviationFromRunwayAbove80KtsAirspeedDuringTakeoff,
    HeadingDeviationFromRunwayAtTOGADuringTakeoff,
    HeadingDeviationFromRunwayAt50FtDuringLanding,
    HeadingDeviationFromRunwayDuringLandingRoll,
    HeadingVariation300To50Ft,
    HeadingVariation500To50Ft,
    HeadingVariationAbove100KtsAirspeedDuringLanding,
    HeadingVariationTouchdownPlus4SecTo60KtsAirspeed,
    HeadingVacatingRunway,
    HeightLossLiftoffTo35Ft,
    HeightLoss35To1000Ft,
    HeightLoss1000To2000Ft,
    HeightMinsToTouchdown,
    IANFinalApproachCourseDeviationMax,
    IANGlidepathDeviationMax,
    ILSFrequencyDuringApproach,
    ILSGlideslopeDeviation1500To1000FtMax,
    ILSGlideslopeDeviation1000To500FtMax,
    ILSGlideslopeDeviation500To200FtMax,
    ILSLocalizerDeviation1500To1000FtMax,
    ILSLocalizerDeviation1000To500FtMax,
    ILSLocalizerDeviation500To200FtMax,
    ILSLocalizerDeviationAtTouchdown,
    LastFlapChangeToTakeoffRollEndDuration,
    LastUnstableStateDuringApproachBeforeGoAround,
    LastUnstableStateDuringLastApproach,
    LatitudeAtLiftoff,
    LatitudeAtTouchdown,
    LatitudeSmoothedAtLiftoff,
    LatitudeSmoothedAtTouchdown,
    LatitudeAtLowestAltitudeDuringApproach,
    LongitudeAtLiftoff,
    LongitudeAtTouchdown,
    LongitudeSmoothedAtLiftoff,
    LongitudeSmoothedAtTouchdown,
    LongitudeAtLowestAltitudeDuringApproach,
    MachWhileGearExtendingMax,
    MachWhileGearRetractingMax,
    MachMax,
    MachWithFlapMax,
    MachWithGearDownMax,
    MagneticVariationAtTakeoffTurnOntoRunway,
    MagneticVariationAtLandingTurnOffRunway,
    PercentApproachStable,
    Pitch1000To500FtMax,
    Pitch1000To500FtMin,
    Pitch35To400FtMax,
    Pitch35To400FtMin,
    Pitch400To1000FtMax,
    Pitch400To1000FtMin,
    Pitch500To50FtMax,
    Pitch500To20FtMin,
    Pitch50FtToTouchdownMax,
    Pitch20FtToTouchdownMin,
    Pitch7FtToTouchdownMin,
    PitchAfterFlapRetractionMax,
    PitchAtLiftoff,
    PitchAtTouchdown,
    PitchAt35FtDuringClimb,
    PitchCyclesDuringFinalApproach,
    PitchDuringGoAroundMax,
    PitchLiftoffTo35FtMax,
    PitchRate35To1000FtMax,
    PitchRate20FtToTouchdownMax,
    PitchRate20FtToTouchdownMin,
    PitchRate2DegPitchTo35FtMax,
    PitchRate2DegPitchTo35FtMin,
    RateOfClimbMax,
    RateOfClimb35To1000FtMin,
    RateOfClimbBelow10000FtMax,
    RateOfClimbDuringGoAroundMax,
    RateOfDescent10000To5000FtMax,
    RateOfDescent5000To3000FtMax,
    RateOfDescent3000To2000FtMax,
    RateOfDescent2000To1000FtMax,
    RateOfDescent1000To500FtMax,
    RateOfDescent500To50FtMax,
    RateOfDescent50FtToTouchdownMax,
    RateOfDescentAtTouchdown,
    RateOfDescentBelow10000FtMax,
    RateOfDescentMax,
    RateOfDescentTopOfDescentTo10000FtMax,
    RateOfDescentDuringGoAroundMax,
    RollLiftoffTo20FtMax,
    Roll20To400FtMax,
    Roll400To1000FtMax,
    RollAbove1000FtMax,
    Roll1000To300FtMax,
    Roll300To20FtMax,
    Roll20FtToTouchdownMax,
    RollCyclesDuringFinalApproach,
    RollCyclesNotDuringFinalApproach,
    RudderDuringTakeoffMax,
    RudderCyclesAbove50Ft,
    RudderReversalAbove50Ft,
    SpeedbrakeDeployedDuringGoAroundDuration,
    SpeedbrakeDeployed1000To20FtDuration,
    SpeedbrakeDeployedWithPowerOnDuration,
    SpeedbrakeDeployedWithConfDuration,
    SpeedbrakeDeployedWithFlapDuration,
    StickPusherActivatedDuration,
    StickShakerActivatedDuration,
    TailClearanceDuringApproachMin,
    TailClearanceDuringLandingMin,
    TailClearanceDuringTakeoffMin,
    TailwindLiftoffTo100FtMax,
    Tailwind100FtToTouchdownMax,
    TCASRAWarningDuration,
    TCASRAReactionDelay,
    TCASRAInitialReactionStrength,
    TCASRAToAPDisengagedDuration,
    TCASTAWarningDuration,
    TOGASelectedDuringFlightDuration,
    TOGASelectedDuringGoAroundDuration,
    TerrainClearanceAbove3000FtMin,
    ThrottleCyclesDuringFinalApproach,
    ThrottleReductionToTouchdownDuration,
    ThrustAsymmetryDuringTakeoffMax,
    ThrustAsymmetryDuringFlightMax,
    ThrustAsymmetryDuringGoAroundMax,
    ThrustAsymmetryDuringApproachMax,
    ThrustAsymmetryWithThrustReversersDeployedMax,
    ThrustAsymmetryDuringApproachDuration,
    ThrustAsymmetryWithThrustReversersDeployedDuration,
    TouchdownTo60KtsDuration,
    TouchdownToElevatorDownDuration,
    TouchdownToThrustReversersDeployedDuration,
    TurbulenceDuringApproachMax,
    TurbulenceDuringCruiseMax,
    TurbulenceDuringFlightMax,
    TwoDegPitchTo35FtDuration,
    WindAcrossLandingRunwayAt50Ft,
    WindDirectionAtAltitudeDuringDescent,
    WindSpeedAtAltitudeDuringDescent,
    ZeroFuelWeight,
    MasterWarningDuration,
    MasterWarningDuringTakeoffDuration,
    MasterCautionDuringTakeoffDuration,
    TakeoffConfigurationWarningDuration,
    TakeoffConfigurationFlapWarningDuration,
    TakeoffConfigurationParkingBrakeWarningDuration,
    TakeoffConfigurationSpoilerWarningDuration,
    TakeoffConfigurationStabilizerWarningDuration,
    TAWSAlertDuration,
    TAWSGeneralWarningDuration,
    TAWSSinkRateWarningDuration,
    TAWSTooLowFlapWarningDuration,
    TAWSTerrainWarningDuration,
    TAWSTerrainPullUpWarningDuration,
    TAWSGlideslopeWarning1500To1000FtDuration,
    TAWSGlideslopeWarning1000To500FtDuration,
    TAWSGlideslopeWarning500To200FtDuration,
    TAWSTooLowTerrainWarningDuration,
    TAWSTooLowGearWarningDuration,
    TAWSPullUpWarningDuration,
    TAWSDontSinkWarningDuration,
    TAWSWindshearWarningBelow1500FtDuration,
    ThrustReversersDeployedDuration,
    PackValvesOpenAtLiftoff,
    IsolationValveOpenAtLiftoff,
)
from analysis_engine.key_time_instances import (
    AltitudeWhenClimbing,
    AltitudeWhenDescending,
    EngStop,
)
from analysis_engine.library import (max_abs_value, max_value, min_value)
from analysis_engine.flight_phase import Fast
from flight_phase_test import buildsection

debug = sys.gettrace() is not None


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(),
            self.operational_combinations,
        )


class CreateKPVsAtKPVsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKPVsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_kpvs = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_kpvs.assert_called_once_with(mock1.array, mock2)


class CreateKPVsAtKTIsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2)


class CreateKPVsWithinSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpvs_within_slices = Mock()
        node.derive(mock1, mock2)
        if hasattr(self, 'second_param_method_calls'):
            mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
            node.create_kpvs_within_slices.assert_called_once_with(\
                mock1.array, mock3.return_value, self.function)
        else:
            self.assertEqual(mock2.method_calls, [])
            node.create_kpvs_within_slices.assert_called_once_with(\
                mock1.array, mock2, self.function)


class CreateKPVFromSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpv_from_slices = Mock()
        node.derive(mock1, mock2)
        if hasattr(self, 'second_param_method_calls'):
            mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
            node.create_kpv_from_slices.assert_called_once_with(\
                mock1.array, mock3.return_value, self.function)
        else:
            self.assertEqual(mock2.method_calls, [])
            node.create_kpv_from_slices.assert_called_once_with(\
                mock1.array, mock2, self.function)


class ILSTest(NodeTest):
    '''
    '''

    def prepare__frequency__basic(self):
        # Let's give this a really hard time with alternate samples invalid and
        # the final signal only tuned just at the end of the data.
        ils_frequency = P(
            name='ILS Frequency',
            array=np.ma.array([108.5] * 6 + [114.05] * 4),
        )
        ils_frequency.array[0:10:2] = np.ma.masked
        ils_ests = buildsection('ILS Localizer Established', 2, 9)
        return ils_frequency, ils_ests

    def prepare__glideslope__basic(self):
        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(1.0 - np.cos(np.arange(0, 6.3, 0.1))),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            # Altitude from 1875 to 325 ft in 63 steps.
            array=np.ma.array((75 - np.arange(63)) * 25),
        )
        ils_ests = buildsection('ILS Glideslope Established', 2, 63)
        return ils_glideslope, alt_aal, ils_ests

    def prepare__glideslope__four_peaks(self):
        ils_glideslope = P(
            name='ILS Glideslope',
            array=np.ma.array(-0.2 - np.sin(np.arange(0, 12.6, 0.1))),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            # Altitude from 1875 to 325 ft in 63 steps.
            array=np.ma.array((75 - np.arange(63)) * 25),
        )
        ils_ests = buildsection('ILS Glideslope Established', 2, 56)
        return ils_glideslope, alt_aal, ils_ests

    def prepare__localizer__basic(self):
        ils_localizer = P(
            name='ILS Localizer',
            array=np.ma.array(np.arange(0, 12.6, 0.1)),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array(np.cos(np.arange(0, 12.6, 0.1)) * -1000 + 1000),
        )
        ils_ests = buildsection('ILS Localizer Established', 30, 115)
        return ils_localizer, alt_aal, ils_ests


##############################################################################
# Test Classes


##############################################################################
# Acceleration


########################################
# Acceleration: Lateral


class TestAccelerationLateralMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralMax
        self.operational_combinations = [
            ('Acceleration Lateral Offset Removed',),
            ('Acceleration Lateral Offset Removed', 'Groundspeed'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralAtTouchdown
        self.operational_combinations = [('Acceleration Lateral Offset Removed', 'Touchdown')]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_lat = Mock()
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3, 'Touchdown'),
            KeyTimeInstance(1, 'Touchdown'),
        ])
        node = AccelerationLateralAtTouchdown()
        node.derive(acc_lat, touchdowns)
        bump.assert_has_calls([
            call(acc_lat, touchdowns[0]),
            call(acc_lat, touchdowns[1]),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Lateral At Touchdown', slice(None, None)),
            KeyPointValue(1, 2.0, 'Acceleration Lateral At Touchdown', slice(None, None)),
        ])


class TestAccelerationLateralDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLateralDuringTakeoffMax
        self.operational_combinations = [('Acceleration Lateral Offset Removed', 'Takeoff Roll')]
        self.function = max_abs_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLateralDuringLandingMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralDuringLandingMax
        self.operational_combinations = [('Acceleration Lateral Offset Removed', 'Landing Roll', 'FDR Landing Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLateralWhileTaxiingStraightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralWhileTaxiingStraightMax
        self.operational_combinations = [('Acceleration Lateral Offset Removed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralWhileTaxiingTurnMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralWhileTaxiingTurnMax
        self.operational_combinations = [('Acceleration Lateral Offset Removed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralOffset(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationLateralOffset
        self.operational_combinations = [('Acceleration Lateral', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


########################################
# Acceleration: Longitudinal


class TestAccelerationLongitudinalDuringTakeoffMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalDuringTakeoffMax
        self.operational_combinations = [('Acceleration Longitudinal', 'Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationLongitudinalDuringLandingMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalDuringLandingMax
        self.operational_combinations = [('Acceleration Longitudinal', 'Landing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Acceleration: Normal


class TestAccelerationNormalMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormalMax
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Mobile')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationNormal20FtToFlareMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormal20FtToFlareMax
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 5), {})]

    def test_derive(self):
        '''
        Depends upon DerivedParameterNode.slices_from_to and library.max_value.
        '''
        # Test height range limit:
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(48, 0, -3))
        acc_norm = P('Acceleration Normal', np.ma.array(range(10, 18) + range(18, 10, -1)) / 10.0)
        node = AccelerationNormal20FtToFlareMax()
        node.derive(acc_norm, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(index=10, value=1.6, name='Acceleration Normal 20 Ft To Flare Max'),
        ])
        # Test peak acceleration:
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(32, 0, -2))
        node = AccelerationNormal20FtToFlareMax()
        node.derive(acc_norm, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(index=8, value=1.8, name='Acceleration Normal 20 Ft To Flare Max'),
        ])


class TestAccelerationNormalWithFlapUpWhileAirborneMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapUpWhileAirborneMax
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalWithFlapUpWhileAirborneMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapUpWhileAirborneMin
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalWithFlapDownWhileAirborneMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapDownWhileAirborneMax
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalWithFlapDownWhileAirborneMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalWithFlapDownWhileAirborneMin
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalAtLiftoff
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Liftoff')]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_norm = Mock()
        liftoffs = KTI('Liftoff', items=[
            KeyTimeInstance(3, 'Liftoff'),
            KeyTimeInstance(1, 'Liftoff'),
        ])
        node = AccelerationNormalAtLiftoff()
        node.derive(acc_norm, liftoffs)
        bump.assert_has_calls([
            call(acc_norm, liftoffs[0]),
            call(acc_norm, liftoffs[1]),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Normal At Liftoff', slice(None, None)),
            KeyPointValue(1, 2.0, 'Acceleration Normal At Liftoff', slice(None, None)),
        ])


class TestAccelerationNormalAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalAtTouchdown
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Touchdown')]

    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        bump.side_effect = [(3, 4), (1, 2)]
        acc_norm = Mock()
        touchdowns = KTI('Touchdown', items=[
            KeyTimeInstance(3, 'Touchdown'),
            KeyTimeInstance(1, 'Touchdown'),
        ])
        node = AccelerationNormalAtTouchdown()
        node.derive(acc_norm, touchdowns)
        bump.assert_has_calls([
            call(acc_norm, touchdowns[0]),
            call(acc_norm, touchdowns[1]),
        ])
        self.assertEqual(node, [
            KeyPointValue(3, 4.0, 'Acceleration Normal At Touchdown', slice(None, None)),
            KeyPointValue(1, 2.0, 'Acceleration Normal At Touchdown', slice(None, None)),
        ])


class TestAccelerationNormalLiftoffTo35FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormalLiftoffTo35FtMax
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAccelerationNormalOffset(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AccelerationNormalOffset
        self.operational_combinations = [('Acceleration Normal', 'Taxiing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Airspeed


########################################
# Airspeed: General


class TestAirspeedAt8000Ft(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AirspeedAt8000Ft
        self.operational_combinations = [('Airspeed', 'Altitude When Climbing', 'Altitude When Descending')]
    
    def test_derive_basic(self):
        air_spd = P('Airspeed', array=np.ma.arange(0, 200, 10))
        alt_climbs = AltitudeWhenClimbing(
            items=[KeyTimeInstance(9, '7000 Ft Climbing'),
                   KeyTimeInstance(10, '8000 Ft Climbing'),
                   KeyTimeInstance(11, '9000 Ft Climbing')])
        alt_descs = AltitudeWhenDescending(
            items=[KeyTimeInstance(15, '9000 Ft Descending'),
                   KeyTimeInstance(16, '8000 Ft Descending'),
                   KeyTimeInstance(17, '7000 Ft Descending'),
                   KeyTimeInstance(18, '8000 Ft Descending'),
                   KeyTimeInstance(19, '6000 Ft Descending')])
        node = self.node_class()
        node.derive(air_spd, alt_climbs, alt_descs)
        self.assertEqual(node,
                         [KeyPointValue(index=10, value=100.0, name='Airspeed At 8000 Ft'),
                          KeyPointValue(index=16, value=160.0, name='Airspeed At 8000 Ft'),
                          KeyPointValue(index=18, value=180.0, name='Airspeed At 8000 Ft')])


class TestAirspeedMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMax
        self.operational_combinations = [('Airspeed', 'Airborne')]
        self.function = max_value

    def test_derive_basic(self):
        testline = np.arange(0, 12.6, 0.1)
        testwave = np.cos(testline) * -100 + 100
        spd = P('Airspeed', np.ma.array(testwave))
        waves=np.ma.clump_unmasked(np.ma.masked_less(testwave, 80))
        airs = []
        for wave in waves:
            airs.append(Section('Airborne', wave, wave.start, wave.stop))
        kpv = AirspeedMax()
        kpv.derive(spd, airs)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 31)
        self.assertGreater(kpv[0].value, 199.9)
        self.assertLess(kpv[0].value, 200)
        self.assertEqual(kpv[1].index, 94)
        self.assertGreater(kpv[1].value, 199.9)
        self.assertLess(kpv[1].value, 200)


class TestAirspeedDuringCruiseMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringCruiseMax
        self.operational_combinations = [('Airspeed', 'Cruise')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedDuringCruiseMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringCruiseMin
        self.operational_combinations = [('Airspeed', 'Cruise')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedGustsDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedGustsDuringFinalApproach
        self.operational_combinations = [('Airspeed', 'Groundspeed', 'Altitude Radio', 'Airborne')]

    def test_derive_basic(self):
        # This function interpolates twice, hence the more complex test case.
        air_spd = P(
            name='Airspeed',
            array=np.ma.array([180, 180, 180, 180, 170, 150, 140, 120, 100]),
            frequency=1.0,
            offset=0.0,
        )
        gnd_spd = P(
            name='Groundspeed',
            array=np.ma.array([180, 180, 180, 180, 170, 100, 100, 100, 100]),
            frequency=1.0,
            offset=0.0,
        )
        alt_rad = P(
            name='Altitude Radio',
            array=np.ma.array([45, 45, 45, 45, 35, 25, 15, 5, 0]),
            frequency=1.0,
            offset=0.0,
        )
        airborne = S(items=[Section('Airborne', slice(3, 9), 3, 9)])
        kpv = AirspeedGustsDuringFinalApproach()
        kpv.get_derived([air_spd, gnd_spd, alt_rad, airborne])
        self.assertEqual(kpv[0].value, 25)
        self.assertEqual(kpv[0].index, 4.75)


########################################
# Airspeed: Climbing


class TestAirspeedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtLiftoff
        self.operational_combinations = [('Airspeed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAt35FtDuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAt35FtDuringTakeoff
        self.operational_combinations = [('Airspeed', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeed35To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed35To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To8000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed1000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 8000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed8000To10000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed8000To10000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (8000, 10000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Descending


class TestAirspeed10000To8000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed10000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (10000, 8000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed8000To5000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed8000To5000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (8000, 5000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed5000To3000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed5000To3000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (5000, 3000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed3000To1000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed3000To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (3000, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500,), {})]

    def test_derive_basic(self):
        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = P('Airspeed', np.ma.array(testwave))
        alt_ph = P('Altitude AAL For Flight Phases', np.ma.array(testwave) * 10)
        kpv = Airspeed1000To500FtMax()
        kpv.derive(spd, alt_ph)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertEqual(kpv[0].value, 91.250101656055278)
        self.assertEqual(kpv[1].index, 110)
        self.assertEqual(kpv[1].value, 99.557430201194919)


class TestAirspeed1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed500To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed500To20FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Airspeed500To20FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtTouchdown
        self.operational_combinations = [('Airspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedTrueAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedTrueAtTouchdown
        self.operational_combinations = [('Airspeed True', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Minus V2


class TestAirspeedMinusV2AtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2AtLiftoff
        self.operational_combinations = [('Airspeed Minus V2', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2At35FtDuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2At35FtDuringTakeoff
        self.operational_combinations = [('Airspeed Minus V2', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedMinusV235To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV235To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2For3Sec35To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2For3Sec35To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Relative

class TestAirspeedRelativeAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedRelativeAtTouchdown
        self.operational_combinations = [('Airspeed Relative', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative500To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelative20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelative20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelative20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Airspeed: Flap


class TestAirspeedWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapMax
        self.operational_combinations = [('Flap', 'Airspeed', 'Fast')]

    def test_derive(self):
        flap = [[0, 5, 10]] * 10
        flap = P('Flap', np.ma.array(reduce(operator.add, zip(*flap))))
        air_spd = P('Airspeed', np.ma.array(range(30)))
        fast = buildsection('Fast', 0, 30)
        flap.array[19] = np.ma.masked  # mask the max value
        air_spd_flap_max = AirspeedWithFlapMax()
        air_spd_flap_max.derive(flap, air_spd, fast)

        self.assertEqual(len(air_spd_flap_max), 2)
        self.assertEqual(air_spd_flap_max[0].name, 'Airspeed With Flap 5 Max')
        self.assertEqual(air_spd_flap_max[0].index, 18)  # 19 was masked
        self.assertEqual(air_spd_flap_max[0].value, 18)
        self.assertEqual(air_spd_flap_max[1].name, 'Airspeed With Flap 10 Max')
        self.assertEqual(air_spd_flap_max[1].index, 29)
        self.assertEqual(air_spd_flap_max[1].value, 29)

    def test_derive_alternative_method(self):
        # Note: This test will produce the following warning:
        #       "No flap settings - rounding to nearest 5"
        flap = [[0, 1, 2, 5, 10, 15, 25, 30, 40, 0]] * 2
        flap = P('Flap', np.ma.masked_array(reduce(operator.add, zip(*flap))))
        air_spd = P('Airspeed', np.ma.arange(20))
        fast = buildsection('Fast', 0, 20)
        step = Flap()
        step.derive(flap)
        air_spd_flap_max = AirspeedWithFlapMax()
        air_spd_flap_max.derive(step, air_spd, fast)

        self.assertEqual(air_spd_flap_max, [
            KeyPointValue(index=7, value=7, name='Airspeed With Flap 5 Max'),
            KeyPointValue(index=9, value=9, name='Airspeed With Flap 10 Max'),
            KeyPointValue(index=11, value=11, name='Airspeed With Flap 15 Max'),
            KeyPointValue(index=13, value=13, name='Airspeed With Flap 25 Max'),
            KeyPointValue(index=15, value=15, name='Airspeed With Flap 30 Max'),
            KeyPointValue(index=17, value=17, name='Airspeed With Flap 40 Max'),
        ])


class TestAirspeedWithFlapMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapMin
        self.operational_combinations = [('Flap', 'Airspeed', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDuringClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringClimbMax
        self.operational_combinations = [('Flap', 'Airspeed', 'Climb')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDuringClimbMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringClimbMin
        self.operational_combinations = [('Flap', 'Airspeed', 'Climb')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDuringDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringDescentMax
        self.operational_combinations = [('Flap', 'Airspeed', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDuringDescentMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapDuringDescentMin
        self.operational_combinations = [('Flap', 'Airspeed', 'Descent To Flare')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedRelativeWithFlapDuringDescentMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeWithFlapDuringDescentMin
        self.operational_combinations = [('Flap', 'Airspeed Relative', 'Descent To Flare')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


########################################
# Airspeed: Landing Gear


class TestAirspeedWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithGearDownMax
        self.operational_combinations = [('Airspeed', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        air_spd = P(
            name='Airspeed',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(air_spd, gear, airs)
        self.assertItemsEqual(node, [
            KeyPointValue(index=1, value=1.0, name='Airspeed With Gear Down Max'),
            KeyPointValue(index=3, value=3.0, name='Airspeed With Gear Down Max'),
            KeyPointValue(index=5, value=5.0, name='Airspeed With Gear Down Max'),
        ])


class TestAirspeedWhileGearRetractingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedWhileGearRetractingMax
        self.operational_combinations = [('Airspeed', 'Gear Retracting')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedWhileGearExtendingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedWhileGearExtendingMax
        self.operational_combinations = [('Airspeed', 'Gear Extending')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAtGearUpSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtGearUpSelection
        self.operational_combinations = [('Airspeed', 'Gear Up Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedAtGearDownSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedAtGearDownSelection
        self.operational_combinations = [('Airspeed', 'Gear Down Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Thrust Reversers


class TestAirspeedWithThrustReversersDeployedMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithThrustReversersDeployedMin
        self.operational_combinations = [('Airspeed True', 'Thrust Reversers', 'Eng (*) N1 Avg', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedAtThrustReversersSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAtThrustReversersSelection
        self.operational_combinations = [('Airspeed', 'Thrust Reversers', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


########################################
# Airspeed: Other


class TestAirspeedVacatingRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AirspeedVacatingRunway
        self.operational_combinations = [('Airspeed True', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedDuringRejectedTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AirspeedDuringRejectedTakeoffMax
        self.operational_combinations = [('Airspeed', 'Rejected Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedBelow10000FtDuringDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedBelow10000FtDuringDescentMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTopOfDescentTo10000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedTopOfDescentTo10000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Altitude QNH', 'FDR Landing Airport', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedDuringLevelFlightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedDuringLevelFlightMax
        self.operational_combinations = [('Airspeed', 'Level Flight')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithSpoilerDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithSpoilerDeployedMax
        self.operational_combinations = [('Airspeed', 'Spoiler')]

    def test_derive_basic(self):
        air_spd = P(
            name='Airspeed',
            array=np.ma.arange(10),
        )
        spoiler = M(
            name='Spoiler',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0]),
            values_mapping={0: '-', 1: 'Deployed'},
        )
        node = self.node_class()
        node.derive(air_spd, spoiler)
        self.assertItemsEqual(node, [
            KeyPointValue(index=5, value=5.0, name='Airspeed With Spoiler Deployed Max'),
            KeyPointValue(index=8, value=8.0, name='Airspeed With Spoiler Deployed Max'),
        ])


##############################################################################
# Angle of Attack


class TestAOAWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOAWithFlapMax
        self.operational_combinations = [('Flap', 'AOA', 'Fast')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAOADuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AOADuringGoAroundMax
        self.operational_combinations = [('AOA', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Autopilot


class TestAPDisengagedDuringCruiseDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APDisengagedDuringCruiseDuration
        self.operational_combinations = [('AP Engaged', 'Cruise')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################

class TestThrustReversersDeployedDuration(unittest.TestCase):
    
    def test_can_operate(self):
        ops = ThrustReversersDeployedDuration.get_operational_combinations()
        self.assertEqual(ops, [('Thrust Reversers', 'Landing')])
        
    def test_derive(self):
        rev = M(array=np.ma.zeros(30), values_mapping={
            0: 'Stowed', 1: 'In Transit', 2: 'Deployed',}, frequency=2)
        ldg = S(frequency=2)
        ldg.create_section(slice(5, 25))
        # no deployment
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 0)

        # deployed for a while
        rev.array[6:13] = 'Deployed'
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(dur[0].index, 5.5)
        self.assertEqual(dur[0].value, 3.5)


class TestTouchdownToThrustReversersDeployedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TouchdownToThrustReversersDeployedDuration
        self.operational_combinations = [('Thrust Reversers', 'Landing', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TouchdownToSpoilersDeployedDuration(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# TOGA Usage


class TestTOGASelectedDuringFlightDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TOGASelectedDuringFlightDuration
        self.operational_combinations = [('Takeoff And Go Around', 'Go Around And Climbout', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTOGASelectedDuringGoAroundDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TOGASelectedDuringGoAroundDuration
        self.operational_combinations = [('Takeoff And Go Around', 'Go Around And Climbout')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestLiftoffToClimbPitchDuration(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
    
    @unittest.skip('Test Not Implemented')    
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')    


##############################################################################
# Landing Gear


##################################
# Braking


class TestBrakePressureInTakeoffRollMax(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = BrakePressureInTakeoffRollMax
        self.operational_combinations = [('Brake Pressure', 'Takeoff Roll',)]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDelayedBrakingAfterTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = DelayedBrakingAfterTouchdown
        self.operational_combinations = [('Landing', 'Groundspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Altitude


########################################
# Altitude: General


class TestAltitudeMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AltitudeMax
        self.operational_combinations = [('Altitude STD Smoothed', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeSTDAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeSTDAtLiftoff
        self.operational_combinations = [('Altitude STD Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeSTDAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeSTDAtTouchdown
        self.operational_combinations = [('Altitude STD Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeQNHAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeQNHAtLiftoff
        self.operational_combinations = [('Altitude QNH', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeQNHAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeQNHAtTouchdown
        self.operational_combinations = [('Altitude QNH', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeDuringGoAroundMin(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeDuringGoAroundMin
        self.operational_combinations = [('Altitude AAL', 'Go Around')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeOvershootAtSuspectedLevelBust(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeOvershootAtSuspectedLevelBust
        self.operational_combinations = [('Altitude STD Smoothed', )]

    def test_derive_handling_no_data(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array([0] + [1000] * 4),
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [])

    def test_derive_up_down_and_up(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array(1.0 + np.sin(np.arange(0, 12.6, 0.1))) * 1000,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [
            KeyPointValue(index=16, value=999.5736030415051,
                name='Altitude Overshoot At Suspected Level Bust'),
            KeyPointValue(index=47, value=-1998.4666029387058,
                name='Altitude Overshoot At Suspected Level Bust'),
            KeyPointValue(index=79, value=1994.3775951461494,
                name='Altitude Overshoot At Suspected Level Bust'),
            # XXX: Was -933.6683091995028, ask Dave if min value is correct:
            KeyPointValue(index=110, value=-834.386031102394,
                name='Altitude Overshoot At Suspected Level Bust'),
        ])

    def test_derive_too_slow(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array(1.0 + np.sin(np.arange(0, 12.6, 0.1))) * 1000,
            frequency=0.02,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [])

    def test_derive_straight_up_and_down(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array(range(0, 10000, 50) + range(10000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [])

    def test_derive_up_and_down_with_overshoot(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array(range(0, 10000, 50) + range(10000, 9000, -50)
                + [9000] * 200 + range(9000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [
            KeyPointValue(index=200, value=1000,
                name='Altitude Overshoot At Suspected Level Bust'),
        ])

    def test_derive_up_and_down_with_undershoot(self):
        alt_std = P(
            name='Altitude STD',
            array=np.ma.array(range(0, 10000, 50) + [10000] * 200
                + range(10000, 9000, -50) + range(9000, 20000, 50)
                + range(20000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [
            KeyPointValue(index=420, value=-1000,
                name='Altitude Overshoot At Suspected Level Bust'),
        ])


########################################
# Altitude: Flap


class TestAltitudeWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeWithFlapMax
        self.operational_combinations = [('Flap', 'Altitude STD Smoothed', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtFlapExtension(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFlapExtension
        self.operational_combinations = [('Flap', 'Altitude AAL', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtFirstFlapChangeAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapChangeAfterLiftoff
        self.operational_combinations = [('Flap', 'Altitude AAL', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtLastFlapChangeBeforeTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtLastFlapChangeBeforeTouchdown
        self.operational_combinations = [('Flap', 'Altitude AAL', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtFirstFlapRetractionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapRetractionDuringGoAround
        self.operational_combinations = [('Altitude AAL', 'Flap Retraction During Go Around', 'Go Around And Climbout')]
        self.alt_aal = P(
            name='Altitude AAL',
            array=np.ma.concatenate([
                np.ma.array([0] * 10),
                np.ma.arange(40) * 1000,
                np.ma.array([40000] * 10),
                np.ma.arange(40, 0, -1) * 1000,
                np.ma.arange(1, 3) * 1000,
                np.ma.array([3000] * 10),
                np.ma.arange(3, -1, -1) * 1000,
                np.ma.array([0] * 10),
            ]),
        )
        self.go_arounds = buildsection('Go Around And Climbout', 97, 112)

    def test_derive_multiple_ktis(self):
        '''
        Create a single KPV within the Go Around And Climbout section.
        '''
        flap_rets = KTI('Go Around Flap Retracted', items=[
            KeyTimeInstance(100, 'Go Around Flap Retracted'),
            KeyTimeInstance(104, 'Go Around Flap Retracted'),
        ])
        node = AltitudeAtFirstFlapRetractionDuringGoAround()
        node.derive(self.alt_aal, flap_rets, self.go_arounds)
        self.assertEqual(node, [
            KeyPointValue(100, 1000, 'Altitude At First Flap Retraction During Go Around'),
        ])

    def test_derive_no_ktis(self):
        '''
        Create no KPVs without a Go Around Flap Retracted KTI.
        '''
        flap_rets = KTI('Go Around Flap Retracted', items=[])
        node = AltitudeAtFirstFlapRetractionDuringGoAround()
        node.derive(self.alt_aal, flap_rets, self.go_arounds)
        self.assertEqual(node, [])


########################################
# Altitude: Gear


class TestAltitudeAtGearDownSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearDownSelection
        self.operational_combinations = [('Altitude AAL', 'Gear Down Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearUpSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearUpSelection
        self.operational_combinations = [('Altitude AAL', 'Gear Up Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearUpSelectionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearUpSelectionDuringGoAround
        self.operational_combinations = [('Altitude AAL', 'Go Around And Climbout', 'Gear Up Selection During Go Around')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


########################################
# Altitude: Automated Systems


class TestAltitudeAtAPEngagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtAPEngagedSelection
        self.operational_combinations = [('Altitude AAL', 'AP Engaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtAPDisengagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtAPDisengagedSelection
        self.operational_combinations = [('Altitude AAL', 'AP Disengaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtATEngagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtATEngagedSelection
        self.operational_combinations = [('Altitude AAL', 'AT Engaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtATDisengagedSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtATDisengagedSelection
        self.operational_combinations = [('Altitude AAL', 'AT Disengaged Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeWithGearDownMax
        self.operational_combinations = [('Altitude AAL', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        alt_aal = P(
            name='Altitude',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(alt_aal, gear, airs)
        self.assertItemsEqual(node, [
            KeyPointValue(index=1, value=1.0, name='Altitude With Gear Down Max'),
            KeyPointValue(index=3, value=3.0, name='Altitude With Gear Down Max'),
            KeyPointValue(index=5, value=5.0, name='Altitude With Gear Down Max'),
        ])


########################################
# Altitude: Mach


class TestAltitudeAtMachMax(unittest.TestCase, CreateKPVsAtKPVsTest):

    def setUp(self):
        self.node_class = AltitudeAtMachMax
        self.operational_combinations = [('Altitude STD Smoothed', 'Mach Max')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################


class TestControlColumnStiffness(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ControlColumnStiffness
        self.operational_combinations = [('Control Column Force', 'Control Column', 'Fast')]

    def test_derive_too_few_samples(self):
        cc_disp = P('Control Column',
                            np.ma.array([0,.3,1,2,2.5,1.4,0,0]))
        cc_force = P('Control Column Force',
                             np.ma.array([0,2,4,7,8,5,2,1]))
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([100]*10)))
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff, [])

    def test_derive_max(self):
        testwave = np.ma.array((1.0 - np.cos(np.arange(0,6.3,0.1)))/2.0)
        cc_disp = P('Control Column', testwave * 10.0)
        cc_force = P('Control Column Force', testwave * 27.0)
        phase_fast = buildsection('Fast',0,63)
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff.get_first().index, 31)
        self.assertAlmostEqual(stiff.get_first().value, 2.7) # lb/deg


##############################################################################
# ILS


class TestILSFrequencyDuringApproach(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSFrequencyDuringApproach
        self.operational_combinations = [(
            'ILS Frequency',
            'ILS Localizer Established',
        )]

    def test_derive_basic(self):
        kpv = ILSFrequencyDuringApproach()
        kpv.derive(*self.prepare__frequency__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 2)
        self.assertEqual(kpv[0].value, 108.5)


class TestILSGlideslopeDeviation1500To1000FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation1500To1000FtMax
        self.operational_combinations = [(
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
        )]

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 31)
        self.assertAlmostEqual(kpv[0].value, 1.99913515027)

    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 16)
        self.assertAlmostEqual(kpv[0].value, -1.1995736)


class TestILSGlideslopeDeviation1000To500FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation1000To500FtMax
        self.operational_combinations = [(
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
        )]

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation1000To500FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 36)
        self.assertAlmostEqual(kpv[0].value, 1.89675842)

    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation1000To500FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 47)
        self.assertAlmostEqual(kpv[0].value, 0.79992326)


class TestILSGlideslopeDeviation500To200FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSGlideslopeDeviation500To200FtMax
        self.operational_combinations = [(
            'ILS Glideslope',
            'Altitude AAL For Flight Phases',
            'ILS Glideslope Established',
        )]

    def test_derive_basic(self):
        kpv = ILSGlideslopeDeviation500To200FtMax()
        kpv.derive(*self.prepare__glideslope__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 56)
        self.assertAlmostEqual(kpv[0].value, 0.22443412)

    # FIXME: Need to amend the test data as it produces no key point value for
    #        the 500-200ft altitude range. Originally this was not a problem
    #        before we split the 1000-250ft range in two.
    @unittest.expectedFailure
    def test_derive_four_peaks(self):
        kpv = ILSGlideslopeDeviation500To200FtMax()
        kpv.derive(*self.prepare__glideslope__four_peaks())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 0)          # FIXME
        self.assertAlmostEqual(kpv[0].value, 0.0)  # FIXME


class TestILSLocalizerDeviation1500To1000FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation1500To1000FtMax
        self.operational_combinations = [(
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
        )]

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation1500To1000FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        self.assertAlmostEqual(kpv[0].value, 4.7)
        self.assertAlmostEqual(kpv[1].value, 10.9)


class TestILSLocalizerDeviation1000To500FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation1000To500FtMax
        self.operational_combinations = [(
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
        )]

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation1000To500FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 52)
        self.assertEqual(kpv[1].index, 114)
        self.assertAlmostEqual(kpv[0].value, 5.2)
        self.assertAlmostEqual(kpv[1].value, 11.4)


class TestILSLocalizerDeviation500To200FtMax(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviation500To200FtMax
        self.operational_combinations = [(
            'ILS Localizer',
            'Altitude AAL For Flight Phases',
            'ILS Localizer Established',
        )]

    def test_derive_basic(self):
        kpv = ILSLocalizerDeviation500To200FtMax()
        kpv.derive(*self.prepare__localizer__basic())
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 56)
        self.assertAlmostEqual(kpv[0].value, 5.6)


class TestILSLocalizerDeviationAtTouchdown(unittest.TestCase, ILSTest):

    def setUp(self):
        self.node_class = ILSLocalizerDeviationAtTouchdown
        self.operational_combinations = [(
            'ILS Localizer',
            'ILS Localizer Established',
            'Touchdown',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive_basic(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestIANGlidepathDeviationMax(unittest.TestCase):

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('IAN Glidepath', 'Altitude AAL For Flight Phases', 'Approach And Landing'),
                    ('IAN Glidepath', 'Altitude AAL For Flight Phases', 'Approach And Landing', 'ILS Glideslope Established')]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANGlidepathDeviationMax

        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.apps = S(items=[Section('Approach And Landing', slice(2, 12), 2, 12)])
        self.ian = P(name='IAN Glidepath', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 3, 0, 0, 0, 0, 0], dtype=np.float,))
        self.ils = S(items=[Section('ILS Glideslope Established', slice(3, 12), 3, 12)])

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, None)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 3)
        self.assertAlmostEqual(kpv[0].name, 'IAN Glidepath Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.ils)
        self.assertEqual(len(kpv), 0)


class TestIANFinalApproachCourseDeviationMax(unittest.TestCase):

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('IAN Final Approach Course', 'Altitude AAL For Flight Phases', 'Approach And Landing'),
                    ('IAN Final Approach Course', 'Altitude AAL For Flight Phases', 'Approach And Landing', 'ILS Localizer Established')]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANFinalApproachCourseDeviationMax

        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.apps = S(items=[Section('Approach And Landing', slice(2, 12), 2, 12)])
        self.ian = P(name='IAN Final Approach Course', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 3, 0, 0, 0, 0, 0], dtype=np.float,))
        self.ils = S(items=[Section('ILS Localizer Established', slice(3, 12), 3, 12)])

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, None)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 3)
        self.assertAlmostEqual(kpv[0].name, 'IAN Final Approach Course Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.ils)
        self.assertEqual(len(kpv), 0)


##############################################################################
# Mach


########################################
# Mach: General


class TestMachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachMax
        self.operational_combinations = [('Mach', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Mach: Flap


class TestMachWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachWithFlapMax
        self.operational_combinations = [('Flap', 'Mach', 'Fast')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Mach: Landing Gear


class TestMachWithGearDownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MachWithGearDownMax
        self.operational_combinations = [('Mach', 'Gear Down', 'Airborne')]

    def test_derive_basic(self):
        mach = P(
            name='Mach',
            array=np.ma.arange(10),
        )
        gear = M(
            name='Gear Down',
            array=np.ma.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            values_mapping={0: 'Up', 1: 'Down'},
        )
        airs = buildsection('Airborne', 0, 7)
        node = self.node_class()
        node.derive(mach, gear, airs)
        self.assertItemsEqual(node, [
            KeyPointValue(index=1, value=1.0, name='Mach With Gear Down Max'),
            KeyPointValue(index=3, value=3.0, name='Mach With Gear Down Max'),
            KeyPointValue(index=5, value=5.0, name='Mach With Gear Down Max'),
        ])


class TestMachWhileGearRetractingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachWhileGearRetractingMax
        self.operational_combinations = [('Mach', 'Gear Retracting')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestMachWhileGearExtendingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = MachWhileGearExtendingMax
        self.operational_combinations = [('Mach', 'Gear Extending')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
########################################


class TestAltitudeFirstStableDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeFirstStableDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_stable_with_one_approach(self):
        firststable = AltitudeFirstStableDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,9,9,3,2,9,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,500,400,300]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 1.5)
        self.assertEqual(firststable[0].value, 850)

    def test_derive_stable_with_one_approach_all_unstable(self):
        firststable = AltitudeFirstStableDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,3,3,3,2,3,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,400,140,0]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 7.5)
        self.assertEqual(firststable[0].value, 0)
        
    def test_derive_two_approaches(self):
        # two approaches
        firststable = AltitudeFirstStableDuringLastApproach()
        #                                            stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9],
                                             mask=[0,0,0,0,  1,0,0,   0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100]))
        firststable.derive(stable, alt2app)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 7.5)
        self.assertEqual(firststable[0].value, 250)


class TestAltitudeFirstStableDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeFirstStableDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches_keeps_only_first(self):
        # two approaches
        firststable = AltitudeFirstStableDuringApproachBeforeGoAround()
        #                                            stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9],
                                             mask=[0,0,0,0,  1,0,0,   0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100]))
        firststable.derive(stable, alt2app)
        self.assertEqual(len(firststable), 1)
        self.assertEqual(firststable[0].index, 1.5)
        self.assertEqual(firststable[0].value, 850)
        
    def test_derive_stable_with_one_approach_is_ignored(self):
        firststable = AltitudeFirstStableDuringApproachBeforeGoAround()
        stable = StableApproach(array=np.ma.array([1,4,9,9,3,2,9,2]))
        alt = P(array=np.ma.array([1000,900,800,700,600,500,400,300]))
        firststable.derive(stable, alt)
        self.assertEqual(len(firststable), 0)
        
        
class TestAltitudeLastUnstableDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeLastUnstableDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches_uses_last_one(self):
        # two approaches
        lastunstable = AltitudeLastUnstableDuringLastApproach()
        #                                                 stable tooshort stable
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,   2,9,9,1,1],
                                             mask=[0,0,0,0,  1,0,0,   0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,20,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)  # will always land with AAL of 0
        
    def test_never_stable_stores_a_value(self):
        # if we were never stable, ensure we record a value at landing (0 feet)
        lastunstable = AltitudeLastUnstableDuringLastApproach()
        # not stable for either approach
        stable = StableApproach(array=np.ma.array([1,4,4,4,  3,2,2,2,2,2,1,1],
                                             mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,50,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 11.5)
        self.assertEqual(lastunstable[0].value, 0)


class TestAltitudeLastUnstableDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = AltitudeLastUnstableDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches(self):
        # two approaches
        lastunstable = AltitudeLastUnstableDuringApproachBeforeGoAround()
        #                                                 stable tooshort stable  last
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,9,2,9,9,1,1, 1,3,9,9,9],
                                             mask=[0,0,0,0,  1,0,0,0,0,0,0,0, 1,0,0,0,0]))
        alt2app = P(array=np.ma.array([1500,1400,1200,1000,
                                       900,800,700,600,500,400,300,200,
                                       100,50,20,0,0]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 2)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 1.5)
        self.assertEqual(lastunstable[0].value, 1300)
        self.assertEqual(lastunstable[1].index, 11.5)
        self.assertEqual(lastunstable[1].value, 0)  # was not stable prior to go around
        
    def test_never_stable_reads_0(self):
        lastunstable = AltitudeLastUnstableDuringApproachBeforeGoAround()
        # not stable for either approach
        stable = StableApproach(array=np.ma.array([1,4,4,4,  3,2,2,2,2,2,1,1],
                                             mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1000,900,800,700,600,500,400,300,200,100,50,20]))
        lastunstable.derive(stable, alt2app)
        self.assertEqual(len(lastunstable), 1)
        # stable to the end of the approach
        self.assertEqual(lastunstable[0].index, 3.5)
        self.assertEqual(lastunstable[0].value, 0)


class TestLastUnstableStateDuringLastApproach(unittest.TestCase):
    def test_can_operate(self):
        ops = LastUnstableStateDuringLastApproach.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach',)])
        
    def test_derive(self):
        state = LastUnstableStateDuringLastApproach()
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,4,2,9,9,9,9],
                                             mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        state.derive(stable)
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].index, 7.5)
        self.assertEqual(state[0].value, 2)

    @unittest.skip('This is so unlikely that its deemed unrealistic')
    def test_last_unstable_state_if_always_stable(self):
        # pas possible
        pass


class TestLastUnstableStateDuringApproachBeforeGoAround(unittest.TestCase):
    def test_can_operate(self):
        ops = LastUnstableStateDuringApproachBeforeGoAround.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach',)])
        
    def test_derive(self):
        state = LastUnstableStateDuringApproachBeforeGoAround()
        stable = StableApproach(array=np.ma.array([1,4,9,9,  3,2,4,2,9,9,9,9],
                                             mask=[0,0,0,0,  1,0,0,0,0,0,0,0]))
        state.derive(stable)
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].index, 1.5)
        self.assertEqual(state[0].value, 4)
        
        
class TestPercentApproachStableBelow(unittest.TestCase):
    def test_can_operate(self):
        ops = PercentApproachStable.get_operational_combinations()
        self.assertEqual(ops, [('Stable Approach', 'Altitude AAL')])

    def test_derive_two_approaches(self):
        percent_stable = PercentApproachStable()
        stable = StableApproach(array=np.ma.array([1,4,9,9,9, 3, 2,9,2,9,9,1,1],
                                             mask=[0,0,0,0,0, 1, 0,0,0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1100,1000,900,800,700,
                                       600,
                                       600,650,200,100,50,20,1]))
        percent_stable.derive(stable, alt2app)
        # both approaches below 
        self.assertEqual(len(percent_stable), 3)

        # First approach reaches only 1000 feet barrier (does not create 500ft)
        self.assertEqual(percent_stable[0].name,
            "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[0].index, 2)
        self.assertEqual(percent_stable[0].value, 75)  #3/4 == 75% - 4 samples below 1000ft
        
        # Last approach is below 1000 and 500 feet
        self.assertEqual(percent_stable[1].name,
            "Percent Approach Stable Below 1000 Ft During Last Approach")
        self.assertEqual(percent_stable[1].index, 7)
        self.assertEqual(percent_stable[1].value, (3/7.0)*100)  #3/7

        self.assertEqual(percent_stable[2].name,
            "Percent Approach Stable Below 500 Ft During Last Approach")
        self.assertEqual(percent_stable[2].index, 9)
        self.assertEqual(percent_stable[2].value, 40)  #2/5 == 40%
      
        
    def test_derive_three_approaches(self):
        # three approaches
        percent_stable = PercentApproachStable()
        stable = StableApproach(array=np.ma.array(
              [1,4,9,9,9, 3, 2,9,2,9,9,1,1, 3, 1,1,1,1,1],
         mask=[0,0,0,0,0, 1, 0,0,0,0,0,0,0, 1, 0,0,0,0,0]))
        alt2app = P(array=np.ma.array([1100,1000,900,800,700,  # approach 1
                                       1000,
                                       600,550,200,100,50,20,10,  # approach 2
                                       1000,
                                       300,200,100,30,0  # approach 3
                                       ]))
        percent_stable.derive(stable, alt2app)
        self.assertEqual(len(percent_stable), 5)
        
        # First Approach
        self.assertEqual(percent_stable[0].name,
            "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[0].index, 2)
        self.assertEqual(percent_stable[0].value, 75)
        
        # Second Approach
        self.assertEqual(percent_stable[1].name,
            "Percent Approach Stable Below 1000 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[1].index, 7)
        self.assertEqual(percent_stable[1].value, (3/7.0)*100)

        self.assertEqual(percent_stable[2].name,
            "Percent Approach Stable Below 500 Ft During Approach Before Go Around")
        self.assertEqual(percent_stable[2].index, 9)
        self.assertEqual(percent_stable[2].value, 40)  # 2/5
        
        # Last Approach (landing)
        # test that there was an approach but non was stable
        self.assertEqual(percent_stable[3].name,
            "Percent Approach Stable Below 1000 Ft During Last Approach")
        self.assertEqual(percent_stable[3].index, 14)
        self.assertEqual(percent_stable[3].value, 0)  # No stability == 0%

        self.assertEqual(percent_stable[4].name,
            "Percent Approach Stable Below 500 Ft During Last Approach")
        self.assertEqual(percent_stable[4].index, 14)
        self.assertEqual(percent_stable[4].value, 0)  # No stability == 0%


class TestDecelerateToStopOnRunwayDuration(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDecelerationFromTouchdownToStopOnRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = DecelerationFromTouchdownToStopOnRunway
        self.operational_combinations = [('Groundspeed', 'Touchdown',
            'Landing', 'Latitude Smoothed At Touchdown', 'Longitude Smoothed At Touchdown',
            'FDR Landing Runway', 'ILS Glideslope Established',
            'ILS Localizer Established', 'Precise Positioning')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFrom60KtToRunwayEnd(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromRunwayStartToTouchdown(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromTouchdownToRunwayEnd(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistancePastGlideslopeAntennaToTouchdown(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Bleed


class TestEngBleedValvesAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngBleedValvesAtLiftoff
        self.operational_combinations = [
            ('Liftoff', 'Eng (1) Bleed', 'Eng (2) Bleed'),
            ('Liftoff', 'Eng (1) Bleed', 'Eng (2) Bleed', 'Eng (3) Bleed'),
            ('Liftoff', 'Eng (1) Bleed', 'Eng (2) Bleed', 'Eng (4) Bleed'),
            ('Liftoff', 'Eng (1) Bleed', 'Eng (2) Bleed', 'Eng (3) Bleed', 'Eng (4) Bleed'),
        ]

    def test_derive(self):
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(name='Liftoff', index=3)])
        b1 = P('Eng (1) Bleed', array=[0, 0, 1, 0, 0])
        b2 = P('Eng (2) Bleed', array=[0, 0, 0, 1, 0])
        b3 = P('Eng (3) Bleed', array=[0, 1, 0, 0, 0])
        b4 = P('Eng (4) Bleed', array=[0, 1, 0, 1, 0])
        # Test with four engines, integer values:
        node = EngBleedValvesAtLiftoff()
        node.derive(liftoff, b1, b2, b3, b4)
        self.assertEqual(node, KPV('Eng Bleed Valves At Liftoff', items=[
            KeyPointValue(name='Eng Bleed Valves At Liftoff', index=3, value=2),
        ]))
        # Test with four engines, float values:
        b4f = P('Eng (4) Bleed', array=[0, 1.5, 0, 1.5, 0])
        node = EngBleedValvesAtLiftoff()
        node.derive(liftoff, b1, b2, b3, b4f)
        self.assertEqual(node, KPV('Eng Bleed Valves At Liftoff', items=[
            KeyPointValue(name='Eng Bleed Valves At Liftoff', index=3, value=2),
        ]))
        # Test with two engines, integer values:
        node = EngBleedValvesAtLiftoff()
        node.derive(liftoff, b1, b2, None, None)
        self.assertEqual(node, KPV('Eng Bleed Valves At Liftoff', items=[
            KeyPointValue(name='Eng Bleed Valves At Liftoff', index=3, value=1),
        ]))


##############################################################################
# Engine EPR


class TestEngEPRDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTaxiMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPRDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPR500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPR500To50FtMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPR500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngEPR500To50FtMin
        self.operational_combinations = [('Eng (*) EPR Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Engine Fire


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestEngFireWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngFireWarningDuration
        self.operational_combinations = [('Eng (*) Fire', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
        
        
##############################################################################
# Engine Shutdown


class TestEngShutdownDuringFlightDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngShutdownDuringFlightDuration
        self.operational_combinations = [('Eng (*) All Running', 'Airborne')]

    def test_derive(self):
        eng_running = M(
            array=np.ma.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
            values_mapping={0: 'Not Running', 1: 'Running'},
        )
        airborne = S(items=[Section('', slice(4, 20), 4.1, 20.1)])
        node = self.node_class(frequency=2)
        node.derive(eng_running=eng_running, airborne=airborne)
        # Note: Should only be single KPV (as must be greater than one second)
        self.assertEqual(node, [
            KeyPointValue(index=10, value=3.5, name='Eng Shutdown During Flight Duration'),
        ])


##############################################################################
# Engine Gas Temperature


class TestEngGasTempDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngGasTempDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngGasTempDuringGoAround5MinRatingMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngGasTempDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringMaximumContinuousPowerForXMinMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringMaximumContinuousPowerForXMinMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringEngStartMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringEngStartMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Eng (*) N2 Min', 'Takeoff Turn Onto Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringEngStartForXSecMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngGasTempDuringEngStartForXSecMax
        self.operational_combinations = [('Eng (*) Gas Temp Max', 'Eng (*) N2 Min', 'Takeoff Turn Onto Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempDuringFlightMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngGasTempDuringFlightMin
        self.operational_combinations = [('Eng (*) Gas Temp Min', 'Airborne')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine N1


class TestEngN1DuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringTaxiMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1DuringApproachMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringApproachMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1MaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1CyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1CyclesDuringFinalApproach
        self.operational_combinations = [('Eng (*) N1 Avg', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1500To50FtMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1500To50FtMin
        self.operational_combinations = [('Eng (*) N1 Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5Sec1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1For5Sec1000To500FtMin
        self.operational_combinations = [('Eng (*) N1 Min For 5 Sec', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5Sec500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN1For5Sec500To50FtMin
        self.operational_combinations = [('Eng (*) N1 Min For 5 Sec', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1WithThrustReversersInTransitMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1WithThrustReversersInTransitMax
        self.operational_combinations = [('Eng (*) N1 Avg', 'Thrust Reversers', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1Below60PercentAfterTouchdownDuration(unittest.TestCase):

    def test_can_operate(self):
        opts = EngN1Below60PercentAfterTouchdownDuration.get_operational_combinations()
        self.assertEqual(('Eng Stop', 'Eng (1) N1', 'Touchdown'), opts[0])
        self.assertEqual(('Eng Stop', 'Eng (2) N1', 'Touchdown'), opts[1])
        self.assertEqual(('Eng Stop', 'Eng (3) N1', 'Touchdown'), opts[2])
        self.assertEqual(('Eng Stop', 'Eng (4) N1', 'Touchdown'), opts[3])
        self.assertTrue(('Eng Stop', 'Eng (1) N1', 'Eng (2) N1', 'Touchdown') in opts)
        self.assertTrue(all(['Touchdown' in avail for avail in opts]))
        self.assertTrue(all(['Eng Stop' in avail for avail in opts]))

    def test_derive_eng_n1_cooldown(self):
        #TODO: Add later if required
        #gnd = S(items=[Section('', slice(10,100))])
        eng_stop = EngStop(items=[KeyTimeInstance(90, 'Eng (1) Stop'),])
        eng = P(array=np.ma.array([100] * 60 + [40] * 40)) # idle for 40
        tdwn = KTI(items=[KeyTimeInstance(30), KeyTimeInstance(50)])
        max_dur = EngN1Below60PercentAfterTouchdownDuration()
        max_dur.derive(eng_stop, eng, eng, None, None, tdwn)
        self.assertEqual(max_dur[0].index, 60) # starts at drop below 60
        self.assertEqual(max_dur[0].value, 30) # stops at 90
        self.assertTrue('Eng (1)' in max_dur[0].name)
        # Eng (2) should not be in the results as it did not have an Eng Stop KTI
        ##self.assertTrue('Eng (2)' in max_dur[1].name)
        self.assertEqual(len(max_dur), 1)


class TestEngN1AtTOGADuringTakeoff(unittest.TestCase):

    def test_can_operate(self):
        opts = EngN1AtTOGADuringTakeoff.get_operational_combinations()
        self.assertEqual([('Eng (*) N1 Min', 'Takeoff And Go Around', 'Takeoff')], opts)

    def test_derive_eng_n1_cooldown(self):
        eng_n1_min = P(array=np.ma.arange(10, 20))
        toga = M(array=np.ma.zeros(10), values_mapping={0: '-', 1:'TOGA'})
        toga.array[3] = 1
        toff = buildsection('Takeoff', 2,6)
        n1_toga = EngN1AtTOGADuringTakeoff()
        n1_toga.derive(eng_n1=eng_n1_min,
                      toga=toga,
                      takeoff=toff)
        self.assertEqual(n1_toga[0].value, 12.5)
        self.assertEqual(n1_toga[0].index, 2.5)


##############################################################################
# Engine N2


class TestEngN2DuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringTaxiMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN2DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2MaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN2DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N2 Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2CyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN2CyclesDuringFinalApproach
        self.operational_combinations = [('Eng (*) N2 Avg', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine N3


class TestEngN3DuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringTaxiMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3DuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN3DuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN3DuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3MaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN3DuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) N3 Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Throttles


class TestThrottleReductionToTouchdownDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrottleReductionToTouchdownDuration
        self.operational_combinations = [('Throttle Levers', 'Landing', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Oil Pressure


class TestEngOilPressMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngOilPressMax
        self.operational_combinations = [('Eng (*) Oil Press Max', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilPressMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilPressMin
        self.operational_combinations = [('Eng (*) Oil Press Min', 'Airborne')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Oil Quantity


class TestEngOilQtyMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilQtyMax
        self.operational_combinations = [('Eng (*) Oil Qty Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilQtyMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilQtyMin
        self.operational_combinations = [('Eng (*) Oil Qty Min', 'Airborne')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Engine Oil Temperature


class TestEngOilTempMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngOilTempMax
        self.operational_combinations = [('Eng (*) Oil Temp Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngOilTempForXMinMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngOilTempForXMinMax
        self.operational_combinations = [('Eng (*) Oil Temp Max', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')

    def test_derive_all_oil_data_masked(self):
        # This has been a specific problem, hence this test.
        oil_temp = P(
            name='Eng (*) Oil Temp Max',
            array=np.ma.array(data=range(123, 128), dtype=float, mask=True),
        )
        node = EngOilTempForXMinMax()
        node.derive(oil_temp)
        self.assertEqual(node, KPV('Eng Oil Temp For X Min Max', items=[]))


##############################################################################
# Engine Torque


class TestEngTorqueDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueDuringTaxiMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTorqueDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorqueDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngTorqueDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorque500To50FtMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500To50FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTorque500To50FtMin
        self.operational_combinations = [('Eng (*) Torque Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Engine Vibration


class TestEngVibN1Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN1Max
        self.operational_combinations = [('Eng (*) Vib N1 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibN2Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN2Max
        self.operational_combinations = [('Eng (*) Vib N2 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibN3Max(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibN3Max
        self.operational_combinations = [('Eng (*) Vib N3 Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################


class TestEventMarkerPressed(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLoss1000To2000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLoss1000To2000Ft
        self.operational_combinations = [(
            'Descend For Flight Phases',
            'Altitude AAL For Flight Phases',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLoss35To1000Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLoss35To1000Ft
        self.operational_combinations = [(
            'Descend For Flight Phases',
            'Altitude AAL For Flight Phases',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLossLiftoffTo35Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeightLossLiftoffTo35Ft
        self.operational_combinations = [(
            'Vertical Speed Inertial',
            'Altitude AAL For Flight Phases',
        )]

    def test_basic(self):
        vs = P(
            name='Vertical Speed Inertial',
            array=np.ma.array([0.0, 0, 1, 2, 1, 0, -1, -2, 0, 4]),
            frequency=2.0,
            offset=0.0,
        )
        alt = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([0.0, 0, 4, 15, 40]),
            frequency=1.0,
            offset=0.5,
        )
        ht_loss = HeightLossLiftoffTo35Ft()
        ht_loss.get_derived((vs, alt))
        self.assertEqual(ht_loss[0].value, 0.75)


class TestHeightOfBouncedLanding(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeAtTouchdown
        self.operational_combinations = [
            ('Latitude', 'Touchdown'),
            ('Touchdown', 'AFR Landing Airport'),
            ('Touchdown', 'AFR Landing Runway'),
            ('Touchdown', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport'),
            ('Latitude', 'Touchdown', 'AFR Landing Runway'),
            ('Latitude', 'Touchdown', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Touchdown', 'AFR Landing Airport', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Latitude (Coarse)')
        ]

    def test_derive_with_latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        tdwns = KTI(name='Touchdown')
        afr_land_rwy = None
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_rwy, afr_land_apt, lat_c)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, tdwns)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_land_rwy(self):
        lat = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = A(name='AFR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, lat_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_land_apt(self):
        lat = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = None
        afr_land_apt = A(name='AFR Landing Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

class TestLatitudeAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LatitudeAtLiftoff
        self.operational_combinations = [
            ('Latitude', 'Liftoff'),
            ('Liftoff', 'AFR Takeoff Airport'),
            ('Liftoff', 'AFR Takeoff Runway'),
            ('Liftoff', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Runway'),
            ('Latitude', 'Liftoff', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Liftoff', 'AFR Takeoff Airport', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
            ('Latitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude (Coarse)'),
        ]

    def test_derive_with_latitude(self):
        lat = P(name='Latitude')
        lat.array = Mock()
        liftoffs = KTI(name='Liftoff')
        afr_toff_rwy = None
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_rwy, afr_toff_apt, None)
        node.create_kpvs_at_ktis.assert_called_once_with(lat.array, liftoffs)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_toff_rwy(self):
        lat = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = A(name='AFR Takeoff Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, lat_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_toff_apt(self):
        lat = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = A(name='AFR Takeoff Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lat, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLatitudeSmoothedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeSmoothedAtTouchdown
        self.operational_combinations = [('Latitude Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLatitudeSmoothedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeSmoothedAtLiftoff
        self.operational_combinations = [('Latitude Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLatitudeAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LatitudeAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Latitude Prepared', 'Lowest Altitude During Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeAtTouchdown
        self.operational_combinations = [
            ('Longitude', 'Touchdown'),
            ('Touchdown', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport'),
            ('Longitude', 'Touchdown', 'AFR Landing Runway'),
            ('Longitude', 'Touchdown', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport', 'AFR Landing Runway', 'Longitude (Coarse)')
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Latitude')
        lon.array = Mock()
        tdwns = KTI(name='Touchdown')
        afr_land_rwy = None
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_rwy, afr_land_apt, lat_c)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, tdwns)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_land_rwy(self):
        lon = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = A(name='AFR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_land_apt = None
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, lon_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_land_apt(self):
        lon = None
        tdwns = KTI(name='Touchdown', items=[KeyTimeInstance(index=0)])
        afr_land_rwy = None
        afr_land_apt = A(name='AFR Landing Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        lat_c = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, tdwns, afr_land_apt, afr_land_rwy, lat_c)
        node.create_kpv.assert_called_once_with(tdwns[-1].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLongitudeAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LongitudeAtLiftoff
        self.operational_combinations = [
            ('Longitude', 'Liftoff'),
            ('Liftoff', 'AFR Takeoff Airport'),
            ('Liftoff', 'AFR Takeoff Runway'),
            ('Liftoff', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Runway'),
            ('Longitude', 'Liftoff', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Liftoff', 'AFR Takeoff Airport', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
            ('Longitude', 'Liftoff', 'AFR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude (Coarse)'),
        ]

    def test_derive_with_longitude(self):
        lon = P(name='Longitude')
        lon.array = Mock()
        liftoffs = KTI(name='Liftoff')
        afr_toff_rwy = None
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_rwy, afr_toff_apt, None)
        node.create_kpvs_at_ktis.assert_called_once_with(lon.array, liftoffs)
        assert not node.create_kpv.called, 'method should not have been called'

    def test_derive_with_afr_toff_rwy(self):
        lon = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = A(name='AFR Takeoff Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 1, 'longitude': 1},
        })
        afr_toff_apt = None
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        lat_m, lon_m = midpoint(0, 0, 1, 1)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, lon_m)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'

    def test_derive_with_afr_toff_apt(self):
        lon = None
        liftoffs = KTI(name='Liftoff', items=[KeyTimeInstance(index=0)])
        afr_toff_rwy = None
        afr_toff_apt = A(name='AFR Takeoff Airport', value={
            'latitude': 1,
            'longitude': 1,
        })
        node = self.node_class()
        node.create_kpv = Mock()
        node.create_kpvs_at_ktis = Mock()
        node.derive(lon, liftoffs, afr_toff_apt, afr_toff_rwy, None)
        node.create_kpv.assert_called_once_with(liftoffs[0].index, 1)
        assert not node.create_kpvs_at_ktis.called, 'method should not have been called'


class TestLongitudeSmoothedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeSmoothedAtTouchdown
        self.operational_combinations = [('Longitude Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLongitudeSmoothedAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeSmoothedAtLiftoff
        self.operational_combinations = [('Longitude Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestLongitudeAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = LongitudeAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Longitude Prepared', 'Lowest Altitude During Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


################################################################################
# Magnetic Variation


class TestMagneticVariationAtTakeoffTurnOntoRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = MagneticVariationAtTakeoffTurnOntoRunway
        self.operational_combinations = [('Magnetic Variation', 'Takeoff Turn Onto Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariationAtLandingTurnOffRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = MagneticVariationAtLandingTurnOffRunway
        self.operational_combinations = [('Magnetic Variation', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


###############################################################################

class TestIsolationValveOpenAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = IsolationValveOpenAtLiftoff
        self.operational_combinations = [('Isolation Valve Open', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPackValvesOpenAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PackValvesOpenAtLiftoff
        self.operational_combinations = [('Pack Valves Open', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDescentToFlare(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearExtending(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGoAround5MinRating(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLevelFlight(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoff5MinRating(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRoll(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRotation(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDuringTakeoff(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingDuringTakeoff
        self.operational_combinations = [('Heading Continuous', 'Takeoff Roll')]

    def test_derive_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        toff = buildsection('Takeoff', 2,6)
        kpv = HeadingDuringTakeoff()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=7.5,
                                  name='Heading During Takeoff')]
        self.assertEqual(kpv, expected)

    def test_derive_modulus(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,6)
        kpv = HeadingDuringTakeoff()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5,
                                  name='Heading During Takeoff')]
        self.assertEqual(kpv, expected)


class TestHeadingTrueDuringTakeoff(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingTrueDuringTakeoff
        self.operational_combinations = [('Heading True Continuous',
                                          'Takeoff Roll')]

    def test_derive_basic(self):
        head = P('Heading True Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        toff = buildsection('Takeoff', 2,6)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=7.5,
                                  name='Heading True During Takeoff')]
        self.assertEqual(kpv, expected)

    def test_derive_modulus(self):
        head = P('Heading True Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,6)
        kpv = self.node_class()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5,
                                  name='Heading True During Takeoff')]
        self.assertEqual(kpv, expected)


class TestHeadingDuringLanding(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingDuringLanding
        self.operational_combinations = [('Heading Continuous', 'Landing Roll')]

    def test_derive_basic(self):
        head = P('Heading Continuous',np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                                                   7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Landing',5,15)
        head.array[13] = np.ma.masked
        kpv = HeadingDuringLanding()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)


class TestHeadingTrueDuringLanding(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeadingTrueDuringLanding
        self.operational_combinations = [('Heading True Continuous',
                                          'Landing Roll')]

    def test_derive_basic(self):
        # Duplicate of TestHeadingDuringLanding.test_derive_basic.
        head = P('Heading True Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Landing', 5, 15)
        head.array[13] = np.ma.masked
        kpv = HeadingDuringLanding()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)


class TestHeadingAtLowestAltitudeDuringApproach(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = HeadingAtLowestAltitudeDuringApproach
        self.operational_combinations = [('Heading Continuous', 'Lowest Altitude During Approach')]

    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        # derive() uses par1 % 360.0, so the par1 needs to be compatible with %
        # operator
        mock1.array = 0
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayAbove80KtsAirspeedDuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayAbove80KtsAirspeedDuringTakeoff
        self.operational_combinations = [(
            'Heading True Continuous',
            'Airspeed',
            'Pitch',
            'Takeoff',
            'FDR Takeoff Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayAtTOGADuringTakeoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayAtTOGADuringTakeoff
        self.operational_combinations = [(
            'Heading True Continuous',
            'Takeoff And Go Around',
            'Takeoff',
            'FDR Takeoff Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayAt50FtDuringLanding(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayAt50FtDuringLanding
        self.operational_combinations = [(
            'Heading True Continuous',
            'Landing',
            'FDR Landing Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationFromRunwayDuringLandingRoll(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingDeviationFromRunwayDuringLandingRoll
        self.operational_combinations = [(
            'Heading True Continuous',
            'Landing Roll',
            'FDR Landing Runway',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariation300To50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariation300To50Ft
        self.operational_combinations = [(
            'Heading Continuous',
            'Altitude AAL For Flight Phases',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariation500To50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariation500To50Ft
        self.operational_combinations = [(
            'Heading Continuous',
            'Altitude AAL For Flight Phases',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariationAbove100KtsAirspeedDuringLanding(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariationAbove100KtsAirspeedDuringLanding
        self.operational_combinations = [(
            'Heading Continuous',
            'Airspeed',
            'Altitude AAL For Flight Phases',
            'Landing',
        )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVariationTouchdownPlus4SecTo60KtsAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVariationTouchdownPlus4SecTo60KtsAirspeed
        self.operational_combinations = [('Heading Continuous', 'Airspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVacatingRunway(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingVacatingRunway
        self.operational_combinations = [('Heading Continuous', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Height


class TestHeightMinsToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = HeightMinsToTouchdown
        self.operational_combinations = [('Altitude AAL', 'Mins To Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Flap


class TestFlapAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FlapAtLiftoff
        self.operational_combinations = [('Flap', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestFlapAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FlapAtTouchdown
        self.operational_combinations = [('Flap', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestFlapAtGearDownSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FlapAtGearDownSelection
        self.operational_combinations = [('Flap', 'Gear Down Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapWithGearUpMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapWithGearUpMax
        self.operational_combinations = [('Flap', 'Gear Down')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapWithSpeedbrakeDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapWithSpeedbrakeDeployedMax
        self.operational_combinations = [('Flap', 'Speedbrake Selected', 'Airborne', 'Landing')]

    def test_derive(self):
        flap = P(
            name='Flap',
            array=np.arange(15),
        )
        spd_brk = M(
            name='Speedbrake Selected',
            array=np.ma.array([0, 1, 2, 0, 0] * 3),
            values_mapping={
                0: 'Stowed',
                1: 'Armed/Cmd Dn',
                2: 'Deployed/Cmd Up',
            },
        )
        airborne = buildsection('Airborne', 5, 15)
        landings = buildsection('Landing', 10, 15)
        node = self.node_class()
        node.derive(flap, spd_brk, airborne, landings)
        self.assertEqual(node, [
            KeyPointValue(7, 7, 'Flap With Speedbrake Deployed Max'),
        ])


##############################################################################


class TestFlareDuration20FtToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = FlareDuration20FtToTouchdown
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Touchdown', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlareDistance20FtToTouchdown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = FlareDistance20FtToTouchdown
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Touchdown', 'Landing', 'Groundspeed')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Fuel Quantity


class TestFuelQtyAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FuelQtyAtLiftoff
        self.operational_combinations = [('Fuel Qty', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestFuelQtyAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FuelQtyAtTouchdown
        self.operational_combinations = [('Fuel Qty', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Groundspeed


class TestGroundspeedMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = GroundspeedMax
        self.operational_combinations = [('Groundspeed', 'Grounded')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestGroundspeedWhileTaxiingStraightMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWhileTaxiingStraightMax
        self.operational_combinations = [('Groundspeed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedWhileTaxiingTurnMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWhileTaxiingTurnMax
        self.operational_combinations = [('Groundspeed', 'Taxiing', 'Turning On Ground')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedDuringRejectedTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = GroundspeedDuringRejectedTakeoffMax
        self.operational_combinations = [('Groundspeed', 'Rejected Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GroundspeedAtTouchdown
        self.operational_combinations = [('Groundspeed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedVacatingRunway(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GroundspeedVacatingRunway
        self.operational_combinations = [('Groundspeed', 'Landing Turn Off Runway')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedAtTOGA(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedAtTOGA
        self.operational_combinations = [('Groundspeed', 'Takeoff And Go Around', 'Takeoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedWithThrustReversersDeployedMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GroundspeedWithThrustReversersDeployedMin
        self.operational_combinations = [('Groundspeed', 'Thrust Reversers', 'Eng (*) N1 Max', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Pitch


class TestPitchAfterFlapRetractionMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAfterFlapRetractionMax
        self.operational_combinations = [('Flap', 'Pitch', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = PitchAtLiftoff
        self.operational_combinations = [('Pitch', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitchAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = PitchAtTouchdown
        self.operational_combinations = [('Pitch', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitchAt35FtDuringClimb(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAt35FtDuringClimb
        self.operational_combinations = [('Pitch', 'Altitude AAL')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchLiftoffTo35FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchLiftoffTo35FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (0, 35), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch35To400FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch35To400FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 400), {})]

    def test_derive_basic(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([100, 101, 102, 103, 700, 105, 104, 103, 102]),
        )
        node = Pitch35To400FtMax()
        node.derive(pitch, alt_aal)
        self.assertEqual(node, KPV('Pitch 35 To 400 Ft Max', items=[
            KeyPointValue(name='Pitch 35 To 400 Ft Max', index=3, value=7),
        ]))


class TestPitch35To400FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch35To400FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 400), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitch400To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch400To1000FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (400, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch400To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch400To1000FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (400, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch1000To500FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch1000To500FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch500To50FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To20FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Pitch500To20FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch50FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch50FtToTouchdownMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (50, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch20FtToTouchdownMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch7FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Pitch7FtToTouchdownMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (7, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchCyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchCyclesDuringFinalApproach
        self.operational_combinations = [('Pitch', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchDuringGoAroundMax
        self.operational_combinations = [('Pitch', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Pitch Rate


class TestPitchRate35To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate35To1000FtMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = PitchRate20FtToTouchdownMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate20FtToTouchdownMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = PitchRate20FtToTouchdownMin
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate2DegPitchTo35FtMax
        self.operational_combinations = [('Pitch Rate', '2 Deg Pitch To 35 Ft')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchRate2DegPitchTo35FtMin
        self.operational_combinations = [('Pitch Rate', '2 Deg Pitch To 35 Ft')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Rate of Climb


class TestRateOfClimbMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfClimbMax
        self.operational_combinations = [('Vertical Speed', 'Climbing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimb35To1000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfClimb35To1000FtMin
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimbBelow10000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfClimbBelow10000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (0, 10000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimbDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfClimbDuringGoAroundMax
        self.operational_combinations = [('Vertical Speed', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Rate of Descent


class TestRateOfDescentMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentMax
        self.operational_combinations = [('Vertical Speed', 'Descending')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentTopOfDescentTo10000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentTopOfDescentTo10000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude STD Smoothed', 'Descent')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentBelow10000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescentBelow10000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (10000, 0), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescent10000To5000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent10000To5000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude STD Smoothed')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (10000, 5000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent5000To3000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent5000To3000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (5000, 3000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent3000To2000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent3000To2000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (3000, 2000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent2000To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent2000To1000FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (2000, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent1000To500FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescent500To50FtMax
        self.operational_combinations = [('Vertical Speed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 50), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent50FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = RateOfDescent50FtToTouchdownMax
        self.operational_combinations = [('Vertical Speed Inertial', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (50, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescentAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentAtTouchdown
        self.operational_combinations = [('Vertical Speed Inertial', 'Altitude AAL', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RateOfDescentDuringGoAroundMax
        self.operational_combinations = [('Vertical Speed', 'Go Around And Climbout')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Roll


class TestRollLiftoffTo20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RollLiftoffTo20FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (1, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll20To400FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll20To400FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (20, 400), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll400To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll400To1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (400, 1000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollAbove1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RollAbove1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1000,), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRoll1000To300FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll1000To300FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (1000, 300), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll300To20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = Roll300To20FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_from_to', (300, 20), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll20FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Roll20FtToTouchdownMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Touchdown')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_to_kti', (20, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesDuringFinalApproach
        self.operational_combinations = [('Roll', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesNotDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollCyclesNotDuringFinalApproach
        self.operational_combinations = [('Roll', 'Airborne', 'Final Approach', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Rudder


class TestRudderDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = RudderDuringTakeoffMax
        self.operational_combinations = [('Rudder', 'Takeoff Roll')]
        self.function = max_abs_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderCyclesAbove50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RudderCyclesAbove50Ft
        self.operational_combinations = [('Rudder', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderReversalAbove50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RudderReversalAbove50Ft
        self.operational_combinations = [('Rudder', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Speedbrake


class TestSpeedbrakeDeployed1000To20FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployed1000To20FtDuration
        self.operational_combinations = [('Speedbrake Selected', 'Altitude AAL For Flight Phases')]

    def test_derive_basic(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    array=np.ma.arange(2000, 0, -10))
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping, 
            array=np.ma.array(
                [0] * 40 + [1] * 20 + [0] * 80 + [1] * 20 + [0] * 40))
        node = self.node_class()
        node.derive(spd_brk, alt_aal)
        self.assertEqual(
            node, [KeyPointValue(140, 20.0,
                                 'Speedbrake Deployed 1000 To 20 Ft Duration')])


class TestSpeedbrakeDeployedWithConfDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithConfDuration
        self.operational_combinations = [('Speedbrake Selected', 'Configuration', 'Airborne')]

    def test_derive_basic(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        conf = P('Configuration', array=np.ma.array([0] * 10 + range(2, 22)))
        airborne = buildsection('Airborne', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, conf, airborne)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0, 'Speedbrake Deployed With Conf Duration')])


class TestSpeedbrakeDeployedWithFlapDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithFlapDuration
        self.operational_combinations = [('Speedbrake Selected', 'Flap', 'Airborne')]
    
    def test_derive_basic(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        flap = P('Flap', array=np.ma.array([0] * 10 + range(1, 21)))
        airborne = buildsection('Airborne', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, flap, airborne)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0, 'Speedbrake Deployed With Flap Duration')])


class TestSpeedbrakeDeployedWithPowerOnDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithPowerOnDuration
        self.operational_combinations = [('Speedbrake Selected', 'Eng (*) N1 Avg', 'Airborne', 'Manufacturer')]

    def test_derive_basic(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        flap = P('Eng (*) N1 Avg',
                 array=np.ma.array([40] * 10 + [60] * 10 + [50] * 10))
        airborne = buildsection('Airborne', 10, 20)
        manufacturer = A('Manufacturer', value='Airbus')
        node = self.node_class()
        node.derive(spd_brk, flap, airborne, manufacturer)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0,
                          'Speedbrake Deployed With Power On Duration')])


class TestSpeedbrakeDeployedDuringGoAroundDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedDuringGoAroundDuration
        self.operational_combinations = [('Speedbrake Selected', 'Go Around And Climbout')]

    def test_derive(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        go_around = buildsection('Go Around And Climbout', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, go_around)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0,
                          'Speedbrake Deployed During Go Around Duration')])


##############################################################################
# Warnings: Stick Pusher/Shaker


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestStickPusherActivatedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = StickPusherActivatedDuration
        self.operational_combinations = [('Stick Pusher', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestStickShakerActivatedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = StickShakerActivatedDuration
        self.operational_combinations = [('Stick Shaker', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Tail Clearance


class TestTailClearanceDuringTakeoffMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TailClearanceDuringTakeoffMin
        self.operational_combinations = [('Altitude Tail', 'Takeoff')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceDuringLandingMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TailClearanceDuringLandingMin
        self.operational_combinations = [('Altitude Tail', 'Landing')]
        self.function = min_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceDuringApproachMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TailClearanceDuringApproachMin
        self.operational_combinations = [('Altitude AAL', 'Altitude Tail', 'Distance To Landing')]

    @unittest.skip('Test Out Of Date')
    def test_derive(self):
        # XXX: The BDUTerrain test files are missing from the repository?
        test_data_dir = os.path.join(test_data_path, 'BDUTerrain')
        alt_aal_array = np.ma.masked_array(np.load(os.path.join(test_data_dir, 'alt_aal.npy')))
        alt_radio_array = np.ma.masked_array(np.load(os.path.join(test_data_dir, 'alt_radio.npy')))
        dtl_array = np.ma.masked_array(np.load(os.path.join(test_data_dir, 'dtl.npy')))
        alt_aal = P(array=alt_aal_array, frequency=8)
        alt_radio = P(array=alt_radio_array, frequency=0.5)
        dtl = P(array=dtl_array, frequency=0.25)
        alt_radio.array = align(alt_radio, alt_aal)
        dtl.array = align(dtl, alt_aal)
        # FIXME: Should tests for the BDUTerrain node be in a separate TestCase?
        ####param = BDUTerrain()
        ####param.derive(alt_aal, alt_radio, dtl)
        ####self.assertEqual(param, [
        ####    KeyPointValue(name='BDU Terrain', index=1008, value=0.037668517049960347),
        ####])


##############################################################################
# Terrain Clearance


class TestTerrainClearanceAbove3000FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TerrainClearanceAbove3000FtMin
        self.operational_combinations = [('Altitude Radio', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_above', (3000.0,), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Tailwind


# FIXME: Make CreateKPVsWithinSlicesTest more generic and then use it again...
class TestTailwindLiftoffTo100FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TailwindLiftoffTo100FtMax
        self.operational_combinations = [('Tailwind', 'Altitude AAL For Flight Phases')]
        #self.second_param_method_calls = [('slices_from_to', (0, 100), {})]
        #self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# FIXME: Make CreateKPVsWithinSlicesTest more generic and then use it again...
class TestTailwind100FtToTouchdownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency is used.
        self.node_class = Tailwind100FtToTouchdownMax
        self.operational_combinations = [('Tailwind', 'Altitude AAL For Flight Phases', 'Touchdown')]
        #self.function = max_value
        #self.second_param_method_calls = [('slices_to_kti', (100, []), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFuelQtyLowWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQtyLowWarningDuration.get_operational_combinations()
        self.assertEqual(opts, [('Fuel Qty (*) Low',)])
        
    def test_derive(self):
        low = FuelQtyLowWarningDuration()
        low.derive(M(array=np.ma.array([0,0,1,1,0]), 
                     values_mapping={1: 'Warning'}))
        self.assertEqual(low[0].index, 2)
        self.assertEqual(low[0].value, 2)


##############################################################################
# Warnings: Takeoff Configuration Warning


class TestMasterWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarningDuration
        self.operational_combinations = [('Master Warning',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMasterWarningDuringTakeoffDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarningDuringTakeoffDuration
        self.operational_combinations = [('Master Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMasterCautionDuringTakeoffDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterCautionDuringTakeoffDuration
        self.operational_combinations = [('Master Caution', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TestTAWSAlertDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSAlertDuration
        self.operational_combinations = [('TAWS Alert', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSGeneralWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGeneralWarningDuration
        self.operational_combinations = [('TAWS General', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSSinkRateWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSSinkRateWarningDuration
        self.operational_combinations = [('TAWS Sink Rate', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowFlapWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowFlapWarningDuration
        self.operational_combinations = [('TAWS Too Low Flap', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTerrainWarningDuration
        self.operational_combinations = [('TAWS Terrain', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainPullUpWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTerrainPullUpWarningDuration
        self.operational_combinations = [('TAWS Terrain Pull Up', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSGlideslopeWarning1500To1000FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning1500To1000FtDuration
        self.operational_combinations = [('TAWS Glideslope', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSGlideslopeWarning1000To500FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning1000To500FtDuration
        self.operational_combinations = [('TAWS Glideslope', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSGlideslopeWarning500To200FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSGlideslopeWarning500To200FtDuration
        self.operational_combinations = [('TAWS Glideslope', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowTerrainWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowTerrainWarningDuration
        self.operational_combinations = [('TAWS Too Low Terrain', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowGearWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSTooLowGearWarningDuration
        self.operational_combinations = [('TAWS Too Low Gear', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSPullUpWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSPullUpWarningDuration
        self.operational_combinations = [('TAWS Pull Up', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTAWSDontSinkWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSDontSinkWarningDuration
        self.operational_combinations = [('TAWS Dont Sink', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSWindshearWarningBelow1500FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSWindshearWarningBelow1500FtDuration
        self.operational_combinations = [('TAWS Windshear Warning', 'Altitude AAL For Flight Phases')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Warnings: Traffic Collision Avoidance System (TCAS)


class TestTCASTAWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASTAWarningDuration
        self.operational_combinations = [('TCAS Combined Control', 'Airborne')]

    def test_derive(self):
        values_mapping = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'Preventive',
        }
        tcas = M(
            'TCAS Combined Control', array=np.ma.array([0,1,2,3,4,6,6,6,4,5]),
            values_mapping=values_mapping)
        airborne = buildsection('Airborne', 2, 7)
        node = self.node_class()
        node.derive(tcas, airborne)
        self.assertEqual([KeyPointValue(5.0, 2.0, 'TCAS TA Warning Duration')],
                         node)


class TestTCASRAWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAWarningDuration
        self.operational_combinations = [('TCAS Combined Control', 'Airborne')]

    def test_derive(self):
        values_mapping = {
            0: 'A',
            1: 'B',
            2: 'Drop Track',
            3: 'Altitude Lost',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'G',
        }
        tcas = M(
            'TCAS Combined Control', array=np.ma.array([0,1,2,3,4,5,4,5,6]),
            values_mapping=values_mapping)
        airborne = buildsection('Airborne', 2, 7)
        node = self.node_class()
        node.derive(tcas, airborne)
        self.assertEqual([KeyPointValue(2, 5.0, 'TCAS RA Warning Duration')],
                         node)


class TestTCASRAReactionDelay(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAReactionDelay
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'TCAS Combined Control', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAInitialReactionStrength(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAInitialReactionStrength
        self.operational_combinations = [('Acceleration Normal Offset Removed', 'TCAS Combined Control', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAToAPDisengagedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASRAToAPDisengagedDuration
        self.operational_combinations = [('AP Disengaged Selection', 'TCAS Combined Control', 'Airborne')]

    def test_derive(self):
        values_mapping = {
            0: 'A',
            1: 'B',
            2: 'Drop Track',
            3: 'Altitude Lost',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'G',
        }
        kti_name = 'AP Disengaged Selection'
        ap_offs = KTI(kti_name, items=[KeyTimeInstance(1, kti_name),
                                       KeyTimeInstance(7, kti_name)])
        tcas = M(
            'TCAS Combined Control', array=np.ma.array([0,1,2,3,4,5,4,4,1,3,0]),
            values_mapping=values_mapping)
        airborne = buildsection('Airborne', 2, 9)
        node = self.node_class()
        node.derive(ap_offs, tcas, airborne)
        self.assertEqual([KeyPointValue(7.0, 5.0,
                                        'TCAS RA To AP Disengaged Duration')],
                         node)


##############################################################################
# Warnings: Takeoff Configuration


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTakeoffConfigurationWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TakeoffConfigurationWarningDuration
        self.operational_combinations = [('Takeoff Configuration Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTakeoffConfigurationFlapWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TakeoffConfigurationFlapWarningDuration
        self.operational_combinations = [('Takeoff Configuration Flap Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTakeoffConfigurationParkingBrakeWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TakeoffConfigurationParkingBrakeWarningDuration
        self.operational_combinations = [('Takeoff Configuration Parking Brake Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTakeoffConfigurationSpoilerWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TakeoffConfigurationSpoilerWarningDuration
        self.operational_combinations = [('Takeoff Configuration Spoiler Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


# TODO: Need a CreateKPVsWhereStateTest super class!
class TestTakeoffConfigurationStabilizerWarningDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TakeoffConfigurationStabilizerWarningDuration
        self.operational_combinations = [('Takeoff Configuration Stabilizer Warning', 'Takeoff Roll')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Throttle


class TestThrottleCyclesDuringFinalApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrottleCyclesDuringFinalApproach
        self.operational_combinations = [('Throttle Levers', 'Final Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Thrust Asymmetry


class TestThrustAsymmetryDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringTakeoffMax
        self.operational_combinations = [('Thrust Asymmetry', 'Takeoff Roll')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringFlightMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringFlightMax
        self.operational_combinations = [('Thrust Asymmetry', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringGoAroundMax
        self.operational_combinations = [('Thrust Asymmetry', 'Go Around And Climbout')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringApproachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringApproachMax
        self.operational_combinations = [('Thrust Asymmetry', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryWithThrustReversersDeployedMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryWithThrustReversersDeployedMax
        self.operational_combinations = [('Thrust Asymmetry', 'Thrust Reversers', 'Mobile')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryDuringApproachDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryDuringApproachDuration
        self.operational_combinations = [('Thrust Asymmetry', 'Approach')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetryWithThrustReversersDeployedDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ThrustAsymmetryWithThrustReversersDeployedDuration
        self.operational_combinations = [('Thrust Asymmetry', 'Thrust Reversers', 'Mobile')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestTouchdownToElevatorDownDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TouchdownToElevatorDownDuration
        self.operational_combinations = [('Airspeed', 'Elevator', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownTo60KtsDuration(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TouchdownTo60KtsDuration
        self.operational_combinations = [
            ('Airspeed', 'Touchdown'),
            ('Airspeed', 'Groundspeed', 'Touchdown'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Turbulence


class TestTurbulenceDuringApproachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TurbulenceDuringApproachMax
        self.operational_combinations = [('Turbulence RMS g', 'Approach')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTurbulenceDuringCruiseMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TurbulenceDuringCruiseMax
        self.operational_combinations = [('Turbulence RMS g', 'Cruise')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTurbulenceDuringFlightMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = TurbulenceDuringFlightMax
        self.operational_combinations = [('Turbulence RMS g', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Wind


class TestWindSpeedAtAltitudeDuringDescent(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindSpeedAtAltitudeDuringDescent
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Wind Speed')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirectionAtAltitudeDuringDescent(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindDirectionAtAltitudeDuringDescent
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Wind Direction Continuous')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindAcrossLandingRunwayAt50Ft(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = WindAcrossLandingRunwayAt50Ft
        self.operational_combinations = [('Wind Across Landing Runway', 'Landing')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################
# Weight


class TestGrossWeightAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestGrossWeightAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = GrossWeightAtTouchdown
        self.operational_combinations = [('Gross Weight Smoothed', 'Touchdown')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestZeroFuelWeight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ZeroFuelWeight
        self.operational_combinations = [('Fuel Qty', 'Gross Weight')]

    def test_derive(self):
        fuel_qty = P('Fuel Qty', np.ma.array([1, 2, 3, 4]))
        gross_wgt = P('Gross Weight', np.ma.array([11, 12, 13, 14]))
        zfw = ZeroFuelWeight()
        zfw.derive(fuel_qty, gross_wgt)
        self.assertEqual(zfw[0].index, 0)  # Note: Index should always be 0!
        self.assertEqual(zfw[0].value, 10.0)


##############################################################################


class TestHoldingDuration(unittest.TestCase):
    # TODO: CreateKPVsFromSliceDurations test superclass.
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestTwoDegPitchTo35FtDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TwoDegPitchTo35FtDuration
        self.operational_combinations = [('2 Deg Pitch To 35 Ft',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLastFlapChangeToTakeoffRollEndDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LastFlapChangeToTakeoffRollEndDuration
        self.operational_combinations = [
            ('Flap', 'Takeoff Roll')
        ]

    def test_derive(self):
        flap = P(
            name='Altitude STD',
            array=np.ma.array([15, 15, 20, 20, 15, 15]),
        )
        takeoff_roll = S(items=[Section('Takeoff Roll', slice(0, 5), 0, 5)])
        node = self.node_class()
        node.derive(flap, takeoff_roll)
        expected = [
            KeyPointValue(
                index=3.5, value=1.5,
                name='Last Flap Change To Takeoff Roll End Duration')
        ]
        self.assertEqual(list(node), expected)
