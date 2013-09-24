import operator
import os
import numpy as np
import sys
import unittest

from mock import Mock, call, patch

from flightdatautilities.geometry import midpoint

from analysis_engine.library import align
from analysis_engine.node import (
    A, App, ApproachItem, KPV, KTI, load, M, P, KeyPointValue, MultistateDerivedParameterNode,
    KeyTimeInstance, Section, S
)

from analysis_engine.multistate_parameters import (
    FlapExcludingTransition,
    FlapIncludingTransition,
    StableApproach,
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
    AccelerationLongitudinalDuringLandingMin,
    AccelerationNormal20FtToFlareMax,
    AccelerationNormalWithFlapDownWhileAirborneMax,
    AccelerationNormalWithFlapDownWhileAirborneMin,
    AccelerationNormalWithFlapUpWhileAirborneMax,
    AccelerationNormalWithFlapUpWhileAirborneMin,
    AccelerationNormalAtLiftoff,
    AccelerationNormalAtTouchdown,
    AccelerationNormalTakeoffMax,
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
    AirspeedAtFlapExtensionWithGearDown,
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
    AirspeedRelativeWithConfigurationDuringDescentMin,
    AirspeedRelativeWithFlapDuringDescentMin,
    AirspeedTopOfDescentTo10000FtMax,
    AirspeedV2Plus20DifferenceAtVNAVModeAndEngThrustModeRequired,
    AirspeedWithThrustReversersDeployedMin,
    AirspeedAtThrustReversersSelection,
    AirspeedTrueAtTouchdown,
    AirspeedVacatingRunway,
    AirspeedWithConfigurationMax,
    AirspeedWithFlapDuringClimbMax,
    AirspeedWithFlapDuringClimbMin,
    AirspeedWithFlapDuringDescentMax,
    AirspeedWithFlapDuringDescentMin,
    AirspeedWithFlapMax,
    AirspeedWithFlapMin,
    AirspeedWithGearDownMax,
    AirspeedWithSpoilerDeployedMax,
    AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft,
    AltitudeAtFirstFlapChangeAfterLiftoff,
    AltitudeAtGearUpSelectionDuringGoAround,
    AltitudeDuringGoAroundMin,
    AltitudeAtLastFlapChangeBeforeTouchdown,
    AltitudeAtMachMax,
    AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft,
    AltitudeAtGearDownSelection,
    AltitudeAtGearDownSelectionWithFlapUp,
    AltitudeAtGearDownSelectionWithFlapDown,
    AltitudeAtGearUpSelection,
    AltitudeAtAPDisengagedSelection,
    AltitudeAtAPEngagedSelection,
    AltitudeAtATDisengagedSelection,
    AltitudeAtATEngagedSelection,
    AltitudeAtVNAVModeAndEngThrustModeRequired,
    AltitudeAtFirstAPEngagedAfterLiftoff,
    AltitudeAtFirstFlapExtensionAfterLiftoff,
    AltitudeAtFirstFlapRetraction,
    AltitudeAtFirstFlapRetractionDuringGoAround,
    AltitudeAtFlapExtension,
    AltitudeAtFlapExtensionWithGearDown,
    AltitudeAtLastAPDisengagedDuringApproach,
    AltitudeFirstStableDuringApproachBeforeGoAround,
    AltitudeFirstStableDuringLastApproach,
    AltitudeLastUnstableDuringApproachBeforeGoAround,
    AltitudeLastUnstableDuringLastApproach,
    AltitudeOvershootAtSuspectedLevelBust,
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
    APUFireWarningDuration,
    BrakePressureInTakeoffRollMax,
    ControlColumnStiffness,
    DecelerationFromTouchdownToStopOnRunway,
    DelayedBrakingAfterTouchdown,
    ElevatorDuringLandingMin,
    EngBleedValvesAtLiftoff,
    EngEPRDuringApproachMax,
    EngEPRDuringApproachMin,
    EngEPRDuringTaxiMax,
    EngEPRDuringTakeoff5MinRatingMax,
    EngEPRDuringGoAround5MinRatingMax,
    EngEPRDuringMaximumContinuousPowerMax,
    EngEPR500To50FtMax,
    EngEPR500To50FtMin,
    EngEPRFor5Sec1000To500FtMin,
    EngEPRFor5Sec500To50FtMin,
    EngEPRAtTOGADuringTakeoffMax,
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
    EngN154to72PercentWithThrustReversersDeployedDurationMax,
    EngN2DuringTaxiMax,
    EngN2DuringTakeoff5MinRatingMax,
    EngN2DuringGoAround5MinRatingMax,
    EngN2DuringMaximumContinuousPowerMax,
    EngN2CyclesDuringFinalApproach,
    EngN3DuringTaxiMax,
    EngN3DuringTakeoff5MinRatingMax,
    EngN3DuringGoAround5MinRatingMax,
    EngN3DuringMaximumContinuousPowerMax,
    EngNpDuringTaxiMax,
    EngNpDuringTakeoff5MinRatingMax,
    EngNpDuringGoAround5MinRatingMax,
    EngNpDuringMaximumContinuousPowerMax,
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
    EngTPRAtTOGADuringTakeoffMin,
    EngTPRDuringGoAround5MinRatingMax,
    EngTPRDuringTakeoff5MinRatingMax,
    #EngTPRLimitDifferenceDuringGoAroundMax,
    #EngTPRLimitDifferenceDuringTakeoffMax,
    EngVibBroadbandMax,
    EngVibN1Max,
    EngVibN2Max,
    EngVibN3Max,
    EngVibAMax,
    EngVibBMax,
    EngVibCMax,
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
    LandingConfigurationGearWarningDuration,
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
    MachDuringCruiseAvg,
    MachWhileGearExtendingMax,
    MachWhileGearRetractingMax,
    MachMax,
    MachWithFlapMax,
    MachWithGearDownMax,
    MagneticVariationAtTakeoffTurnOntoRunway,
    MagneticVariationAtLandingTurnOffRunway,
    ModeControlPanelAirspeedSelectedAt8000Ft,
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
    PitchAlternateLawDuration,
    PitchAtLiftoff,
    PitchAtTouchdown,
    PitchAt35FtDuringClimb,
    PitchAtVNAVModeAndEngThrustModeRequired,
    PitchCyclesDuringFinalApproach,
    PitchDirectLawDuration,
    PitchDuringGoAroundMax,
    PitchTakeoffMax,
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
    AltitudeWhenDescending,
    EngStop,
)
from analysis_engine.library import (max_abs_value, max_value, min_value)
from analysis_engine.flight_phase import Fast
from flight_phase_test import buildsection, buildsections

debug = sys.gettrace() is not None


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        if not hasattr(self, 'node_class'):
            return

        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            combinations = map(set, self.node_class.get_operational_combinations())
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)


class CreateKPVsWhereTest(NodeTest):
    '''
    Basic test for KPVs created with `create_kpvs_where()` method.

    The rationale for this class is to be able to use very simple test case
    boilerplate for the "multi state parameter duration in given flight phase"
    scenario.

    This test checks basic mechanics of specific type of KPV: duration of a
    given state in multistate parameter.

    The test supports multiple parameters and optionally a phase name
    within which the duration is measured.

    What is tested this class:
        * kpv.can_operate() results
        * parameter and KPV names
        * state names
        * basic logic to measure the time of a state duration within a phase
          (slice)

    What isn't tested:
        * potential edge cases of specific parameters
    '''
    def basic_setup(self):
        '''
        Setup for test_derive_basic.

        In the most basic use case the test which derives from this class
        should declare the attributes used to build the test case and then call
        self.basic_setup().

        You need to declare:

        self.node_class::
            class of the KPV node to be used to derive

        self.param_name::
            name of the parameter to be passed to the KPVs `derive()` method

        self.phase_name::
            name of the flight phase to be passed to the `derive()` or None if
            the KPV does not use phases

        self.values_mapping::
            "state to state name" mapping for multistate parameter

        Optionally:

        self.additional_params::
            list of additional parameters to be passed to the `derive()` after
            the main parameter. If unset, only one parameter will be used.


        The method performs the following operations:

            1. Builds the main parameter using self.param_name,
               self.values_array and self.values_mapping

            2. Builds self.params list from the main parameter and
               self.additional_params, if given
            3. Optionally builds self.phases with self.phase_name if given
            4. Builds self.operational_combinations from self.params and
               self.phases
            5. Builds self.expected list of expected values using
               self.node_class and self.phases

        Any of the built attributes can be overridden in the derived class to
        alter the expected test results.
        '''
        if not hasattr(self, 'values_array'):
            self.values_array = np.ma.array([0] * 3 + [1] * 6 + [0] * 3)

        if not hasattr(self, 'phase_slice'):
            self.phase_slice = slice(2, 7)

        if not hasattr(self, 'expected_index'):
            self.expected_index = 3

        if not hasattr(self, 'params'):
            self.params = [
                MultistateDerivedParameterNode(
                    self.param_name,
                    array=self.values_array,
                    values_mapping=self.values_mapping
                )
            ]

            if hasattr(self, 'additional_params'):
                self.params += self.additional_params

        if hasattr(self, 'phase_name') and self.phase_name:
            self.phases = buildsection(self.phase_name,
                                       self.phase_slice.start,
                                       self.phase_slice.stop)
        else:
            self.phases = []

        if not hasattr(self, 'operational_combinations'):
            combinations = [p.name for p in self.params]

            self.operational_combinations = [combinations]
            if self.phases:
                combinations.append(self.phases.name)

        if not hasattr(self, 'expected'):
            self.expected = []
            if self.phases:
                slices = [p.slice for p in self.phases]
            else:
                slices = [slice(None)]

            for sl in slices:
                expected_value = np.count_nonzero(
                    self.values_array[sl])
                if expected_value:
                    self.expected.append(
                        KeyPointValue(
                            name=self.node_class().get_name(),
                            index=self.expected_index,
                            value=expected_value
                        )
                    )

    def test_can_operate(self):
        '''
        Test the operational combinations.
        '''
        # sets of sorted tuples of node combinations must match exactly
        kpv_operational_combinations = \
            self.node_class.get_operational_combinations()

        kpv_combinations = set(
            tuple(sorted(c)) for c in kpv_operational_combinations)

        expected_combinations = set(
            tuple(sorted(c)) for c in self.operational_combinations)

        self.assertSetEqual(kpv_combinations, expected_combinations)

    def test_derive_basic(self):
        '''
        Basic test of state duration in given phase.

        self.node_class: the class of the tested node
        '''
        if hasattr(self, 'node_class'):
            node = self.node_class()
            node.derive(*(self.params + self.phases))
            self.assertEqual(node, self.expected)


class CreateKPVsAtKPVsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKPVsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
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
        kwargs = {}
        if hasattr(self, 'interpolate'):
            kwargs = {'interpolate': self.interpolate}
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2, **kwargs)


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
            node.create_kpvs_within_slices.assert_called_once_with(
                mock1.array, mock3.return_value, self.function)
        else:
            self.assertEqual(mock2.method_calls, [])
            node.create_kpvs_within_slices.assert_called_once_with(
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


class TestAccelerationLongitudinalDuringLandingMin(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = AccelerationLongitudinalDuringLandingMin
        self.operational_combinations = [('Acceleration Longitudinal', 'Landing')]
        self.function = min_value

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


class TestAccelerationNormalTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = AccelerationNormalTakeoffMax
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
        self.operational_combinations = [('Airspeed', 'Altitude When Descending')]
    
    def test_derive_basic(self):
        air_spd = P('Airspeed', array=np.ma.arange(0, 200, 10))
        alt_std_desc = AltitudeWhenDescending(
            items=[KeyTimeInstance(8, '6000 Ft Descending'),
                   KeyTimeInstance(10, '8000 Ft Descending'),
                   KeyTimeInstance(16, '8000 Ft Descending'),
                   KeyTimeInstance(18, '8000 Ft Descending')])
        node = self.node_class()
        node.derive(air_spd, alt_std_desc)
        self.assertEqual(node,
                         [KeyPointValue(index=10, value=100.0, name='Airspeed At 8000 Ft'),
                          KeyPointValue(index=16, value=160.0, name='Airspeed At 8000 Ft'),
                          KeyPointValue(index=18, value=180.0, name='Airspeed At 8000 Ft')])


class TestModeControlPanelAirspeedSelectedAt8000Ft(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = ModeControlPanelAirspeedSelectedAt8000Ft
        self.operational_combinations = [('Mode Control Panel Airspeed Selected', 'Altitude When Descending')]

    def test_derive_basic(self):
        air_spd = P('Mode Control Panel Airspeed Selected', array=np.ma.arange(0, 200, 5))
        alt_std_desc = AltitudeWhenDescending(
            items=[KeyTimeInstance(13, '8000 Ft Descending'),
                   KeyTimeInstance(26, '8000 Ft Descending'),
                   KeyTimeInstance(32, '8000 Ft Descending'),
                   KeyTimeInstance(35, '4000 Ft Descending')])
        node = self.node_class()
        node.derive(air_spd, alt_std_desc)
        self.assertEqual(node,
                         [KeyPointValue(index=13, value=65.0, name='Mode Control Panel Airspeed Selected At 8000 Ft'),
                          KeyPointValue(index=26, value=130.0, name='Mode Control Panel Airspeed Selected At 8000 Ft'),
                          KeyPointValue(index=32, value=160.0, name='Mode Control Panel Airspeed Selected At 8000 Ft')])


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


class TestAirspeedV2Plus20DifferenceAtVNAVModeAndEngThrustModeRequired(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = AirspeedV2Plus20DifferenceAtVNAVModeAndEngThrustModeRequired
        self.operational_combinations = [('Airspeed', 'V2', 'VNAV Mode And Eng Thrust Mode Required')]
    
    def test_derive(self):
        airspeed_array = np.ma.arange(0, 200, 10)
        airspeed = P('Airspeed', array=airspeed_array)
        v2_array = np.ma.array([200] * 20)
        v2_array.mask = [False] * 15 + [True] * 3 + [False] * 2
        v2 = P('V2', array=v2_array)
        kti_name = 'VNAV Mode And Eng Thrust Mode Required'
        vnav_thrusts = KTI(kti_name, items=[
            KeyTimeInstance(index=5, name=kti_name),
            KeyTimeInstance(index=15, name=kti_name)])
        node = self.node_class()
        node.derive(airspeed, v2, vnav_thrusts)
        self.assertEqual(
            node,
            [KeyPointValue(index=5, value=170.0, name='Airspeed V2 Plus 20 Difference At Vnav Mode And Eng Thrust Mode Required'),
             KeyPointValue(index=15, value=70.0, name='Airspeed V2 Plus 20 Difference At Vnav Mode And Eng Thrust Mode Required')])


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


class TestAirspeed35To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed35To1000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed35To1000FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value
        
    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To8000FtMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = Airspeed1000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases',
                                          'Altitude STD Smoothed', 'Climb')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 8000), {})]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')

    def test_basic(self):
        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = P('Airspeed', np.ma.array(testwave))
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.array(testwave) * 50)
        alt_std = P('Altitude STD Smoothed', np.ma.array(testwave) * 50 + 2000)
        climb = buildsections('Climb', [3, 28], [65, 91])
        event=Airspeed1000To8000FtMax()
        event.derive(spd, alt_aal, alt_std, climb)
        self.assertEqual(event[0].index, 22)
        self.assertEqual(event[1].index, 84)
        self.assertGreater(event[0].value, 150.0)
        self.assertGreater(event[1].value, 150.0)
        

class TestAirspeed8000To10000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed8000To10000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


########################################
# Airspeed: Descending


class TestAirspeed10000To8000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed10000To8000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude STD Smoothed', 'Descent')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

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


class TestAirspeed5000To3000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed5000To3000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Descent')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed3000To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed3000To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Initial Approach')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeed1000To500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = P('Airspeed', np.ma.array(testwave))
        alt_ph = P('Altitude AAL For Flight Phases', np.ma.array(testwave) * 10)
        final_app = buildsections('Final Approach', [47, 60], [109, 123])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertEqual(kpv[0].value, 91.250101656055278)
        self.assertEqual(kpv[1].index, 110)
        self.assertEqual(kpv[1].value, 99.557430201194919)


class TestAirspeed1000To500FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed1000To500FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

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


class TestAirspeedMinusV2For3Sec35To1000FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedMinusV2For3Sec35To1000FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec35To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2 For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

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


class TestAirspeedRelativeFor3Sec1000To500FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec500To20FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAirspeedRelativeFor3Sec20FtToTouchdownMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec', 'Altitude AAL For Flight Phases', 'Touchdown', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Airspeed: Configuration


class TestAirspeedWithConfigurationMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.values_mapping = {
            0 : '0',
            1 : '1',
            2 : '1+F',
            3 : '1*',
            4 : '2',
            5 : '2*',
            6 : '3',
            7 : '4',
            8 : '5',
            9 : 'Full',
        }
        self.node_class = AirspeedWithConfigurationMax
        self.operational_combinations = [
            ('Configuration', 'Airspeed', 'Fast'),
        ]

    def test_derive(self):
        conf_array = np.ma.array(
            [0, 1, 1, 2, 2, 4, 6, 4, 4, 2, 1, 0, 0, 0, 0, 0])
        conf = M('Configuration', conf_array,
                 values_mapping=self.values_mapping)
        air_spd = P('Airspeed', np.ma.arange(16))
        fast = buildsection('Fast', 0, 16)
        node = self.node_class()
        node.derive(conf, air_spd, fast)

        self.assertEqual(len(node), 4)
        self.assertEqual(node[0].name, 'Airspeed With Configuration 1 Max')
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[1].name, 'Airspeed With Configuration 1+F Max')
        self.assertEqual(node[1].index, 9)
        self.assertEqual(node[1].value, 9)
        self.assertEqual(node[2].name, 'Airspeed With Configuration 2 Max')
        self.assertEqual(node[2].index, 8)
        self.assertEqual(node[2].value, 8)
        self.assertEqual(node[3].name, 'Airspeed With Configuration 3 Max')
        self.assertEqual(node[3].index, 6)
        self.assertEqual(node[3].value, 6)




class TestAirspeedRelativeWithConfigurationDuringDescentMin(unittest.TestCase, NodeTest):


    def setUp(self):
        self.values_mapping = {
            0 : '0',
            1 : '1',
            2 : '1+F',
            3 : '1*',
            4 : '2',
            5 : '2*',
            6 : '3',
            7 : '4',
            8 : '5',
            9 : 'Full',
        }
        self.node_class = AirspeedRelativeWithConfigurationDuringDescentMin
        self.operational_combinations = [
            ('Configuration', 'Airspeed Relative', 'Descent To Flare'),
        ]

    def test_derive(self):
        conf_array = np.ma.array(
            [0, 1, 1, 2, 2, 4, 6, 4, 4, 2, 1, 0, 0, 0, 0, 0])
        conf_array = np.ma.concatenate((conf_array, conf_array[::-1]))
        conf = M('Configuration', conf_array,
                 values_mapping=self.values_mapping)
        air_spd_array = np.ma.concatenate((np.ma.arange(16), np.ma.arange(16, -1, -1)))
        air_spd = P('Airspeed Relative', air_spd_array)
        fast = buildsection('Descent To Flare', 16, 30)
        node = self.node_class()
        node.derive(conf, air_spd, fast)

        self.assertEqual(len(node), 4)
        self.assertEqual(node[0].name, 'Airspeed Relative With Configuration 1 During Descent Min')
        self.assertEqual(node[0].index, 30)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(node[1].name, 'Airspeed Relative With Configuration 1+F During Descent Min')
        self.assertEqual(node[1].index, 28)
        self.assertEqual(node[1].value, 4)
        self.assertEqual(node[2].name, 'Airspeed Relative With Configuration 2 During Descent Min')
        self.assertEqual(node[2].index, 26)
        self.assertEqual(node[2].value, 6)
        self.assertEqual(node[3].name, 'Airspeed Relative With Configuration 3 During Descent Min')
        self.assertEqual(node[3].index, 25)
        self.assertEqual(node[3].value, 7)


##############################################################################
# Airspeed: Flap


class TestAirspeedWithFlapMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedWithFlapMax
        self.operational_combinations = [
            ('Airspeed', 'Fast', 'Flap Lever', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Fast', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Fast', 'Flap Lever'),
        ]

    def test_derive(self):
        flap = [[0, 5, 10]] * 10
        flap_array = np.ma.array(reduce(operator.add, zip(*flap)))
        values_mapping = {f: str(f) for f in set(flap_array)}
        flap_inc_trans = M('Flap Including Transition', flap_array.copy(),
                           values_mapping=values_mapping)
        flap_exc_trans = M('Flap Excluding Transition', flap_array.copy(),
                           values_mapping=values_mapping)
        air_spd = P('Airspeed', np.ma.arange(30))
        fast = buildsection('Fast', 0, 30)
        flap_inc_trans.array[19] = np.ma.masked  # mask the max value
        air_spd_flap_max = AirspeedWithFlapMax()
        air_spd_flap_max.derive(None, None, flap_inc_trans, flap_exc_trans, air_spd, fast)

        self.assertEqual(len(air_spd_flap_max), 4)
        self.assertEqual(air_spd_flap_max[1].name, 'Airspeed With Flap Including Transition 5 Max')
        self.assertEqual(air_spd_flap_max[1].index, 18)  # 19 was masked
        self.assertEqual(air_spd_flap_max[1].value, 18)
        self.assertEqual(air_spd_flap_max[0].name, 'Airspeed With Flap Including Transition 10 Max')
        self.assertEqual(air_spd_flap_max[0].index, 29)
        self.assertEqual(air_spd_flap_max[0].value, 29)
        self.assertEqual(air_spd_flap_max[3].name, 'Airspeed With Flap Excluding Transition 5 Max')
        self.assertEqual(air_spd_flap_max[3].index, 19)
        self.assertEqual(air_spd_flap_max[3].value, 19)
        self.assertEqual(air_spd_flap_max[2].name, 'Airspeed With Flap Excluding Transition 10 Max')
        self.assertEqual(air_spd_flap_max[2].index, 29)
        self.assertEqual(air_spd_flap_max[2].value, 29)

    def test_derive_alternative_method(self):
        # Note: This test will produce the following warning:
        #       "No flap settings - rounding to nearest 5"
        flap = [[0, 1, 2, 5, 10, 15, 25, 30, 40, 0]] * 2
        flap_array = np.ma.array(reduce(operator.add, zip(*flap)))
        flap_angle = P('Flap Angle', flap_array)
        air_spd = P('Airspeed', np.ma.arange(20))
        fast = buildsection('Fast', 0, 20)
        flap_inc_trans = FlapIncludingTransition()
        flap_inc_trans.derive(flap_angle)
        flap_exc_trans = FlapExcludingTransition()
        flap_exc_trans.derive(flap_angle)
        air_spd_flap_max = AirspeedWithFlapMax()
        air_spd_flap_max.derive(None, None, flap_inc_trans, flap_exc_trans, air_spd, fast)

        self.assertEqual(air_spd_flap_max.get_ordered_by_index(), [
            KeyPointValue(index=7, value=7, name='Airspeed With Flap Including Transition 5 Max'),
            KeyPointValue(index=7, value=7, name='Airspeed With Flap Excluding Transition 5 Max'),
            KeyPointValue(index=9, value=9, name='Airspeed With Flap Including Transition 10 Max'),
            KeyPointValue(index=9, value=9, name='Airspeed With Flap Excluding Transition 10 Max'),
            KeyPointValue(index=11, value=11, name='Airspeed With Flap Including Transition 15 Max'),
            KeyPointValue(index=11, value=11, name='Airspeed With Flap Excluding Transition 15 Max'),
            KeyPointValue(index=13, value=13, name='Airspeed With Flap Including Transition 25 Max'),
            KeyPointValue(index=13, value=13, name='Airspeed With Flap Excluding Transition 25 Max'),
            KeyPointValue(index=15, value=15, name='Airspeed With Flap Including Transition 30 Max'),
            KeyPointValue(index=15, value=15, name='Airspeed With Flap Excluding Transition 30 Max'),
            KeyPointValue(index=17, value=17, name='Airspeed With Flap Including Transition 40 Max'),
            KeyPointValue(index=17, value=17, name='Airspeed With Flap Excluding Transition 40 Max'),
        ])

    @patch.dict('analysis_engine.key_point_values.AirspeedWithFlapMax.NAME_VALUES',
            {'flap': (5.5, 10.1, 20.9)})
    def test_derive_fractional_settings(self):
        flap = [[0, 5.5, 10.1, 20.85]] * 5
        flap_array = np.ma.array(reduce(operator.add, zip(*flap)))
        values_mapping = {f: str(f) for f in set(flap_array)}
        flap_lever = M('Flap Lever', flap_array.copy(),
                       values_mapping=values_mapping)
        flap = M('Flap', flap_array.copy(),
                 values_mapping=values_mapping)
        flap_inc_trans = M('Flap Including Transition', flap_array.copy(),
                           values_mapping=values_mapping)
        flap_exc_trans = M('Flap Excluding Transition', flap_array.copy(),
                           values_mapping=values_mapping)
        air_spd = P('Airspeed', np.ma.arange(30))
        fast = buildsection('Fast', 0, 30)
        
        node = self.node_class()
        node.derive(None, None, flap_inc_trans, None, air_spd, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap Including Transition 5.5 Max')
        self.assertEqual(node[1].name, 'Airspeed With Flap Including Transition 10.1 Max')
        self.assertEqual(node[2].name, 'Airspeed With Flap Including Transition 20.9 Max')
        
        node = self.node_class()
        node.derive(None, None, None, flap_exc_trans, air_spd, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap Excluding Transition 5.5 Max')
        
        node = self.node_class()
        node.derive(flap_lever, None, None, None, air_spd, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')
        
        node = self.node_class()
        node.derive(None, flap, None, None, air_spd, fast)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')
        
        node = self.node_class()
        node.derive(flap_lever, flap, flap_inc_trans, flap_exc_trans, air_spd, fast)
        self.assertEqual(len(node), 9)
        self.assertEqual(node[0].name, 'Airspeed With Flap 5.5 Max')
        self.assertEqual(node[3].name, 'Airspeed With Flap Including Transition 5.5 Max')
        self.assertEqual(node[6].name, 'Airspeed With Flap Excluding Transition 5.5 Max')


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
        self.operational_combinations = [
            ('Airspeed', 'Climb', 'Flap Lever', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Climb', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Climb', 'Flap Lever'),
        ]

    def test_derive_basic(self):
        flap_inc_trans_array = np.ma.array(
            [0, 0, 5, 10, 10, 10, 15, 15, 15, 35])
        flap_inc_trans_values_mapping = {f: str(f) for f in np.ma.unique(flap_inc_trans_array)}
        flap_inc_trans = M('Flap Including Transition', flap_inc_trans_array,
                           values_mapping=flap_inc_trans_values_mapping)
        flap_exc_trans_array = np.ma.array([0, 0, 5, 10, 15, 35, 35, 15, 10, 0])
        flap_exc_trans_values_mapping = {f: str(f) for f in np.ma.unique(flap_exc_trans_array)}
        flap_exc_trans = M('Flap Excluding Transition', flap_exc_trans_array,
                           values_mapping=flap_exc_trans_values_mapping)
        airspeed = P('Airspeed', np.ma.arange(0, 100, 10))
        climb = buildsection('Climbing', 2, 8)
        node = self.node_class()
        node.derive(None, None, flap_inc_trans, flap_exc_trans, airspeed, climb)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=2.0, value=20.0, name='Airspeed With Flap Including Transition 5 During Climb Max'),
            KeyPointValue(index=2.0, value=20.0, name='Airspeed With Flap Excluding Transition 5 During Climb Max'),
            KeyPointValue(index=5.0, value=50.0, name='Airspeed With Flap Including Transition 10 During Climb Max'),
            KeyPointValue(index=6.0, value=60.0, name='Airspeed With Flap Excluding Transition 35 During Climb Max'),
            KeyPointValue(index=7.0, value=70.0, name='Airspeed With Flap Excluding Transition 15 During Climb Max'),
            KeyPointValue(index=8.0, value=80.0, name='Airspeed With Flap Including Transition 15 During Climb Max'),
            KeyPointValue(index=8.0, value=80.0, name='Airspeed With Flap Excluding Transition 10 During Climb Max'),
        ])


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
        self.operational_combinations = [
            ('Airspeed', 'Descent', 'Flap Lever', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Descent', 'Flap', 'Flap Including Transition', 'Flap Excluding Transition'),
            ('Airspeed', 'Descent', 'Flap Lever'),
        ]

    def test_derive_basic(self):
        flap_inc_trans_array = np.ma.array([0, 0, 5, 10, 10, 10, 15, 15, 15, 35])
        flap_inc_trans = M('Flap Including Transition', flap_inc_trans_array,
                           values_mapping={f: str(f) for f in np.ma.unique(flap_inc_trans_array)})
        flap_exc_trans_array = np.ma.array([0, 0, 5, 10, 15, 35, 35, 15, 10, 0])
        flap_exc_trans = M('Flap Excluding Transition', flap_exc_trans_array,
                           values_mapping={f: str(f) for f in np.ma.unique(flap_exc_trans_array)})
        airspeed = P('Airspeed', np.ma.arange(100, 0, -10))
        desc = buildsection('Descending', 2, 8)
        node = self.node_class()
        node.derive(None, None, flap_inc_trans, flap_exc_trans, airspeed, desc)
        self.assertEqual(node.get_ordered_by_index(), [
            KeyPointValue(index=2.0, value=80.0, name='Airspeed With Flap Including Transition 5 During Descent Max'),
            KeyPointValue(index=2.0, value=80.0, name='Airspeed With Flap Excluding Transition 5 During Descent Max'),
            KeyPointValue(index=3.0, value=70.0, name='Airspeed With Flap Including Transition 10 During Descent Max'),
            KeyPointValue(index=3.0, value=70.0, name='Airspeed With Flap Excluding Transition 10 During Descent Max'),
            KeyPointValue(index=4.0, value=60.0, name='Airspeed With Flap Excluding Transition 15 During Descent Max'),
            KeyPointValue(index=5.0, value=50.0, name='Airspeed With Flap Excluding Transition 35 During Descent Max'),
            KeyPointValue(index=6.0, value=40.0, name='Airspeed With Flap Including Transition 15 During Descent Max'),])


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
        
        # deployed the whole time
        rev.array[:] = 'Deployed'
        dur = ThrustReversersDeployedDuration()
        dur.derive(rev, ldg)
        self.assertEqual(len(dur), 1)
        self.assertEqual(dur[0].index, 5)
        self.assertEqual(dur[0].value, 10)


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


class TestMainGearOnGroundToNoseGearOnGroundDuration(unittest.TestCase,
                                                     NodeTest):

    def test_derive(self):
        from analysis_engine.key_point_values import \
            MainGearOnGroundToNoseGearOnGroundDuration

        self.node_class = MainGearOnGroundToNoseGearOnGroundDuration
        self.operational_combinations = [('Brake Pressure', 'Takeoff Roll',)]
        self.function = max_value

        gog_array = np.ma.array([0] * 20 + [1] * 15)
        gog = M(
            name='Gear On Ground',
            array=gog_array,
            values_mapping={0: 'Air', 1: 'Ground'},
        )
        gogn_array = np.ma.array([0] * 25 + [1] * 10)
        gogn = M(
            name='Gear (N) On Ground',
            array=gogn_array,
            values_mapping={0: 'Air', 1: 'Ground'},
        )
        landing = buildsection('Landing', 10, 30)
        node = self.node_class()
        node.derive(gog, gogn, landing)
        self.assertEqual(node, [
            KeyPointValue(
                19.5, 5.0,
                'Main Gear On Ground To Nose Gear On Ground Duration'),
        ])


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


class TestAutobrakeRejectedTakeoffNotSetDuringTakeoff(unittest.TestCase,
                                                      CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            AutobrakeRejectedTakeoffNotSetDuringTakeoff

        self.values_array = np.ma.array([1] * 3 + [0] * 6 + [1] * 3)
        self.expected = [KeyPointValue(
            index=3, value=4.0,
            name='Autobrake Rejected Takeoff Not Set During Takeoff')]

        self.param_name = 'Autobrake Selected RTO'
        self.phase_name = 'Takeoff'
        self.node_class = AutobrakeRejectedTakeoffNotSetDuringTakeoff
        self.values_mapping = {0: '-', 1: 'Selected'}

        self.basic_setup()


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
        self.operational_combinations = [('Altitude STD Smoothed', 'Altitude AAL')]

    def test_derive_handling_no_data(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array([0] + [1000] * 4),
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std)
        self.assertEqual(node, [])

    def test_derive_up_down_and_up(self):
        alt_aal = P(name='Altitude AAL',
                    array=np.ma.array((1.0-np.cos(np.arange(0,12.6,0.1)))*1000)+10000.0,
                    )
        alt_std = P(name='Altitude STD Smoothed',
                    array=alt_aal.array+3000.0)
        node = AltitudeOvershootAtSuspectedLevelBust()        
        node.derive(alt_std, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(index=31, value=1995.6772472964967,
                name='Altitude Overshoot At Suspected Level Bust'),
            KeyPointValue(index=63, value=-1992.0839618360187,
                name='Altitude Overshoot At Suspected Level Bust'),
            KeyPointValue(index=94, value=1985.8853443140706,
                name='Altitude Overshoot At Suspected Level Bust')
        ])
        
    def test_comparable_to_alt_aal(self):
        # If the overshoot or undershoot is comparable to the height above the airfield, no level bust is declared.
        alt_aal = P(name='Altitude AAL',
                    array=np.ma.array((1.0-np.cos(np.arange(0,12.6,0.1)))*1000)+1000.0,
                    )
        alt_std = P(name='Altitude STD Smoothed',
                    array=alt_aal.array+3000.0)
        node = AltitudeOvershootAtSuspectedLevelBust()
        
        node.derive(alt_std, alt_aal)
        self.assertEqual(node, [])

    def test_derive_too_slow(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array(1.0 + np.sin(np.arange(0, 12.6, 0.1))) * 1000,
            frequency=0.02,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std, alt_std) # Pretend the airfield is at sea level :o)
        self.assertEqual(node, [])

    def test_derive_straight_up_and_down(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array(range(0, 10000, 50) + range(10000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std, alt_std) # Pretend the airfield is at sea level :o)
        self.assertEqual(node, [])

    def test_derive_up_and_down_with_overshoot(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array(range(0, 10000, 50) + range(10000, 9000, -50)
                + [9000] * 200 + range(9000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std, alt_std) # Pretend the airfield is at sea level :o)
        self.assertEqual(node, [
            KeyPointValue(index=200, value=1000,
                name='Altitude Overshoot At Suspected Level Bust'),
        ])

    def test_derive_up_and_down_with_undershoot(self):
        alt_std = P(
            name='Altitude STD Smoothed',
            array=np.ma.array(range(0, 10000, 50) + [10000] * 200
                + range(10000, 9000, -50) + range(9000, 20000, 50)
                + range(20000, 0, -50)),
            frequency=1,
        )
        node = AltitudeOvershootAtSuspectedLevelBust()
        node.derive(alt_std, alt_std) # Pretend the airfield is at sea level :o)
        self.assertEqual(node, [
            KeyPointValue(index=420, value=-1000,
                name='Altitude Overshoot At Suspected Level Bust'),
        ])
        
    def test_derive_with_real_go_around_data_ignores_overshoot(self):
        alt_std = load(os.path.join(test_data_path,
                                    'alt_std_smoothed_go_around.nod'))
        alt_aal = load(os.path.join(test_data_path,
                                    'alt_aal_go_around.nod'))
        alt_aal_aligned = alt_aal.get_aligned(alt_std)
        bust = AltitudeOvershootAtSuspectedLevelBust()
        bust.derive(alt_std, alt_aal_aligned)
        self.assertEqual(len(bust), 0)


class TestAltitudeAtVNAVModeAndEngThrustModeRequired(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = AltitudeAtVNAVModeAndEngThrustModeRequired
        self.operational_combinations = [('Altitude AAL', 'VNAV Mode And Eng Thrust Mode Required')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestAltitudeAtCabinPressureLowWarningDuration(unittest.TestCase,
                                                    CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            AltitudeAtCabinPressureLowWarningDuration

        self.param_name = 'Cabin Altitude'
        self.phase_name = 'Airborne'
        self.node_class = AltitudeAtCabinPressureLowWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


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
        self.operational_combinations = [('Flap Extension While Airborne', 'Altitude AAL')]

    def test_derive_multiple_ktis(self):
        alt_aal = P('Altitude AAL', np.ma.array([1234.0]*15+[2345.0]*15))
        flap_exts = KTI('Flap Extension While Airborne', items=[
            KeyTimeInstance(10, 'Flap Extension While Airborne'),
            KeyTimeInstance(20, 'Flap Extension While Airborne'),
            ])
        node = AltitudeAtFlapExtension()
        node.derive(flap_exts, alt_aal)
        self.assertEqual(node, [
            KeyPointValue(10, 1234, 'Altitude At Flap Extension'),
            KeyPointValue(20, 2345, 'Altitude At Flap Extension'),
        ])

    def test_derive_no_ktis(self):
        '''
        Create no KPVs without a Go Around Flap Retracted KTI.
        '''
        alt_aal = P('Altitude AAL', np.ma.array([1234.0]*15+[2345.0]*15))
        flap_exts = KTI('Flap Extension While Airborne', items=[])
        node = AltitudeAtFlapExtension()
        node.derive(flap_exts, alt_aal)
        self.assertEqual(node, [])

class TestAltitudeAtFirstFlapExtensionAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapExtensionAfterLiftoff
        self.operational_combinations = [('Altitude At Flap Extension',)]

    def test_derive_basic(self):
        flap_exts = KPV('Altitude At Flap Extension',
                        items=[KeyPointValue(index=7, value=21),
                               KeyPointValue(index=14, value=43)])
        node = self.node_class()
        node.derive(flap_exts)
        self.assertEqual(node, [KeyPointValue(name='Altitude At First Flap Extension After Liftoff',
                                              index=7, value=21)])


class TestAltitudeAtFirstFlapChangeAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapChangeAfterLiftoff
        self.operational_combinations = [
            ('Flap Lever', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
            ('Flap', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
            ('Flap Lever', 'Flap', 'Flap At Liftoff', 'Altitude AAL', 'Airborne'),
        ]

    def test_derive(self):
        flap_takeoff = KPV('Flap At Liftoff', items=[
            KeyPointValue(name='Flap At Liftoff', index=2, value=5.0),
        ])
        flap_array = np.ma.array(
            [0, 5, 5, 5, 5, 0, 0, 0, 0, 15, 30, 30, 30, 30, 15, 0])
        flap = M('Flap', flap_array,
                 values_mapping={f: str(f) for f in np.ma.unique(flap_array)})
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array,alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airs = buildsection('Airborne', 2, 14)

        node = AltitudeAtFirstFlapChangeAfterLiftoff()
        node.derive(flap, None, flap_takeoff, alt_aal, airs)

        expected = KPV('Altitude At First Flap Change After Liftoff', items=[
            KeyPointValue(name='Altitude At First Flap Change After Liftoff', index=4.5, value=150),
        ])
        node = AltitudeAtFirstFlapChangeAfterLiftoff()
        node.derive(None, flap, flap_takeoff, alt_aal, airs)
        self.assertEqual(node, expected)

    def test_derive_no_flap_takeoff(self):
        flap_takeoff = KPV('Flap At Liftoff', items=[
            KeyPointValue(name='Flap At Liftoff', index=2, value=0.0),
        ])
        flap_array = np.ma.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 30, 30, 30, 30, 15, 0])
        flap = M('Flap', flap_array,
                 values_mapping={f: str(f) for f in np.ma.unique(flap_array)})
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array,alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airs = buildsection('Airborne', 2, 14)

        node = AltitudeAtFirstFlapChangeAfterLiftoff()
        node.derive(flap, None, flap_takeoff, alt_aal, airs)

        expected = KPV('Altitude At First Flap Change After Liftoff', items=[])
        self.assertEqual(node, expected)
        
        node = AltitudeAtFirstFlapChangeAfterLiftoff()
        node.derive(None, flap, flap_takeoff, alt_aal, airs)
        self.assertEqual(node, expected)


class TestAltitudeAtFlapExtensionWithGearDown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFlapExtensionWithGearDown
        self.operational_combinations = [
            ('Flap Lever', 'Altitude AAL', 'Gear Extended', 'Airborne'),
            ('Flap', 'Altitude AAL', 'Gear Extended', 'Airborne'),
            ('Flap Lever', 'Flap', 'Altitude AAL', 'Gear Extended', 'Airborne'),
        ]

    def test_derive(self):
        flap_array = np.ma.array([0, 5, 5, 0, 0, 0, 1, 1, 10, 20, 20, 20, 35, 35, 15, 0.0])
        flap = M('Flap', flap_array, values_mapping={f: str(f) for f in np.ma.unique(flap_array)})
        gear = buildsection('Gear Extended', 7, None)
        
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array,alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airs = buildsection('Airborne', 2, 14)

        node = AltitudeAtFlapExtensionWithGearDown()
        node.derive(flap, None, alt_aal, gear, airs)
        first = node.get_first()
        self.assertEqual(first.index, 8)
        self.assertEqual(first.value, 400)
        self.assertEqual(first.name, 'Altitude At Flap 10 Extension With Gear Down')
        second = node.get_next(8)
        self.assertEqual(second.index, 9)
        self.assertEqual(second.value, 300)
        self.assertEqual(second.name, 'Altitude At Flap 20 Extension With Gear Down')
        third = node.get_last()
        self.assertEqual(third.index, 12)
        self.assertEqual(third.value, 50)
        self.assertEqual(third.name, 'Altitude At Flap 35 Extension With Gear Down')
        
        node = AltitudeAtFlapExtensionWithGearDown()
        node.derive(None, flap, alt_aal, gear, airs)
        self.assertEqual(first.index, 8)
        self.assertEqual(first.value, 400)
        self.assertEqual(first.name, 'Altitude At Flap 10 Extension With Gear Down')        


class TestAirspeedAtFlapExtensionWithGearDown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAtFlapExtensionWithGearDown
        self.operational_combinations = [
            ('Flap Lever', 'Airspeed', 'Gear Extended', 'Airborne'),
            ('Flap', 'Airspeed', 'Gear Extended', 'Airborne'),
            ('Flap Lever', 'Flap', 'Airspeed', 'Gear Extended', 'Airborne'),
        ]

    def test_derive(self):
        flap_array = np.ma.array([0, 5, 5, 0, 0, 0, 1, 1, 10, 20, 20, 20, 35, 35, 15, 0.0])
        flap = M('Flap', np.ma.array([0, 5, 5, 0, 0, 0, 1, 1, 10, 20, 20, 20, 35, 35, 15, 0.0]),
                 values_mapping={f: str(f) for f in np.ma.array(flap_array)})
        gear = buildsection('Gear Extended', 7, None)
        
        air_spd_array = np.ma.array([0, 0, 0, 50, 100, 200, 250, 280])
        air_spd_array2 = np.ma.concatenate((air_spd_array,air_spd_array[::-1]))
        air_spd = P('Airspeed', air_spd_array2)
        airs = buildsection('Airborne', 2, 14)

        node = AirspeedAtFlapExtensionWithGearDown()
        node.derive(flap, None, air_spd, gear, airs)
        first = node.get_first()
        self.assertEqual(first.index, 8)
        self.assertEqual(first.value, 280)
        self.assertEqual(first.name, 'Airspeed At Flap 10 Extension With Gear Down')
        second = node.get_next(8)
        self.assertEqual(second.index, 9)
        self.assertEqual(second.value, 250)
        self.assertEqual(second.name, 'Airspeed At Flap 20 Extension With Gear Down')
        third = node.get_last()
        self.assertEqual(third.index, 12)
        self.assertEqual(third.value, 50)
        self.assertEqual(third.name, 'Airspeed At Flap 35 Extension With Gear Down')
        node = AirspeedAtFlapExtensionWithGearDown()
        node.derive(None, flap, air_spd, gear, airs)
        first = node.get_first()
        self.assertEqual(first.index, 8)
        self.assertEqual(first.value, 280)
        self.assertEqual(first.name, 'Airspeed At Flap 10 Extension With Gear Down')



class TestAltitudeAtLastFlapChangeBeforeTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtLastFlapChangeBeforeTouchdown
        self.operational_combinations = [
            ('Flap Lever', 'Altitude AAL', 'Touchdown'),
            ('Flap', 'Altitude AAL', 'Touchdown'),
            ('Flap Lever', 'Flap', 'Altitude AAL', 'Touchdown'),
        ]

    def test_derive(self):
        flap_array = np.ma.array(([10] * 8) + ([15] * 7))
        flap_lever = M('Flap Lever', flap_array,
                       values_mapping={f: str(f) for f in np.ma.unique(flap_array)})
        alt_aal = P('Altitude AAL',
                    array=np.ma.concatenate([np.ma.arange(1000, 0, -100),
                                             [0] * 5]))
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10)])
        node = self.node_class()
        node.derive(flap_lever, None, alt_aal, touchdowns)
        self.assertEqual(
            node,
            [KeyPointValue(index=8.0, value=200.0,
                           name='Altitude At Last Flap Change Before Touchdown')])


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
        flap_rets = KTI('Flap Retraction During Go Around', items=[
            KeyTimeInstance(100, 'Flap Retraction During Go Around'),
            KeyTimeInstance(104, 'Flap Retraction During Go Around'),
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
        flap_rets = KTI('Flap Retraction During Go Around', items=[])
        node = AltitudeAtFirstFlapRetractionDuringGoAround()
        node.derive(self.alt_aal, flap_rets, self.go_arounds)
        self.assertEqual(node, [])


class TestAltitudeAtFirstFlapRetraction(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstFlapRetraction
        self.operational_combinations = [('Altitude AAL', 'Flap Retraction While Airborne')]
        self.alt_aal = P(
            name='Altitude AAL',
            array=np.ma.concatenate([
                np.ma.array([0] * 10),
                np.ma.arange(40) * 1000,
            ]),
        )

    def test_derive_basic(self):
        '''
        Create a single KPV within the Go Around And Climbout section.
        '''
        flap_rets = KTI('Flap Retraction While Airborne', items=[
            KeyTimeInstance(30, 'Flap Retraction While Airborne'),
            KeyTimeInstance(40, 'Flap Retraction While Airborne'),
        ])
        node = AltitudeAtFirstFlapRetraction()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, [
            KeyPointValue(30, 20000, 'Altitude At First Flap Retraction'),
        ])

    def test_derive_no_ktis(self):
        '''
        Create no KPVs without a Go Around Flap Retracted KTI.
        '''
        flap_rets = KTI('Flap Retraction While Airborne', items=[])
        node = AltitudeAtFirstFlapRetractionDuringGoAround()
        node.derive(self.alt_aal, flap_rets)
        self.assertEqual(node, [])


class TestAltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = AltitudeAtClimbThrustDerateDeselectedDuringClimbBelow33000Ft
        self.operational_combinations = [('Altitude AAL',
                                          'Climb Thrust Derate Deselected',
                                          'Climbing')]
    
    def test_derive_basic(self):
        alt_aal_array = np.ma.concatenate(
            [np.ma.arange(0, 40000, 4000), [40000] * 10,
             np.ma.arange(40000, 0, -4000)])
        alt_aal = P('Altitude AAL', array=alt_aal_array)
        climb_thrust_derate = KTI('Climb Thrust Derate Deselected', items=[
            KeyTimeInstance(5, 'Climb Thrust Derate Deselected'),
            KeyTimeInstance(12, 'Climb Thrust Derate Deselected'),
            KeyTimeInstance(17, 'Climb Thrust Derate Deselected')])
        climbs = buildsection('Climbing', 0, 14)
        node = self.node_class()
        node.derive(alt_aal, climb_thrust_derate, climbs)
        self.assertEqual(node, [
            KeyPointValue(5, 20000.0, 'Altitude At Climb Thrust Derate Deselected During Climb Below 33000 Ft')])


########################################
# Altitude: Gear


class TestAltitudeAtGearDownSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtGearDownSelection
        self.operational_combinations = [('Altitude AAL', 'Gear Down Selection')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearDownSelectionWithFlapDown(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeAtGearDownSelectionWithFlapDown
        self.operational_combinations = [('Altitude AAL', 'Gear Down Selection', 'Flap')]
    
    def test_derive_basic(self):
        alt_aal = P('Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI('Gear Down Selection', items=[KeyTimeInstance(2),
                                                       KeyTimeInstance(4),
                                                       KeyTimeInstance(6),
                                                       KeyTimeInstance(8)])
        flap = M('Flap', array=np.ma.array([5] * 3 + [0] * 5 + [20] * 2),
                 values_mapping={f: str(f) for f in [0, 5, 20]})
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap)
        self.assertEqual(node, [KeyPointValue(2, 200, 'Altitude At Gear Down Selection With Flap Down'),
                                KeyPointValue(8, 800, 'Altitude At Gear Down Selection With Flap Down')])


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


class TestAltitudeAtGearDownSelectionWithFlapUp(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = AltitudeAtGearDownSelectionWithFlapUp
        self.operational_combinations = [('Altitude AAL', 'Gear Down Selection', 'Flap')]
    
    def test_derive_basic(self):
        alt_aal = P('Altitude AAL', array=np.ma.arange(0, 1000, 100))
        gear_downs = KTI('Gear Down Selection', items=[KeyTimeInstance(2),
                                                       KeyTimeInstance(4),
                                                       KeyTimeInstance(6),
                                                       KeyTimeInstance(8)])
        flap = M('Flap', array=np.ma.array([5] * 3 + [0] * 5 + [20] * 2),
                 values_mapping={f: str(f) for f in [0, 5, 20]})
        node = self.node_class()
        node.derive(alt_aal, gear_downs, flap)
        self.assertEqual(node, [KeyPointValue(4, 400, 'Altitude At Gear Down Selection With Flap Up'),
                                KeyPointValue(6, 600, 'Altitude At Gear Down Selection With Flap Up')])


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


class TestAltitudeAtFirstAPEngagedAfterLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AltitudeAtFirstAPEngagedAfterLiftoff
        self.operational_combinations = [('AP Engaged', 'Altitude AAL', 'Airborne')]

    def test_derive_basic(self):

        ap = M('AP Engaged', np.ma.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), values_mapping={1:'Engaged'})
        alt_aal_array = np.ma.array([0, 0, 0, 50, 100, 200, 300, 400])
        alt_aal_array = np.ma.concatenate((alt_aal_array,alt_aal_array[::-1]))
        alt_aal = P('Altitude AAL', alt_aal_array)
        airs = buildsection('Airborne', 2, 14)

        node = self.node_class()
        node.derive(ap=ap,alt_aal=alt_aal, airborne=airs)

        expected = KPV('Altitude At First AP Engaged After Liftoff', items=[
            KeyPointValue(name='Altitude At First AP Engaged After Liftoff', index=4.5, value=150),
        ])
        self.assertEqual(node, expected)


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


class TestControlColumnForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.key_point_values import ControlColumnForceMax

        self.node_class = ControlColumnForceMax
        self.operational_combinations = [('Control Column Force', 'Fast')]

    def test_derive(self):
        ccf = P(
            name='Control Column Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Control Column Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Control Column Force Max')]))


class TestControlWheelForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.key_point_values import ControlWheelForceMax

        self.node_class = ControlWheelForceMax
        self.operational_combinations = [('Control Wheel Force', 'Fast')]

    def test_derive(self):
        cwf = P(
            name='Control Wheel Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 9)
        node = self.node_class()
        node.derive(cwf, phase_fast)
        self.assertEqual(
            node,
            KPV('Control Wheel Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Control Wheel Force Max')]))


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
        expected = [('IAN Glidepath', 'Altitude AAL For Flight Phases', 'Approach Information', 'Displayed App Source (Capt)', 'Displayed App Source (FO)')]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANGlidepathDeviationMax
        approaches = [ApproachItem('LANDING', slice(2, 12)),]
        self.apps = App('Approach Information', items=approaches)
        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.ian = P(name='IAN Glidepath', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 3, 0, 0, 0, 0, 0], dtype=np.float,))
        self.app_src_capt = M(
                    name='Displayed App Source (Capt)',
                    array=np.ma.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    values_mapping={0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'},
                )
        self.app_src_fo = M(
                    name='Displayed App Source (FO)',
                    array=np.ma.array([0, 0, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5]),
                    values_mapping={0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'},
                )

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.app_src_capt, self.app_src_fo)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 3)
        self.assertAlmostEqual(kpv[0].name, 'IAN Glidepath Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        self.apps[0].gs_est = slice(3, 12)
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.app_src_capt, self.app_src_fo)
        self.assertEqual(len(kpv), 0)


class TestIANFinalApproachCourseDeviationMax(unittest.TestCase):

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('IAN Final Approach Course', 'Altitude AAL For Flight Phases', 'Approach Information', 'Displayed App Source (Capt)', 'Displayed App Source (FO)'),]
        self.assertEqual(ops, expected)

    def setUp(self):
        self.node_class = IANFinalApproachCourseDeviationMax
        approaches = [ApproachItem('LANDING', slice(2, 12)),]
        self.height = P(name='Altitude AAL For Flight Phases', array=np.ma.arange(600, 300, -25))
        self.apps = App('Approach Information', items=approaches)
        self.ian = P(name='IAN Final Approach Course', array=np.ma.array([4, 2, 2, 1, 0.5, 0.5, 3, 0, 0, 0, 0, 0], dtype=np.float,))
        self.app_src_capt = M(
                    name='Displayed App Source (Capt)',
                    array=np.ma.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    values_mapping={0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'},
                )
        self.app_src_fo = M(
                    name='Displayed App Source (FO)',
                    array=np.ma.array([0, 0, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5]),
                    values_mapping={0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'},
                )

    def test_derive_basic(self):
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.app_src_capt, self.app_src_fo)
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 6)
        self.assertAlmostEqual(kpv[0].value, 3)
        self.assertAlmostEqual(kpv[0].name, 'IAN Final Approach Course Deviation 500 To 200 Ft Max')

    def test_derive_with_ils_established(self):
        self.apps[0].loc_est = slice(3, 12)
        kpv = self.node_class()
        kpv.derive(self.ian, self.height, self.apps, self.app_src_capt, self.app_src_fo)
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


class TestMachDuringCruiseAvg(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = MachDuringCruiseAvg
        self.operational_combinations = [('Mach', 'Cruise')]
    
    def test_derive_basic(self):
        mach_array = np.ma.concatenate([np.ma.arange(0, 1, 0.1),
                                        np.ma.arange(1, 0, -0.1)])
        mach = P('Mach', array=mach_array)
        cruise = buildsection('Cruise', 5, 10)
        node = self.node_class()
        node.derive(mach, cruise)
        self.assertEqual(node[0].index, 7)
        self.assertAlmostEqual(node[0].value,0.7)
        self.assertEqual(node[0].name, 'Mach During Cruise Avg')


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


class TestAltitudeAtLastAPDisengagedDuringApproach(unittest.TestCase):
    '''
    '''
    def test_can_operate(self):
        ops = AltitudeAtLastAPDisengagedDuringApproach.get_operational_combinations()
        self.assertEqual(ops, [('Altitude AAL', 'AP Disengaged Selection', 'Approach Information')])
    
    def test_derive_basic(self):
        alt_array = np.ma.concatenate([np.ma.arange(10, 0, -1),
                                       np.ma.arange(10),
                                       np.ma.arange(10, 0, -1)])
        alt_aal = P('Altitude AAL', array=alt_array)
        ap_dis = KTI('AP Disengaged Selection',
                     items=[KeyTimeInstance(name='AP Disengaged', index=3),
                            KeyTimeInstance(name='AP Disengaged', index=7),
                            KeyTimeInstance(name='AP Disengaged', index=25)])
        apps = App('Approach Information',
                   items=[ApproachItem('TOUCH_AND_GO', slice(0, 10)),
                          ApproachItem('LANDING', slice(20, 30)),])
        node = AltitudeAtLastAPDisengagedDuringApproach()
        node.derive(alt_aal, ap_dis, apps)
        self.assertEqual(node,
                         [KeyPointValue(index=7, value=3.0, name='Altitude At Last AP Disengaged During Approach'),
                          KeyPointValue(index=25, value=5.0, name='Altitude At Last AP Disengaged During Approach')])


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


class TestEngEPRDuringApproachMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRDuringApproachMax

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('Eng (*) EPR Max', 'Approach')]
        self.assertEqual(ops, expected)

    def test_derive(self):
        approaches = buildsection('Approach', 70, 120)
        epr_array = np.round(10 + np.ma.array(10 * np.sin(np.arange(0, 12.6, 0.1))))
        epr = P(name='Eng (*) EPR Max', array=epr_array)
        node = self.node_class()
        node.derive(epr, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 76)
        self.assertEqual(node[0].value, 20)
        self.assertEqual(node[0].name, 'Eng EPR During Approach Max')


class TestEngEPRDuringApproachMin(unittest.TestCase):

    def setUp(self):
        self.node_class = EngEPRDuringApproachMin

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        expected = [('Eng (*) EPR Min', 'Approach')]
        self.assertEqual(ops, expected)


    def test_derive(self):
        approaches = buildsection('Approach', 70, 120)
        epr_array = np.round(10 + np.ma.array(10 * np.sin(np.arange(0, 12.6, 0.1))))
        epr = P(name='Eng (*) EPR Max', array=epr_array)
        node = self.node_class()
        node.derive(epr, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 107)
        self.assertEqual(node[0].value, 0)
        self.assertEqual(node[0].name, 'Eng EPR During Approach Min')


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


class TestEngTPRDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngTPRDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) TPR Max', 'Takeoff 5 Min Rating')]
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


class TestEngEPRDuringMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

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


class TestEngEPRFor5Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRFor5Sec1000To500FtMin
        self.operational_combinations = [('Eng (*) EPR Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRFor5Sec500To50FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngEPRFor5Sec500To50FtMin
        self.operational_combinations = [('Eng (*) EPR Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngEPRAtTOGADuringTakeoffMax(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = EngEPRAtTOGADuringTakeoffMax
        self.operational_combinations = [('Eng (*) EPR Max', 'Takeoff And Go Around', 'Takeoff')]
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRAtTOGADuringTakeoffMin(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = EngTPRAtTOGADuringTakeoffMin
        self.operational_combinations = [('Eng (*) TPR Min', 'Takeoff And Go Around', 'Takeoff')]
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    
    def setUp(self):
        self.node_class = EngTPRDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Takeoff 5 Min Rating')]
        self.function = max_value
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngTPRDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    
    def setUp(self):
        self.node_class = EngTPRDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng TPR Limit Difference', 'Go Around 5 Min Rating')]
        self.function = max_value
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


#class TestEngTPRLimitDifferenceDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    
    #def setUp(self):
        #self.node_class = EngTPRLimitDifferenceDuringTakeoffMax
        #self.operational_combinations = [('Eng TPR Limit Difference', 'Takeoff')]
        #self.function = max_value
    
    #@unittest.skip('Test Not Implemented')
    #def test_derive(self):
        #self.assertTrue(False, msg='Test Not Implemented')


#class TestEngTPRLimitDifferenceDuringGoAroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    
    #def setUp(self):
        #self.node_class = EngTPRLimitDifferenceDuringGoAroundMax
        #self.operational_combinations = [('Eng TPR Limit Difference', 'Go Around')]
        #self.function = max_value
    
    #@unittest.skip('Test Not Implemented')
    #def test_derive(self):
        #self.assertTrue(False, msg='Test Not Implemented')


##############################################################################
# Engine Fire


class TestEngFireWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import EngFireWarningDuration

        self.param_name = 'Eng (*) Fire'
        self.phase_name = 'Airborne'
        self.node_class = EngFireWarningDuration
        self.values_mapping = {0: '-', 1: 'Fire'}

        self.basic_setup()


##############################################################################
# APU Fire


class TestAPUFireWarningDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = APUFireWarningDuration

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        expected = [('Fire APU Single Bottle System',),
                    ('Fire APU Dual Bottle System',),
                    ('Fire APU Single Bottle System', 'Fire APU Dual Bottle System')]
        self.assertEqual(expected, opts)

    def test_derive_basic(self):
        values_mapping = {
            1: 'Fire',
        }
        single_fire = M(name='Fire APU Single Bottle System',
                        array=np.ma.zeros(10),
                        values_mapping=values_mapping)
        single_fire.array[5:7] = 'Fire'
        
        node = self.node_class()
        node.derive(single_fire, None)

        self.assertEqual(node[0].name, 'APU Fire Warning Duration')
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(len(node), 1)


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


class TestEngN1For5Sec1000To500FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1For5Sec1000To500FtMin
        self.operational_combinations = [('Eng (*) N1 Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngN1For5Sec500To50FtMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngN1For5Sec500To50FtMin
        self.operational_combinations = [('Eng (*) N1 Min For 5 Sec', 'Altitude AAL For Flight Phases', 'HDF Duration')]

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


class TestEngN154to72PercentWithThrustReversersDeployedDurationMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngN154to72PercentWithThrustReversersDeployedDurationMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        expected = [('Eng (1) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Thrust Reversers'),
                    ('Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers'),
                    ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1', 'Thrust Reversers')]

        self.assertEqual(expected, opts)

    def test_derive(self):
        values_mapping = {
            0: 'Stowed',
            1: 'In Transit',
            2: 'Deployed',
        }
        n1_array = 30*(2+np.sin(np.arange(0, 12.6, 0.1)))
        eng_1 = P(name='Eng (1) N1', array=np.ma.array(n1_array))
        thrust_reversers_array = np.ma.zeros(126)
        thrust_reversers_array[55:94] = 2
        thrust_reversers = M('Thrust Reversers', array=thrust_reversers_array, values_mapping=values_mapping)

        node = self.node_class()
        node.derive(eng_1, None, None, None, thrust_reversers)

        self.assertEqual(node[0].name, 'Eng (1) N1 54 To 72 Percent With Thrust Reversers Deployed Duration Max')
        self.assertEqual(node[0].index, 64)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(len(node), 1)

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
# Engine Np


class TestEngNpDuringTaxiMax(unittest.TestCase, CreateKPVFromSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringTaxiMax
        self.operational_combinations = [('Eng (*) Np Max', 'Taxiing')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpDuringTakeoff5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringTakeoff5MinRatingMax
        self.operational_combinations = [('Eng (*) Np Max', 'Takeoff 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngNpDuringGoAround5MinRatingMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngNpDuringGoAround5MinRatingMax
        self.operational_combinations = [('Eng (*) Np Max', 'Go Around 5 Min Rating')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngNpMaximumContinuousPowerMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngNpDuringMaximumContinuousPowerMax
        self.operational_combinations = [('Eng (*) Np Max', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating', 'Grounded')]

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


class TestEngOilQtyDuringTaxiInMax(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.key_point_values import EngOilQtyDuringTaxiInMax

        self.node_class = EngOilQtyDuringTaxiInMax
        self.operational_combinations = [('Eng (1) Oil Qty', 'Taxi In')]
        self.function = max_value

    def test_derive(self):
        oil_qty = P(
            name='Eng (1) Oil Qty',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        taxi_in = S(items=[Section('Taxi In', slice(3, 9), 3, 9)])
        node = self.node_class()
        node.derive(oil_qty, None, None, None, taxi_in)
        self.assertEqual(
            node,
            KPV('Eng (1) Oil Qty During Taxi In Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Eng (1) Oil Qty During Taxi In Max')]))


class TestEngOilQtyDuringTaxiOutMax(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.key_point_values import EngOilQtyDuringTaxiOutMax

        self.node_class = EngOilQtyDuringTaxiOutMax
        self.operational_combinations = [('Eng (1) Oil Qty', 'Taxi Out')]
        self.function = max_value

    def test_derive(self):
        oil_qty = P(
            name='Eng (1) Oil Qty',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        taxi_out = S(items=[Section('Taxi Out', slice(3, 9), 3, 9)])
        node = self.node_class()
        node.derive(oil_qty, None, None, None, taxi_out)
        self.assertEqual(
            node,
            KPV('Eng (1) Oil Qty During Taxi Out Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Eng (1) Oil Qty During Taxi Out Max')]))

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

    '''
    This data set produces overlaps which extend beyond the bounds of the
    array with 45 minute clip periods (daft, but that's what the KPV calls
    for). The consequence was that the np.ma.average was being called with no
    data, returning "nan" and this led to a large number of KTP problems. Not
    certain how to replicate this in a test.
    
    def test_derive_real_case(self):
        oil_temp = P(
            name='Eng (*) Oil Temp Max',
            array=np.ma.array(data=[[67.0,79,81,82,84,85,87,88,90,90,92,93,93,94,
                                    94,95,97,103,109,112,115,118,119,121,121,
                                    123,123,124,124,125,125,124,122,121,121,120,
                                    120]+[119]*34+[117]*17+[115]*98+[112]*5+[106]*65+[103]*80][0]))
        oil_temp.array[-3:]=np.ma.masked
        node = EngOilTempForXMinMax()
        node.derive(oil_temp)
        '''

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


class TestEngVibAMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibAMax
        self.operational_combinations = [('Eng (*) Vib A Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibBMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibBMax
        self.operational_combinations = [('Eng (*) Vib B Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibCMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngVibCMax
        self.operational_combinations = [('Eng (*) Vib C Max', 'Airborne')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestEngVibBroadbandMax(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = EngVibBroadbandMax
        self.operational_combinations = [('Eng (*) Vib Broadband Max',)]
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
            ('Touchdown', 'AFR Landing Airport'),
            ('Touchdown', 'AFR Landing Runway'),
            ('Touchdown', 'Longitude (Coarse)'),
            ('Longitude', 'Touchdown', 'AFR Landing Airport'),
            ('Longitude', 'Touchdown', 'AFR Landing Runway'),
            ('Longitude', 'Touchdown', 'Longitude (Coarse)'),
            ('Touchdown', 'AFR Landing Airport', 'AFR Landing Runway'),
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


class TestElevatorDuringLandingMin(unittest.TestCase,
                                   CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ElevatorDuringLandingMin
        self.operational_combinations = [('Elevator', 'Landing')]
        self.function = min_value

    def test_derive(self):
        ccf = P(
            name='Elevator During Landing',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Landing', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Elevator During Landing Min',
                items=[KeyPointValue(
                    index=8.0, value=42.0,
                    name='Elevator During Landing Min')]))


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
        self.interpolate = False

    def test_derive(self):
        flap = P(
            name='Flap',
            array=np.ma.repeat([0, 1, 5, 15, 5, 1, 0], 5),
        )
        for index, value in (14.25, 5), (14.75, 15), (15.00, 15), (15.25, 15):
            liftoffs = KTI(name='Liftoff', items=[
                KeyTimeInstance(index=index, name='Liftoff'),
            ])
            node = self.node_class()
            node.derive(flap, liftoffs)
            self.assertEqual(node, KPV(name='Flap At Liftoff', items=[
                KeyPointValue(index=index, value=value, name='Flap At Liftoff'),
            ]))


class TestFlapAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FlapAtTouchdown
        self.operational_combinations = [('Flap', 'Touchdown')]
        self.interpolate = False

    def test_derive(self):
        flap = P(
            name='Flap',
            array=np.ma.repeat([0, 1, 5, 15, 20, 25, 30], 5),
        )
        for index, value in (29.25, 25), (29.75, 30), (30.00, 30), (30.25, 30):
            touchdowns = KTI(name='Touchdown', items=[
                KeyTimeInstance(index=index, name='Touchdown'),
            ])
            node = self.node_class()
            node.derive(flap, touchdowns)
            self.assertEqual(node, KPV(name='Flap At Touchdown', items=[
                KeyPointValue(index=index, value=value, name='Flap At Touchdown'),
            ]))


class TestFlapAtGearDownSelection(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = FlapAtGearDownSelection
        self.operational_combinations = [('Flap', 'Gear Down Selection')]
        self.interpolate = False

    def test_derive(self):
        flap = P(
            name='Flap',
            array=np.ma.repeat([0, 1, 5, 15, 20, 25, 30], 5),
        )
        flap.array[29] = np.ma.masked
        gear = KTI(name='Gear Down Selection', items=[
            KeyTimeInstance(index=19.25, name='Gear Down Selection'),
            KeyTimeInstance(index=19.75, name='Gear Down Selection'),
            KeyTimeInstance(index=20.00, name='Gear Down Selection'),
            KeyTimeInstance(index=20.25, name='Gear Down Selection'),
            KeyTimeInstance(index=29.25, name='Gear Down Selection'),
            KeyTimeInstance(index=29.75, name='Gear Down Selection'),
            KeyTimeInstance(index=30.00, name='Gear Down Selection'),
            KeyTimeInstance(index=30.25, name='Gear Down Selection'),
        ])
        node = self.node_class()
        node.derive(flap, gear)
        self.assertEqual(node, KPV(name='Flap At Gear Down Selection', items=[
            KeyPointValue(index=19.25, value=15, name='Flap At Gear Down Selection'),
            KeyPointValue(index=19.75, value=20, name='Flap At Gear Down Selection'),
            KeyPointValue(index=20.00, value=20, name='Flap At Gear Down Selection'),
            KeyPointValue(index=20.25, value=20, name='Flap At Gear Down Selection'),
            # Note: Index 29 is masked so we get a value of 30, not 25!
            KeyPointValue(index=29.25, value=30, name='Flap At Gear Down Selection'),
            KeyPointValue(index=29.75, value=30, name='Flap At Gear Down Selection'),
            KeyPointValue(index=30.00, value=30, name='Flap At Gear Down Selection'),
            KeyPointValue(index=30.25, value=30, name='Flap At Gear Down Selection'),
        ]))


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
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Touchdown', 'Landing', 'Altitude Radio')]

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


class TestFuelJettisonDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import FuelJettisonDuration

        self.param_name = 'Jettison Nozzle'
        self.phase_name = 'Airborne'
        self.node_class = FuelJettisonDuration
        self.values_mapping = {0: '-', 1: 'Jettison'}

        self.basic_setup()


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
        self.operational_combinations = [('Pitch', 'Altitude AAL', 'Initial Climb')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = PitchTakeoffMax
        self.operational_combinations = [('Pitch', 'Takeoff')]
        self.function = max_value

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch35To400FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35To400FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    def test_derive_basic(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]),
        )
        alt_aal = P(
            name='Altitude AAL For Flight Phases',
            array=np.ma.array([100, 101, 102, 103, 700, 105, 104, 103, 102]),
        )
        climb = buildsection('Combined Climb', 0, 4)
        node = Pitch35To400FtMax()
        node.derive(pitch, alt_aal, climb)
        self.assertEqual(node, KPV('Pitch 35 To 400 Ft Max', items=[
            KeyPointValue(name='Pitch 35 To 400 Ft Max', index=3, value=7),
        ]))


class TestPitch35To400FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch35To400FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestPitch400To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch400To1000FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch400To1000FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch400To1000FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch1000To500FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch1000To500FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = min_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To50FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch500To50FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)


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


class TestPitchAtVNAVModeAndEngThrustModeRequired(unittest.TestCase, CreateKPVsAtKTIsTest):

    def setUp(self):
        self.node_class = PitchAtVNAVModeAndEngThrustModeRequired
        self.operational_combinations = [('Pitch', 'VNAV Mode And Eng Thrust Mode Required')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


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


class TestRateOfClimb35To1000FtMin(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimb35To1000FtMin.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Initial Climb')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 500, 25), np.ma.arange(500, 1000, 100), [1050, 950, 990], [1100]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([25]*19, [43, 62, 81, 100, 112, 62, 47, 50, 12, 37, 27, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        climb = buildsection('Initial Climb', 1.4, 28)

        node = RateOfClimb35To1000FtMin()
        node.derive(vert_spd, climb)

        expected = KPV('Rate Of Climb 35 To 1000 Ft Min', items=[
            KeyPointValue(name='Rate Of Climb 35 To 1000 Ft Min', index=27, value=12),
        ])
        self.assertEqual(node, expected)


class TestRateOfClimbBelow10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfClimbBelow10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 5000, 250), np.ma.arange(5000, 10000, 1000), [10500, 9500, 9900], [11000]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude STD Smoothed', array)
        roc_array = np.ma.concatenate(([250]*19, [437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, 1-roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        node = RateOfClimbBelow10000FtMax()
        node.derive(vert_spd, alt)

        expected = KPV('Rate Of Climb Below 10000 Ft Max', items=[
            KeyPointValue(name='Rate Of Climb Below 10000 Ft Max', index=23, value=1125),
            KeyPointValue(name='Rate Of Climb Below 10000 Ft Max', index=26, value=500),
        ])
        self.assertEqual(node, expected)


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


class TestRateOfDescentTopOfDescentTo10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescentTopOfDescentTo10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Combined Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescentBelow10000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescentBelow10000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Combined Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent10000To5000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent10000To5000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude STD Smoothed', 'Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent5000To3000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent5000To3000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Descent')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestRateOfDescent3000To2000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent3000To2000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Initial Approach')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 1500, 100), np.ma.arange(1500, 3000, 200), [3050, 2850, 2990], [3150]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([100]*14, [125, 150, 175, 200, 200, 200, 200, 187, 87, 72, 62, 25, 75, 40, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Initial Approach', 38, 62)

        node = RateOfDescent3000To2000FtMax()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 3000 To 2000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent 3000 To 2000 Ft Max', index=41, value=-200),
        ])
        self.assertEqual(node, expected)



class TestRateOfDescent2000To1000FtMax(unittest.TestCase):


    def test_can_operate(self):
        opts = RateOfDescent2000To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Initial Approach')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 500, 25), np.ma.arange(500, 2000, 100), [2050, 1850, 1990], [2150]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([25]*19, [43, 62, 81, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 112, 37, 47, 62, 25, 75, 40, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Initial Approach', 48, 60)

        node = RateOfDescent2000To1000FtMax()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 2000 To 1000 Ft Max', items=[
            KeyPointValue(name='Rate Of Descent 2000 To 1000 Ft Max', index=49, value=-62),
            KeyPointValue(name='Rate Of Descent 2000 To 1000 Ft Max', index=52, value=-112),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent1000To500FtMax(unittest.TestCase):


    def test_can_operate(self):
        opts = RateOfDescent1000To500FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 500, 25), np.ma.arange(500, 1000, 100), [1050, 950, 990], [1090]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([25]*19, [43, 62, 81, 100, 112, 62, 47, 47, 10, 35, 25, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Final Approach', 37, 63)

        node = RateOfDescent1000To500FtMax()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 1000 To 500 Ft Max', items=[
            KeyPointValue(index=39.0, value=-47.0, name='Rate Of Descent 1000 To 500 Ft Max'),
            KeyPointValue(index=42.0, value=-112.0, name='Rate Of Descent 1000 To 500 Ft Max')
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent500To50FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent500To50FtMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 50, 25), np.ma.arange(50, 500, 100), [550, 450, 540], [590]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([25]*2, [62, 81, 100, 100, 50, 47, 35, 10, 35, 12, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed', roc_array)

        descents = buildsection('Final Approach', 19, 27)

        node = RateOfDescent500To50FtMax()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 500 To 50 Ft Max', items=[
            KeyPointValue(index=21.0, value=-35.0, name='Rate Of Descent 500 To 50 Ft Max'),
            KeyPointValue(index=24.0, value=-100.0, name='Rate Of Descent 500 To 50 Ft Max')
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent50FtToTouchdownMax(unittest.TestCase):

    def test_can_operate(self):
        opts = RateOfDescent50FtToTouchdownMax.get_operational_combinations()
        self.assertEqual(opts, [('Vertical Speed Inertial', 'Altitude AAL For Flight Phases', 'Touchdown')])

    def test_derive(self):
        array = np.ma.concatenate((np.ma.arange(0, 50, 5), [55, 45, 54], [59]*5))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate(([5]*8, [6, 2, 3, 3, 1, 3, 1, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        roc_array[33] = -26
        vert_spd = P('Vertical Speed Inertial', roc_array)

        touch_down = KTI('Touchdown', items=[KeyTimeInstance(34, 'Touchdown')])

        node = RateOfDescent50FtToTouchdownMax()
        node.derive(vert_spd, alt, touch_down)

        expected = KPV('Rate Of Descent 50 Ft To Touchdown Max', items=[
            KeyPointValue(name='Rate Of Descent 50 Ft To Touchdown Max', index=33, value=-26),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescentAtTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RateOfDescentAtTouchdown
        self.operational_combinations = [('Vertical Speed Inertial', 'Touchdown')]

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


class TestRoll400To1000FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Roll400To1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Initial Climb')]
        self.function = max_abs_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

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


class TestRoll1000To300FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Roll1000To300FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases', 'Final Approach')]
        self.function = max_abs_value

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, self.operational_combinations)

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


class TestRudderPedalForceMax(unittest.TestCase, NodeTest):
    def setUp(self):
        from analysis_engine.key_point_values import RudderPedalForceMax

        self.node_class = RudderPedalForceMax
        self.operational_combinations = [('Rudder Pedal Force', 'Fast')]

    def test_derive(self):
        ccf = P(
            name='Rudder Pedal Force',
            array=np.ma.array(data=range(50, 30, -1), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 9)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Rudder Pedal Force Max',
                items=[KeyPointValue(
                    index=3.0, value=47.0,
                    name='Rudder Pedal Force Max')]))

    def test_big_left_boot(self):
        ccf = P(
            name='Rudder Pedal Force',
            array=np.ma.array(data=range(30, -50, -5), dtype=float),
        )
        phase_fast = buildsection('Fast', 3, 13)
        node = self.node_class()
        node.derive(ccf, phase_fast)
        self.assertEqual(
            node,
            KPV('Rudder Pedal Force Max',
                items=[KeyPointValue(
                    index=12.0, value=-30.0,
                    name='Rudder Pedal Force Max')]))


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
        flap = M('Flap', array=np.ma.array([0] * 10 + range(1, 21)),
                 values_mapping={f: str(f) for f in range(0, 21)})
        airborne = buildsection('Airborne', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, flap, airborne)
        self.assertEqual(node, [
            KeyPointValue(14, 2.0, 'Speedbrake Deployed With Flap Duration')])


class TestSpeedbrakeDeployedWithPowerOnDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedbrakeDeployedWithPowerOnDuration
        self.operational_combinations = [('Speedbrake Selected', 'Eng (*) N1 Avg', 'Airborne')]

    def test_derive_basic(self):
        spd_brk_loop = [0] * 4 + [1] * 2 + [0] * 4
        values_mapping = {0: 'Undeployed/Cmd Down', 1: 'Deployed/Cmd Up'}
        spd_brk = M(
            'Speedbrake Selected', values_mapping=values_mapping,
            array=np.ma.array(spd_brk_loop * 3))
        power = P('Eng (*) N1 Avg',
                 array=np.ma.array([40] * 10 + [60] * 10 + [50] * 10))
        airborne = buildsection('Airborne', 10, 20)
        node = self.node_class()
        node.derive(spd_brk, power, airborne)
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


class TestStickPusherActivatedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            StickPusherActivatedDuration

        self.param_name = 'Stick Pusher'
        self.phase_name = 'Airborne'
        self.node_class = StickPusherActivatedDuration
        self.values_mapping = {0: '-', 1: 'Push'}

        self.basic_setup()


class TestStickShakerActivatedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            StickShakerActivatedDuration

        self.param_name = 'Stick Shaker'
        self.phase_name = 'Airborne'
        self.node_class = StickShakerActivatedDuration
        self.values_mapping = {0: '-', 1: 'Shake'}

        self.basic_setup()


class TestOverspeedDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import OverspeedDuration

        self.param_name = 'Overspeed Warning'
        self.phase_name = None
        self.node_class = OverspeedDuration
        self.values_mapping = {0: '-', 1: 'Overspeed'}

        self.basic_setup()


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
# Warnings: Landing Configuration Warning


class TestLandingConfigurationGearWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        opts = LandingConfigurationGearWarningDuration.get_operational_combinations()
        self.assertEqual(opts, [('Landing Configuration Gear Warning', 'Airborne',)])
        
    def test_derive(self):
        node = LandingConfigurationGearWarningDuration()
        airs = buildsection('Airborne', 2, 8)
        warn = M(array=np.ma.array([0,0,0,0,0,1,1,0,0,0]),
                             values_mapping={1: 'Warning'})
        node.derive(warn, airs)
        self.assertEqual(node[0].index, 5)


class TestLandingConfigurationSpeedbrakeCautionDuration(unittest.TestCase,
                                                        CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            LandingConfigurationSpeedbrakeCautionDuration

        self.param_name = 'Landing Configuration Speedbrake Caution'
        self.phase_name = 'Airborne'
        self.node_class = LandingConfigurationSpeedbrakeCautionDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


##############################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TestTAWSAlertDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TAWSAlertDuration
        self.operational_combinations = [('TAWS Alert', 'Airborne')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import TAWSWarningDuration

        self.param_name = 'TAWS Warning'
        self.phase_name = 'Airborne'
        self.node_class = TAWSWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


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


class TestTAWSDontSinkWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSDontSinkWarningDuration

        self.param_name = 'TAWS Dont Sink'
        self.phase_name = 'Airborne'
        self.node_class = TAWSDontSinkWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSCautionObstacleDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSCautionObstacleDuration

        self.param_name = 'TAWS Caution Obstacle'
        self.phase_name = 'Airborne'
        self.node_class = TAWSCautionObstacleDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSCautionTerrainDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import TAWSCautionTerrainDuration

        self.param_name = 'TAWS Caution Terrain'
        self.phase_name = 'Airborne'
        self.node_class = TAWSCautionTerrainDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSTerrainCautionDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import TAWSTerrainCautionDuration

        self.param_name = 'TAWS Terrain Caution'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainCautionDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.basic_setup()


class TestTAWSFailureDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import TAWSFailureDuration

        self.param_name = 'TAWS Failure'
        self.phase_name = 'Airborne'
        self.node_class = TAWSFailureDuration
        self.values_mapping = {0: '-', 1: 'Failed'}

        self.basic_setup()


class TestTAWSObstacleWarningDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSObstacleWarningDuration

        self.param_name = 'TAWS Obstacle Warning'
        self.phase_name = 'Airborne'
        self.node_class = TAWSObstacleWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSPredictiveWindshearDuration(unittest.TestCase,
                                          CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSPredictiveWindshearDuration

        self.param_name = 'TAWS Predictive Windshear'
        self.phase_name = 'Airborne'
        self.node_class = TAWSPredictiveWindshearDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSTerrainAheadDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSTerrainAheadDuration

        self.param_name = 'TAWS Terrain Ahead'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainAheadDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSTerrainAheadPullUpDuration(unittest.TestCase,
                                         CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSTerrainAheadPullUpDuration

        self.param_name = 'TAWS Terrain Pull Up Ahead'
        self.phase_name = 'Airborne'
        self.node_class = TAWSTerrainAheadPullUpDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTAWSWindshearCautionBelow1500FtDuration(unittest.TestCase,
                                                  CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSWindshearCautionBelow1500FtDuration

        self.param_name = 'TAWS Windshear Caution'
        self.phase_name = None
        self.node_class = TAWSWindshearCautionBelow1500FtDuration
        self.values_mapping = {0: '-', 1: 'Caution'}

        self.additional_params = [
            P(
                'Altitude AAL For Flight Phases',
                array=np.ma.array([
                    1501, 1502, 1501, 1499, 1498, 1499, 1499, 1499, 1499, 1501,
                    1502, 1501
                ]),
            )
        ]

        self.basic_setup()


class TestTAWSWindshearSirenBelow1500FtDuration(unittest.TestCase,
                                                CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TAWSWindshearSirenBelow1500FtDuration

        self.param_name = 'TAWS Windshear Siren'
        self.phase_name = None
        self.node_class = TAWSWindshearSirenBelow1500FtDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.additional_params = [
            P(
                'Altitude AAL For Flight Phases',
                array=np.ma.array([
                    1501, 1502, 1501, 1499, 1498, 1499, 1499, 1499, 1499, 1501,
                    1502, 1501
                ]),
            )
        ]

        self.basic_setup()


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


class TestTCASFailureDuration(unittest.TestCase, CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import TCASFailureDuration

        self.param_name = 'TCAS Failure'
        self.phase_name = 'Airborne'
        self.node_class = TCASFailureDuration
        self.values_mapping = {0: '-', 1: 'Failed'}

        self.basic_setup()


##############################################################################
# Warnings: Takeoff Configuration


class TestTakeoffConfigurationWarningDuration(unittest.TestCase,
                                              CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TakeoffConfigurationWarningDuration

        self.param_name = 'Takeoff Configuration Warning'
        self.phase_name = 'Takeoff Roll'
        self.node_class = TakeoffConfigurationWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTakeoffConfigurationFlapWarningDuration(unittest.TestCase,
                                                  CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TakeoffConfigurationFlapWarningDuration

        self.param_name = 'Takeoff Configuration Flap Warning'
        self.phase_name = 'Takeoff Roll'
        self.node_class = TakeoffConfigurationFlapWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTakeoffConfigurationParkingBrakeWarningDuration(unittest.TestCase,
                                                          CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TakeoffConfigurationParkingBrakeWarningDuration

        self.param_name = 'Takeoff Configuration Parking Brake Warning'
        self.phase_name = 'Takeoff Roll'
        self.node_class = TakeoffConfigurationParkingBrakeWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTakeoffConfigurationSpoilerWarningDuration(unittest.TestCase,
                                                     CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TakeoffConfigurationSpoilerWarningDuration

        self.param_name = 'Takeoff Configuration Spoiler Warning'
        self.phase_name = 'Takeoff Roll'
        self.node_class = TakeoffConfigurationSpoilerWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


class TestTakeoffConfigurationStabilizerWarningDuration(unittest.TestCase,
                                                        CreateKPVsWhereTest):
    def setUp(self):
        from analysis_engine.key_point_values import \
            TakeoffConfigurationStabilizerWarningDuration

        self.param_name = 'Takeoff Configuration Stabilizer Warning'
        self.phase_name = 'Takeoff Roll'
        self.node_class = TakeoffConfigurationStabilizerWarningDuration
        self.values_mapping = {0: '-', 1: 'Warning'}

        self.basic_setup()


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
        self.operational_combinations = [('Airspeed', 'Elevator', 'Touchdown', 'Landing')]

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


class TestGrossWeightAtLiftoff(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight Smoothed', 'Liftoff')]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test Not Implemented')


class TestGrossWeightAtTouchdown(unittest.TestCase, NodeTest):

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
            ('Flap Lever', 'Takeoff Roll'),
            ('Flap', 'Takeoff Roll'),
            ('Flap Lever', 'Flap', 'Takeoff Roll'),
        ]

    def test_derive(self):
        flap_array = np.ma.array([15, 15, 20, 20, 15, 15])
        flap_lever = M(
            name='Flap Lever', array=flap_array,
            values_mapping={f: str(f) for f in np.ma.unique(flap_array)},
        )
        takeoff_roll = S(items=[Section('Takeoff Roll', slice(0, 5), 0, 5)])
        node = self.node_class()
        node.derive(flap_lever, None, takeoff_roll)
        expected = [
            KeyPointValue(
                index=3.5, value=1.5,
                name='Last Flap Change To Takeoff Roll End Duration')
        ]
        self.assertEqual(list(node), expected)
        flap = M(
            name='Flap',
            array=np.ma.array([15, 15, 20, 15, 15, 15]),
            values_mapping={f: str(f) for f in np.ma.unique(flap_array)},
        )
        node = self.node_class()
        node.derive(None, flap, takeoff_roll)
        expected = [
            KeyPointValue(
                index=2.5, value=2.5,
                name='Last Flap Change To Takeoff Roll End Duration')
        ]
        self.assertEqual(list(node), expected)



class TestPitchAlternateLawDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAlternateLawDuration
        self.operational_combinations = [('Pitch Alternate Law',)]

    def test_derive(self):
        alt_law = M(name='Pitch Alternate Law',
                    array=np.ma.array([0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0],dtype=int),
                    values_mapping={1:'Alternate'})
        node = self.node_class()
        node.derive(alt_law)

        expected = [
            KeyPointValue(
                index=4, value=3,
                name='Pitch Alternate Law Duration'),
            KeyPointValue(
                index=10, value=6,
                name='Pitch Alternate Law Duration')
        ]
        self.assertEqual(node, expected)
        


class TestPitchDirectLawDuration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchDirectLawDuration
        self.operational_combinations = [('Pitch Direct Law',)]

    def test_derive(self):
        dir_law = M(name='Pitch Direct Law',
                    array=np.ma.array([0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0],dtype=int),
                    values_mapping={1:'Direct'})
        node = self.node_class()
        node.derive(dir_law)

        expected = [
            KeyPointValue(
                index=4, value=3,
                name='Pitch Direct Law Duration'),
            KeyPointValue(
                index=10, value=6,
                name='Pitch Direct Law Duration')
        ]
        self.assertEqual(node, expected)
