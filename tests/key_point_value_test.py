import os
import numpy as np
import sys
import unittest

from mock import Mock, patch

from analysis_engine.derived_parameters import Flap
from analysis_engine.library import align
from analysis_engine.node import (KeyTimeInstance, KTI, KeyPointValue, 
                                  Parameter, P, Section)

from analysis_engine.key_point_values import (
    AccelerationLateralAtTouchdown,
    AccelerationLateralMax,
    AccelerationLateralTaxiingStraightMax,
    AccelerationLateralTaxiingTurnsMax,
    AccelerationLongitudinalPeakTakeoff,
    AccelerationNormal20FtToFlareMax,
    AccelerationNormalAirborneFlapsDownMax,
    AccelerationNormalAirborneFlapsDownMin,
    AccelerationNormalAirborneFlapsUpMax,
    AccelerationNormalAirborneFlapsUpMin,
    AccelerationNormalAtLiftoff,
    AccelerationNormalAtTouchdown,
    AccelerationNormalLiftoffTo35FtMax,
    AccelerationNormalMax,
    AccelerationNormalOffset,
    Airspeed10000ToLandMax,
    Airspeed1000To500FtMax,
    Airspeed1000To500FtMin,
    Airspeed2000To30FtMin,
    Airspeed35To1000FtMax,
    Airspeed35To1000FtMin,
    Airspeed400To1500FtMin,
    Airspeed500To20FtMax,
    Airspeed500To20FtMin,
    AirspeedAsGearExtendingMax,
    AirspeedAsGearRetractingMax,
    AirspeedAt35FtInTakeoff,
    AirspeedAtGearDownSelection,
    AirspeedAtGearUpSelection,
    AirspeedAtLiftoff,
    AirspeedAtTouchdown,
    AirspeedBelowAltitudeMax,
    AirspeedBetween90SecToTouchdownAndTouchdownMax,
    AirspeedBetween1000And3000FtMax,
    AirspeedBetween3000And5000FtMax,
    AirspeedBetween5000And8000FtMax,
    AirspeedBetween8000And10000FtMax,
    AirspeedCruiseMax,
    AirspeedCruiseMin,
    AirspeedLevelFlightMax,
    AirspeedMax,
    AirspeedMinusV235To1000FtMax,
    AirspeedMinusV235To1000FtMin,
    AirspeedMinusV2AtLiftoff,
    AirspeedRelativeAtTouchdown,
    AirspeedRelative1000To500FtMax,
    AirspeedRelative1000To500FtMin,
    AirspeedRelative20FtToTouchdownMax,
    AirspeedRelative20FtToTouchdownMin,
    AirspeedRelative500To20FtMax,
    AirspeedRelative500To20FtMin,        
    AirspeedRelativeFor3Sec1000To500FtMax,
    AirspeedRelativeFor3Sec1000To500FtMin,
    AirspeedRelativeFor3Sec20FtToTouchdownMax,
    AirspeedRelativeFor3Sec20FtToTouchdownMin,
    AirspeedRelativeFor3Sec500To20FtMax,
    AirspeedRelativeFor3Sec500To20FtMin,    
    AirspeedRelativeFor5Sec1000To500FtMax,
    AirspeedRelativeFor5Sec1000To500FtMin,
    AirspeedRelativeFor5Sec20FtToTouchdownMax,
    AirspeedRelativeFor5Sec20FtToTouchdownMin,
    AirspeedRelativeFor5Sec500To20FtMax,
    AirspeedRelativeFor5Sec500To20FtMin,
    AirspeedRelativeWithFlapDescentMin,
    AirspeedRTOMax,
    AirspeedThrustReversersDeployedMin,
    AirspeedTODTo10000Max,
    AirspeedTrueAtTouchdown,
    AirspeedVacatingRunway,
    AirspeedWithFlapMax,
    AirspeedWithFlapMin,
    AirspeedWithFlapClimbMax,
    AirspeedWithFlapClimbMin,
    AirspeedWithFlapDescentMax,
    AirspeedWithFlapDescentMin,
    AirspeedWithGearDownMax,
    AltitudeAtFirstFlapChangeAfterLiftoff,
    AltitudeAtGoAroundMin,
    AltitudeAtLastFlapChangeBeforeLanding,
    AltitudeAtLiftoff,
    AltitudeAtMachMax,
    AltitudeAtSuspectedLevelBust,
    AltitudeAtTouchdown,
    AltitudeAutopilotDisengaged,
    AltitudeAutopilotEngaged,
    AltitudeAutothrottleDisengaged,
    AltitudeAutothrottleEngaged,    
    AltitudeFlapExtensionMax,
    AltitudeGoAroundFlapRetracted,
    AltitudeGoAroundGearRetracted,
    AltitudeMax,
    ControlColumnStiffness,
    EngEPR500FtToTouchdownMin,
    EngN1500To20FtMin,
    EngN1TakeoffMax,
    EngOilTempMax,
    EngOilTemp15MinuteMax,
    EngVibN1Max,
    EngVibN2Max,
    Eng_N1MaxDurationUnder60PercentAfterTouchdown,
    FlapAtLiftoff,
    FlapAtTouchdown,
    FuelQtyAtLiftoff,
    FuelQtyAtTouchdown,
    GrossWeightAtLiftoff,
    GrossWeightAtTouchdown,
    HeadingAtLanding,
    HeadingAtTakeoff,
    ILSFrequencyOnApproach,
    ILSGlideslopeDeviation1500To1000FtMax,
    ILSGlideslopeDeviation1000To250FtMax,
    ILSLocalizerDeviation1500To1000FtMax,
    ILSLocalizerDeviation1000To250FtMax,    
    LatitudeAtLanding,
    LatitudeAtLiftoff,
    LatitudeAtTakeoff,
    LatitudeAtTouchdown,
    LongitudeAtLanding,
    LongitudeAtLiftoff,
    LongitudeAtTakeoff,
    LongitudeAtTouchdown,
    MachMax,
    Pitch35To400FtMax,
    Pitch35To400FtMin,
    PitchAtLiftoff,
    PitchAtTouchdown,
    RateOfDescent10000To5000FtMax,
    RateOfDescent5000To3000FtMax,
    RateOfDescent3000To2000FtMax,
    RateOfDescent2000To1000FtMax,
    RateOfDescent1000To500FtMax,
    RateOfDescent500To20FtMax,
    RateOfDescent500FtToTouchdownMax,
    RollAbove1000FtMax,
    TailClearanceOnApproach,
    ZeroFuelWeight,
)
from analysis_engine.library import (max_abs_value, max_value, min_value)
from analysis_engine.flight_phase import Fast
from flight_phase_test import buildsection

debug = sys.gettrace() is not None


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


class NodeTest(object):
    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         self.operational_combinations)


class CreateKPVsAtKPVsTest(NodeTest):
    '''
    Example of subclass inheriting tests:
    
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
    Example of subclass inheriting tests:
    
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
    Example of subclass inheriting tests:
    
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
    Example of subclass inheriting tests:
    
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



"""
class TestAccelerationLateralTaxiingMax(unittest.TestCase,
                                         CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationLateralTaxiingMax
        self.operational_combinations = [('Acceleration Lateral', 'Grounded')]
        self.function = max_value
        """


"""
class TestAccelerationNormalAirborneMax(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationNormalAirborneMax
        self.operational_combinations = [('Acceleration Normal', 'Airborne')]
        self.function = max_value
        """


"""
class TestAccelerationNormalAirborneMin(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationNormalAirborneMin
        self.operational_combinations = [('Acceleration Normal', 'Airborne')]
        self.function = min_value
        """

"""
class TestAccelerationNormalDuringTakeoffMax(unittest.TestCase,
                                             CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationNormalDuringTakeoffMax
        self.operational_combinations = [('Acceleration Normal', 'Takeoff')]
        self.function = max_value
        """


class TestAccelerationNormalMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(AccelerationNormalMax.get_operational_combinations(),
                         [('Acceleration Normal Offset Removed',),
                          ('Acceleration Normal Offset Removed', 'Mobile'),])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        acc_norm_max = AccelerationNormalMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        acc_norm_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(acc_norm_max,
                         [KeyPointValue(index=index, value=value,
                                        name=acc_norm_max.name)])


class TestAccelerationNormal20FtToFlareMax(unittest.TestCase,
                                            CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationNormal20FtToFlareMax
        self.operational_combinations = [('Acceleration Normal Offset Removed',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 5,), {})]
    
    def test_derive(self):
        '''
        Depends upon DerivedParameterNode.slices_from_to and library.max_value.
        '''
        # Test height range limit
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(48, 0, -3))
        acceleration_normal = \
            P('Acceleration Normal',
              np.ma.array(range(10,18) + range(18, 10, -1)) / 10.0)
        node = AccelerationNormal20FtToFlareMax()
        node.derive(acceleration_normal, alt_aal)
        self.assertEqual(node,
                [KeyPointValue(index=10, value=1.6,
                               name='Acceleration Normal 20 Ft To Flare Max')])
        # Test peak acceleration
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(32, 0, -2))
        node = AccelerationNormal20FtToFlareMax()
        node.derive(acceleration_normal, alt_aal)
        self.assertEqual(node,
                [KeyPointValue(index=8, value=1.8,
                               name='Acceleration Normal 20 Ft To Flare Max')])


class TestAirspeed1000To500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed1000To500FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500,), {})]
        
    def test_airspeed_1000_500_basic(self):
        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = Parameter('Airspeed', np.ma.array(testwave))
        alt_ph = Parameter('Altitude AAL For Flight Phases', 
                           np.ma.array(testwave) * 10)
        kpv = Airspeed1000To500FtMax()
        kpv.derive(spd, alt_ph)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertEqual(kpv[0].value, 91.250101656055278)
        self.assertEqual(kpv[1].index, 110)
        self.assertEqual(kpv[1].value, 99.557430201194919)


class TestAirspeed2000To30FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed2000To30FtMin
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (2000, 30,), {})]


class TestAirspeed400To1500FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed400To1500FtMin
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (400, 1500,), {})]


"""
class TestAirspeed50To1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed50To1000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (50, 1000,), {})]
"""


"""
class TestAirspeed500To50FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed50To1000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50,), {})]
"""


class TestAirspeedAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedAtTouchdown
        self.operational_combinations = [('Airspeed', 'Touchdown')]


class TestAirspeedBelowAltitudeMax(unittest.TestCase):
    def test_derive(self):
        airspeed = P(array=np.ma.arange(20))
        alt_aal = P(array=np.ma.arange(0, 10000, 500))
        param = AirspeedBelowAltitudeMax()
        param.derive(airspeed, alt_aal)
        self.assertEqual(param,
            [KeyPointValue(index=19, value=19.0, 
                           name='Airspeed Below 10000 Ft Max', 
                           slice=slice(None, None, None), datetime=None), 
             KeyPointValue(index=15, value=15.0, 
                           name='Airspeed Below 8000 Ft Max', 
                           slice=slice(None, None, None), datetime=None), 
             KeyPointValue(index=13, value=13.0, 
                           name='Airspeed Below 7000 Ft Max', 
                           slice=slice(None, None, None), datetime=None), 
             KeyPointValue(index=9, value=9.0, 
                           name='Airspeed Below 5000 Ft Max', 
                           slice=slice(None, None, None), datetime=None), 
             KeyPointValue(index=5, value=5.0, 
                           name='Airspeed Below 3000 Ft Max', 
                           slice=slice(None, None, None), datetime=None)])


class TestAirspeedMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedMax
        self.operational_combinations = [('Airspeed', 'Airborne')]
        self.function = max_value
        
    def test_airspeed_max_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-100))+100
        spd = Parameter('Airspeed', np.ma.array(testwave))
        waves=np.ma.clump_unmasked(np.ma.masked_less(testwave,80))
        airs=[]
        for wave in waves:
            airs.append(Section('Airborne',wave, wave.start, wave.stop))
        ##from analysis_engine.node import FlightPhaseNode
        ##wave_phases = FlightPhaseNode(items=airs)
        
        kpv = AirspeedMax()
        kpv.derive(spd, airs)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 31)
        self.assertGreater(kpv[0].value, 199.9)
        self.assertLess(kpv[0].value, 200)
        self.assertEqual(kpv[1].index, 94)
        self.assertGreater(kpv[1].value, 199.9)
        self.assertLess(kpv[1].value, 200)


"""
class TestAirspeedMinusV235To400FtMin(unittest.TestCase,
                                      CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedMinusV235To400FtMin
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 400,), {})]
        """


"""
class TestAirspeedMinusV2400To1500FtMin(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedMinusV2400To1500FtMin
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (400, 1500), {})]
        """
        

class TestAirspeedMinusV2AtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedMinusV2AtLiftoff
        self.operational_combinations = [('Airspeed Minus V2', 'Liftoff')]


####class TestAirspeedRelative500FtToTouchdownMax(unittest.TestCase,
####                                               CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = AirspeedRelative500FtToTouchdownMax
####        self.operational_combinations = [('Airspeed Relative',
####                                          'Altitude AAL For Flight Phases')]
####        self.function = max_value
####        self.second_param_method_calls = [('slices_from_to', (500, 0,), {})]


class TestAirspeedRelativeAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedRelativeAtTouchdown
        self.operational_combinations = [('Airspeed Relative', 'Touchdown')]

    
class TestAirspeedWithFlapMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(\
            AirspeedWithFlapMax.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Fast')])
        
    def test_airspeed_with_flaps(self):
        spd = P('Airspeed', np.ma.array(range(30)))
        flap = P('Flap', np.ma.array([0]*10 + [5]*10 + [10]*10))
        fast = buildsection('Fast',0,30)
        flap.array[19] = np.ma.masked # mask the max val
        spd_flap = AirspeedWithFlapMax()
        spd_flap.derive(flap, spd, fast)
        self.assertEqual(len(spd_flap), 2)
        self.assertEqual(spd_flap[0].name, 'Airspeed With Flap 5 Max')
        self.assertEqual(spd_flap[0].index, 18) # 19 was masked
        self.assertEqual(spd_flap[0].value, 18)
        self.assertEqual(spd_flap[1].name, 'Airspeed With Flap 10 Max')
        self.assertEqual(spd_flap[1].index, 29)
        self.assertEqual(spd_flap[1].value, 29)

    def test_derive_alternative_method(self):
        # This test will produce a warning "No flap settings - rounding to nearest 5"
        airspeed = P('Airspeed', np.ma.arange(20))
        flap = P('Flap',
                 np.ma.masked_array([0] * 2 + [1] * 2 + [2] * 2 + [5] * 2 +
                                    [10] * 2 +  [15] * 2 + [25] * 2 +
                                    [30] * 2 + [40] * 2 + [0] * 2))
        fast = buildsection('Fast',0,20)
        step = Flap()
        step.derive(flap)
        
        airspeed_with_flap_max = AirspeedWithFlapMax()
        airspeed_with_flap_max.derive(step, airspeed, fast)
        self.assertEqual(airspeed_with_flap_max,
          [KeyPointValue(index=7, value=7, name='Airspeed With Flap 5 Max'),
           KeyPointValue(index=9, value=9, name='Airspeed With Flap 10 Max'),
           KeyPointValue(index=11, value=11, name='Airspeed With Flap 15 Max'),
           KeyPointValue(index=13, value=13, name='Airspeed With Flap 25 Max'),
           KeyPointValue(index=15, value=15, name='Airspeed With Flap 30 Max'),
           KeyPointValue(index=17, value=17, name='Airspeed With Flap 40 Max')])


class TestAltitudeAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAtTouchdown
        self.operational_combinations = [('Altitude STD', 'Touchdown')]


class TestAltitudeAtMachMax(unittest.TestCase, CreateKPVsAtKPVsTest):
    def setUp(self):
        self.node_class = AltitudeAtMachMax
        self.operational_combinations = [('Altitude STD', 'Mach Max')]


class TestAltitudeMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AltitudeMax
        self.operational_combinations = [('Altitude STD', 'Airborne')]
        self.function = max_value

class TestAltitudeAtSuspectedLevelBust(unittest.TestCase):
    def test_handling_no_data(self):
        alt=Parameter('Altitude STD',np.ma.array([0,1000,1000,1000,1000]))
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[]
        self.assertEqual(kpv,expected)
        
    def test_up_down_and_up(self):
        testwave = np.ma.array(1.0+np.sin(np.arange(0,12.6,0.1)))*1000
        alt=Parameter('Altitude STD',testwave)
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[KeyPointValue(index=16, value=999.5736030415051, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None), 
                  KeyPointValue(index=47, value=-1998.4666029387058, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None), 
                  KeyPointValue(index=79, value=1994.3775951461494, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None), 
                  KeyPointValue(index=110, value=-933.6683091995028, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None)]
        self.assertEqual(kpv,expected)
        
    def test_too_slow(self):
        testwave = np.ma.array(1.0+np.sin(np.arange(0,12.6,0.1)))*1000
        alt=Parameter('Altitude STD',testwave,0.02)
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[]
        self.assertEqual(kpv,expected)

    def test_straight_up_and_down(self):
        testwave = np.ma.array(range(0,10000,50)+range(10000,0,-50))
        alt=Parameter('Altitude STD',testwave,1)
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[]
        self.assertEqual(kpv,expected)
        
    def test_up_and_down_with_overshoot(self):
        testwave = np.ma.array(range(0,10000,50)+range(10000,9000,-50)+[9000]*200+range(9000,0,-50))
        alt=Parameter('Altitude STD',testwave,1)
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[KeyPointValue(index=200, value=1000, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None)] 
        self.assertEqual(kpv,expected)

    def test_up_and_down_with_undershoot(self):
        testwave = np.ma.array(range(0,10000,50)+
                               [10000]*200+
                               range(10000,9000,-50)+
                               range(9000,20000,50)+
                               range(20000,0,-50))
        alt=Parameter('Altitude STD',testwave,1)
        kpv=AltitudeAtSuspectedLevelBust()
        kpv.derive(alt)
        expected=[KeyPointValue(index=420, value=-1000, 
                                name='Altitude At Suspected Level Bust', 
                                slice=slice(None, None, None), datetime=None)]
        self.assertEqual(kpv,expected)
              
    
"""
class TestAPEngaged1AtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = APEngaged1AtLiftoff
        self.operational_combinations = [('AP Engaged 1', 'Liftoff')]


class TestAPEngaged1AtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = APEngaged1AtTouchdown
        self.operational_combinations = [('AP Engaged 1', 'Touchdown')]


class TestAPEngaged2AtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = APEngaged2AtLiftoff
        self.operational_combinations = [('AP Engaged 2', 'Liftoff')]


class TestAPEngaged2AtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = APEngaged2AtTouchdown
        self.operational_combinations = [('AP Engaged 2', 'Touchdown')]
        """


"""
class TestAltitudeRadioDividedByDistanceToLanding3000To50FtMinTerrain(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeRadioDividedByDistanceToLanding3000To50FtMin.get_operational_combinations(),
                         [('Altitude AAL', 'Altitude Radio',
                           'Distance To Landing')])
    
    def test_derive(self):
        test_data_dir = os.path.join('test_data', 'BDUTerrain')
        alt_aal_array = np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                                'alt_aal.npy')))
        alt_radio_array = np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                                  'alt_radio.npy')))
        dtl_array = np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                            'dtl.npy')))
        alt_aal = P(array=alt_aal_array, frequency=8)
        alt_radio = P(array=alt_radio_array, frequency=0.5)
        dtl = P(array=dtl_array, frequency=0.25)
        alt_radio.array = align(alt_radio, alt_aal)
        dtl.array = align(dtl, alt_aal)        
        param = AltitudeRadioDividedByDistanceToLanding3000To50FtMin()
        param.derive(alt_aal, alt_radio, dtl)
        self.assertEqual(param, [KeyPointValue(name='BDU Terrain', index=1008, value=0.037668517049960347)])
        """


class TestControlColumnStiffness(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ControlColumnStiffness.get_operational_combinations(),
                         [('Control Column Force','Control Column','Fast')])

    def test_control_column_stiffness_too_few_samples(self):
        cc_disp = Parameter('Control Column', 
                            np.ma.array([0,.3,1,2,2.5,1.4,0,0]))
        cc_force = Parameter('Control Column Force',
                             np.ma.array([0,2,4,7,8,5,2,1]))
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', np.ma.array([100]*10)))
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff, [])
        
    def test_control_column_stiffness_max(self):
        testwave = np.ma.array((1.0 - np.cos(np.arange(0,6.3,0.1)))/2.0)
        cc_disp = Parameter('Control Column', testwave * 10.0)
        cc_force = Parameter('Control Column Force', testwave * 27.0)
        phase_fast = buildsection('Fast',0,63)
        stiff = ControlColumnStiffness()
        stiff.derive(cc_force,cc_disp,phase_fast)
        self.assertEqual(stiff.get_first().index, 31) 
        self.assertAlmostEqual(stiff.get_first().value, 2.7) # lb/deg 
        

"""
class TestEngGasTempMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngGasTempMax.get_operational_combinations(),
                         [('Eng (*) Gas Temp Max',)])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_egt_max = EngGasTempMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_egt_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_egt_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_egt_max.name)])
                                        """


class TestEngEPR500FtToTouchdownMin(unittest.TestCase,
                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = EngEPR500FtToTouchdownMin
        self.operational_combinations = [('Eng (*) EPR Min',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 0,), {})]


class TestEngN1500To20FtMin(unittest.TestCase,
                                   CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = EngN1500To20FtMin
        self.operational_combinations = [('Eng (*) N1 Min',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20,), {})]


class EngGasTempTakeoffMax(unittest.TestCase):
    def setUp(self):
        self.node_class = EngGasTempTakeoffMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff')]
        self.function = max_value


"""
class TestEngN13000To20FtMax(unittest.TestCase,
                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = EngN13000To20FtMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (3000, 0,), {})]
"""

class TestEngN1TakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = EngN1TakeoffMax
        self.function = max_value
        self.operational_combinations = [('Eng (*) N1 Max',
                                          'Takeoff 5 Min Rating')]


"""
class TestEngN1Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngN1Max.get_operational_combinations(),
                         [('Eng (*) N1 Max',)])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_n1_max = EngN1Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_n1_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_n1_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_n1_max.name)])


class TestEngN2Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngN2Max.get_operational_combinations(),
                         [('Eng (*) N2 Max',)])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_n2_max = EngN2Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_n2_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_n2_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_n2_max.name)])
                                        """


class TestEngOilTempMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngOilTempMax.get_operational_combinations(),
                         [('Eng (*) Oil Temp Max', 'Airborne')])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_oil_temp_max = EngOilTempMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_oil_temp_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_oil_temp_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_oil_temp_max.name)])


class TestEngOilTemp15MinuteMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(EngOilTempMax.get_operational_combinations(),
                         [('Eng (*) Oil Temp Max', 'Airborne')])
        
    def test_all_oil_data_masked(self):
        # This has been a specific problem, hence this test.
        oil_temp=np.ma.array(data=[123,124,125,126,127], dtype=float,
                             mask=[1,1,1,1,1])
        kpv = EngOilTemp15MinuteMax()
        kpv.derive(P('Eng (*) Oil Temp Max', oil_temp))
        


class TestEngVibN1Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngVibN1Max.get_operational_combinations(),
                         [('Eng (*) Vib N1 Max', 'Airborne')])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_vib_n1_max = EngVibN1Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_vib_n1_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_vib_n1_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_vib_n1_max.name)])


class TestEngVibN2Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngVibN2Max.get_operational_combinations(),
                         [('Eng (*) Vib N2 Max', 'Airborne')])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_vib_n2_max = EngVibN2Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_vib_n2_max.derive(param)
        max_value.assert_called_once_with(param.array)
        self.assertEqual(eng_vib_n2_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_vib_n2_max.name)])


class TestEng_N1MaxDurationUnder60PercentAfterTouchdown(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_N1MaxDurationUnder60PercentAfterTouchdown.get_operational_combinations()
        self.assertEqual(
            ('Eng (1) N1', 'Touchdown', 'Eng (*) Stop'), opts[0]) 
        self.assertEqual(
            ('Eng (2) N1', 'Touchdown', 'Eng (*) Stop'), opts[1]) 
        self.assertEqual(
            ('Eng (3) N1', 'Touchdown', 'Eng (*) Stop'), opts[2])
        self.assertEqual(
            ('Eng (4) N1', 'Touchdown', 'Eng (*) Stop'), opts[3])
        self.assertTrue(
            ('Eng (1) N1', 'Eng (2) N1', 'Touchdown', 'Eng (*) Stop') in opts) 
        self.assertTrue(all(['Touchdown' in avail for avail in opts]))
        self.assertTrue(all(['Eng (*) Stop' in avail for avail in opts]))
        
    def test_eng_n1_cooldown(self):
        #TODO: Add later if required
        #gnd = S(items=[Section('', slice(10,100))]) 
        
        eng = P(array=np.ma.array([100] * 60 + [40] * 40)) # idle for 40        
        tdwn = KTI(items=[KeyTimeInstance(30), KeyTimeInstance(50)])
        eng_stop = KTI(items=[KeyTimeInstance(90, 'Eng (1) Stop'),])
        max_dur = Eng_N1MaxDurationUnder60PercentAfterTouchdown()
        max_dur.derive(eng, eng, None, None, tdwn, eng_stop)
        self.assertEqual(max_dur[0].index, 60) # starts at drop below 60
        self.assertEqual(max_dur[0].value, 30) # stops at 90
        self.assertTrue('Eng (1)' in max_dur[0].name)
        # Eng (2) should not be in the results as it did not have an Eng Stop KTI
        ##self.assertTrue('Eng (2)' in max_dur[1].name)
        self.assertEqual(len(max_dur), 1)


class TestFlapAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = FlapAtLiftoff
        self.operational_combinations = [('Flap', 'Liftoff')]


class TestFlapAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = FlapAtTouchdown
        self.operational_combinations = [('Flap', 'Touchdown')]


class TestFuelQtyAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = FuelQtyAtLiftoff
        self.operational_combinations = [('Fuel Qty', 'Liftoff')]
        

class TestFuelQtyAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = FuelQtyAtTouchdown
        self.operational_combinations = [('Fuel Qty', 'Touchdown')]

        
"""
class TestFuelQtyAirborneMin(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = FuelQtyAirborneMin
        self.operational_combinations = [('Fuel Qty', 'Airborne')]
        self.function = min_value
        """


class TestILSGlideslopeDeviation1500To1000FtMax(unittest.TestCase):
        
    def test_ils_glide_1500_1000_basic(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        
        testwave = np.ma.array(1.0 - np.cos(np.arange(0,6.3,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        
        gs_estab = buildsection('ILS Glideslope Established', 2,63)
        
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 31)
        self.assertAlmostEqual(kpv[0].value, 1.99913515027)

    def test_ils_glide_1500_1000_four_peaks(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        testwave = np.ma.array(-0.2-np.sin(np.arange(0,12.6,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        gs_estab = buildsection('ILS Glideslope Established', 2,56)
        kpv = ILSGlideslopeDeviation1500To1000FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertAlmostEqual(kpv[0].value, -1.1995736)

"""
class TestILSGlideslopeDeviationBelow1000FtMax(unittest.TestCase,
                                            CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = ILSGlideslopeDeviationBelow1000FtMax
        self.operational_combinations = [('ILS Glideslope', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_below', (1000,), {})]
        """


class TestILSGlideslopeDeviation1000To250FtMax(unittest.TestCase):
        
    def test_ils_glide_1000_250_basic(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        
        testwave = np.ma.array(1.0 - np.cos(np.arange(0,6.3,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        
        gs_estab = buildsection('ILS Glideslope Established', 2,63)
        
        kpv = ILSGlideslopeDeviation1000To250FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 36)
        self.assertAlmostEqual(kpv[0].value, 1.89675842)

    def test_ils_glide_1000_250_four_peaks(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        testwave = np.ma.array(-0.2-np.sin(np.arange(0,12.6,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        gs_estab = buildsection('ILS Glideslope Established', 2,56)
        kpv = ILSGlideslopeDeviation1000To250FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertAlmostEqual(kpv[0].value, 0.79992326)

'''
class TestGroundSpeedOnGroundMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = GroundSpeedOnGroundMax
        self.operational_combinations = [('Groundspeed', 'Grounded')]
        self.function = max_value
'''

class TestGrossWeightAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight Smoothed', 'Liftoff')]


class TestGrossWeightAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = GrossWeightAtTouchdown
        self.operational_combinations = [('Gross Weight Smoothed', 'Touchdown')]


class TestHeadingAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous', 'Takeoff')]
        opts = HeadingAtTakeoff.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_takeoff_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        toff = buildsection('Takeoff', 2,6)
        kpv = HeadingAtTakeoff()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=7.5,
                                  name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)
        
    def test_takeoff_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,6)
        kpv = HeadingAtTakeoff()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5,
                                  name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)

class TestHeadingAtLanding(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous', 'Landing')]
        opts = HeadingAtLanding.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                                                   7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Landing',5,15)
        head.array[13] = np.ma.masked
        kpv = HeadingAtLanding()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=10, value=6.0,
                                  name='Heading At Landing')]
        self.assertEqual(kpv, expected)


"""
class TestHeightAtGoAroundMin(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = HeightAtGoAroundMin
        self.operational_combinations = [('Altitude AAL For Flight Phases', 'Go Around')]
        """


class TestILSFrequencyOnApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Frequency', 'ILS Localizer Established',)]
        opts = ILSFrequencyOnApproach.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ILS_frequency_on_approach_basic(self):
        # Let's give this a really hard time with alternate samples invalid and
        # the final signal only tuned just at the end of the data.
        frq = P('ILS Frequency',np.ma.array([108.5]*6+[114.05]*4))
        frq.array[0:10:2] = np.ma.masked
        ils = buildsection('ILS Localizer Established', 2, 9)
        kpv = ILSFrequencyOnApproach()
        kpv.derive(frq, ils)
        expected = [KeyPointValue(index=2, value=108.5, 
                                  name='ILS Frequency On Approach')]
        self.assertEqual(kpv, expected)


class TestILSLocalizerDeviation1500To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases',
                     'ILS Localizer Established')]
        opts = ILSLocalizerDeviation1500To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1500_1000_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases',
                           np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        loc_est = buildsection('ILS Localizer Established', 30,115)
        kpv = ILSLocalizerDeviation1500To1000FtMax()
        kpv.derive(ils_loc, alt_ph, loc_est)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        
        
class TestILSLocalizerDeviation1000To250FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases',
                     'ILS Localizer Established')]
        opts = ILSLocalizerDeviation1000To250FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1000_250_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases',
                           np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        loc_est = buildsection('ILS Localizer Established', 30,115)
        kpv = ILSLocalizerDeviation1000To250FtMax()
        kpv.derive(ils_loc, alt_ph, loc_est)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 57)
        self.assertEqual(kpv[1].index, 114)


class TestMachMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = MachMax
        self.operational_combinations = [('Mach', 'Airborne')]
        self.function = max_value


####class TestPitch1000To100FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = Pitch1000To100FtMax
####        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
####        self.function = max_value
####        self.second_param_method_calls = [('slices_from_to', (1000, 100,), {})]
####
####
####class TestPitch1000To100FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = Pitch1000To100FtMin
####        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
####        self.function = min_value
####        self.second_param_method_calls = [('slices_from_to', (1000, 100,), {})]
####
####
####class TestPitch20FtToTouchdownMin(unittest.TestCase,
####                                  CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = Pitch20FtToTouchdownMin
####        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
####        self.function = min_value
####        self.second_param_method_calls = [('slices_from_to', (20, 0,), {})]


class TestPitch35To400FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Pitch35To400FtMax
        self.operational_combinations = [('Pitch',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 400,), {})]
        
    def test_pitch_35_400_basic(self):
        pch = [0,2,4,7,9,8,6,3,-1]
        alt = [100,101,102,103,700,105,104,103,102]
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(alt))
        pitch = Parameter('Pitch', np.ma.array(pch))
        kpv = Pitch35To400FtMax()
        kpv.derive(pitch, alt_ph)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 3)
        self.assertEqual(kpv[0].value, 7)


class TestPitch35To400FtMin(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Pitch35To400FtMin
        self.operational_combinations = [('Pitch',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 400,), {})]


####class TestPitch5FtToTouchdownMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = Pitch5FtToTouchdownMax
####        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
####        self.function = max_value
####        self.second_param_method_calls = [('slices_from_to', (5, 0,), {})]


class TestPitchAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = PitchAtLiftoff
        self.operational_combinations = [('Pitch', 'Liftoff')]


class TestPitchAtTouchdown(unittest.TestCase, CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = PitchAtTouchdown
        self.operational_combinations = [('Pitch', 'Touchdown')]


"""
class TestPitchDuringFinalApproachMin(unittest.TestCase,
                                      CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = PitchDuringFinalApproachMin
        self.operational_combinations = [('Pitch', 'Final Approach')]
        self.function = min_value
        """


"""
class TestPitchDuringTakeoffMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = PitchDuringTakeoffMax
        self.operational_combinations = [('Pitch', 'Takeoff')]
        self.function = max_value
        """


"""
class TestPitchRate35To1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = PitchRate35To1500FtMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1500), {})]
        """


####class TestPitchRateDuringTakeoffMax(unittest.TestCase,
####                                    CreateKPVsWithinSlicesTest):
####    def setUp(self):
####        self.node_class = PitchRateDuringTakeoffMax
####        self.operational_combinations = [('Pitch Rate', 'Takeoff')]
####        self.function = max_value
####
####
####class TestPitchRateDuringTakeoffMin(unittest.TestCase):
####    def test_derive(self):
####        node = PitchRateDuringTakeoffMin()
####        self.assertTrue(False)


class TestRateOfDescent10000To5000FtMax(unittest.TestCase,
                                        CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent10000To5000FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (10000, 5000), {})]


class TestRateOfDescent5000To3000FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent5000To3000FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (5000, 3000), {})]


class TestRateOfDescent3000To2000FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent3000To2000FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (3000, 2000), {})]


class TestRateOfDescent2000To1000FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent2000To1000FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (2000, 1000), {})]


class TestRateOfDescent1000To500FtMax(unittest.TestCase,
                                      CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent1000To500FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]
        

class TestRateOfDescent500To20FtMax(unittest.TestCase,
                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RateOfDescent500To20FtMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]
        

# FIXME: Uses slices_to_kti(), not slices_from_to()!
class TestRateOfDescent500FtToTouchdownMax(unittest.TestCase,
                                           CreateKPVsWithinSlicesTest):
    def setUp(self):
        # XXX: This test does not explicitly test how the Touchdown dependency
        #      is used.
        self.node_class = RateOfDescent500FtToTouchdownMax
        self.operational_combinations = [('Vertical Speed',
                                          'Altitude AAL For Flight Phases',
                                          'Touchdown',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_to_kti', (500, []), {})]


class TestRollAbove1000FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RollAbove1000FtMax
        self.operational_combinations = [('Roll',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1000,), {})]


"""
class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RollAbove1500FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1500,), {})]
        """


"""
class TestRollBelow20FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RollBelow20FtMax
        self.operational_combinations = [('Roll', 'Altitude Radio')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_between', (1,20), {})]
    
    @patch('analysis_engine.key_point_values.max_abs_value')
    def test_derive(self, max_abs_value):
        roll_below_20ft_max = RollBelow20FtMax()
        index, value = 10, 30
        max_abs_value.return_value = index, value
        param1 = Mock()
        param1.array = Mock()
        param2 = Mock()
        param2.array = Mock()
        param2.slices_below = Mock()
        param2.slices_below.return_value = [slice(0, 10)]
        roll_below_20ft_max.derive(param1, param2)
        max_abs_value.assert_called_once_with(param1.array, slice(0,10))
        self.assertEqual(roll_below_20ft_max,
                         [KeyPointValue(index=index, value=value,
                                        name=roll_below_20ft_max.name)])
                                        """


"""
class TestRollBetween100And500FtMax(unittest.TestCase,
                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RollBetween100And500FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_between', (100, 500), {})]
        """


"""
class TestRollBetween500And1500FtMax(unittest.TestCase,
                                     CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = RollBetween500And1500FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_between', (500, 1500), {})]
        """

class TestTailClearanceOnApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TailClearanceOnApproach.get_operational_combinations(),
                         [('Altitude AAL', 'Altitude Tail',
                           'Distance To Landing')])
    
    def test_derive(self):
        # XXX: The BDUTerrain test files are missing from the repository?
        test_data_dir = os.path.join(test_data_path, 'BDUTerrain')
        alt_aal_array = np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                                'alt_aal.npy')))
        alt_radio_array = \
            np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                    'alt_radio.npy')))
        dtl_array = np.ma.masked_array(np.load(os.path.join(test_data_dir,
                                                            'dtl.npy')))
        alt_aal = P(array=alt_aal_array, frequency=8)
        alt_radio = P(array=alt_radio_array, frequency=0.5)
        dtl = P(array=dtl_array, frequency=0.25)
        alt_radio.array = align(alt_radio, alt_aal)
        dtl.array = align(dtl, alt_aal)        
        # Q: Should tests for the BDUTerrain node be in a separate TestCase?
        param = BDUTerrain()
        param.derive(alt_aal, alt_radio, dtl)
        self.assertEqual(param, [KeyPointValue(name='BDU Terrain', index=1008,
                                               value=0.037668517049960347)])


class TestZeroFuelWeight(unittest.TestCase):
    def test_derive(self):
        fuel = P('Fuel Qty', np.ma.array([1,2,3,4]))
        weight = P('Gross Weight', np.ma.array([11,12,13,14]))
        zfw = ZeroFuelWeight()
        zfw.derive(fuel, weight)
        self.assertEqual(zfw[0].value, 10.0)
    

class TestAccelerationLateralAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationLateralAtTouchdown.get_operational_combinations(),
            [('Acceleration Lateral', 'Touchdown',)])
        
    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        values = [(1, 2,), (3, 4,)]
        bump.side_effect = lambda *args, **kwargs: values.pop()
        node = AccelerationLateralAtTouchdown()
        acc = Mock()
        tdwn = [Section('Touchdown', slice(10, 20), 10, 20),
                Section('Touchdown', slice(30, 40), 30, 40),]
        node.derive(acc, tdwn)
        self.assertEqual(bump.call_args_list[0][0], (acc, tdwn[0]))
        self.assertEqual(bump.call_args_list[1][0], (acc, tdwn[1]))
        self.assertEqual(
            node,
            [KeyPointValue(3, 4.0, 'Acceleration Lateral At Touchdown',
                           slice(None, None)),
             KeyPointValue(1, 2.0, 'Acceleration Lateral At Touchdown',
                           slice(None, None))])        


class TestAccelerationLateralMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AccelerationLateralMax.get_operational_combinations(),
                         [('Acceleration Lateral',),
                          ('Acceleration Lateral', 'Groundspeed',),])        
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralTaxiingStraightMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationLateralTaxiingStraightMax.get_operational_combinations(),
            [('Acceleration Lateral', 'Taxiing', 'Turning On Ground',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationLateralTaxiingTurnsMax(unittest.TestCase,
                                             CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationLateralTaxiingTurnsMax
        self.operational_combinations = [('Acceleration Lateral',
                                          'Turning On Ground',)]
        self.function = max_abs_value


class TestAccelerationLongitudinalPeakTakeoff(unittest.TestCase,
                                              CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AccelerationLongitudinalPeakTakeoff
        self.operational_combinations = [('Acceleration Longitudinal',
                                          'Takeoff',)]
        self.function = max_value


class TestAccelerationNormalAirborneFlapsDownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAirborneFlapsDownMax.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Flap', 
              'Airborne',)])        
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalAirborneFlapsDownMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAirborneFlapsDownMin.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Flap', 
              'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalAirborneFlapsUpMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAirborneFlapsUpMax.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Flap', 
              'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalAirborneFlapsUpMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAirborneFlapsUpMin.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Flap', 
              'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAccelerationNormalAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAtLiftoff.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Liftoff',)])
        
    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        values = [(1, 2,), (3, 4,)]
        bump.side_effect = lambda *args, **kwargs: values.pop()
        node = AccelerationNormalAtLiftoff()
        acc = Mock()
        tdwn = [Section('Liftoff', slice(10, 20), 10, 20),
                Section('Liftoff', slice(30, 40), 30, 40),]
        node.derive(acc, tdwn)
        self.assertEqual(bump.call_args_list[0][0], (acc, tdwn[0]))
        self.assertEqual(bump.call_args_list[1][0], (acc, tdwn[1]))
        self.assertEqual(
            node,
            [KeyPointValue(3, 4.0, 'Acceleration Normal At Liftoff',
                           slice(None, None)),
             KeyPointValue(1, 2.0, 'Acceleration Normal At Liftoff',
                           slice(None, None))])


class TestAccelerationNormalAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalAtTouchdown.get_operational_combinations(),
            [('Acceleration Normal Offset Removed', 'Touchdown',)])
        
    @patch('analysis_engine.key_point_values.bump')
    def test_derive(self, bump):
        values = [(1, 2,), (3, 4,)]
        bump.side_effect = lambda *args, **kwargs: values.pop()
        node = AccelerationNormalAtTouchdown()
        acc = Mock()
        tdwn = [Section('Touchdown', slice(10, 20), 10, 20),
                Section('Touchdown', slice(30, 40), 30, 40),]
        node.derive(acc, tdwn)
        self.assertEqual(bump.call_args_list[0][0], (acc, tdwn[0]))
        self.assertEqual(bump.call_args_list[1][0], (acc, tdwn[1]))
        self.assertEqual(
            node,
            [KeyPointValue(3, 4.0, 'Acceleration Normal At Touchdown',
                           slice(None, None)),
             KeyPointValue(1, 2.0, 'Acceleration Normal At Touchdown',
                           slice(None, None))])                


class TestAccelerationNormalLiftoffTo35FtMax(unittest.TestCase,
                                             CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AccelerationNormalLiftoffTo35FtMax
        self.operational_combinations = [('Acceleration Normal Offset Removed',
                                          'Takeoff',)]
        self.function = max_value


class TestAccelerationNormalOffset(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AccelerationNormalOffset.get_operational_combinations(),
            [('Acceleration Normal', 'Taxiing',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeed10000ToLandMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            Airspeed10000ToLandMax.get_operational_combinations(),
            [('Airspeed', 'Altitude STD', 'Altitude QNH', 'FDR Landing Airport',
              'Descent')])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeed1000To500FtMin(unittest.TestCase,
                                 CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed1000To500FtMin
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeed35To1000FtMax(unittest.TestCase,
                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed35To1000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]



class TestAirspeed35To1000FtMin(unittest.TestCase,
                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed35To1000FtMin
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]


class TestAirspeed500To20FtMax(unittest.TestCase,
                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed500To20FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeed500To20FtMin(unittest.TestCase,
                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = Airspeed500To20FtMin
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedAsGearExtendingMax(unittest.TestCase,
                                     CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedAsGearExtendingMax
        self.operational_combinations = [('Airspeed', 'Gear Extending',)]
        self.function = max_value


class TestAirspeedAsGearRetractingMax(unittest.TestCase,
                                      CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedAsGearRetractingMax
        self.operational_combinations = [('Airspeed', 'Gear Retracting',)]
        self.function = max_value


class TestAirspeedAt35FtInTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedAt35FtInTakeoff.get_operational_combinations(),
            [('Airspeed', 'Takeoff',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedAtGearDownSelection(unittest.TestCase,
                                      CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedAtGearDownSelection
        self.operational_combinations = [('Airspeed', 'Gear Down Selection',)]


class TestAirspeedAtGearUpSelection(unittest.TestCase,
                                    CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedAtGearUpSelection
        self.operational_combinations = [('Airspeed', 'Gear Up Selection',)]


class TestAirspeedAtLiftoff(unittest.TestCase,
                            CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedAtLiftoff
        self.operational_combinations = [('Airspeed', 'Liftoff',)]


class TestAirspeedBetween1000And3000FtMax(unittest.TestCase,
                                          CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedBetween1000And3000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_between', (1000, 3000), {})]


class TestAirspeedBetween3000And5000FtMax(unittest.TestCase,
                                          CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedBetween3000And5000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_between', (3000, 5000), {})]


class TestAirspeedBetween5000And8000FtMax(unittest.TestCase,
                                          CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedBetween5000And8000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_between', (5000, 8000), {})]


class TestAirspeedBetween8000And10000FtMax(unittest.TestCase,
                                           CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedBetween8000And10000FtMax
        self.operational_combinations = [('Airspeed',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_between', (8000, 10000), {})]


class TestAirspeedBetween90SecToTouchdownAndTouchdownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedBetween90SecToTouchdownAndTouchdownMax.get_operational_combinations(),
            [('Secs To Touchdown', 'Airspeed',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedCruiseMax(unittest.TestCase,
                                           CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedCruiseMax
        self.operational_combinations = [('Airspeed', 'Cruise',)]
        self.function = max_value


class TestAirspeedCruiseMin(unittest.TestCase,
                                           CreateKPVFromSlicesTest):
    def setUp(self):
        self.node_class = AirspeedCruiseMin
        self.operational_combinations = [('Airspeed', 'Cruise',)]
        self.function = min_value


class TestAirspeedLevelFlightMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedLevelFlightMax.get_operational_combinations(),
            [('Airspeed', 'Level Flight',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedMax3Sec(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedMinusV235To1000FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMax
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]


class TestAirspeedMinusV235To1000FtMin(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedMinusV235To1000FtMin
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 1000), {})]


class TestAirspeedMinusV2At35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedRTOMax(unittest.TestCase,
                         CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRTOMax
        self.operational_combinations = [('Airspeed', 'Rejected Takeoff',)]
        self.function = max_value


class TestAirspeedRelative1000To500FtMax(unittest.TestCase,
                                         CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMax
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelative1000To500FtMin(unittest.TestCase,
                                         CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative1000To500FtMin
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelative20FtToTouchdownMax(unittest.TestCase,
                                             CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelative20FtToTouchdownMin(unittest.TestCase,
                                             CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelative500To20FtMax(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMax
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelative500To20FtMin(unittest.TestCase,
                                       CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelative500To20FtMin
        self.operational_combinations = [('Airspeed Relative',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelativeFor3Sec1000To500FtMax(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelativeFor3Sec1000To500FtMin(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelativeFor3Sec20FtToTouchdownMax(unittest.TestCase,
                                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelativeFor3Sec20FtToTouchdownMin(unittest.TestCase,
                                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelativeFor3Sec500To20FtMax(unittest.TestCase,
                                              CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMax
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelativeFor3Sec500To20FtMin(unittest.TestCase,
                                              CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec500To20FtMin
        self.operational_combinations = [('Airspeed Relative For 3 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelativeFor5Sec1000To500FtMax(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec1000To500FtMax
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelativeFor5Sec1000To500FtMin(unittest.TestCase,
                                                CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec1000To500FtMin
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]


class TestAirspeedRelativeFor5Sec20FtToTouchdownMax(unittest.TestCase,
                                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec20FtToTouchdownMax
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelativeFor5Sec20FtToTouchdownMin(unittest.TestCase,
                                                    CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec20FtToTouchdownMin
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (20, 0), {})]


class TestAirspeedRelativeFor5Sec500To20FtMax(unittest.TestCase,
                                              CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec500To20FtMax
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelativeFor5Sec500To20FtMin(unittest.TestCase,
                                              CreateKPVsWithinSlicesTest):
    def setUp(self):
        self.node_class = AirspeedRelativeFor5Sec500To20FtMin
        self.operational_combinations = [('Airspeed Relative For 5 Sec',
                                          'Altitude AAL For Flight Phases',)]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 20), {})]


class TestAirspeedRelativeWithFlapDescentMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedRelativeWithFlapDescentMin.get_operational_combinations(),
            [('Flap', 'Airspeed Relative', 'Descent To Flare',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTODTo10000Max(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedTODTo10000Max.get_operational_combinations(),
            [('Airspeed', 'Altitude STD', 'Altitude QNH', 'FDR Landing Airport',
              'Descent')])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedThrustReversersDeployedMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedThrustReversersDeployedMin.get_operational_combinations(),
            [('Airspeed True', 'Thrust Reversers', 'Eng (*) N1 Avg',
              'Landing',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedTrueAtTouchdown(unittest.TestCase,
                                  CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedTrueAtTouchdown
        self.operational_combinations = [('Airspeed True', 'Touchdown',)]


class TestAirspeedVacatingRunway(unittest.TestCase,
                                 CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AirspeedVacatingRunway
        self.operational_combinations = [('Airspeed True',
                                          'Landing Turn Off Runway',)]


class TestAirspeedWithFlapClimbMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithFlapClimbMax.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Climb',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapClimbMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithFlapClimbMin.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Climb',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDescentMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithFlapDescentMax.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Descent',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapDescentMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithFlapDescentMin.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Descent',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithFlapMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithFlapMin.get_operational_combinations(),
            [('Flap', 'Airspeed', 'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAirspeedWithGearDownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AirspeedWithGearDownMax.get_operational_combinations(),
            [('Airspeed', 'Gear Down', 'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtFirstFlapChangeAfterLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeAtFirstFlapChangeAfterLiftoff.get_operational_combinations(),
            [('Flap', 'Altitude AAL', 'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearDownSelection(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGearUpSelection(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtGoAroundMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeAtGoAroundMin.get_operational_combinations(),
            [('Altitude AAL', 'Go Around',),
             ('Altitude AAL', 'Go Around', 'Altitude Radio',),])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtLastFlapChangeBeforeLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeAtLastFlapChangeBeforeLanding.get_operational_combinations(),
            [('Flap', 'Altitude AAL', 'Touchdown',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeAtLiftoff(unittest.TestCase,
                            CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAtLiftoff
        self.operational_combinations = [('Altitude STD', 'Liftoff',)]


class TestAltitudeAutopilotDisengaged(unittest.TestCase,
                                      CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAutopilotDisengaged
        self.operational_combinations = [('Altitude AAL',
                                          'AP Disengaged Selection',)]


class TestAltitudeAutopilotEngaged(unittest.TestCase,
                                   CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAutopilotEngaged
        self.operational_combinations = [('Altitude AAL',
                                          'AP Engaged Selection',)]


class TestAltitudeAutothrottleDisengaged(unittest.TestCase,
                                         CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAutothrottleDisengaged
        self.operational_combinations = [('Altitude AAL',
                                          'AT Disengaged Selection',)]


class TestAltitudeAutothrottleEngaged(unittest.TestCase,
                                      CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeAutothrottleEngaged
        self.operational_combinations = [('Altitude AAL',
                                          'AT Engaged Selection',)]


class TestAltitudeFlapExtensionMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeFlapExtensionMax.get_operational_combinations(),
            [('Flap', 'Altitude AAL', 'Airborne',)])
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeGoAroundFlapRetracted(unittest.TestCase,
                                        CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeGoAroundFlapRetracted
        self.operational_combinations = [('Altitude AAL',
                                          'Go Around Flap Retracted',)]


class TestAltitudeGoAroundGearRetracted(unittest.TestCase,
                                        CreateKPVsAtKTIsTest):
    def setUp(self):
        self.node_class = AltitudeGoAroundGearRetracted
        self.operational_combinations = [('Altitude AAL',
                                          'Go Around Gear Retracted',)]


class TestAltitudeMinsToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeWithFlapsMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDecelerateToStopOnRunwayDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDecelerationLongitudinalPeakLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDecelerationToStopOnRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFrom60KtToRunwayEnd(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromRunwayStartToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistanceFromTouchdownToRunwayEnd(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDistancePastGlideslopeAntennaToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng1GasTempStartMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng2GasTempStartMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngBleedValvesAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRAboveFL100Max(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngEPRToFL100Max(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempGoAroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngGasTempMaximumContinuousPowerMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1500To20FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1CyclesInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1GoAroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1MaximumContinuousPowerMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN1TaxiMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2CyclesInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2GoAroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2MaximumContinuousPowerMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2TakeoffMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN2TaxiMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3GoAroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3MaximumContinuousPowerMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3TakeoffMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngN3TaxiMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilPressMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilPressMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilQtyMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngOilQtyMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500FtToTouchdownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorque500FtToTouchdownMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueAbove10000FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueAbove10000FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueAboveFL100Max(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueGoAroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueMaximumContinuousPowerMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueTakeoffMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTorqueToFL100Max(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEventMarkerPressed(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapAtGearDownSelection(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapWithGearUpMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlapWithSpeedbrakesDeployedMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlareDistance20FtToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFlareDuration20FtToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGenericDescent(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedOnGroundMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedRTOMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedTaxiingStraightMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedTaxiingTurnsMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedThrustReversersDeployedMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGroundspeedVacatingRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingAtLowestPointOnApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviation500To20Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationOnLandingAbove100Kts(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationOnTakeoffAbove100Kts(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingDeviationTouchdownPlus4SecTo60Kts(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeadingVacatingRunway(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLost1000To2000Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLost35To1000Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightLostTakeoffTo35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHeightOfBouncedLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestHoldingDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestIsolationValveOpenAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudeAtLowestPointOnApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudeAtLowestPointOnApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMachAsGearExtendingMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMachAsGearRetractingMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMachMax3Sec(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMachWithGearDownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariationAtLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariationAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPackValvesOpenAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch1000To500FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch20FtToLandingMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch400To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch400To1000FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To20FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch500To50FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch50FtToLandingMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch7FtToLandingMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchAt35FtInClimb(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchCyclesInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate20FtToTouchdownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate20FtToTouchdownMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtAverage(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate2DegPitchTo35FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchRate35To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchTakeoffTo35FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimb35To1000FtMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfClimbMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescent20ToTouchdownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRateOfDescentTopOfDescentTo10000FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll1000To300FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll20FtToLandingMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll20To400FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll300To20FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll400To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollCyclesInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRollTakeoffTo20FtMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderReversalAbove50Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployed1000To20FtDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployedInGoAroundDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployedWithConfDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployedWithFlapDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployedWithPowerOnDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSpeedbrakesDeployedWithPowerOnInHeightBandsDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestStickPusherActivatedDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestStickShakerActivatedDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSAlertDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSDontSinkWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSGlideslopeWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSPullUpWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSSinkRateWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainPullUpWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTerrainWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowFlapWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowGearWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSTooLowTerrainWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTAWSWindshearWarningBelow1500FtDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAInitialReaction(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAReactionDelay(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAToAPDisengageDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASRAWarningDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceOnLandingMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailClearanceOnTakeoffMin(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTailwind100FtToTouchdownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrottleCyclesInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownTo60KtsDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTouchdownToElevatorDownDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTwoDegPitchTo35FtDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindAcrossLandingRunwayAt50Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirectionInDescent(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindSpeedInDescent(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestDescentToFlare(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearExtending(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGoAround5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLevelFlight(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoff5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRotation(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTwoDegPitchTo35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')
        
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
