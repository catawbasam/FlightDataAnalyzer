import os
import numpy as np
import sys
import unittest

from mock import Mock, patch

from analysis_engine.derived_parameters import Flap
from analysis_engine.library import align
from analysis_engine.node import (KeyTimeInstance, KTI, KeyPointValue, 
                           KeyPointValueNode, Parameter, P, Section, S)

from analysis_engine.key_point_values import (
    ##AccelerationLateralTaxiingMax,
    AccelerationNormal20FtToGroundMax,
    ##AccelerationNormalAirborneMax,
    ##AccelerationNormalAirborneMin,
    ##AccelerationNormalDuringTakeoffMax,
    AccelerationNormalMax,
    Airspeed1000To500FtMax,
    Airspeed2000To30FtMin,
    Airspeed400To1500FtMin,
    ##Airspeed50To1000FtMax,
    ##Airspeed500To50FtMax,
    AirspeedAtTouchdown,
    AirspeedBelowAltitudeMax,
    AirspeedMax,
    ##AirspeedMinusV235To400FtMin,
    ##AirspeedMinusV2400To1500FtMin,
    AirspeedMinusV2AtLiftoff,
    AirspeedMinusVref500FtToTouchdownMax,
    AirspeedMinusVrefAtTouchdown,
    AirspeedWithFlapMax,
    ##AirspeedWithGearSelectedDownMax,
    AltitudeAtMachMax,
    AltitudeAtTouchdown,
    AltitudeMax,
    ##AltitudeRadioDividedByDistanceToLanding3000To50FtMin,
    ##AutopilotEngaged1AtLiftoff,
    ##AutopilotEngaged1AtTouchdown,
    ##AutopilotEngaged2AtLiftoff,
    ##AutopilotEngaged2AtTouchdown,
    ControlColumnStiffness,
    ##EngEGTMax,
    EngEPR500FtToTouchdownMin,
    EngN13000FtToTouchdownMax,
    EngN1500FtToTouchdownMin,
    ##EngN1Max,
    EngN1TakeoffMax,
    ##EngN2Max,
    EngOilTempMax,
    EngVibN1Max,
    EngVibN2Max,
    HeadingAtTakeoff,
    Eng_N1MaxDurationUnder60PercentAfterTouchdown,
    ##FlapAtGearSelectedDown,
    FlapAtLiftoff,
    FlapAtTouchdown,
    FuelQtyAtLiftoff,
    FuelQtyAtTouchdown,
    ##FuelQtyAirborneMin,
    GrossWeightAtLiftoff,
    GrossWeightAtTouchdown,
    ##GroundSpeedOnGroundMax,
    ILSFrequencyOnApproach,
    ILSGlideslopeDeviation1500To1000FtMax,
    ILSGlideslopeDeviation1000To150FtMax,
    ##ILSGlideslopeDeviationBelow1000FtMax,
    HeadingAtLanding,
    ## HeadingAtLowPointOnApproach,
    ##HeightAtGoAroundMin,
    LatitudeAtLanding,
    ## LatitudeAtLowPointOnApproach,
    LatitudeAtTakeoff,
    LongitudeAtLanding,
    ## LongitudeAtLowPointOnApproach,
    LongitudeAtTakeoff,
    ILSLocalizerDeviation1500To1000FtMax,
    ILSLocalizerDeviation1000To150FtMax,
    MachMax,
    Pitch1000To100FtMax,
    Pitch1000To100FtMin,
    Pitch20FtToTouchdownMin,
    Pitch35To400FtMax,
    Pitch35To400FtMin,
    Pitch5FtToTouchdownMax,
    PitchAtLiftoff,
    PitchAtTouchdown,
    ##PitchRate35To1500FtMax,
    PitchRateDuringTakeoffMax,
    PitchRateDuringTakeoffMin,
    ##PitchDuringFinalApproachMin,
    ##PitchDuringTakeoffMax,
    RateOfDescent500FtToTouchdownMax,
    RateOfDescent1000To500FtMax,
    ###RateOfDescent1000To50FtMax,
    RateOfDescent2000To1000FtMax,
    RollAbove1000FtMax,
    ##RollAbove1500FtMax,
    ##RollBelow20FtMax,
    ##RollBetween100And500FtMax,
    ##RollBetween500And1500FtMax,
    TailClearanceOnApproach,
    ZeroFuelWeight,
)
from analysis_engine.library import (max_abs_value, max_value, min_value)
from analysis_engine.flight_phase import Fast
from flight_phase_test import buildsection, buildsections

debug = sys.gettrace() is not None


class TestNode(object):
    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         self.operational_combinations)


class TestCreateKPVsAtKPVs(TestNode):
    '''
    Example of subclass inheriting tests:
    
class TestAltitudeAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
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


class TestCreateKPVsAtKTIs(TestNode):
    '''
    Example of subclass inheriting tests:
    
class TestAltitudeAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
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


class TestCreateKPVsWithinSlices(TestNode):
    '''
    Example of subclass inheriting tests:
    
class TestRollAbove1500FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
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
        self.assertEqual
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


"""
class TestAccelerationLateralTaxiingMax(unittest.TestCase,
                                         TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AccelerationLateralTaxiingMax
        self.operational_combinations = [('Acceleration Lateral', 'On Ground')]
        self.function = max_value
        """


"""
class TestAccelerationNormalAirborneMax(unittest.TestCase,
                                        TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AccelerationNormalAirborneMax
        self.operational_combinations = [('Acceleration Normal', 'Airborne')]
        self.function = max_value
        """


"""
class TestAccelerationNormalAirborneMin(unittest.TestCase,
                                        TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AccelerationNormalAirborneMin
        self.operational_combinations = [('Acceleration Normal', 'Airborne')]
        self.function = min_value
        """

"""
class TestAccelerationNormalDuringTakeoffMax(unittest.TestCase,
                                             TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AccelerationNormalDuringTakeoffMax
        self.operational_combinations = [('Acceleration Normal', 'Takeoff')]
        self.function = max_value
        """


class TestAccelerationNormalMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(AccelerationNormalMax.get_operational_combinations(),
                         [('Acceleration Normal',)])
    
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


class TestAccelerationNormal20FtToGroundMax(unittest.TestCase,
                                            TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AccelerationNormal20FtToGroundMax
        self.operational_combinations = [('Acceleration Normal',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (20, 0,), {})]
    
    def test_derive(self):
        '''
        Depends upon DerivedParameterNode.slices_from_to and library.max_value.
        '''
        # Test height range limit
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(48,0,-3))
        acceleration_normal = P('Acceleration Normal', np.ma.array(range(10,18)+range(18,10,-1))/10.0)
        node = AccelerationNormal20FtToGroundMax()
        node.derive(acceleration_normal, alt_aal)
        self.assertEqual(node,
                [KeyPointValue(index=10, value=1.6,
                               name='Acceleration Normal 20 Ft To Ground Max')])
        # Test peak acceleration
        alt_aal = P('Altitude AAL For Flight Phases', np.ma.arange(32,0,-2))
        node = AccelerationNormal20FtToGroundMax()
        node.derive(acceleration_normal, alt_aal)
        self.assertEqual(node,
                [KeyPointValue(index=8, value=1.8,
                               name='Acceleration Normal 20 Ft To Ground Max')])


class TestAirspeed1000To500FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Airspeed1000To500FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500,), {})]
        
    def test_airspeed_1000_500_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-100))+100
        spd = Parameter('Airspeed', np.ma.array(testwave))
        alt_ph = Parameter('Altitude AAL For Flight Phases', 
                           np.ma.array(testwave)*10)
        kpv = Airspeed1000To500FtMax()
        kpv.derive(spd, alt_ph)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertEqual(kpv[0].value, 91.250101656055278)
        self.assertEqual(kpv[1].index, 110)
        self.assertEqual(kpv[1].value, 99.557430201194919)


class TestAirspeed2000To30FtMin(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Airspeed2000To30FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (2000, 30,), {})]


class TestAirspeed400To1500FtMin(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Airspeed400To1500FtMin
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (400, 1500,), {})]


"""
class TestAirspeed50To1000FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Airspeed50To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (50, 1000,), {})]
        """


"""
class TestAirspeed500To50FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Airspeed50To1000FtMax
        self.operational_combinations = [('Airspeed', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 50,), {})]
        """


class TestAirspeedAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
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


class TestAirspeedMax(unittest.TestCase, TestCreateKPVsWithinSlices):
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
                                      TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AirspeedMinusV235To400FtMin
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 400,), {})]
        """


"""
class TestAirspeedMinusV2400To1500FtMin(unittest.TestCase,
                                        TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AirspeedMinusV2400To1500FtMin
        self.operational_combinations = [('Airspeed Minus V2',
                                          'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (400, 1500), {})]
        """
        

class TestAirspeedMinusV2AtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AirspeedMinusV2AtLiftoff
        self.operational_combinations = [('Airspeed Minus V2', 'Liftoff')]


class TestAirspeedMinusVref500FtToTouchdownMax(unittest.TestCase,
                                               TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AirspeedMinusVref500FtToTouchdownMax
        self.operational_combinations = [('Airspeed Minus Vref',
                                          'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (500, 0,), {})]


class TestAirspeedMinusVrefAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AirspeedMinusVrefAtTouchdown
        self.operational_combinations = [('Airspeed Minus Vref', 'Touchdown')]

    
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
        flap = P('Flap', np.ma.masked_array([0] * 2 + [1] * 2 + [2] * 2 + [5] * 2 + \
                                      [10] * 2 +  [15] * 2 + [25] * 2 + \
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


"""
class TestAirspeedWithGearSelectedDownMax(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(\
            AirspeedWithGearSelectedDownMax.get_operational_combinations(),
            [('Airspeed', 'Gear Selected Down')])
    
    def test_derive(self):
        airspeed = P('Airspeed', np.ma.masked_array(np.ma.arange(0,10),
                                   mask=[False] * 4 + [True] * 1 + [False] * 5))
        gear_sel_down = P('Gear Selected Down',
                          np.ma.masked_array([0,1,1,1,1,0,0,0,0,0],
                                   mask=[False] * 3 + [True] * 1 + [False] * 6))
        airspeed_with_gear_max = AirspeedWithGearSelectedDownMax()
        airspeed_with_gear_max.derive(airspeed, gear_sel_down)
        self.assertEqual(airspeed_with_gear_max,
          [KeyPointValue(index=2, value=2,
                         name='Airspeed With Gear Selected Down Max',
                         slice=slice(None, None, None), datetime=None)])
                         """


class TestAltitudeAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AltitudeAtTouchdown
        self.operational_combinations = [('Altitude STD', 'Touchdown')]


class TestAltitudeAtMachMax(unittest.TestCase, TestCreateKPVsAtKPVs):
    def setUp(self):
        self.node_class = AltitudeAtMachMax
        self.operational_combinations = [('Altitude STD', 'Mach Max')]


class TestAltitudeMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = AltitudeMax
        self.operational_combinations = [('Altitude STD', 'Airborne')]
        self.function = max_value


"""
class TestAutopilotEngaged1AtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AutopilotEngaged1AtLiftoff
        self.operational_combinations = [('Autopilot Engaged 1', 'Liftoff')]


class TestAutopilotEngaged1AtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AutopilotEngaged1AtTouchdown
        self.operational_combinations = [('Autopilot Engaged 1', 'Touchdown')]


class TestAutopilotEngaged2AtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AutopilotEngaged2AtLiftoff
        self.operational_combinations = [('Autopilot Engaged 2', 'Liftoff')]


class TestAutopilotEngaged2AtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AutopilotEngaged2AtTouchdown
        self.operational_combinations = [('Autopilot Engaged 2', 'Touchdown')]
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
class TestEngEGTMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngEGTMax.get_operational_combinations(),
                         [('Eng (*) EGT Max',)])
    
    @patch('analysis_engine.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_egt_max = EngEGTMax()
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
                                    TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = EngEPR500FtToTouchdownMin
        self.operational_combinations = [('Eng (*) EPR Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 0,), {})]


class TestEngN1500FtToTouchdownMin(unittest.TestCase,
                                   TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = EngN1500FtToTouchdownMin
        self.operational_combinations = [('Eng (*) N1 Min', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 0,), {})]


class EngEGTTakeoffMax(unittest.TestCase):
    def setUp(self):
        self.node_class = EngEGTTakeoffMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Takeoff')]
        self.function = max_value


class TestEngN13000FtToTouchdownMax(unittest.TestCase,
                                    TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = EngN13000FtToTouchdownMax
        self.operational_combinations = [('Eng (*) N1 Max', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (3000, 0,), {})]


class TestEngN1TakeoffMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = EngN1TakeoffMax
        self.function = max_value
        self.operational_combinations = [('Eng (*) N1 Max', 'TOGA 5 Min Rating')]


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
                         [('Eng (*) Oil Temp Max',)])
    
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


class TestEngVibN1Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngVibN1Max.get_operational_combinations(),
                         [('Eng (*) Vib N1 Max',)])
    
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
                         [('Eng (*) Vib N2 Max',)])
    
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
            ('Eng (3) N1', 'Touchdown', 'Eng (*) Stop'),  opts[2])
        self.assertEqual(
            ('Eng (4) N1', 'Touchdown', 'Eng (*) Stop'), opts[3])
        self.assertTrue(
            ('Eng (1) N1', 'Eng (2) N1', 'Touchdown', 'Eng (*) Stop') in opts) 
        self.assertTrue(all(['Touchdown' in avail for avail in opts]))
        self.assertTrue(all(['Eng (*) Stop' in avail for avail in opts]))
        
    def test_eng_n1_cooldown(self):
        #TODO: Add later if required
        #gnd = S(items=[Section('', slice(10,100))]) 
        
        eng = P(array=np.ma.array([100]*60 + [40]*40)) # idle for 40        
        tdwn = KTI(items=[KeyTimeInstance(30),KeyTimeInstance(50)])
        eng_stop = KTI(items=[KeyTimeInstance(90, 'Eng (1) Stop'),])
        max_dur = Eng_N1MaxDurationUnder60PercentAfterTouchdown()
        max_dur.derive(eng, eng, None, None, tdwn, eng_stop)
        self.assertEqual(max_dur[0].index, 60) # starts at drop below 60
        self.assertEqual(max_dur[0].value, 30) # stops at 90
        self.assertTrue('Eng (1)' in max_dur[0].name)
        # Eng (2) should not be in the results as it did not have an Eng Stop KTI
        ##self.assertTrue('Eng (2)' in max_dur[1].name)
        self.assertEqual(len(max_dur), 1)

"""
class TestFlapAtGearSelectedDown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FlapAtGearSelectedDown
        self.operational_combinations = [('Flap', 'Gear Selected Down')]
        """


class TestFlapAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FlapAtLiftoff
        self.operational_combinations = [('Flap', 'Liftoff')]


class TestFlapAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FlapAtTouchdown
        self.operational_combinations = [('Flap', 'Touchdown')]


class TestFuelQtyAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FuelQtyAtLiftoff
        self.operational_combinations = [('Fuel Qty', 'Liftoff')]
        

class TestFuelQtyAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FuelQtyAtTouchdown
        self.operational_combinations = [('Fuel Qty', 'Touchdown')]

        
"""
class TestFuelQtyAirborneMin(unittest.TestCase, TestCreateKPVsWithinSlices):
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
                                            TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = ILSGlideslopeDeviationBelow1000FtMax
        self.operational_combinations = [('ILS Glideslope', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_below', (1000,), {})]
        """


class TestILSGlideslopeDeviation1000To150FtMax(unittest.TestCase):
    def test_ils_glide_1000_150_basic(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        
        testwave = np.ma.array(1.0 - np.cos(np.arange(0,6.3,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        
        gs_estab = buildsection('ILS Glideslope Established', 2,63)
        
        kpv = ILSGlideslopeDeviation1000To150FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].index, 36)
        self.assertAlmostEqual(kpv[0].value, 1.89675842)

    def test_ils_glide_1000_150_four_peaks(self):
        testline = np.ma.array((75 - np.arange(63))*25) # 1875 to 325 ft in 63 steps.
        alt_ph = Parameter('Altitude AAL For Flight Phases', testline)
        testwave = np.ma.array(-0.2-np.sin(np.arange(0,12.6,0.1)))
        ils_gs = Parameter('ILS Glideslope', testwave)
        gs_estab = buildsection('ILS Glideslope Established', 2,56)
        kpv = ILSGlideslopeDeviation1000To150FtMax()
        kpv.derive(ils_gs, alt_ph, gs_estab)
        # 'KeyPointValue', 'index' 'value' 'name'
        self.assertAlmostEqual(kpv[0].value, 0.79992326)

'''
class TestGroundSpeedOnGroundMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = GroundSpeedOnGroundMax
        self.operational_combinations = [('Groundspeed', 'On Ground')]
        self.function = max_value
'''

class TestGrossWeightAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight Smoothed', 'Liftoff')]


class TestGrossWeightAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
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
        expected = [KeyPointValue(index=4, value=7.5, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)
        
    def test_takeoff_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3])*-1.0)
        toff = buildsection('Takeoff', 2,6)
        kpv = HeadingAtTakeoff()
        kpv.derive(head, toff)
        expected = [KeyPointValue(index=4, value=360-7.5, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)

class TestHeadingAtLanding(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous', 'Landing')]
        opts = HeadingAtLanding.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Landing',5,15)
        head.array[13] = np.ma.masked
        kpv = HeadingAtLanding()
        kpv.derive(head, landing)
        expected = [KeyPointValue(index=10, value=6.0, name='Heading At Landing')]
        self.assertEqual(kpv, expected)


"""
class TestHeadingAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = HeadingAtLowPointOnApproach
        self.operational_combinations = [('Heading Continuous',
                                          'Approach And Landing Lowest')]
"""

"""
class TestHeightAtGoAroundMin(unittest.TestCase, TestCreateKPVsAtKTIs):
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

        
class TestLatitudeAtLanding(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LatitudeAtLanding
        self.operational_combinations = [('Latitude',
                                          'Touchdown')]

class TestLongitudeAtLanding(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LongitudeAtLanding
        self.operational_combinations = [('Longitude',
                                          'Touchdown')]


class TestLatitudeAtTakeoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LatitudeAtTakeoff
        self.operational_combinations = [('Latitude',
                                          'Liftoff')]

class TestLongitudeAtTakeoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LongitudeAtTakeoff
        self.operational_combinations = [('Longitude',
                                          'Liftoff')]


"""
class TestLatitudeAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LatitudeAtLowPointOnApproach
        self.operational_combinations = [('Latitude Smoothed',
                                          'Approach And Landing Lowest')]


class TestLongitudeAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LongitudeAtLowPointOnApproach
        self.operational_combinations = [('Longitude Smoothed',
                                          'Approach And Landing Lowest')]
"""

class TestILSLocalizerDeviation1500To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases',
                     'ILS Localizer Established')]
        opts = ILSLocalizerDeviation1500To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1500_1000_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        loc_est = buildsection('ILS Localizer Established', 30,115)
        kpv = ILSLocalizerDeviation1500To1000FtMax()
        kpv.derive(ils_loc, alt_ph, loc_est)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        
        
class TestILSLocalizerDeviation1000To150FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases',
                     'ILS Localizer Established')]
        opts = ILSLocalizerDeviation1000To150FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1000_150_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        loc_est = buildsection('ILS Localizer Established', 30,115)
        kpv = ILSLocalizerDeviation1000To150FtMax()
        kpv.derive(ils_loc, alt_ph, loc_est)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 57)
        self.assertEqual(kpv[1].index, 114)


class TestMachMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = MachMax
        self.operational_combinations = [('Mach', 'Airborne')]
        self.function = max_value


class TestPitch1000To100FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch1000To100FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (1000, 100,), {})]


class TestPitch1000To100FtMin(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch1000To100FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 100,), {})]


class TestPitch20FtToTouchdownMin(unittest.TestCase,
                                  TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch20FtToTouchdownMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (20, 0,), {})]


class TestPitch35To400FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch35To400FtMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
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


class TestPitch35To400FtMin(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch35To400FtMin
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (35, 400,), {})]


class TestPitch5FtToTouchdownMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = Pitch5FtToTouchdownMax
        self.operational_combinations = [('Pitch', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (5, 0,), {})]


class TestPitchAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = PitchAtLiftoff
        self.operational_combinations = [('Pitch', 'Liftoff')]


class TestPitchAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = PitchAtTouchdown
        self.operational_combinations = [('Pitch', 'Touchdown')]


"""
class TestPitchDuringFinalApproachMin(unittest.TestCase,
                                      TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = PitchDuringFinalApproachMin
        self.operational_combinations = [('Pitch', 'Final Approach')]
        self.function = min_value
        """


"""
class TestPitchDuringTakeoffMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = PitchDuringTakeoffMax
        self.operational_combinations = [('Pitch', 'Takeoff')]
        self.function = max_value
        """


"""
class TestPitchRate35To1500FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = PitchRate35To1500FtMax
        self.operational_combinations = [('Pitch Rate', 'Altitude AAL For Flight Phases')]
        self.function = max_value
        self.second_param_method_calls = [('slices_from_to', (35, 1500), {})]
        """


class TestPitchRateDuringTakeoffMax(unittest.TestCase,
                                    TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = PitchRateDuringTakeoffMax
        self.operational_combinations = [('Pitch Rate', 'Takeoff')]
        self.function = max_value


class TestPitchRateDuringTakeoffMin(unittest.TestCase):
    def test_derive(self):
        '''
        TODO
        '''
        self.assertTrue(False)


class TestRateOfDescent500FtToTouchdownMax(unittest.TestCase,
                                           TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RateOfDescent500FtToTouchdownMax
        self.operational_combinations = [('Rate Of Climb', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (500, 0), {})]


class TestRateOfDescent1000To500FtMax(unittest.TestCase,
                                      TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RateOfDescent1000To500FtMax
        self.operational_combinations = [('Rate Of Climb', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 500), {})]
        

"""
class TestRateOfDescent1000To50FtMax(unittest.TestCase,
                                      TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RateOfDescent1000To50FtMax
        self.operational_combinations = [('Rate Of Climb', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (1000, 50), {})]
        """


class TestRateOfDescent2000To1000FtMax(unittest.TestCase,
                                       TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RateOfDescent2000To1000FtMax
        self.operational_combinations = [('Rate Of Climb', 'Altitude AAL For Flight Phases')]
        self.function = min_value
        self.second_param_method_calls = [('slices_from_to', (2000, 1000), {})]


class TestRollAbove1000FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RollAbove1000FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1000,), {})]


"""
class TestRollAbove1500FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RollAbove1500FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_above', (1500,), {})]
        """


"""
class TestRollBelow20FtMax(unittest.TestCase, TestCreateKPVsWithinSlices):
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
                                    TestCreateKPVsWithinSlices):
    def setUp(self):
        self.node_class = RollBetween100And500FtMax
        self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
        self.function = max_abs_value
        self.second_param_method_calls = [('slices_between', (100, 500), {})]
        """


"""
class TestRollBetween500And1500FtMax(unittest.TestCase,
                                     TestCreateKPVsWithinSlices):
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
        param = BDUTerrain()
        param.derive(alt_aal, alt_radio, dtl)
        self.assertEqual(param, [KeyPointValue(name='BDU Terrain', index=1008, value=0.037668517049960347)])
        
class TestZeroFuelWeight(unittest.TestCase):
    def test_derive(self):
        fuel = P('Fuel Qty', np.ma.array([1,2,3,4]))
        weight = P('Gross Weight', np.ma.array([11,12,13,14]))
        zfw = ZeroFuelWeight()
        zfw.derive(fuel, weight)
        self.assertEqual(zfw[0].value, 10.0)
    