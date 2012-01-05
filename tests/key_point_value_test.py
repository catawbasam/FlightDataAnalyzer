import unittest
from mock import Mock, patch
import numpy as np
import sys
debug = sys.gettrace() is not None

from analysis.node import (KeyTimeInstance, KTI, KeyPointValue, Parameter, P,
                           Section, S)
from analysis.key_point_values import (AccelerationNormalMax,
                                       Airspeed1000To500FtMax,
                                       AirspeedAtTouchdown,
                                       AirspeedMax,
                                       AirspeedWithGearSelectedDownMax,
                                       AltitudeAtTouchdown,
                                       AutopilotEngaged1AtLiftoff,
                                       AutopilotEngaged1AtTouchdown,
                                       AutopilotEngaged2AtLiftoff,
                                       AutopilotEngaged2AtTouchdown,
                                       EngEGTMax,
                                       EngN1Max,
                                       EngN2Max,
                                       EngOilTempMax,
                                       EngVibN1Max,
                                       EngVibN2Max,
                                       HeadingAtTakeoff,
                                       Eng_N1MaxDurationUnder60PercentAfterTouchdown,
                                       FlapAtLiftoff,
                                       FuelQtyAtLiftoff,
                                       FuelQtyAtTouchdown,
                                       GlideslopeDeviation1500To1000FtMax,
                                       GlideslopeDeviation1000To150FtMax,
                                       GrossWeightAtLiftoff,
                                       GrossWeightAtTouchdown,
                                       ILSFrequencyOnApproach,
                                       HeadingAtLanding,
                                       HeadingAtLowPointOnApproach,
                                       LatitudeAtLanding,
                                       LatitudeAtLowPointOnApproach,
                                       LongitudeAtLanding,
                                       LongitudeAtLowPointOnApproach,
                                       LocalizerDeviation1500To1000FtMax,
                                       LocalizerDeviation1000To150FtMax,
                                       Pitch35To400FtMax,
                                       PitchAtLiftoff,
                                       RollBelow20FtMax)

import sys
debug = sys.gettrace() is not None


class TestCreateKPVsAtKTIs(object):
    '''
    Example of subclass inheriting tests:
    
class TestAltitudeAtLiftoff(unittest.TestCase, TestKPV):
    def setUp(self):
        self.node_class = AltitudeAtLiftoff
        self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         self.operational_combinations)
    
    def test_derive(self):
        if debug:
            mock1, mock2 = Mock(), Mock()
            mock1.array = Mock()
            node = self.node_class()
            node.create_kpvs_at_ktis = Mock()
            node.derive(mock1, mock2)
            self.assertEqual(node.create_kpvs_at_ktis.call_args,
                         ((mock1.array, mock2), {}))


class TestAccelerationNormalMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(AccelerationNormalMax.get_operational_combinations(),
                         [('Normal Acceleration',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        acc_norm_max = AccelerationNormalMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        acc_norm_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(acc_norm_max,
                         [KeyPointValue(index=index, value=value,
                                        name=acc_norm_max.name)])


class TestAirspeedAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AirspeedAtTouchdown
        self.operational_combinations = [('Airspeed', 'Touchdown')]


class TestAirspeed1000To500FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Altitude AAL For Flight Phases')]
        opts = Airspeed1000To500FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
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


class TestAirspeedMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Airborne')]
        opts = AirspeedMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_airspeed_max_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-100))+100
        spd = Parameter('Airspeed', np.ma.array(testwave))
        waves=np.ma.clump_unmasked(np.ma.masked_less(testwave,80))
        airs=[]
        for wave in waves:
            airs.append(Section('Airborne',wave))
        ##from analysis.node import FlightPhaseNode
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


class TestAirspeed1000To500FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Altitude AAL For Flight Phases')]
        opts = Airspeed1000To500FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
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


class TestAltitudeAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AltitudeAtTouchdown
        self.operational_combinations = [('Altitude STD', 'Touchdown')]


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


class TestEngEGTMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngEGTMax.get_operational_combinations(),
                         [('Eng (*) EGT Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_egt_max = EngEGTMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_egt_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(eng_egt_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_egt_max.name)])


class TestEngN1Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngN1Max.get_operational_combinations(),
                         [('Eng (*) N1 Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_n1_max = EngN1Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_n1_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(eng_n1_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_n1_max.name)])


class TestEngN2Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngN2Max.get_operational_combinations(),
                         [('Eng (*) N2 Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_n2_max = EngN2Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_n2_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(eng_n2_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_n2_max.name)])


class TestEngOilTempMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngOilTempMax.get_operational_combinations(),
                         [('Eng (*) Oil Temp Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_oil_temp_max = EngOilTempMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_oil_temp_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(eng_oil_temp_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_oil_temp_max.name)])


class TestEngVibN1Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngVibN1Max.get_operational_combinations(),
                         [('Eng (*) Vib N1 Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_vib_n1_max = EngVibN1Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_vib_n1_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
        self.assertEqual(eng_vib_n1_max,
                         [KeyPointValue(index=index, value=value,
                                        name=eng_vib_n1_max.name)])


class TestEngVibN2Max(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(EngVibN2Max.get_operational_combinations(),
                         [('Eng (*) Vib N2 Max',)])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        eng_vib_n2_max = EngVibN2Max()
        index, value = 10, 30
        max_value.return_value = index, value
        param = Mock()
        param.array = Mock()
        eng_vib_n2_max.derive(param)
        self.assertEqual(max_value.call_args, ((param.array,), {}))
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


class TestFlapAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FlapAtLiftoff
        self.operational_combinations = [('Flap', 'Liftoff')]


class TestFuelQtyAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FuelQtyAtLiftoff
        self.operational_combinations = [('Fuel Qty', 'Liftoff')]
        

class TestFuelQtyAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = FuelQtyAtTouchdown
        self.operational_combinations = [('Fuel Qty', 'Touchdown')]


class TestGlideslopeDeviation1500To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Glideslope','Altitude AAL For Flight Phases')]
        opts = GlideslopeDeviation1500To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_glide_1500_1000_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_gs = Parameter('ILS Glideslope', np.ma.array(testline))
        kpv = GlideslopeDeviation1500To1000FtMax()
        kpv.derive(ils_gs, alt_ph)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        
        
class TestGlideslopeDeviation1000To150FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Glideslope','Altitude AAL For Flight Phases')]
        opts = GlideslopeDeviation1000To150FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_glide_1000_150_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_gs = Parameter('ILS Glideslope', np.ma.array(testline))
        kpv = GlideslopeDeviation1000To150FtMax()
        kpv.derive(ils_gs, alt_ph)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 57)
        self.assertEqual(kpv[1].index, 120)


class TestGrossWeightAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = GrossWeightAtLiftoff
        self.operational_combinations = [('Gross Weight', 'Liftoff')]


class TestGrossWeightAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = GrossWeightAtTouchdown
        self.operational_combinations = [('Gross Weight', 'Touchdown')]


class TestHeadingAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff Peak Acceleration','Heading Continuous')]
        opts = HeadingAtTakeoff.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_takeoff_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = KTI('Takeoff Peak Acceleration',
                 items=[KeyTimeInstance(index=3,
                                        name='Takeoff Peak Acceleration')])        
        kpv = HeadingAtTakeoff()
        kpv.derive(acc, head)
        expected = [KeyPointValue(index=3, value=4.5, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)
        #############################################
        ## I KNOW THIS FAILS
        ## TODO: write a version for takeoff and landing that is more robust
        #############################################
        
    def test_takeoff_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([-1,-2,-4,-7,-9,-8,-6,-3]))
        acc = KTI('Takeoff Peak Acceleration',
                 items=[KeyTimeInstance(index=4,
                                        name='Takeoff Peak Acceleration')])        
        kpv = HeadingAtTakeoff()
        kpv.derive(acc, head)
        expected = [KeyPointValue(index=4, value=357, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)


class TestHeadingAtLanding(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing Peak Deceleration','Heading Continuous')]
        opts = HeadingAtLanding.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = [KeyTimeInstance(index=5, name='Landing Peak Deceleration'),
                   KeyTimeInstance(index=16, name='Landing Peak Deceleration')]
        head.array[13] = np.ma.masked
        kpv = HeadingAtLanding()
        kpv.derive(landing, head)
        expected = [KeyPointValue(index=5, value=4.5, name='Heading At Landing'),
                    KeyPointValue(index=16, value=359.0, name='Heading At Landing')]
        self.assertEqual(kpv, expected)


class TestHeadingAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = HeadingAtLowPointOnApproach
        self.operational_combinations = [('Heading Continuous',
                                          'Approach And Landing Lowest')]

        
class TestILSFrequencyOnApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer Established', 
                     'Approach And Landing Lowest Point', 'ILS Frequency')]
        opts = ILSFrequencyOnApproach.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ILS_frequency_on_approach_basic(self):
        # Let's give this a really hard time with alternate samples invalid and
        # the final signal only tuned just at the end of the data.
        frq = P('ILS Frequency',np.ma.array([108.5]*6+[114.05]*4))
        frq.array[0:10:2] = np.ma.masked
        ils = S('ILS Localizer Established', items=[Section('ILS Localizer Established', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', 
                  items=[KeyTimeInstance(index=8, 
                                         name='Approach And Landing Lowest Point')])
        kpv = ILSFrequencyOnApproach()
        kpv.derive(ils, low, frq)
        expected = [KeyPointValue(index=2, value=108.5, 
                                  name='ILS Frequency On Approach')]
        self.assertEqual(kpv, expected)

        
class TestLatitudeAtLanding(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LatitudeAtLanding
        self.operational_combinations = [('Latitude',
                                          'Landing Peak Deceleration')]

class TestLongitudeAtLanding(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LongitudeAtLanding
        self.operational_combinations = [('Longitude',
                                          'Landing Peak Deceleration')]


class TestLatitudeAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LatitudeAtLowPointOnApproach
        self.operational_combinations = [('Latitude',
                                          'Approach And Landing Lowest')]


class TestLongitudeAtLowPointOnApproach(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = LongitudeAtLowPointOnApproach
        self.operational_combinations = [('Longitude',
                                          'Approach And Landing Lowest')]


class TestLocalizerDeviation1500To1000FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases')]
        opts = LocalizerDeviation1500To1000FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1500_1000_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        kpv = LocalizerDeviation1500To1000FtMax()
        kpv.derive(ils_loc, alt_ph)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 47)
        self.assertEqual(kpv[1].index, 109)
        
        
class TestLocalizerDeviation1000To150FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer','Altitude AAL For Flight Phases')]
        opts = LocalizerDeviation1000To150FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ils_loc_1000_150_basic(self):
        testline = np.arange(0,12.6,0.1)
        testwave = (np.cos(testline)*(-1000))+1000
        alt_ph = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        ils_loc = Parameter('ILS Localizer', np.ma.array(testline))
        kpv = LocalizerDeviation1000To150FtMax()
        kpv.derive(ils_loc, alt_ph)
        # 'KeyPointValue', 'index value name'
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 57)
        self.assertEqual(kpv[1].index, 120)
        
        
class TestPitch35To400FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch','Altitude AAL For Flight Phases')]
        opts = Pitch35To400FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
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


class TestPitchAtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = PitchAtLiftoff
        self.operational_combinations = [('Pitch', 'Liftoff')]


class TestRollBelow20FtMax(unittest.TestCase):
    def test_can_operate(self, eng=P()):
        self.assertEqual(RollBelow20FtMax.get_operational_combinations(),
                         [('Roll', 'Altitude AAL For Flight Phases')])
    
    @patch('analysis.key_point_values.max_value')
    def test_derive(self, max_value):
        roll_below_20ft_max = RollBelow20FtMax()
        index, value = 10, 30
        max_value.return_value = index, value
        param1 = Mock()
        param1.array = Mock()
        param2 = Mock()
        param2.array = Mock()
        param2.slices_below = Mock()
        param2.slices_below.return_value = [slice(0, 10)]
        roll_below_20ft_max.derive(param1, param2)
        self.assertEqual(max_value.call_args, ((param1.array,),
                                               {'_slice':slice(0,10)}))
        self.assertEqual(roll_below_20ft_max,
                         [KeyPointValue(index=index, value=value,
                                        name=roll_below_20ft_max.name)])
