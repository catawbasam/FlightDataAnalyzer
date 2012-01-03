import unittest
from mock import Mock
import numpy as np

from analysis.node import KeyTimeInstance, KTI, KeyPointValue, Parameter, P, Section, S
from analysis.key_point_values import (Airspeed1000To500FtMax,
                                       AirspeedAtTouchdown,
                                       AirspeedMax,
                                       AirspeedWithGearSelectedDownMax,
                                       AltitudeAtTouchdown,
                                       AutopilotEngaged1AtLiftoff,
                                       AutopilotEngaged1AtTouchdown,
                                       AutopilotEngaged2AtLiftoff,
                                       AutopilotEngaged2AtTouchdown,
                                       HeadingAtTakeoff,
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
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        self.assertEqual(node.create_kpvs_at_ktis.call_args,
                         ((mock1.array, mock2), {}))


class TestAirspeedAtTouchdown(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AirspeedAtTouchdown
        self.operational_combinations = [('Airspeed', 'Touchdown')]


class TestAirspeed1000To500FtMax(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Altitude AAL For Flight Phases')]
        opts = Airspeed1000To500FtMax.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_airspeed_1000_150_basic(self):
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
        
    def test_airspeed_1000_150_basic(self):
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


class TestAutopilotEngaged1AtLiftoff(unittest.TestCase):
    def setUp(self):
        self.node_class = AutopilotEngaged1AtLiftoff
        self.operational_combinations = [('Autopilot Engaged 1', 'Liftoff')]


class TestAutopilotEngaged1AtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AutopilotEngaged1AtTouchdown
        self.operational_combinations = [('Autopilot Engaged 1', 'Touchdown')]


class TestAutopilotEngaged2AtLiftoff(unittest.TestCase, TestCreateKPVsAtKTIs):
    def setUp(self):
        self.node_class = AutopilotEngaged2AtLiftoff
        self.operational_combinations = [('Autopilot Engaged 2', 'Liftoff')]


class TestAutopilotEngaged2AtTouchdown(unittest.TestCase):
    def setUp(self):
        self.node_class = AutopilotEngaged2AtTouchdown
        self.operational_combinations = [('Autopilot Engaged 2', 'Touchdown')]


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
        expected = [KeyPointValue(index=3, value=7.0, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)
        #############################################
        ## I KNOWW THIS FAILS
        ## TODO: write a version for takeoff and landing that is more robust
        #############################################
        
    def test_takeoff_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([-1,-2,-4,-7,-9,-8,-6,-3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,.1,.2,.35,.2,.1,0]))
        toff_ph = [Section('Takeoff',slice(2,6,None))]
        kpv = HeadingAtTakeoff()
        kpv.derive(toff_ph, head, acc)
        expected = [KeyPointValue(index=4, value=351, name='Heading At Takeoff')]
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
    def test_can_operate(self):
        self.assertEqual(RollBelow20FtMax.get_operational_combinations(),
                         [])
    
    def test_derive(self):
        roll = P('Roll', array=np.ma.arange(20))
        desc_alts = KTI('Altitude When Descending', items=2000)
        climb_alts