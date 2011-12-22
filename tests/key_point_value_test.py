import unittest
import numpy as np

from analysis.library import rate_of_change
from analysis.plot_flight import plot_parameter
from analysis.node import A, KPV, KeyTimeInstance, KTI, KeyPointValue, Parameter, P, Section, S
from analysis.key_point_values import (Airspeed1000To500FtMax,
                                       AirspeedMax,
                                       AltitudeAtLiftoff,
                                       AltitudeAtTouchdown,
                                       AutopilotEngaged1AtLiftoff,
                                       AutopilotEngaged1AtTouchdown,
                                       AutopilotEngaged2AtLiftoff,
                                       AutopilotEngaged2AtTouchdown,
                                       HeadingAtTakeoff,
                                       FuelQtyAtLiftoff,
                                       FuelQtyAtTouchdown,
                                       GlideslopeDeviation1500To1000FtMax,
                                       GlideslopeDeviation1000To150FtMax,
                                       GrossWeightAtLiftoff,
                                       GrossWeightAtTouchdown,
                                       ILSFrequencyOnApproach,
                                       HeadingAtLanding,
                                       LatitudeAtLanding,
                                       LongitudeAtLanding,
                                       LocalizerDeviation1500To1000FtMax,
                                       LocalizerDeviation1000To150FtMax,
                                       Pitch35To400FtMax,
                                       PitchAtLiftoff)

import sys
debug = sys.gettrace() is not None

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


class TestAltitudeAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeAtLiftoff.get_operational_combinations(),
                         [('Liftoff', 'Altitude STD')])
    
    def test_derive(self):
        node = AltitudeAtLiftoff()
        altitude_std = P('Altitude STD', array=np.ma.masked_array([2,4,6]))
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(1, 'a')])
        node.derive(liftoff, altitude_std)
        self.assertEqual(node,
                         [KeyPointValue(1, 4, 'Altitude At Liftoff')])


class TestAltitudeAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeAtTouchdown.get_operational_combinations(),
                         [('Touchdown', 'Altitude STD')])
    
    def test_derive(self):
        node = AltitudeAtTouchdown()
        altitude_std = P('Altitude STD', array=np.ma.masked_array([2,4,6]))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(1, 'a')])
        node.derive(touchdown, altitude_std)
        self.assertEqual(node,
                         [KeyPointValue(1, 4, 'Altitude At Touchdown')])


class TestAutopilotEngaged1AtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AutopilotEngaged1AtLiftoff.get_operational_combinations(),
                         [('Autopilot Engaged 1', 'Liftoff')])
    
    def test_derive(self):
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(2, 'a')])
        autopilot = P('Autopilot Engaged 1', array=np.ma.array([0,2,4,6,8]))
        autopilot_at_liftoff = AutopilotEngaged1AtLiftoff()
        autopilot_at_liftoff.derive(autopilot, liftoff)
        self.assertEqual(autopilot_at_liftoff,
                         [KeyPointValue(2, 4, 'Autopilot Engaged 1 At Liftoff')])


class TestAutopilotEngaged2AtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AutopilotEngaged2AtLiftoff.get_operational_combinations(),
                         [('Autopilot Engaged 2', 'Liftoff')])
    
    def test_derive(self):
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(2, 'a')])
        autopilot = P('Autopilot Engaged 2', array=np.ma.array([0,2,4,6,8]))
        autopilot_at_liftoff = AutopilotEngaged2AtLiftoff()
        autopilot_at_liftoff.derive(autopilot, liftoff)
        self.assertEqual(autopilot_at_liftoff,
                         [KeyPointValue(2, 4, 'Autopilot Engaged 2 At Liftoff')])


class TestAutopilotEngaged1AtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AutopilotEngaged1AtTouchdown.get_operational_combinations(),
                         [('Autopilot Engaged 1', 'Touchdown')])
    
    def test_derive(self):
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(2, 'a')])
        autopilot = P('Autopilot Engaged 1', array=np.ma.array([0,2,4,6,8]))
        autopilot_at_touchdown = AutopilotEngaged1AtTouchdown()
        autopilot_at_touchdown.derive(autopilot, touchdown)
        self.assertEqual(autopilot_at_touchdown,
                         [KeyPointValue(2, 4, 'Autopilot Engaged 1 At Touchdown')])


class TestAutopilotEngaged2AtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AutopilotEngaged2AtTouchdown.get_operational_combinations(),
                         [('Autopilot Engaged 2', 'Touchdown')])
    
    def test_derive(self):
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(2, 'a')])
        autopilot = P('Autopilot Engaged 2', array=np.ma.array([0,2,4,6,8]))
        autopilot_at_touchdown = AutopilotEngaged2AtTouchdown()
        autopilot_at_touchdown.derive(autopilot, touchdown)
        self.assertEqual(autopilot_at_touchdown,
                         [KeyPointValue(2, 4, 'Autopilot Engaged 2 At Touchdown')])


class TestHeadingAtTakeoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Heading Continuous', 'Acceleration Forwards For Flight Phases')]
        opts = HeadingAtTakeoff.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_takeoff_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,.2,.3,.2,.1,0,0]))
        toff_ph = [Section('Takeoff',slice(2,5,None))]
        kpv = HeadingAtTakeoff()
        kpv.derive(toff_ph, head, acc)
        expected = [KeyPointValue(index=3, value=7.0, name='Heading At Takeoff')]
        self.assertEqual(kpv, expected)
        
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
        head = P('Heading Continuous',np.ma.array([0,2,4,7,0,-3,-7,-93]))
        landing = [KeyTimeInstance(index=2, name='Landing Peak Deceleration'),
                   KeyTimeInstance(index=6, name='Landing Peak Deceleration')]
        kpv = HeadingAtLanding()
        kpv.derive(landing, head)
        expected = [KeyPointValue(index=2, value=4.0, name='Heading At Landing'),
                    KeyPointValue(index=6, value=353.0, name='Heading At Landing')]
        self.assertEqual(kpv, expected)


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


class TestFuelQtyAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FuelQtyAtLiftoff.get_operational_combinations(),
                         [('Fuel Qty', 'Liftoff')])
    
    def test_derive(self):
        node = FuelQtyAtLiftoff()
        fuel_qty = P('Fuel Qty', array=np.ma.masked_array([2,4,6]))
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(1, 'a')])
        node.derive(fuel_qty, liftoff)
        self.assertEqual(node, [KeyPointValue(1, 4, 'Fuel Qty At Liftoff')])


class TestFuelQtyAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FuelQtyAtTouchdown.get_operational_combinations(),
                         [('Fuel Qty', 'Touchdown')])
    
    def test_derive(self):
        node = FuelQtyAtTouchdown()
        fuel_qty = P('Fuel Qty', array=np.ma.masked_array([2,4,6]))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(1, 'a')])
        node.derive(fuel_qty, touchdown)
        self.assertEqual(node,
                         [KeyPointValue(1, 4, 'Fuel Qty At Touchdown')])


class TestGrossWeightAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GrossWeightAtLiftoff.get_operational_combinations(),
                         [('Gross Weight', 'Liftoff')])
    
    def test_derive(self):
        node = GrossWeightAtLiftoff()
        gross_weight = P('Gross Weight', array=np.ma.masked_array([2,4,6]))
        liftoff = KTI('Liftoff', items=[KeyTimeInstance(1, 'a')])
        node.derive(gross_weight, liftoff)
        self.assertEqual(node, [KeyPointValue(1, 4, 'Gross Weight At Liftoff')])


class TestGrossWeightAtTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GrossWeightAtTouchdown.get_operational_combinations(),
                         [('Gross Weight', 'Touchdown')])
    
    def test_derive(self):
        node = GrossWeightAtTouchdown()
        gross_weight = P('Gross Weight', array=np.ma.masked_array([2,4,6]))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(1, 'a')])
        node.derive(gross_weight, touchdown)
        self.assertEqual(node,
                         [KeyPointValue(1, 4, 'Gross Weight At Touchdown')])

        
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

        
class TestLatitudeAtLanding(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing Peak Deceleration','Latitude')]
        opts = LatitudeAtLanding.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        data = P('Longitude',np.ma.array([0,66,99]))
        landing = [KeyTimeInstance(1, 'Landing Peak Deceleration')]
        kpv = LatitudeAtLanding()
        kpv.derive(landing, data)
        expected = [KeyPointValue(1, 66.0, 'Latitude At Landing')]
        self.assertEqual(kpv, expected)


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


class TestLongitudeAtLanding(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing Peak Deceleration','Longitude')]
        opts = LongitudeAtLanding.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        data = P('Longitude',np.ma.array([0,66,77]))
        landing = [KeyTimeInstance(2, 'Landing Peak Deceleration')]
        kpv = LongitudeAtLanding()
        kpv.derive(landing, data)
        expected = [KeyPointValue(2, 77.0, 'Longitude At Landing')]
        self.assertEqual(kpv, expected)
        
        
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


class TestPitchAtLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(PitchAtLiftoff.get_operational_combinations(),
                         [('Liftoff', 'Pitch')])
        
    def test_pitch_35_400_basic(self):
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(2, 'a'),
                                         KeyTimeInstance(4, 'a')])
        pitch = P('Autopilot Engaged 2', array=np.ma.array([0,2,4,6,8]))
        node = PitchAtLiftoff()
        node.derive(pitch, liftoffs)
        self.assertEqual(node,
                         [KeyPointValue(2, 4, 'Pitch At Liftoff'),
                          KeyPointValue(4, 8, 'Pitch At Liftoff')])

