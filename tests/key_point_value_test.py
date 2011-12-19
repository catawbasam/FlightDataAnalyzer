import unittest
import numpy as np

from analysis.library import rate_of_change
from analysis.plot_flight import plot_parameter
from analysis.node import A, KPV, KeyTimeInstance, KTI, KeyPointValue, Parameter, P, Section, S
#from analysis.flight_phase import ()

from analysis.key_point_values import (GlideslopeDeviation1500To1000FtMax,
                                       GlideslopeDeviation1000To150FtMax,
                                       ILSFrequencyInApproach,
                                       LandingHeading,
                                       LocalizerDeviation1500To1000FtMax,
                                       LocalizerDeviation1000To150FtMax,
                                       Pitch35To400FtMax,
                                       TakeoffHeading
                                         )

import sys
debug = sys.gettrace() is not None

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
        
class TestILSFrequencyInApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Approach', 'ILS Frequency')]
        opts = ILSFrequencyInApproach.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_ILS_frequency_in_approach_basic(self):
        # Let's give this a really hard time with alternate samples invalid and
        # the final signal only tuned just at the end of the data.
        frq = P('ILS Frequency',np.ma.array([108.5]*6+[114.05]*4))
        frq.array[0:10:2] = np.ma.masked
        app = [Section('Approach And Landing',slice(2,10,None))]
        kpv = ILSFrequencyInApproach()
        kpv.derive(app,frq)
        expected = [KeyPointValue(index=9, value=114.05, 
                                  name='Ils Frequency In Approach')]
        self.assertEqual(kpv, expected)


class TestLandingHeading(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing','Heading Continuous', 'Acceleration Forwards For Flight Phases')]
        opts = LandingHeading.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,-.1,-.1,-.2,-.1,0,0]))
        landing = [Section('Landing',slice(2,5,None))]
        kpv = LandingHeading()
        kpv.derive(landing, head, acc)
        expected = [KeyPointValue(index=4, value=9.0, name='Landing Heading')]
        self.assertEqual(kpv, expected)
        
    def test_landing_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([0,-2,-4,-7,-9,-8,-6,-3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,-.1,-.2,-.2,-.1,0,0]))
        landing = [Section('Landing',slice(2,5,None))]
        kpv = LandingHeading()
        kpv.derive(landing, head, acc)
        expected = [KeyPointValue(index=3, value=353, name='Landing Heading')]
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
        
        
class TestLocaliserDeviation1000To150FtMax(unittest.TestCase):
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
        
        
class TestTakeoffHeading(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Heading Continuous', 'Acceleration Forwards For Flight Phases')]
        opts = TakeoffHeading.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_takeoff_heading_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,.2,.3,.2,.1,0,0]))
        toff_ph = [Section('Takeoff',slice(2,5,None))]
        kpv = TakeoffHeading()
        kpv.derive(toff_ph, head, acc)
        expected = [KeyPointValue(index=3, value=7.0, name='Takeoff Heading')]
        self.assertEqual(kpv, expected)
        
    def test_takeoff_heading_modulus(self):
        head = P('Heading Continuous',np.ma.array([-1,-2,-4,-7,-9,-8,-6,-3]))
        acc = P('Acceleration Forwards For Flight Phases',np.ma.array([0,0,.1,.2,.35,.2,.1,0]))
        toff_ph = [Section('Takeoff',slice(2,6,None))]
        kpv = TakeoffHeading()
        kpv.derive(toff_ph, head, acc)
        expected = [KeyPointValue(index=4, value=351, name='Takeoff Heading')]
        self.assertEqual(kpv, expected)
