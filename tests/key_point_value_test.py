import unittest
import numpy as np

from analysis.library import rate_of_change
from analysis.plot_flight import plot_parameter
from analysis.node import A, KPV, KeyTimeInstance, KTI, KeyPointValue, Parameter, P, Section, S
#from analysis.flight_phase import ()

from analysis.key_point_values import (GlideslopeDeviation1500To1000FtMax,
                                       GlideslopeDeviation1000To150FtMax,
                                       LocalizerDeviation1500To1000FtMax,
                                       LocalizerDeviation1000To150FtMax,
                                       Pitch35To400FtMax
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