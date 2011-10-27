try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from datetime import datetime

from analysis.library import (calculate_timebase, create_phase_inside,
                              create_phase_outside, rate_of_change,
                              running_average, shift, slope,
                              straighten_headings)


class TestClock(unittest.TestCase):
    def test_calculate_timebase(self):
        # 6th second is the first valid datetime(2020,12,25,23,59,0)
        years = [None] * 6 + [2020] * 19  # 6 sec offset
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [00] * 15
        mins = [None] * 2 + [59] * 10 + [01] * 13
        secs = [None] * 1 + range(55, 60) + range(19)  # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        
        #>>> datetime(2020,12,25,00,01,19) - timedelta(seconds=25)
        #datetime.datetime(2020, 12, 25, 0, 0, 50)
        self.assertEqual(start_dt, datetime(2020, 12, 25, 0, 0, 54))
            

class TestPhaseMasking(unittest.TestCase):
    def test_phase_inside(self):
        #create_phase_inside()
        self.assertTrue(False)
        
    def test_phase_outside(self):
        #create_phase_outside()
        self.assertTrue(False)
      
        
class TestRunningAverage(unittest.TestCase):
    def test_running_average(self):
        #running_average()
        self.assertTrue(False)
        

    
class TestShift(unittest.TestCase):
    def test_shift(self):
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
                
        first = DumParam()
        first.hz = 8
        first.fdr_offset = 1.6
        first.data = np.ma.array(range(10))
        
        second = DumParam()
        second.hz = 4
        second.fdr_offset = 1.4
        second.data = np.ma.array(range(10))
        
        result = shift(first, second)
        self.assertEqual(result.mask, [False]*10)
        
        
class TestSlope(unittest.TestCase):
    def test_slope(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
        sloped = slope(array, 2)

        expected_results = np.ma.array(
            data=[None, None, 0.25, 0.0, 0.0, 0.0, -0.25, 0.5, None, None],
            mask=[True, True, False, False, False,
                  False, False, False, True, True],
            fill_value=1e+20)
        # tests repr are equal - more difinitive tests at lower granularity
        # would be beneficial here.
        self.assertEqual(sloped.__repr__(), expected_results.__repr__())
        
        
class TestStraightenHeadings(unittest.TestCase):
    def test_straight_headings(self):
        data = [35.5,
                29.5,
                11.3,
                0.0,
                348.4,
                336.8,
                358.9,
                2.5,
                8.1,
                14.4]
        expected = [35.5,
                    29.5,
                    11.3,
                    0.0,
                    -11.6,
                    -23.2,
                    -1.1,
                    2.5,
                    8.1,
                    14.4]
        for index, val in enumerate(straighten_headings(data)):
            self.assertEqual(
                '%.2f' % val,
                '%.2f' % expected[index],
                msg="Failed at %s == %s at %s" % (val, expected[index], index)
            )
       
