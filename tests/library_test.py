try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from datetime import datetime

from analysis.library import (align, calculate_timebase, create_phase_inside,
                              create_phase_outside, first_order_lag,
                              hysteresis, powerset, 
                              rate_of_change, straighten_headings)


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
            
            
class TestPowerset(unittest.TestCase):
    def test_powerset(self):
        deps = ['aaa',  'bbb', 'ccc']
        res = list(powerset(deps))
        expected = [(),
                    ('aaa',),
                    ('bbb',), 
                    ('ccc',), 
                    ('aaa', 'bbb'),
                    ('aaa', 'ccc'),
                    ('bbb', 'ccc'),
                    ('aaa', 'bbb', 'ccc')]
        self.assertEqual(res, expected)

class TestPhaseMasking(unittest.TestCase):
    def test_phase_inside(self):
        #create_phase_inside()
        self.assertTrue(False)
        
    def test_phase_outside(self):
        #create_phase_outside()
        self.assertTrue(False)
      
        
'''
Running average superceded, so test no longer required.
class TestRunningAverage(unittest.TestCase):
    def test_running_average(self):
        #running_average()
        self.assertTrue(False)
'''

    
class TestAlign(unittest.TestCase):
    def test_align(self):
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = None
                self.data = []
                
        first = DumParam()
        first.hz = 4
        first.fdr_offset = 0.1
        first.data = np.ma.array(range(8))
        
        second = DumParam()
        second.hz = 4
        second.fdr_offset = 0.2
        second.data = np.ma.array(range(8))
        
        result = align(first, second)
        np.testing.assert_array_equal(result.data, [0, 0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result.mask, [True,False,False,False
                                                    ,False,False,False,False])
        
    def test_align_same_hz_delayed(self):
        # Both arrays at 1Hz, master behind slave in time
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.array([0,1,2,3],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 1
        master.fdr_offset = 0.5
        slave = DumParam()
        slave.data = np.ma.array([10,11,12,13],dtype=float)
        slave.hz = 1
        slave.fdr_offset = 0.2
        result = align(master, slave)
        np.testing.assert_array_almost_equal(result.data, [10.3,11.3,12.3,0])
        np.testing.assert_array_equal(result.mask, [False,False,False,True])
        
    def test_align_same_hz_advanced(self):
        # Both arrays at 1Hz, master ahead of slave in time
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.array([0,1,2,3],dtype=float)
        master.hz = 1
        master.fdr_offset = 0.2
        slave = DumParam()
        slave.data = np.ma.array([10,11,12,13],dtype=float)
        slave.hz = 1
        slave.fdr_offset = 0.5
        result = align(master, slave)
        np.testing.assert_array_almost_equal(result.data, [0,10.7,11.7,12.7])
        np.testing.assert_array_equal(result.mask, [True,False,False,False])
        
    def test_align_increasing_hz_delayed(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.array([0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 4
        master.fdr_offset = 0.15
        slave = DumParam()
        slave.data = np.ma.array([10,11,12,13],dtype=float)
        slave.hz = 2
        slave.fdr_offset = 0.1
        result = align(master, slave)
        np.testing.assert_array_almost_equal(result.data, [10.1,10.6,11.1,11.6,
                                                           12.1,12.6,0.0,0.0])
        np.testing.assert_array_equal(result.mask, [False,False,False,False,
                                                    False,False,True,True])
        
    def test_align_increasing_hz_advanced(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 8
        master.fdr_offset = 0.1
        slave = DumParam()
        slave.data = np.ma.array([10,11,12,13],dtype=float)
        slave.hz = 2
        slave.fdr_offset = 0.15
        result = align(master, slave)
        np.testing.assert_array_almost_equal(result.data, [0.,10.15,10.4,10.65,
                                                           10.9,11.15,11.4,11.65,
                                                           11.9,12.15,12.4,12.65,
                                                           12.9, 0.  , 0. , 0.])
        np.testing.assert_array_equal(result.mask, [True,False,False,False,
                                                    False,False,False,False,
                                                    False,False,False,False,
                                                    False,True,True,True])
        
    def test_align_increasing_hz_extreme(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 8
        master.fdr_offset = 0.1
        slave = DumParam()
        slave.data = np.ma.array([10,11],dtype=float)
        slave.hz = 1
        slave.fdr_offset = 0.95
        result = align(master, slave)
        np.testing.assert_array_almost_equal(result.data,[ 0.  , 0.   , 0. , 0.   ,
                                                           0.  , 0.   , 0. ,10.025,
                                                          10.15,10.275,10.4,10.525,
                                                          10.65,10.775,10.9, 0.   ])
        np.testing.assert_array_equal(result.mask, [True,True,True,True,
                                                    True,True,True,False,
                                                    False,False,False,False,
                                                    False,False,False,True])
        
class TestRateOfChange(unittest.TestCase):
    '''
    must be a way to get the array defined once like this...
    def setUp(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
    ...but I can't get it to work
    '''
        
    def test_rate_of_change(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
        sloped = rate_of_change(array, 2)
        expected_results = np.ma.array(
            data=[None, None, 0.25, 0.0, 0.0, 0.0, -0.25, 0.5, None, None],
            mask=[True, True, False, False, False,
                  False, False, False, True, True],
            fill_value=1e+20)
        ## tests repr are equal - more difinitive tests at lower granularity
        ## would be beneficial here.
        #self.assertEqual(sloped.__repr__(), expected_results.__repr__())
        np.testing.assert_array_equal(sloped, expected_results)
        
    def test_rate_of_change_half_width_too_big(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
        self.assertRaises(ValueError, rate_of_change, array, 25)
        
    def test_rate_of_change_half_width_zero(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
        self.assertRaises(ValueError, rate_of_change, array, 0)
        
    def test_rate_of_change_half_width_negative(self):
        array = np.ma.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
        self.assertRaises(ValueError, rate_of_change, array, -2)
        
        
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


class TestHysteresis(unittest.TestCase):
    def test_hysteresis(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        data[4] = np.ma.masked
        result = hysteresis(data,2)
        np.testing.assert_array_equal(result.data,[0,0,1,1,1,0,4,5,6,1])
        np.testing.assert_array_equal(result.mask,[0,0,0,0,1,0,0,0,0,0])

    def test_hysteresis_change_of_threshold(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        result = hysteresis(data,1)
        np.testing.assert_array_equal(result.data,[0,0.5,1.5,1.5,0.5,-0.5,4.5,5.5,6.5,0.5])
        
class TestFirstOrderLag(unittest.TestCase):
    def test_firstorderlag_decay(self):
        array = np.ma.array([1,1,1,1,1],dtype=float)
        array[2] = np.ma.masked
        result = first_order_lag (array, 1.0, 1.0)
        np.testing.assert_array_almost_equal(result.data,[0.66666667,0.22222222,
                                                   0.07407407,0.02469136,
                                                   0.00823045])
        np.testing.assert_array_equal(result.mask,[0,0,1,0,0])

        