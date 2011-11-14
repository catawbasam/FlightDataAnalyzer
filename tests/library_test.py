try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

# A set of masked array test utilities from Pierre GF Gerard-Marchant
# http://www.java2s.com/Open-Source/Python/Math/Numerical-Python/numpy/numpy/ma/testutils.py.htm
import utilities.masked_array_testutils as ma_test

from datetime import datetime

from analysis.library import (align, calculate_timebase, create_phase_inside,
                              create_phase_outside, first_order_lag,
                               first_order_washout,
                              hysteresis, merge_alternate_sensors, powerset, 
                              rate_of_change, seek, straighten_headings,
                              value_at_time)


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
    def test_align_basic(self):
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
        np.testing.assert_array_equal(result.mask, False)
                
    def test_align_assert_array_lengths(self):
        class DumParam():
            def __init__(self):
                self.fdr_offset = 0.0
                self.hz = 1
                self.data = []
                
        first = DumParam()
        first.hz = 4
        first.data = np.ma.array(range(8))
        second = DumParam()
        second.hz = 4
        second.data = np.ma.array(range(7)) # Unmatched array length !
        self.assertRaises (AssertionError, align, first, second)
                
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
        np.testing.assert_array_almost_equal(result.data, [10.3,11.3,12.3,13.0])
        np.testing.assert_array_equal(result.mask, False)
        
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
        np.testing.assert_array_almost_equal(result.data, [10.0,10.7,11.7,12.7])
        np.testing.assert_array_equal(result.mask, False)
        
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
                                                           12.1,12.6,13.0,13.0])
        np.testing.assert_array_equal(result.mask, False)
        
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
        np.testing.assert_array_almost_equal(result.data, [10.0,10.15,10.4,10.65,
                                                           10.9,11.15,11.4,11.65,
                                                           11.9,12.15,12.4,12.65,
                                                           12.9,13.0 ,13.0,13.0 ])
        
    def test_align_mask_propogation(self):
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
        slave.data[2] = np.ma.masked
        slave.hz = 2
        slave.fdr_offset = 0.15
        result = align(master, slave)
        answer = np.ma.array(data = [10.0,10.15,10.4,10.65,
                                     10.9,0,0,0,
                                     0,0,0,0,
                                     0,13.0,13.0,13.0],
                             mask = [False,False,False,False,
                                     False, True, True, True,
                                     True , True, True, True,
                                     True ,False,False,False])
        ma_test.assert_masked_array_approx_equal(result, answer)

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
        np.testing.assert_array_almost_equal(result.data,[10.0 ,10.0  ,10.0,10.0  ,
                                                          10.0 ,10.0  ,10.0,10.025,
                                                          10.15,10.275,10.4,10.525,
                                                          10.65,10.775,10.9,11.0  ])
        np.testing.assert_array_equal(result.mask, False)
        
    def test_align_across_frame_increasing(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.zeros(64, dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 8
        master.fdr_offset = 0.1
        slave = DumParam()
        slave.data = np.ma.array([10,11],dtype=float)
        slave.hz = 0.25
        slave.fdr_offset = 3.95
        result = align(master, slave, interval='Frame')
        # Build the correct answer...
        answer=np.ma.ones(64)*10
        answer[31] = answer[31] + 0.00625
        for i in range(31):
            answer [31+i+1] = answer[31+i] + 1/32.0
        answer[-1] = 11.0
        # ...and check the resulting array in one hit.
        ma_test.assert_masked_array_approx_equal(result, answer)
        

    def test_align_across_frame_decreasing(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.fdr_offset = None
                self.hz = 1
                self.data = []
        master = DumParam()
        master.data = np.ma.zeros(4, dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.hz = 0.5
        master.fdr_offset = 1.5
        slave = DumParam()
        # Fill a two-frame sample with linear data
        slave.data = np.ma.arange(32,dtype=float)
        slave.hz = 4
        slave.fdr_offset = 0.1
        result = align(master, slave, interval='Frame')
        # Build the correct answer...
        answer=np.ma.array([5.6,13.6,21.6,29.6])
        ma_test.assert_masked_array_approx_equal(result, answer)
        

class TestMergeAlternateSensors(unittest.TestCase):
    def test_merge_alternage_sensors_basic(self):
        array = np.ma.array([0, 5, 0, 5, 1, 6, 1, 6],dtype=float)
        result = merge_alternate_sensors (array)
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,2.75,3.25,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, False)

    def test_merge_alternage_sensors_mask(self):
        array = np.ma.array([0, 5, 0, 5, 1, 6, 1, 6],dtype=float)
        array[4] = np.ma.masked
        result = merge_alternate_sensors (array)
        np.testing.assert_array_equal(result.data[0:3], [2.5,2.5,2.5])
        np.testing.assert_array_equal(result.data[6:8], [3.5,3.5])
        np.testing.assert_array_equal(result.mask, [False,False,False,
                                                    True,True,True,
                                                    False,False])


class TestRateOfChange(unittest.TestCase):
    
    # Reminder: rate_of_change(to_diff, half_width, hz) - half width in seconds.
    
    def test_rate_of_change_basic(self):
        array = np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],dtype=float)
        sloped = rate_of_change(array, 2, 1)
        answer = np.ma.array(data=[-1.0,-1.0,0.0,0.75,1.25,1.0,1.0,1.0,-1.0,2.0],
                             mask=False)
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_increased_frequency(self):
        array = np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],dtype=float)
        sloped = rate_of_change(array, 2, 2)
        answer = np.ma.array(data=[-2.0,-2.0,6.0,-2.0,1.0,1.75,2.0,4.0,-2.0,4.0],
                             mask=False)
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_transfer_mask(self):
        array = np.ma.array(data = [1, 0, -1, 2, 1, 3, 4, 6, 5, 7],dtype=float,
                            mask = [0, 1,  0, 0, 0, 1, 0, 0, 0, 1])
        sloped = rate_of_change(array, 1, 1)
        answer = np.ma.array(data = [0,-1.0,0,1.0,0,1.5,0,0.5,0,0],
             mask = [True,False,True,False,True,False,True,False,True,True])
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_half_width_zero(self):
        array = np.ma.array([0, 1, 0])
        self.assertRaises(ValueError, rate_of_change, array, 0, 1)
        
    def test_rate_of_change_half_width_negative(self):
        array = np.ma.array([0, 1, 0])
        self.assertRaises(ValueError, rate_of_change, array, -2, 1)
        
        
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
        np.testing.assert_array_almost_equal(straighten_headings(data), expected)

        #for index, val in enumerate(straighten_headings(data)):
            #self.assertEqual(
                #'%.2f' % val,
                #'%.2f' % expected[index],
                #msg="Failed at %s == %s at %s" % (val, expected[index], index)
            #)


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

    # first_order_lag (in_param, time_constant, hz, gain = 1.0, initial_value = 0.0)
    
    def test_firstorderlag_time_constant(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        # The result of processing this data is...
        result = first_order_lag (array, 2.0, 1.0, initial_value = 1.0)
        # The correct answer is...
        answer = np.ma.array(data=[0.8,0.48,0.288,0.1728,0.10368,0.062208,
                                   0.0373248,0.02239488,0.01343693,0.00806216],
                             mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderlag_sample_rate_chage(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        # The result of processing this data is...
        result = first_order_lag (array, 2.0, 2.0, initial_value = 1.0)
        # The correct answer is...
        answer = np.ma.array(data=[6.66666667e-01,2.22222222e-01,7.40740741e-02,
                                   2.46913580e-02,8.23045267e-03,2.74348422e-03,
                                   9.14494742e-04,3.04831581e-04,1.01610527e-04,
                                   3.38701756e-05], mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderlag_gain(self):
        array = np.ma.ones(20)
        result = first_order_lag (array, 1.0, 1.0, gain = 10.0)
        # With a short time constant and more samples, the end result will
        # reach the input level (1.0) multiplied by the gain.
        self.assertAlmostEquals(result.data[-1], 10.0)

    def test_firstorderlag_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_lag, array, 1.0, 4.0)

    def test_firstorderlag_mask_retained(self):
        array = np.ma.zeros(5)
        array[3] = np.ma.masked
        result = first_order_lag (array, 1.0, 1.0, initial_value = 1.0)
        ma_test.assert_mask_eqivalent(result.mask, [0,0,0,1,0], err_msg='Masks are not equal')

class TestFirstOrderWashout(unittest.TestCase):

    # first_order_washout (in_param, time_constant, hz, gain = 1.0, initial_value = 0.0)
    
    def test_firstorderwashout_time_constant(self):
        array = np.ma.ones(10)
        result = first_order_washout (array, 2.0, 1.0, initial_value = 0.0)
        # The correct answer is the same as for the first order lag test, but in
        # this case we are starting from zero and the input data is all 1.0.
        # The washout responds transiently then washes back out to zero, 
        # providing the high pass filter that matches the low pass lag filter.
        answer = np.ma.array(data=[0.8,0.48,0.288,0.1728,0.10368,0.062208,
                                   0.0373248,0.02239488,0.01343693,0.00806216],
                             mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderwashout_sample_rate_chage(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        # The result of processing this data is...
        result = first_order_washout (array, 2.0, 2.0, initial_value = 1.0)
        # The correct answer is...
        answer = np.ma.array(data=[6.66666667e-01,2.22222222e-01,7.40740741e-02,
                                   2.46913580e-02,8.23045267e-03,2.74348422e-03,
                                   9.14494742e-04,3.04831581e-04,1.01610527e-04,
                                   3.38701756e-05], mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderwashout_gain(self):
        array = np.ma.ones(20)
        result = first_order_washout (array, 1.0, 1.0, gain = 10.0)
        # With a short time constant and more samples, the end result will
        # reach the input level (1.0) multiplied by the gain.
        self.assertAlmostEquals(result.data[0], 6.6666667)

    def test_firstorderwashout_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_washout, array, 1.0, 4.0)

    def test_firstorderwashout_mask_retained(self):
        array = np.ma.zeros(5)
        array[3] = np.ma.masked
        result = first_order_washout (array, 1.0, 1.0, initial_value = 1.0)
        ma_test.assert_mask_eqivalent(result.mask, [0,0,0,1,0], err_msg='Masks are not equal')
        
        
class TestValueAtTime(unittest.TestCase):

    # Reminder: value_at_time (array, hz, fdr_offset, time_index)
    
    def test_value_at_time_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (value_at_time(array, 1, 0.0, 2.5), 2.5)
        
    def test_value_at_time_right_at_start_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEquals (value_at_time(array, 1, 0.0, 0.0), 22.3)
        
    def test_value_at_time_right_at_end_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEquals (value_at_time(array, 4, 0.0, 0.75), 25.3)
        
    def test_value_at_time_assertion_below_range(self):
        array = np.ma.arange(4)
        self.assertRaises (ValueError, value_at_time, array, 1, 0.1, 0.0)
        
    def test_value_at_time_assertion_above_range(self):
        array = np.ma.arange(4)
        self.assertRaises (ValueError, value_at_time, array, 1, 0.0, 7.0)
        
    def test_value_at_time_with_lower_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[1] = np.ma.masked
        self.assertEquals (value_at_time(array, 2.0, 0.2, 1.0), 9.4)
        
    def test_value_at_time_with_higher_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[2] = np.ma.masked
        self.assertEquals (value_at_time(array, 2.0, 0.2, 1.0), 8.4)
        
    def test_value_at_time_with_neither_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[3] = np.ma.masked
        self.assertEquals (value_at_time(array, 2.0, 0.2, 1.0), 9.0)
        
    def test_value_at_time_with_both_values_masked(self):
        array = np.ma.arange(4) + 7.4
        array[1] = np.ma.masked
        array[2] = np.ma.masked
        self.assertEquals (value_at_time(array, 2.0, 0.2, 1.0), None)


class TestSeek(unittest.TestCase):
    
    # Reminder: seek (array, hz, fdr_offset, scan_start, scan_end, threshold):

    def test_seek_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (seek(array, 1, 0.0, 0, 3, 1.5), 1.5)
        
    def test_seek_backwards(self):
        array = np.ma.arange(8)
        self.assertEquals (seek(array, 1, 0.0, 6, 2, 2.5), 2.5)

    def test_seek_right_at_start(self):
        array = np.ma.arange(4)
        self.assertEquals (seek(array, 1, 0.0, 1, 3, 1.0), 1.0)
                           
    def test_seek_right_at_end(self):
        array = np.ma.arange(4)
        self.assertEquals (seek(array, 1, 0.0, 1, 3, 3.0), 3.0)
        
    def test_seek_threshold_not_crossed(self):
        array = np.ma.arange(4)
        self.assertEquals (seek(array, 1, 0.0, 0, 3, 7.5), None)
        
    def test_seek_errors(self):
        array = np.ma.arange(4)
        self.assertRaises(ValueError, seek, array, 1, 0.0, 2, 2, 7.5)
        self.assertRaises(ValueError, seek, array, 1, 0.0, -1, 2, 7.5)
        self.assertRaises(ValueError, seek, array, 1, 0.0, 2, 5, 7.5)
        self.assertRaises(ValueError, seek, array, 1, 0.0, 5, 2, 7.5)
        self.assertRaises(ValueError, seek, array, 1, 0.0, 2, -1, 7.5)
        
    def test_seek_masked(self):
        array = np.ma.arange(4)
        array[1] = np.ma.masked
        self.assertEquals (seek(array, 1, 0.0, 0, 3, 1.5), None)
        
        
'''
ma_test.assert_masked_array_approx_equal(result, answer)

self.assertAlmostEquals(result.data[-1], 10.0)

    def test_firstorderlag_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_lag, array, 1.0, 4.0)
'''