import csv
import unittest
import numpy as np

from math import sqrt

from datetime import datetime

# A set of masked array test utilities from Pierre GF Gerard-Marchant
# http://www.java2s.com/Open-Source/Python/Math/Numerical-Python/numpy/numpy/ma/testutils.py.htm
import utilities.masked_array_testutils as ma_test

from analysis.library import (align, blend_alternate_sensors,
                              calculate_timebase, create_phase_inside,
                              create_phase_outside, duration, 
                              first_order_lag, first_order_washout, hash_array,
                              hysteresis, index_at_value, interleave,
                              is_index_within_slice, is_slice_within_slice,
                              min_value, mask_inside_slices,
                              mask_outside_slices, max_continuous_unmasked,
                              max_value, max_abs_value, blend_two_parameters,
                              peak_curvature, peak_index, rate_of_change, repair_mask, 
                              rms_noise, slices_above, slices_below, slices_between, 
                              slices_from_to, straighten_headings,
                              #time_at_value, time_at_value_wrapped,
                              value_at_time, vstack_params, InvalidDatetime)

from analysis.node import P, S
from analysis.library import *

class TestAlign(unittest.TestCase):
    
    def test_align_returns_same_array_if_aligned(self):
        slave = P('slave', np.ma.array(range(10)))
        master = P('master', np.ma.array(range(30)))
        aligned = align(slave, master)
        self.assertEqual(id(slave.array), id(aligned))
    
    def test_align_section_param(self):
        alt_aal = P('Altitude AAL', np.ma.arange(0, 5), frequency=1, offset=1)
        fast = S('Fast', frequency=4, offset=0.5)
        aligned = align(alt_aal, fast)
        self.assertEqual(len(aligned), 20)
        np.testing.assert_array_equal(aligned,
                                      [0, 0, 0, 0.25, 0.5, 0.75, 1, 1.25,
                                       1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,
                                       3.25, 3.5, 3.75, 4, 4])
        
    def test_align_basic(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = None
                self.array = []
                
        first = DumParam()
        first.frequency = 4
        first.offset = 0.1
        first.array = np.ma.array(range(8))
        
        second = DumParam()
        second.frequency = 4
        second.offset = 0.2
        second.array = np.ma.array(range(8))
        
        result = align(second, first) #  sounds more natural so order reversed 20/11/11
        np.testing.assert_array_equal(result.data,
                                      [0.0, 0.6, 1.6, 2.6, 3.6,
                                       4.6, 5.6, 6.6000000000000005])
        np.testing.assert_array_equal(result.mask, False)
                
    def test_align_discrete(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = None
                self.array = []
                
        first = DumParam()
        first.frequency = 1
        first.offset = 0.0
        first.array = np.ma.array([0,0,1,1,0,1,0,1],dtype=float)
        
        second = DumParam()
        second.frequency = 1
        second.offset = 0.7
        second.array = np.ma.array([0,0,1,1,0,1,0,1],dtype=float)
        
        result = align(second, first, signaltype='Discrete') #  sounds more natural so order reversed 20/11/11
        np.testing.assert_array_equal(result.data, [0,0,0,1,1,0,1,0])
        np.testing.assert_array_equal(result.mask, False)
                        
                        
    def test_align_multi_state(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = None
                self.array = []
                
        first = DumParam()
        first.frequency = 1
        first.offset = 0.6
        first.array = np.ma.array([11,12,13,14,15],dtype=float)
        
        second = DumParam()
        second.frequency = 1
        second.offset = 0.0
        second.array = np.ma.array([0,1,2,3,4],dtype=float)
        
        result = align(second, first, signaltype='Discrete') #  sounds more natural so order reversed 20/11/11
        np.testing.assert_array_equal(result.data, [1,2,3,4,4])
        np.testing.assert_array_equal(result.mask, False)
    
    def test_align_same_hz_delayed(self):
        # Both arrays at 1Hz, master behind slave in time
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 1
        master.offset = 0.5
        slave = DumParam()
        slave.array = np.ma.array([10,11,12,13],dtype=float)
        slave.frequency = 1
        slave.offset = 0.2
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [10.3,11.3,12.3,13.0])
        np.testing.assert_array_equal(result.mask, False)
        
    def test_align_same_hz_advanced(self):
        # Both arrays at 1Hz, master ahead of slave in time
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3],dtype=float)
        master.frequency = 1
        master.offset = 0.2
        slave = DumParam()
        slave.array = np.ma.array([10,11,12,13],dtype=float)
        slave.frequency = 1
        slave.offset = 0.5
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [10.0,10.7,11.7,12.7])
        np.testing.assert_array_equal(result.mask, False)
        
    def test_align_increasing_hz_delayed(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 4
        master.offset = 0.15
        slave = DumParam()
        slave.array = np.ma.array([10,11,12,13],dtype=float)
        slave.frequency = 2
        slave.offset = 0.1
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [10.1,10.6,11.1,11.6,
                                                           12.1,12.6,13.0,13.0])
        np.testing.assert_array_equal(result.mask, False)
        
    def test_align_increasing_hz_advanced(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 8
        master.offset = 0.1
        slave = DumParam()
        slave.array = np.ma.array([10,11,12,13],dtype=float)
        slave.frequency = 2
        slave.offset = 0.15
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [10.0,10.15,10.4,10.65,
                                                           10.9,11.15,11.4,11.65,
                                                           11.9,12.15,12.4,12.65,
                                                           12.9,13.0 ,13.0,13.0 ])
        
    def test_align_mask_propogation(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 8
        master.offset = 0.1
        slave = DumParam()
        slave.array = np.ma.array([10,11,12,13],dtype=float)
        slave.array[2] = np.ma.masked
        slave.frequency = 2
        slave.offset = 0.15
        result = align(slave, master)
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
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 8
        master.offset = 0.1
        slave = DumParam()
        slave.array = np.ma.array([10,11],dtype=float)
        slave.frequency = 1
        slave.offset = 0.95
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data,[10.0 ,10.0  ,10.0,10.0  ,
                                                          10.0 ,10.0  ,10.0,10.025,
                                                          10.15,10.275,10.4,10.525,
                                                          10.65,10.775,10.9,11.0  ])
        np.testing.assert_array_equal(result.mask, False)

    def test_align_across_frame_increasing(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.zeros(64, dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 8
        master.offset = 0.1
        slave = DumParam()
        slave.array = np.ma.array([10,11],dtype=float)
        slave.frequency = 0.25
        slave.offset = 3.95
        result = align(slave, master, interval='Frame')
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
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.zeros(4, dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 0.5
        master.offset = 1.5
        slave = DumParam()
        # Fill a two-frame sample with linear data
        slave.array = np.ma.arange(32,dtype=float)
        slave.frequency = 4
        slave.offset = 0.1
        result = align(slave, master, interval='Frame')
        # Build the correct answer...
        answer=np.ma.array([5.6,13.6,21.6,29.6])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_align_superframe_master(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.offset = 0.0
                self.array = []
        master = DumParam()
        master.array = np.ma.array([1,2])
        master.frequency = 1/64.0
        slave = DumParam()
        slave.array = np.ma.arange(128)
        slave.frequency = 1
        result = align(slave, master)
        expected = [0,64]
        np.testing.assert_array_equal(result.data,expected)

    def test_align_superframe_slave(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.offset = 0.0
                self.array = []
        master = DumParam()
        master.array = np.ma.arange(64)
        master.frequency = 2
        slave = DumParam()
        slave.array = np.ma.array([1,3,6,9])
        slave.frequency = 1/8.0
        result = align(slave, master)
        expected = [1]*16+[3]*16+[6]*16+[9]*16
        np.testing.assert_array_equal(result.data,expected)

    def test_align_superframes_both(self):
        class DumParam():
            def __init__(self):
                self.offset = None
                self.frequency = 1
                self.offset = 0.0
                self.array = []
        master = DumParam()
        master.array = np.ma.arange(16)
        master.frequency = 1/8.0
        slave = DumParam()
        slave.array = np.ma.arange(4)+100
        slave.frequency = 1/32.0
        result = align(slave, master)
        expected = [100]*4+[101]*4+[102]*4+[103]*4
        np.testing.assert_array_equal(result.data,expected)
                

class TestBlendAlternateSensors(unittest.TestCase):
    def test_blend_alternage_sensors_basic(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        result = blend_alternate_sensors (array_1, array_2, 'Follow')
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,3,3.5,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, [False,False,False,False,
                                                   False,False,False,True])

    def test_blend_alternage_sensors_mask(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        array_1[2] = np.ma.masked
        result = blend_alternate_sensors (array_1, array_2, 'Follow')
        np.testing.assert_array_equal(result.data[0:3], [2.5,2.5,2.5])
        np.testing.assert_array_equal(result.data[6:8], [3.5,3.5])
        np.testing.assert_array_equal(result.mask, [False,False,False,
                                                    True,True,False,
                                                    False,True])

    def test_blend_alternage_sensors_reverse(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        result = blend_alternate_sensors (array_1, array_2, 'Precede')
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,2.5,3,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, [True,False,False,False,
                                                    False,False,False,False])


class TestCalculateTimebase(unittest.TestCase):
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
        
    def test_no_valid_datetimes_raises_valueerror(self):
        years = [None] * 20
        months = [None] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [00] * 15
        mins = [None] * 2 + [59] * 10 + [01] * 13
        secs = [None] * 1 + range(55, 60) + range(19)  # 6th second in next hr
        self.assertRaises(InvalidDatetime, calculate_timebase, years, months, days, hours, mins, secs)
        
        
    def test_uneven_length_arrays(self):
        "Tests that the uneven drabs at the end are ignored"
        # You should always pass in complete arrays at the moment!
        years = [None] * 1 + [2020] * 10  # uneven
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [00] * 1 # uneven
        mins = [None] * 2 + [59] * 10 + [01] * 13
        secs = [None] * 1 + range(55, 60) + range(19)
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        self.assertEqual(start_dt, datetime(2020,12,24,23,58,54))
        
    def test_no_change_in_dt_picks_it_as_start(self):
        # also tests using numpy masked arrays
        years = np.ma.array([2020] * 20)  # 6 sec offset
        months = np.ma.array([12] * 20)
        days = np.ma.array([25] * 20)
        hours = np.ma.array([23] * 20)
        mins = np.ma.array([0] * 20)
        secs = np.ma.array([0] * 20) # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        self.assertEqual(start_dt, datetime(2020,12,25,23,0,0))
        
    @unittest.skip("Implement if this is a requirement")
    def test_using_offset_for_seconds(self):
        # check offset milliseconds are applied to the timestamps
        self.assertFalse(True)
        

class TestDuration(unittest.TestCase):
    def setUp(self):
        test_list = []
        result_list = []
        with open('test_data/duration_test_data.csv', 'rb') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                test_list.append(float(row['input']))
                result_list.append(float(row['output']))
        self.test_array = np.array(test_list)
        self.result_array = np.array(result_list)

    def test_duration_example_of_use(self):
        # Engine temperature at startup limit = 900 C for 5 seconds, say.

        # Pseudo-POLARIS exceedance would be like this:
        # Exceedance = duration(Eng_1_EGT,5sec,1Hz) > 900
        
        # In this case it was over 910 for 5 seconds, hence is an exceedance.
        
        # You get 6 values in the output array for a 5-second duration event.
        # Remember, fenceposts and panels.

        engine_egt = np.array([600.0,700.0,800.0,910.0,950.0,970.0,940.0,\
                                960.0,920.0,890.0,840.0,730.0])
        output_array = np.array([600.0,700.0,800.0,910.0,910.0,910.0,910.0,\
                                910.0,910.0,890.0,840.0,730.0])
        result = duration(engine_egt, 5)
        np.testing.assert_array_equal(result, output_array)
        
    def test_duration_correct_result(self):
        result = duration(self.test_array, 3)
        np.testing.assert_array_almost_equal(result, self.result_array)
    
    def test_duration_rejects_negative_period(self):
        an_array = np.array([0,1])
        self.assertRaises(ValueError, duration, an_array, -0.2)
        
    def test_duration_rejects_negative_hz(self):
        an_array = np.array([0,1])
        self.assertRaises(ValueError, duration, an_array, 0.2, hz=-2)
        
    def test_duration_rejects_zero_period(self):
        an_array = np.array([0,1])
        self.assertRaises(ValueError, duration, an_array, 0.0)
        
    def test_duration_rejects_zero_hz(self):
        an_array = np.array([0,1])
        self.assertRaises(ValueError, duration, an_array, 1.0, hz=0.0)
        
    def test_duration_no_change_below_period(self):
        input_array = np.array([0,1,2,2,2,1,0])
        output_array = input_array
        result = duration(input_array, 1, hz=2)
        np.testing.assert_array_equal(result, output_array)
        
    def test_duration_change_at_period(self):
        input_array = np.array([0.6,1.1,2.1,3.5,1.9,1.0,0])
        output_array = np.array([0.6,1.1,1.1,1.1,1.1,1.0,0.0])
        result = duration(input_array, 6, hz=0.5)
        np.testing.assert_array_equal(result, output_array)
                    
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
        answer = np.ma.array(data=[0.88888889,0.69135802,0.53772291,0.41822893,
                                 0.32528917,0.25300269,0.19677987,0.15305101,
                                 0.11903967,0.09258641], mask = False)
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
        self.assertRaises(ValueError, first_order_lag, array, 0.2, 1.0)

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
        '''
        OK. Tricky test that needs some explanation.
        
        The initial value is the steady state input condition prior to the data 
        we supply. This filter is a washout (high pass) filter, so the steady 
        state output will always be zero.
        
        The initial condition is set to -1.0, then when the data arrives, 
        array[0]=0.0 gives a +1.0 step change to the input and we get a positive 
        kick on the output.
        '''
        result = first_order_washout (array, 2.0, 0.5, initial_value = -1.0)
        # The correct answer is...
        answer = np.ma.array(data=[6.66666667e-01,2.22222222e-01,7.40740741e-02,
                                   2.46913580e-02,8.23045267e-03,2.74348422e-03,
                                   9.14494742e-04,3.04831581e-04,1.01610527e-04,
                                   3.38701756e-05], mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderwashout_gain(self):
        array = np.ma.ones(20)
        result = first_order_washout (array, 1.0, 1.0, gain = 10.0, initial_value = 0.0)
        # With a short time constant and more samples, the end result will
        # reach the input level (1.0) multiplied by the gain.
        self.assertAlmostEquals(result.data[0], 6.6666667)

    def test_firstorderwashout_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_washout, array, 0.2, 1.0)

    def test_firstorderwashout_mask_retained(self):
        array = np.ma.zeros(5)
        array[3] = np.ma.masked
        result = first_order_washout (array, 1.0, 1.0, initial_value = 1.0)
        ma_test.assert_mask_eqivalent(result.mask, [0,0,0,1,0], err_msg='Masks are not equal')
        

class TestHashArray(unittest.TestCase):
    def test_hash_array(self):
        '''
        
        '''
        self.assertEqual(hash_array(np.ma.arange(10)),
                         hash_array(np.ma.arange(10)))
        self.assertNotEqual(hash_array(np.ma.arange(10)),
                            hash_array(np.ma.arange(1,11)))
        # Tests that mask contents affect the generated hash.
        ma1 = np.ma.array(np.ma.arange(100,200), mask=[False] * 100)
        ma2 = np.ma.array(np.ma.arange(100,200),
                          mask=[False] * 50 + [True] + 49 * [False])
        self.assertNotEqual(hash_array(ma1), hash_array(ma2))
        self.assertEqual(hash_array(ma2), hash_array(ma2))
        self.assertEqual(hash_array(np.ma.arange(10, dtype=np.float_)),
            'c29605eb4e50fbb653a19f1a28c4f0955721419f989f1ffd8cb2ed6f4914bbea')


class TestHysteresis(unittest.TestCase):
    def test_hysteresis(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        data[4] = np.ma.masked
        result = hysteresis(data,2)
        np.testing.assert_array_equal(result.filled(999),
                                      [0.5,1,1,1,999,0,5,6,6,0.5])

    def test_hysteresis_change_of_threshold(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        result = hysteresis(data,1)
        np.testing.assert_array_equal(result.data,[0.25,1.,1.5,1.,0.,
                                                   -0.5,5.,6.,6.5,0.25])

    def test_hysteresis_phase_stability(self):
        data = np.ma.array([0,1,2,3,4,5,5,4,3,2,1,0],dtype=float)
        result = hysteresis(data,2)
        # Hysteresis range of 2 = +/- 1.0, so top 1 gets "choppeed off".
        # Slopes remain unaltered, and endpoints are amended by half the
        # hysteresis, being affected only on one pass of this two-pass
        # process.
        np.testing.assert_array_equal(result.data,[0.5,1,2,3,4,4,4,4,3,2,1,0.5])


class TestIndexAtValue(unittest.TestCase):
    
    # Reminder: index_at_value (array, section, threshold):

    def test_index_at_value_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (index_at_value(array, 1.5, slice(0, 3)), 1.5)
        
    def test_index_at_value_no_slice(self):
        array = np.ma.arange(4)
        self.assertEquals (index_at_value(array, 1.5), 1.5)
        self.assertEquals (index_at_value(array, 1.5, slice(None, None, None)), 1.5)
        
    def test_index_at_value_backwards(self):
        array = np.ma.arange(8)
        self.assertEquals (index_at_value(array, 3.2, slice(6, 2, -1)), 3.2)

    def test_index_at_value_backwards_with_negative_values_a(self):
        array = np.ma.arange(8)*(-1.0)
        self.assertEquals (index_at_value(array, -3.2, slice(6, 2, -1)), 3.2)
        
    def test_index_at_value_backwards_with_negative_values_b(self):
        array = np.ma.arange(8)-10
        self.assertEquals (index_at_value(array, -5.2, slice(6, 2, -1)), 4.8)

    def test_index_at_value_right_at_start(self):
        array = np.ma.arange(4)
        self.assertEquals (index_at_value(array, 1.0, slice(1, 3)), 1.0)
                           
    def test_index_at_value_right_at_end(self):
        array = np.ma.arange(4)
        self.assertEquals (index_at_value(array, 3.0, slice(1, 4)), 3.0)
        
    def test_index_at_value_threshold_not_crossed(self):
        array = np.ma.arange(4)
        self.assertEquals (index_at_value(array, 7.5, slice(0, 3)), None)
        
    def test_index_at_value_masked(self):
        array = np.ma.arange(4)
        array[1] = np.ma.masked
        self.assertEquals (index_at_value(array, 1.5, slice(0, 3)), None)
    
    def test_index_at_value_slice_too_small(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 25, slice(25,26)), None)

    def test_index_at_value_slice_beyond_top_end_of_data(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 55, slice(40,60)), None)

    def test_index_at_value_slice_beyond_bottom_end_of_data(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 55, slice(-20,20)), None)


class TestIndexClosestValue(unittest.TestCase):
    def test_index_closest_value(self):
        array = np.ma.array([1,2,3,4,5,4,3])
        self.assertEqual(index_closest_value(array, 6, slice(0,6)), 4)

    def test_index_closest_value_at_start(self):
        array = np.ma.array([6,2,3,4,5,4,3])
        self.assertEqual(index_closest_value(array, 7, slice(0,6)), 0)

    def test_index_closest_value_at_end(self):
        array = np.ma.array([1,2,3,4,5,6,7])
        self.assertEqual(index_closest_value(array, 99, slice(0,6)), 6)

    def test_index_closest_value_negative(self):
        array = np.ma.array([3,2,1,4,5,6,7])
        self.assertEqual(index_closest_value(array, -9, slice(0,6)), 2)

    def test_index_closest_value_sliced(self):
        array = np.ma.array([1,2,3,4,5,4,3])
        self.assertEqual(index_closest_value(array, 6, slice(2,5)), 4)

    def test_index_closest_value_backwards(self):
        array = np.ma.array([3,2,1,4,5,6,7])
        self.assertEqual(index_closest_value(array, -9, slice(5,1,-1)), 2)



class TestIndexOfDatetime(unittest.TestCase):
    def test_index_of_datetime(self):
        start_datetime = datetime.now()
        difference = timedelta(seconds=40)
        index_datetime = start_datetime + difference
        frequency = 4
        index = index_of_datetime(start_datetime, index_datetime, frequency)
        self.assertEqual(index, 160)


class TestInterleave(unittest.TestCase):
    def test_interleave(self):
        param1 = P('A1',np.ma.array(range(4),dtype=float),1,0.2)
        param2 = P('A2',np.ma.array(range(4),dtype=float)+10,1,0.7)
        result = interleave(param1, param2)
        np.testing.assert_array_equal(result.data,[0,10,1,11,2,12,3,13])
        np.testing.assert_array_equal(result.mask, False)

    def test_merge_alternage_sensors_mask(self):
        param1 = P('A1',np.ma.array(range(4),dtype=float),1,0.2)
        param2 = P('A2',np.ma.array(range(4),dtype=float)+10,1,0.7)
        param1.array[1] = np.ma.masked
        param2.array[2] = np.ma.masked
        result = interleave(param1, param2)
        np.testing.assert_array_equal(result.data[0:2], [0,10])
        np.testing.assert_array_equal(result.data[3:5], [11,2])
        np.testing.assert_array_equal(result.data[6:], [3,13])
        np.testing.assert_array_equal(result.mask, [False,False,True,
                                                    False,False,True,
                                                    False,False])


"""
class TestInterpolateParams(unittest.TestCase):
    def test_interpolate_params(self):
        param1 = P('A1',np.ma.arange(10),
                   frequency=1,offset=0.2)
        param2 = P('A2',np.ma.arange(0.2, 10, 0.5),
                   frequency=2,offset=0.7)
        param1.array[1] = np.ma.masked
        param2.array[2] = np.ma.masked
        array, freq, off = interpolate_params(param1, param2)
        np.testing.assert_array_equal(array.data[:5], 
            [0.0, 0.27222222222222225, 0.49444444444444441, 0.71666666666666656,
             0.93888888888888866])
        np.testing.assert_array_equal(array[-5:], 
            [8.9047619047619033, 9.1481481481481435, 9.3833333333333311,
             9.6055555555555525, np.nan])
        array.mask[-1] = True
        self.assertEqual(freq, 3)
        self.assertEqual(off, 3 * param1.offset)
"""

class TestIsIndexWithinSlice(unittest.TestCase):
    def test_is_index_within_slice(self):
        self.assertTrue(is_index_within_slice(1, slice(0,2)))
        self.assertTrue(is_index_within_slice(5, slice(5,7)))
        # Slice is not inclusive of last index.
        self.assertFalse(is_index_within_slice(7, slice(5,7)))
        self.assertTrue(is_index_within_slice(10, slice(8,None)))
        self.assertTrue(is_index_within_slice(10, slice(None, 12)))
        

class TestIsSliceWithinSlice(unittest.TestCase):
    def test_is_slice_within_slice(self):
        self.assertTrue(is_slice_within_slice(slice(5,6), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(4,6), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(4,7), slice(4,7)))
        self.assertFalse(is_slice_within_slice(slice(4,8), slice(4,7)))
        self.assertFalse(is_slice_within_slice(slice(3,7), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(None, None),
                                              slice(None, None)))
        self.assertFalse(is_slice_within_slice(slice(None, None),
                                               slice(None, 20)))
        self.assertFalse(is_slice_within_slice(slice(None, 15), slice(4, None)))
        self.assertTrue(is_slice_within_slice(slice(-1000, 15),
                                              slice(None, None)))
        self.assertTrue(is_slice_within_slice(slice(-1000, None),
                                              slice(None, None)))
        self.assertTrue(is_slice_within_slice(slice(None, 15),
                                              slice(None, None)))
        

class TestMaskInsideSlices(unittest.TestCase):
    def test_mask_inside_slices(self):
        slices = [slice(10, 20), slice(30, 40)]
        array = np.ma.arange(50)
        array.mask = np.array([True] * 5 + [False] * 45)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 5 + [False] * 5 + [True] *  10 + [False] * 10 + [True] * 10)
        ma_test.assert_equal(mask_inside_slices(array, slices),
                             expected_result)


class TestMaxContinuousUnmasked(unittest.TestCase):
    def test_max_continuous_unmasked(self):
        data = np.ma.array(range(20),
                           mask=[1,0,1,1,1,0,0,0,0,1,
                                 0,0,0,0,0,0,0,1,1,1])
        _max = max_continuous_unmasked(data)
        # test duration
        self.assertEqual(_max.stop-_max.start, 7)
        self.assertEqual(_max.start, 10)
        self.assertEqual(_max.stop, 17)
        self.assertFalse(np.any(data[_max].mask)) # none should be masked
    
    def test_max_continuous_unmasked_no_mask(self):
        # no mask
        data = np.ma.array(range(20), mask=False)
        _max = max_continuous_unmasked(data)
        self.assertEqual(_max.stop-_max.start, 20)
        self.assertEqual(_max.start, 0)
        self.assertEqual(_max.stop, 20)
        
        # all masked
        data = np.ma.array(range(5), mask=[1,1,1,1,1])
        _max = max_continuous_unmasked(data)
        self.assertEqual(_max, None)
        
        # no data
        data = np.ma.array([])
        _max = max_continuous_unmasked(data, slice(110,120))
        self.assertEqual(_max, None)
        
    def test_max_continuous_unmasked_with_slice(self):
        data = np.ma.array(range(30),
                           mask=[0,1,0,0,0,1,1,1,1,0,
                                 1,1,1,1,1,1,1,0,0,0,
                                 1,1,1,1,1,0,0,1,1,1,])
        _max = max_continuous_unmasked(data, slice(20,30))
        # test duration
        self.assertEqual(_max.stop-_max.start, 2)
        self.assertEqual(_max.start, 25)
        self.assertEqual(_max.stop, 27)
                


class TestMaskOutsideSlices(unittest.TestCase):
    def test_mask_outside_slices(self):
        slices = [slice(10, 20), slice(30, 40)]
        array = np.ma.arange(50)
        array.mask = np.array([False] * 10 + [True] * 5 + [False] * 35)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 15 + [False] * 5 + [True] * 10 + [False] * 10 + [True] * 10)
        ma_test.assert_equal(mask_outside_slices(array, slices),
                             expected_result)
        
        
class TestMaxValue(unittest.TestCase):
    def test_max_value(self):
        array = np.ma.array(range(50,100) + range(100,50,-1))
        i, v = max_value(array)
        self.assertEqual(i, 50)
        self.assertEqual(v, 100)
        
        subslice = slice(80, 90)
        res = max_value(array, subslice)
        self.assertEqual(res.index, 80)
        self.assertEqual(res.value, 70)
        
        neg_step = slice(100,65,-10)
        self.assertRaises(ValueError, max_value, array, neg_step)
        ##self.assertEqual(res, (69, 81)) # you can get this if you use slice.stop!
        
class TestMaxAbsValue(unittest.TestCase):
    def test_max_abs_value(self):
        array = np.ma.array(range(-20,30) + range(10,-41, -1) + range(10))
        self.assertEqual(max_abs_value(array), (100, -40))


class TestMinValue(unittest.TestCase):
    def test_min_value(self):
        array = np.ma.array(range(50,100) + range(100,50,-1))
        i, v = min_value(array)
        self.assertEqual(i, 0)
        self.assertEqual(v, 50)
        
        subslice = slice(80, 90)
        res = min_value(array, subslice)
        self.assertEqual(res.index, 89)
        self.assertEqual(res.value, 61)
        
        neg_step = slice(100,65,-10)
        self.assertRaises(ValueError, min_value, array, neg_step)


class TestBlendTwoParameters(unittest.TestCase):
    def test_blend_two_parameters_offset_ordered_forward(self):
        p1 = P(array=[0]*4, frequency=1, offset=0.0)
        p2 = P(array=[1,2,3,4], frequency=1, offset=0.2)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[1], 0.5)
        self.assertEqual(freq, 2)
        self.assertEqual(off, 0.1)

    def test_blend_two_parameters_offset_ordered_backward(self):
        p1 = P(array=[5,10,7,8], frequency=2, offset=0.1)
        p2 = P(array=[1,2,3,4], frequency=2, offset=0.0)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[2], 3.5)
        self.assertEqual(freq, 4)
        self.assertEqual(off, 0.05)

    def test_blend_two_parameters_assertion_error(self):
        p1 = P(array=[0]*4, frequency=1, offset=0.0)
        p2 = P(array=[1]*4, frequency=2, offset=0.2)
        self.assertRaises(AssertionError, blend_two_parameters, p1, p2)

    def test_blend_two_parameters_array_mismatch_error(self):
        p1 = P(array=[0]*4, frequency=1, offset=0.0)
        p2 = P(array=[1]*3, frequency=2, offset=0.2)
        self.assertRaises(AssertionError, blend_two_parameters, p1, p2)

        
class TestPeakCurvature(unittest.TestCase):
    # Note: The results from the first two tests are in a range format as the
    # artificial data results in multiple maxima.

    def test_peak_curvature_basic(self):
        array = np.ma.array([0]*20+range(20))
        pc = peak_curvature(array)
        self.assertEqual(pc,18.5) 
        #  Very artificial case returns first location of many seconds of
        #  high curvature.

    def test_peak_curvature(self):
        array = np.ma.array([0]*40+range(40))
        pc = peak_curvature(array)
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_no_peak(self):
        array = np.ma.array([0]*40+range(40))*(-1.0)
        pc = peak_curvature(array, curve_sense='Concave')
        self.assertEqual(pc,None)

    def test_peak_curvature_bipolar(self):
        array = np.ma.array([0]*40+range(40))*(-1.0)
        pc = peak_curvature(array, curve_sense='Bipolar')
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_failure(self):
        array = np.ma.array([0]*2+range(2))
        self.assertRaises(ValueError, peak_curvature, array)
       
    def test_peak_curvature_real_data(self):
        array = np.ma.array([37.9,37.9,37.9,37.9,37.9,38.2,38.2,38.2,38.2,38.8,
                             38.2,38.8,39.1,39.7,40.6,41.5,42.7,43.6,44.5,46,
                             47.5,49.6,52,53.2,54.7,57.4,60.7,61.9,64.3,66.1,
                             69.4,70.6,74.2,74.8])
        pc = peak_curvature(array)
        self.assertGreaterEqual(pc,15.0)
        self.assertLessEqual(pc,15.1)
        
    def test_peak_curvature_with_slice(self):
        array = np.ma.array([0]*100)
        pc = peak_curvature(array, slice(10, 50))
        self.assertEqual(pc, 18)

class TestPeakIndex(unittest.TestCase):
    def test_peak_index_no_data(self):
        self.assertRaises(ValueError, peak_index, [])
        
    def test_peak_index_one_sample(self):
        self.assertEqual(peak_index([4]),0)
        
    def test_peak_index_two_samples_rising(self):
        self.assertEqual(peak_index([2,4]),1)
        
    def test_peak_index_two_samples_falling(self):
        self.assertEqual(peak_index([4,2]),0)
        
    def test_peak_index_three_samples_falling(self):
        self.assertEqual(peak_index([6,4,2]),0)
        
    def test_peak_index_three_samples_rising(self):
        self.assertEqual(peak_index([1,2,3]),2)
        
    def test_peak_index_three_samples_with_peak(self):
        self.assertEqual(peak_index([1,2,1]),1)
        
    def test_peak_index_three_samples_trap_linear(self):
        self.assertEqual(peak_index([0,0.0000001,0]),1)
        
    def test_peak_index_real_peak(self):
        peak=np.sin(np.arange(10)/3.)
        self.assertEqual(peak_index(peak),4.7141807866121832)
        
class TestPhaseMasking(unittest.TestCase):
    def test_phase_inside_basic(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.0,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,0,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_inside_reversed(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.1,5,2) # 2,5 > 5,2
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,1,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_inside_positive_offset(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.1,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,1,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_inside_negative_offset(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,-0.1,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,1,0,0,0,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_inside_low_rate(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)*4
        result = create_phase_inside(array, 0.25,0.0,12,25)
        answer = np.ma.array(data = [0,4,8,12,16,20,24,28],
                             mask = [1,1,1,0,0,0,0,1])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_outside_low_rate(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)*4
        result = create_phase_outside(array, 0.25,0.0,7,21)
        answer = np.ma.array(data = [0,4,8,12,16,20,24,28],
                             mask = [0,0,1,1,1,1,0,0])
        ma_test.assert_masked_array_approx_equal(result, answer)
        
    def test_phase_inside_errors(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        self.assertRaises(ValueError, create_phase_inside, array, 1,0, -1, 5)
        self.assertRaises(ValueError, create_phase_inside, array, 1,0, 10, 5)
        self.assertRaises(ValueError, create_phase_inside, array, 1,0, 2, -1)
        self.assertRaises(ValueError, create_phase_inside, array, 1,0, 2, 11)
    

class TestRateOfChange(unittest.TestCase):
    
    # Reminder: rate_of_change(to_diff, half_width, hz) - half width in seconds.
    
    def test_rate_of_change_basic(self):
        sloped = rate_of_change(P('Test', 
                                  np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],
                                              dtype=float), 1), 2)
        answer = np.ma.array(data=[-1.0,-1.0,0.0,0.75,1.25,1.0,1.0,1.0,-1.0,2.0],
                             mask=False)
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_increased_frequency(self):
        sloped = rate_of_change(P('Test', 
                                  np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],
                                              dtype=float), 2), 2)
        answer = np.ma.array(data=[-2.0,-2.0,6.0,-2.0,1.0,1.75,2.0,4.0,-2.0,4.0],
                             mask=False)
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_transfer_mask(self):
        sloped = rate_of_change(P('Test', 
                                  np.ma.array(data = [1, 0, -1, 2, 1, 3, 4, 6, 5, 7],dtype=float,
                            mask = [0, 1,  0, 0, 0, 1, 0, 0, 0, 1]), 1), 1)
        answer = np.ma.array(data = [0,-1.0,0,1.0,0,1.5,0,0.5,0,0],
             mask = [True,False,True,False,True,False,True,False,True,True])
        ma_test.assert_mask_eqivalent(sloped, answer)
        
    def test_rate_of_change_half_width_zero(self):
        self.assertRaises(ValueError, 
                          rate_of_change, 
                          P('Test',np.ma.array([0, 1, 0]), 1), 0)
        
    def test_rate_of_change_half_width_negative(self):
        self.assertRaises(ValueError, 
                          rate_of_change, 
                          P('Test',np.ma.array([0, 1, 0]), 1), -2)
        
    def test_rate_of_change_small_values(self):
        sloped = rate_of_change(P('Test',np.ma.arange(10)/100.0, 1), 2)
        answer = np.ma.array(data=[0.01]*10,mask=False)
        ma_test.assert_masked_array_approx_equal(sloped, answer)
        
        
class TestRepairMask(unittest.TestCase):
    def test_repair_mask_basic(self):
        array = np.ma.arange(10)
        array[3] = np.ma.masked
        self.assertTrue(np.ma.is_masked(array[3]))
        array[6:8] = np.ma.masked
        res = repair_mask(array)
        np.testing.assert_array_equal(res.data,range(10))
        # test mask is now unmasked
        self.assertFalse(np.ma.is_masked(res[3]))
        self.assertFalse(np.ma.is_masked(res[4]))
        self.assertFalse(np.ma.is_masked(res[5]))
        self.assertFalse(np.ma.is_masked(res[6]))
        self.assertFalse(np.ma.is_masked(res[7]))
        self.assertFalse(np.ma.is_masked(res[8]))
        
    def test_repair_mask_too_much_invalid(self):
        array = np.ma.arange(20)
        array[4:15] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)
        
    def test_repair_mask_not_at_start(self):
        array = np.ma.arange(10)
        array[0] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)
        
    def test_repair_mask_not_at_end(self):
        array = np.ma.arange(10)
        array[9] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)

class test_rms_noise(unittest.TestCase):
    def test_rms_noise_basic(self):
        array = np.ma.array([0,0,1,0,0])
        result = rms_noise(array)
        expected = sqrt(6.0/5.0)
        self.assertAlmostEqual(result, expected)
        
    def test_rms_noise_masked(self):
        array = np.ma.array([0,0,1,0,0])
        array[2]=np.ma.masked
        result = rms_noise(array)
        expected = 0.0
        self.assertAlmostEqual(result, expected)
        
class TestSlicesAbove(unittest.TestCase):
    def test_slices_above(self):
        array = np.ma.concatenate([np.ma.arange(10), np.ma.arange(10)])
        array.mask = [False] * 18 + [True] * 2
        repaired_array, slices = slices_above(array, 5)
        self.assertEqual(slices, [slice(5, 10, None), slice(15, 18, None)])


class TestSlicesBelow(unittest.TestCase):
    def test_slices_below(self):
        array = np.ma.concatenate([np.ma.arange(10), np.ma.arange(10)])
        array.mask = [True] * 2 + [False] * 18
        repaired_array, slices = slices_below(array, 5)
        self.assertEqual(slices, [slice(2, 6, None), slice(10, 16, None)])


class TestSlicesBetween(unittest.TestCase):
    def test_slices_between(self):
        array = np.ma.arange(20)
        array.mask = [True] * 10 + [False] * 10
        repaired_array, slices = slices_between(array, 5, 15)
        self.assertEqual(slices, [slice(10, 16)])


class TestSlicesFromTo(unittest.TestCase):
    def test_slices_from_to(self):
        array = np.ma.arange(20)
        array.mask = [True] * 10 + [False] * 10
        # Ascending.
        repaired_array, slices = slices_from_to(array, 5, 15)
        self.assertEqual(slices, [slice(10, 16)])
        # Descending.
        repaired_array, slices = slices_from_to(array, 18, 3)
        self.assertEqual(slices, [])
        array = np.ma.arange(20, 0, -1)
        array.mask = [True] * 10 + [False] * 10
        repaired_array, slices = slices_from_to(array, 18, 3)
        self.assertEqual(slices, [slice(10, 18)])


class TestStraightenHeadings(unittest.TestCase):
    def test_straight_headings(self):
        data = np.ma.array([35.5,29.5,11.3,0.0,348.4,336.8,358.9,2.5,8.1,14.4])
        expected = np.ma.array([35.5,29.5,11.3,0.0,-11.6,-23.2,-1.1,2.5,8.1,14.4])
        np.testing.assert_array_almost_equal(straighten_headings(data),expected)

    def test_straight_headings_starting_masked(self):
        data=np.ma.array(data=[0]*10+[6]*8+[1]*4+[10,5,0,355,350]+[0]*4,
                         mask=[True]*10+[False]*8+[True]*4+[False]*5+[True]*4)
        expected=np.ma.array(data=[0]*10+[6]*8+[0]*4+[10,5,0,-5,-10]+[0]*4,
                         mask=[True]*10+[False]*8+[True]*4+[False]*5+[True]*4)
        ma_test.assert_masked_array_approx_equal(straighten_headings(data), expected)
            
class TestSmoothTrack(unittest.TestCase):
    def test_smooth_track_latitude(self):
        lat = np.ma.array([0,0,0,1,1,1], dtype=float)
        lon = np.ma.zeros(6, dtype=float)
        lat_s, lon_s, cost = smooth_track(lat, lon)
        self.assertLess (cost,26)
        
    def test_smooth_track_longitude(self):
        lon = np.ma.array([0,0,0,1,1,1], dtype=float)
        lat = np.ma.zeros(6, dtype=float)
        lat_s, lon_s, cost = smooth_track(lat, lon)
        self.assertLess (cost,26)
        
    def test_smooth_track_masked(self):
        lon = np.ma.array([0,0,0,1,1,1], dtype=float)
        lat = np.ma.zeros(6, dtype=float)
        lon[4]=np.ma.masked
        lat_s, lon_s, cost = smooth_track(lat, lon)
        self.assertLess (cost,26)
        
    def test_smooth_track_speed(self):
        from time import clock
        lon = np.ma.arange(10000, dtype=float)
        lon = lon%27
        lat = np.ma.zeros(10000, dtype=float)
        start = clock()
        lat_s, lon_s, cost = smooth_track(lat, lon)
        end = clock()      
        print end-start      
        self.assertLess (end-start,1.0)
            
        
class TestSubslice(unittest.TestCase):
    def test_subslice(self):
        """ Does not test using negative slice start/stop values e.g. (-2,2)
        """
        # test basic
        orig = slice(2,10)
        new = slice(2, 4)
        res = subslice(orig, new)
        self.assertEqual(res, slice(4, 6))
        fifty = range(50)
        self.assertEqual(fifty[orig][new], fifty[res])
        
        orig = slice(10,20,2)
        new = slice(2, 4, 1)
        res = subslice(orig, new)
        thirty = range(30)
        self.assertEqual(thirty[orig][new], thirty[res])
        self.assertEqual(res, slice(14, 18, 2))
        
        
        # test step
        orig = slice(100,200,10)
        new = slice(1, 5, 2)
        sub = subslice(orig, new)
        two_hundred = range(0,200)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(110, 150, 20))
        
        # test negative step
        orig = slice(200,100,-10)
        new = slice(1, 5, 2)
        sub = subslice(orig, new)
        two_hundred = range(201)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(190, 150, -20))
        
        orig = slice(100,200,10)
        new = slice(5, 1, -2)
        sub = subslice(orig, new)
        two_hundred = range(201)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(150, 110, -20))
        self.assertEqual(two_hundred[sub], [150, 130]) #fix
        
        # test invalid back step
        orig = slice(0,200,10)
        new = slice(1, 5, -2)
        sub = subslice(orig, new)
        two_hundred = range(201)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], []) # invalid returns no data
        self.assertEqual(sub, slice(10, 50, -20))
        
        # test no start
        orig = slice(None,100,10)
        new = slice(5, 1, -2)
        sub = subslice(orig, new)
        two_hundred = range(200)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [50,30])
        self.assertEqual(sub, slice(50, 10, -20))

        orig = slice(0,10,2)
        new = slice(None, 4)
        sub = subslice(orig, new)
        two_hundred = range(5)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [0,2,4]) # also tests outside of range
        self.assertEqual(sub, slice(0, 8, 2))
        
        # test None start and invalid back step
        orig = slice(None,200,10)
        new = slice(1, 5, -2)
        sub = subslice(orig, new)
        two_hundred = range(201)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [])
        self.assertEqual(sub, slice(10, 50, -20))

        # test None at end of second slice
        orig = slice(0,10,2)
        new = slice(1, None)
        sub = subslice(orig, new)
        two_hundred = range(5)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [2,4])
        self.assertEqual(sub, slice(2, 20, 2))
                    
        #TODO: test negative start, stop and step

"""
------------------------------------------------------------
Time functions replaced by index operations for consistency.
------------------------------------------------------------

class TestTimeAtValue(unittest.TestCase):
    
    # Reminder: time_at_value (array, hz, offset, scan_start, scan_end, threshold):

    def test_time_at_value_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (time_at_value(array, 1, 0.0, 0, 3, 1.5), 1.5)
        
    def test_time_at_value_backwards(self):
        array = np.ma.arange(8)
        self.assertEquals (time_at_value(array, 1, 0.0, 6, 2, 2.5), 2.5)

    def test_time_at_value_right_at_start(self):
        array = np.ma.arange(4)
        self.assertEquals (time_at_value(array, 1, 0.0, 1, 3, 1.0), 1.0)
                           
    def test_time_at_value_right_at_end(self):
        array = np.ma.arange(4)
        self.assertEquals (time_at_value(array, 1, 0.0, 1, 3, 3.0), 3.0)
        
    def test_time_at_value_threshold_not_crossed(self):
        array = np.ma.arange(4)
        self.assertEquals (time_at_value(array, 1, 0.0, 0, 3, 7.5), None)
        
    def test_time_at_value_masked(self):
        array = np.ma.arange(4)
        array[1] = np.ma.masked
        self.assertEquals (time_at_value(array, 1, 0.0, 0, 3, 1.5), None)
      
        
class TestTimeAtValueWrapped(unittest.TestCase):
    # Reminder: time_at_value_wrapped(parameter, block, value):
  
    def test_time_at_value_wrapped_basic(self):
        test_param = P('TAVW_param',np.ma.array(range(4),dtype=float),1,0.0)
        test_section = Section('TAVW_section',slice(0,4))
        self.assertEquals(time_at_value_wrapped(test_param,test_section,2.5),2.5)

    def test_time_at_value_wrapped_backwards(self):
        test_param = P('TAVW_param',np.ma.array([0,4,0,4]),1,0.0)
        test_section = Section('TAVW_section',slice(0,4))
        self.assertEquals(time_at_value_wrapped(test_param,test_section,2,'Backwards'),2.5)
------------------------------------------------------------
Time functions replaced by index operations for consistency.
------------------------------------------------------------
"""
        
class TestValueAtTime(unittest.TestCase):

    # Reminder: value_at_time (array, hz, offset, time_index)
    
    def test_value_at_time_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (value_at_time(array, 1, 0.0, 2.5), 2.5)
        
    def test_value_at_time_right_at_start_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEquals (value_at_time(array, 1, 0.0, 0.0), 22.3)
        
    def test_value_at_time_right_at_end_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEquals (value_at_time(array, 1.0, 0.0, 3.0), 25.3)
        
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

        
class TestValueAtIndex(unittest.TestCase):

    # Reminder: value_at_time (array, index) This function is thoroughly
    # tested by the higher level value_at_time function, so this single test
    # just establishes confidence in the ability to access the lower level
    # function directly.
    
    def test_value_at_index_basic(self):
        array = np.ma.arange(4)
        self.assertEquals (value_at_index(array, 1.5), 1.5)
        
    def test_value_at_index_above_range(self):
        array = np.ma.arange(4)
        self.assertRaises(ValueError, value_at_index, array, 7)
        
    def test_value_at_index_below_range(self):
        array = np.ma.arange(4)
        self.assertRaises(ValueError, value_at_index, array, -33)
        
    def test_value_at_index_masked(self):
        array = np.ma.arange(4)
        array[2]=np.ma.masked
        expected=np.ma.array(1)
        expected 
        self.assertEquals (value_at_index(array, 2), None)


class TestVstackParams(unittest.TestCase):
    def test_vstack_params(self):
        a = P('a', array=np.ma.array(range(0, 10)))
        b = np.ma.array(range(10,20))
        a.array[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        c = None
        ma_test.assert_array_equal(
            np.ma.filled(vstack_params(a), 99), 
            np.array([[99, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        )
        # test mixed types (Parameter, Masked Array, None)
        ma_test.assert_array_equal(
            np.ma.filled(vstack_params(None, a, b, c), 99),
            np.array([[99,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                      [99, 11, 12, 13, 14, 15, 16, 17, 18, 99]])
        )
        self.assertRaises(ValueError, vstack_params, None, None, None)
                              