try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import csv

# A set of masked array test utilities from Pierre GF Gerard-Marchant
# http://www.java2s.com/Open-Source/Python/Math/Numerical-Python/numpy/numpy/ma/testutils.py.htm
import utilities.masked_array_testutils as ma_test

from datetime import datetime

from analysis.library import (align, calculate_timebase, create_phase_inside,
                              create_phase_outside, duration, 
                              first_order_lag, first_order_washout, hash_array,
                              hysteresis, interleave, merge_alternate_sensors,
                              rate_of_change, repair_mask, straighten_headings,
                              time_at_value, time_at_value_wrapped, value_at_time,
                              InvalidDatetime)

from analysis.node import A, KPV, KTI, Parameter, P, S, Section

class TestAlign(unittest.TestCase):
    
    def test_align_returns_same_array_if_aligned(self):
        slave = P('slave', np.ma.array(range(10)))
        master = P('master', np.ma.array(range(30)))
        aligned = align(slave, master)
        self.assertEqual(id(slave.array), id(aligned))
        
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
        np.testing.assert_array_equal(result.data, [0, 0, 1, 2, 3, 4, 5, 6])
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
                        
    def test_align_assert_array_lengths(self):
        class DumParam():
            def __init__(self):
                self.offset = 0.0
                self.frequency = 1
                self.array = []
                
        first = DumParam()
        first.frequency = 4
        first.array = np.ma.array(range(8))
        second = DumParam()
        second.frequency = 2
        second.array = np.ma.array(range(7)) # Unmatched array length !
        self.assertRaises (AssertionError, align, first, second)
                
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
        '''
        OK. Tricky test that needs some explanation.
        
        The initial value is the steady state input condition prior to the data 
        we supply. This filter is a washout (high pass) filter, so the steady 
        state output will always be zero.
        
        The initial condition is set to -1.0, then when the data arrives, 
        array[0]=0.0 gives a +1.0 step change to the input and we get a positive 
        kick on the output.
        '''
        result = first_order_washout (array, 2.0, 2.0, initial_value = -1.0)
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
        self.assertRaises(ValueError, first_order_washout, array, 1.0, 4.0)

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
        np.testing.assert_array_equal(result.data,[0,0,1,1,1,0,4,5,6,1])
        np.testing.assert_array_equal(result.mask,[0,0,0,0,1,0,0,0,0,0])

    def test_hysteresis_change_of_threshold(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        result = hysteresis(data,1)
        np.testing.assert_array_equal(result.data,[0,0.5,1.5,1.5,0.5,-0.5,4.5,5.5,6.5,0.5])


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
        array = np.ma.array([0, 1, 0])
        self.assertRaises(ValueError, 
                          rate_of_change, 
                          P('Test',np.ma.array([0, 1, 0]), 1), 0)
        
    def test_rate_of_change_half_width_negative(self):
        array = np.ma.array([0, 1, 0])
        self.assertRaises(ValueError, 
                          rate_of_change, 
                          P('Test',np.ma.array([0, 1, 0]), 1), -2)
        
        
class TestRepairMask(unittest.TestCase):
    def test_repair_mask_basic(self):
        array = np.ma.arange(10)
        array[3] = np.ma.masked
        array[6:8] = np.ma.masked
        repair_mask(array)
        np.testing.assert_array_equal(array.data,range(10))
        
    def test_repair_mask_too_much_invalid(self):
        array = np.ma.arange(20)
        array[4:15] = np.ma.masked
        unchanged = array
        repair_mask(array)
        ma_test.assert_masked_array_approx_equal(array, unchanged)
        
    def test_repair_mask_not_at_start(self):
        array = np.ma.arange(10)
        array[0] = np.ma.masked
        unchanged = array
        repair_mask(array)
        ma_test.assert_masked_array_approx_equal(array, unchanged)
        
    def test_repair_mask_not_at_end(self):
        array = np.ma.arange(10)
        array[9] = np.ma.masked
        unchanged = array
        repair_mask(array)
        ma_test.assert_masked_array_approx_equal(array, unchanged)
        
        
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
        
    def test_time_at_value_errors(self):
        array = np.ma.arange(4)
        self.assertRaises(ValueError, time_at_value, array, 1, 0.0, 2, 2, 7.5)
        self.assertRaises(ValueError, time_at_value, array, 1, 0.0, -1, 2, 7.5)
        self.assertRaises(ValueError, time_at_value, array, 1, 0.0, 2, 5, 7.5)
        self.assertRaises(ValueError, time_at_value, array, 1, 0.0, 5, 2, 7.5)
        self.assertRaises(ValueError, time_at_value, array, 1, 0.0, 2, -1, 7.5)
        
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


'''
ma_test.assert_masked_array_approx_equal(result, answer)

self.assertAlmostEquals(result.data[-1], 10.0)

    def test_firstorderlag_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_lag, array, 1.0, 4.0)
'''