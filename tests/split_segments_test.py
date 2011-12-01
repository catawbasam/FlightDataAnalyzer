try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from analysis import settings
from analysis.plot_flight import plot_parameter
from analysis.split_segments import split_segments, subslice,\
     _split_by_frame_counter, _split_by_flight_data

class TestSplitSegments(unittest.TestCase):
    
    def test_split_segments(self):
        a_flight = [0]*50 + [100]*100 + [0]*50 
        # 5 * 200 (flight) samples of airspeed
        airspeed = a_flight * 5
        
        # 1000 samples of dfc
        dfc = range(100,390) + range(2010,2600)
        segs = split_segments(np.ma.array(airspeed), dfc=np.ma.array(dfc))
        self.assertEqual(len(segs), 5)  # Fails with 6 as splits on DFC also!
        exp = [slice(0, 200),
               slice(200, 390), #200,290
               slice(2010, 2210), #0,110
               slice(2210, 2400), #110,310
               slice(2400, 2600),
               ] #31
        self.assertEqual([s.slice for s in segs], exp)
                         
    
    def test_split_flights_by_frame_counter(self):
        normal_range = range(0,400) # creates values in list 0 to 399
        res = _split_by_frame_counter(normal_range)
        self.assertEqual(res, [slice(0,400)])
        
        normal_wrap = [4094, 4095, 4096, 1, 2, 3]
        res = _split_by_frame_counter(normal_wrap)
        self.assertEqual(res, [slice(0,6)])
                         
        flat_line = [100, 101, 101, 101, 102]
        res = _split_by_frame_counter(flat_line)
        self.assertEqual(res, [slice(0,5)])
        
        jump_fwd = [100,101,2000,2001]
        res = _split_by_frame_counter(jump_fwd)
        self.assertEqual(res, [slice(0,2), slice(2,4)])
        
        jump_back = [2000,2001,100,101]
        res = _split_by_frame_counter(jump_back)        
        self.assertEqual(res, [slice(0,2), slice(2,4)])
        
        #TODO: test mixed = [4094, 4095, 4096, 1, 2, 3, 3, 3, 400] #??
        
    def test_split_by_flight_data(self):
        #Two offset: 2  3   4   5  6  7   8   9 10 11 12  13  14  15 16
        airspeed = [10,10,200,200,10,10,200,200,10,10,10,200,200,200,10]
        mask_below_min_aispeed = np.ma.masked_less(airspeed, settings.AIRSPEED_THRESHOLD)
        res = _split_by_flight_data(mask_below_min_aispeed, 2)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], slice(2,7))
        self.assertEqual(res[1], slice(7,12))
        self.assertEqual(res[2], slice(12,17))
        
    def test_split_by_flight_data_no_splits(self):
        # try without masked array
        airspeed_high = np.ma.array([100,100,100,100])
        res = _split_by_flight_data(airspeed_high, 0)
        self.assertEqual(res[0], slice(0,4))
        
        # try with fully masked array
        airspeed_high = np.ma.array([100,100,100,100])
        # NOTE: Array isn't masked fully until we touch an ellement in the array
        airspeed_high[0] = np.ma.masked
        airspeed_high[0] = np.ma.nomask
        res = _split_by_flight_data(airspeed_high, 0)
        self.assertEqual(res[0], slice(0,4))
        
    def test_split_by_flight_data_low_airspeed(self):
        # no splits expected
        airspeed_low = np.ma.array([10,10,10,10,10])
        res = _split_by_flight_data(airspeed_low, 0)
        self.assertEqual(res[0], slice(0,5))
        
    def test_split_segments_no_dfc(self):
        airspeed_data = np.load('test_data/airspeed_sample.npy')
        airspeed = np.ma.array(airspeed_data)
        
        segs = split_segments(airspeed, dfc=None)
        # test 7 complete flights returned
        self.assertEqual(len(segs), 7)
        self.assertEqual([s.type for s in segs], ['START-AND-STOP']*7)
        
        # Note: These slices are due to DFC jumping - test case to be updated with correct values!!
        exp = [slice(0, 2559, None),
               slice(2559, 8955, None),
               slice(8955, 15371, None),
               slice(15371, 21786, None),
               slice(21786, 27731, None),
               slice(27731, 34103, None),
               slice(34103, 41396, None)]
        self.assertEqual([seg.slice for seg in segs], exp)
        
    def test_split_segments_with_dodgy_dfc(self):
        # Question: Is this a valid test? In real-life you'd use the test above (ignoring the DFC)
        airspeed_data = np.load('test_data/airspeed_sample.npy')
        airspeed = np.ma.array(airspeed_data)
        dfc_data = np.load('test_data/dfc_sample.npy')
        dfc = np.ma.array(dfc_data)
        
        segs = split_segments(airspeed, dfc=dfc)
        # test 7 complete flights returned
        
        # Note: These slices are due to DFC jumping - test case to be updated with correct values!!
        exp = [slice(0, 100), slice(100, 542), slice(542, 798), 
               slice(798, 2335), slice(2335, 3829), slice(3829, 3988),
               slice(3988, 5531), slice(5531, 6863), slice(6863, 6947),
               slice(6947, 7100), slice(7100, 8468), slice(8468, 8691), 
               slice(8691, 10349)]
        self.assertEqual([seg.slice for seg in segs], exp)
        
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
        self.assertEqual(sub, slice(2, 10, 2))
                    
        #TODO: test negative start, stop and step
        
    