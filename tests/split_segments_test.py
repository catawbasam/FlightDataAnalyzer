try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from analysis.split_segments import split_segments, subslice,\
     _split_by_frame_counter, _split_by_flight_data

class TestSplitSegments(unittest.TestCase):
    
    def test_split_segments(self):
        a_flight = [0]*50 + [100]*100 + [0]*50 
        # 1000 samples of dfc
        dfc = range(100,390) + range(2010,2600)
        # 5 * 200 (flight) samples of airspeed
        airspeed = a_flight * 5
        res = split_segments(dfc, airspeed)
        self.assertEqual(res, [slice(0, 200),
                              slice(200, 390), #200,290
                              slice(2010, 2210), #0,110
                              slice(2210, 2400), #110,310
                              slice(2400, 2600),
                              ]) #31
    
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
        #           0   1  2   3   4  5  6   7   8  9 10  11  12  13 14
        airspeed = [10,10,200,200,10,10,200,200,10,10,10,200,200,200,10]
        res = _split_by_flight_data(airspeed)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], slice(0,5))
        self.assertEqual(res[1], slice(5,9))
        self.assertEqual(res[2], slice(9,15))
        
    def test_split_by_flight_data_no_splits(self):
        # try without masked array
        airspeed_high = [100,100,100,100]
        res = _split_by_flight_data(airspeed_high)
        self.assertEqual(res[0], slice(0,4))
        
        # try with fully masked array
        airspeed_high = np.ma.array([100,100,100,100])
        # NOTE: Array isn't masked fully until we touch an ellement in the array
        airspeed_high[0] = np.ma.masked
        airspeed_high[0] = np.ma.nomask
        res = _split_by_flight_data(airspeed_high)
        self.assertEqual(res[0], slice(0,4))
        
    def test_split_by_flight_data_low_airspeed(self):
        # no splits expected
        airspeed_low = [10,10,10,10,10]
        res = _split_by_flight_data(airspeed_low)
        self.assertEqual(res[0], slice(0,5))
        
    def test_subslice(self):
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
        self.assertEqual(sub, slice(0, 4, 2))
        
        # test None start and invalid back step
        orig = slice(None,200,10)
        new = slice(1, 5, -2)
        sub = subslice(orig, new)
        two_hundred = range(201)
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [])
        self.assertEqual(sub, slice(110, 150, -20))


                    
        # test no end
        
    