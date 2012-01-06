import mock
import numpy as np
import unittest

from datetime import datetime

from analysis.node import P
from analysis.settings import AIRSPEED_THRESHOLD
from analysis.split_segments import (append_segment_info, split_segments2, 
                                     _identify_segment_type, 
                                     _split_by_frame_counter, 
                                     _split_by_flight_data)

class TestSplitSegments(unittest.TestCase):
    
    def test_split_segments(self):
        a_flight = [0]*50 + [100]*100 + [0]*50 
        # 5 * 200 (flight) samples of airspeed
        airspeed = a_flight * 5
        
        # 1000 samples of dfc
        dfc = range(1,392) + range(2010,2600)
        segs = split_segments2(P('Airspeed', np.ma.array(airspeed)), 
                              dfc=P('Frame Counter', np.ma.array(dfc), 0.25))
        self.assertEqual(len(segs), 5)
        exp = [slice(0, 200),
               slice(200, 390),
               slice(390, 600),
               slice(600, 800),
               slice(800, None),
               ]
        self.assertEqual([s for s in segs], exp)
                         
    
    def test_split_flights_by_frame_counter(self):
        normal_range = range(0,400) # creates values in list 0 to 399
        res = _split_by_frame_counter(normal_range)
        self.assertEqual(res, [slice(0,1600)])
        
        normal_wrap = [4093, 4094, 4095, 1, 2, 3]
        res = _split_by_frame_counter(normal_wrap)
        self.assertEqual(res, [slice(0,24)])
                         
        flat_line = [100, 101, 101, 101, 102]
        res = _split_by_frame_counter(flat_line)
        self.assertEqual(res, [slice(0,20)])
        
        jump_fwd = [100,101,2000,2001]
        res = _split_by_frame_counter(jump_fwd)
        self.assertEqual(res, [slice(0,8), slice(8,16)])
        
        jump_back = [2000,2001,100,101]
        res = _split_by_frame_counter(jump_back)        
        self.assertEqual(res, [slice(0,8), slice(8,16)])
        
        #TODO: test mixed = [4094, 4095, 4096, 1, 2, 3, 3, 3, 400] #??
        
    def test_split_by_flight_data(self):
        #Two offset: 2  3   4   5  6  7   8   9 10 11 12  13  14  15 16
        airspeed = [10,10,200,200,10,10,200,200,10,10,10,200,200,200,10]
        mask_below_min_aispeed = np.ma.masked_less(airspeed, AIRSPEED_THRESHOLD)
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
        airspeed = P('Airspeed', np.ma.array(airspeed_data))
        
        segs = split_segments(airspeed, dfc=None)
        # test 7 complete flights returned
        self.assertEqual(len(segs), 7)
        
        # Note: These slices are due to DFC jumping - test case to be updated with correct values!!
        exp = [slice(0, 2559, None),
               slice(2559, 8955, None),
               slice(8955, 15371, None),
               slice(15371, 21786, None),
               slice(21786, 27731, None),
               slice(27731, 34103, None),
               slice(34103, 41396, None)]
        self.assertEqual(segs, exp)
        
    def test_split_segments_with_dodgy_dfc(self):
        # Question: Is this a valid test? In real-life you'd use the test above (ignoring the DFC)
        airspeed_data = np.load('test_data/airspeed_sample.npy')
        airspeed = P('Airspeed', np.ma.array(airspeed_data))
        dfc_data = np.load('test_data/dfc_sample.npy')
        dfc = P('Frame Counter', np.ma.array(dfc_data), 0.25)
        
        segs = split_segments(airspeed, dfc=dfc)
        # test 7 complete flights returned
        
        # Note: These slices are due to DFC jumping - test case to be updated with correct values!!
        exp = [slice(0, 400, None),
               slice(400, 2168, None),
               slice(2168, 3192, None),
               slice(3192, 9340, None),
               slice(9340, 15316, None),
               slice(15316, 15952, None),
               slice(15952, 22124, None),
               slice(22124, 27452, None),
               slice(27452, 27788, None),
               slice(27788, 28400, None),
               slice(28400, 33872, None),
               slice(33872, 34764, None),
               slice(34764, 41396, None)]
        self.assertEqual(segs, exp)
        
    def test_split_segments_1hz_dfc(self):
        airspeed_data = np.load('test_data/airspeed_1hz_3_L382-Hercules.npy')
        airspeed = P('Airspeed', np.ma.array(airspeed_data), frequency=1, offset=0.31)
        dfc_data = np.load('test_data/dfc_1hz_3_L382-Hercules.npy')
        dfc = P('Frame Counter', np.ma.array(dfc_data), frequency=1, offset=0.20)
        segs = split_segments(airspeed, dfc=dfc)
        self.assertEqual(len(segs), 1)
        
        

        
class TestIdentifySegment(unittest.TestCase):
    def test_ground_only(self):
        # test all slow
        slow = np.ma.array(range(0,75) + range(75,0,-1))
        self.assertEqual(_identify_segment_type(slow), 'GROUND_ONLY')

        # test with all invalid data
        invalid_airspeed = np.ma.array(range(50,100) + range(100,50,-1), mask=[True]*100)
        self.assertEqual(_identify_segment_type(invalid_airspeed), 'GROUND_ONLY')

        # test mid-flight
        mid_flight = np.ma.array(range(100,200) + range(200,100,-1))
        self.assertEqual(_identify_segment_type(mid_flight), 'MID_FLIGHT')

        # test stop only
        # test start only
        # test stop and start
        airspeed_data = np.load('test_data/airspeed_sample.npy')
        airspeed = P('Airspeed', np.ma.array(airspeed_data))
        segs = split_segments(airspeed, dfc=None)        
        self.assertEqual([_identify_segment_type(airspeed.array[s]) for s in segs], 
                         ['START_AND_STOP']*7)
        
        
        
class TestSegmentInfo(unittest.TestCase):
    def setUp(self):
        import analysis.split_segments as splitseg
        class mocked_hdf(object):
            def __init__(self, path):
                self.path = path
                if path == 'slow':
                    self.airspeed = np.ma.array(range(10,20)*5)
                else:
                    self.airspeed = np.ma.array(
                        np.load('test_data/4_3377853_146-301_airspeed.npy'))
                self.duration = len(self.airspeed)
                
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
            
            def __getitem__(self, key):
                if key == 'Airspeed':
                    data = self.airspeed
                
                if self.path == 'invalid timestamps':
                    if key == 'Year':
                        data = np.ma.array([0] * 60)
                    elif key == 'Month':
                        data = np.ma.array([13] * 60)
                    elif key == 'Day':
                        data = np.ma.array([31] * 60)
                    else:
                        data = np.ma.array(range(1,59))
                else:
                    if key == 'Year':
                        data = np.ma.array([2020] * 60)
                    elif key == 'Month':
                        data = np.ma.array([12] * 60)
                    elif key == 'Day':
                        data = np.ma.array([25] * 60)
                    else:
                        data = np.ma.array(range(1,59))
                return P(key, array=data)
            
        splitseg.hdf_file = mocked_hdf
        splitseg.sha_hash_file = mock.Mock()
        splitseg.sha_hash_file.return_value = 'ABCDEFG'
        
    def test_append_segment_info(self):
        # example where it goes fast
        seg = append_segment_info('fast', slice(10,1000), 4) # TODO: Increase slice to be realitic for duration of data
        self.assertEqual(seg.path, 'fast')
        self.assertEqual(seg.part, 4)
        self.assertEqual(seg.type, 'START_AND_STOP')   
        self.assertEqual(seg.start_dt, datetime(2020,12,25,1,1,1))
        self.assertEqual(seg.go_fast_dt, datetime(2020,12,25,3,21,54)) # this is not right!
        self.assertEqual(seg.stop_dt, datetime(2020,12,26,17,52,45))
                         
    def test_append_segment_info_no_gofast(self):
        # example where it does not go fast
        seg = append_segment_info('slow', slice(10,110), 1)
        self.assertEqual(seg.path, 'slow')
        self.assertEqual(seg.go_fast_dt, None) # didn't go fast
        self.assertEqual(seg.start_dt, datetime(2020,12,25,1,1,1)) # still has a start
        self.assertEqual(seg.part, 1)
        self.assertEqual(seg.type, 'GROUND_ONLY')
        self.assertEqual(seg.hash, 'ABCDEFG') # taken from the "file"
        self.assertEqual(seg.stop_dt, datetime(2020,12,25,1,1,51)) # +50 seconds of airspeed
        
    def test_invalid_datetimes(self):
        seg = append_segment_info('invalid timestamps', slice(10,110), 2)
        self.assertEqual(seg.start_dt, datetime(1970,1,1,1,0)) # start of time!
        self.assertEqual(seg.go_fast_dt, datetime(1970, 1, 1, 3, 20, 53)) # went fast