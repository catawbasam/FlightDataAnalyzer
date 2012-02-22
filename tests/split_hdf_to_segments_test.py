import mock
import numpy as np
import os.path
import unittest

from datetime import datetime

from analysis_engine.split_hdf_to_segments import (
    _calculate_start_datetime, append_segment_info, split_segments)
from analysis_engine.node import P,  Parameter

from hdfaccess.file import hdf_file

class TestSplitSegments(unittest.TestCase):
    @mock.patch('analysis_engine.split_segments.hysteresis')
    def test_split_segments(self, hysteresis_patched):
        # Do not apply hysteresis as test results will become obfuscated.
        hysteresis_patched.side_effect = lambda x,y: x        
        # Mock hdf
        airspeed_array = np.ma.concatenate([np.ma.arange(200),
                                            np.ma.arange(200, 0, -1)])
        
        airspeed_frequency = 2
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        
        heading_array = np.ma.zeros(len(airspeed_array) / 4)
        heading_frequency = 1
        heading_array.mask = True
        
        dfc_array = np.ma.arange(0, 200, 2)
        
        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.return_value = None
        hdf.reliable_frame_counter = False
        
        def hdf_getitem(self, key):
            if key == 'Airspeed':
                return Parameter('Airspeed', array=airspeed_array,
                                 frequency=airspeed_frequency)
            elif key == 'Frame Counter':
                return Parameter('Frame Counter', array=dfc_array,
                                 frequency=0.25)
            elif key == 'Heading':
                # TODO: Give heading specific data.
                return Parameter('Heading', array=heading_array,
                                 frequency=heading_frequency)
            else:
                raise KeyError
        hdf.__getitem__ = hdf_getitem
        
        # Unmasked single flight.
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # DFC should not affect result.
        hdf.reliable_frame_counter = True
        # Mask within slow data should not affect result.
        airspeed_array[:50] = np.ma.masked
        airspeed_array[-50:] = np.ma.masked
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked beginning of speedy data will affect result.
        airspeed_array[:100] = np.ma.masked
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]         
        self.assertEqual(segment_type, 'STOP_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked end of speedy data will affect result.
        airspeed_array = np.ma.concatenate([np.ma.arange(200),
                                            np.ma.arange(200, 0, -1)])
        airspeed_array[-100:] = np.ma.masked
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked beginning and end of speedy data will affect result.
        airspeed_array[:100] = np.ma.masked
        airspeed_array[-100:] = np.ma.masked
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'MID_FLIGHT')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Airspeed always slow.
        airspeed_array = np.ma.concatenate([np.ma.arange(50),
                                            np.ma.arange(50, 0, -1)])
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'GROUND_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        
        # Two flights, split will be made using DFC.
        airspeed_array = np.ma.concatenate([np.ma.arange(0, 200, 0.5),
                                            np.ma.arange(200, 0, -0.5),
                                            np.ma.arange(0, 200, 0.5),
                                            np.ma.arange(200, 0, -0.5),])
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        # DFC jumps exactly half way.
        dfc_array = np.ma.concatenate([np.ma.arange(0, 100),
                                       np.ma.arange(200, 300)])
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 2)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 398)
        segment_type, segment_slice = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 398)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        
        # Split using Turning where DFC does not jump.
        dfc_array = np.ma.concatenate([np.ma.arange(4000, 4096),
                                       np.ma.arange(0, 105)])
        heading_array = np.ma.concatenate([np.ma.arange(390, 0, -1),
                                           np.ma.zeros(10),
                                           np.ma.arange(400, 800)])
        segment_tuples = split_segments(hdf)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 395)
        segment_type, segment_slice = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 395)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        
        # Same split conditions, but does not split on jumping DFC because 
        # reliable_frame_counter is False.
        hdf.reliable_frame_counter = False
        dfc_array = np.ma.masked_array(np.random.randint(1000, size=((len(dfc_array),))))
        segment_tuples = split_segments(hdf)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 395)
        segment_type, segment_slice = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 395)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        
        # TODO: Test engine parameters.
    
    @unittest.skipIf(not os.path.isfile("test_data/1_7295949_737-3C.hdf5"),
                     "Test file not present")
    def test_split_segments_737_3C(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf = hdf_file("test_data/1_7295949_737-3C.hdf5")
        segment_tuples = split_segments(hdf)
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3168.0, None)),
                          ('START_AND_STOP', slice(3168.0, 6014.0, None)),
                          ('START_AND_STOP', slice(6014.0, 9504.0, None)),
                          ('START_AND_STOP', slice(9504.0, 12373.0, None)),
                          ('START_AND_STOP', slice(12373.0, 15410.0, None)),
                          ('START_AND_STOP', slice(15410.0, 18752.0, None))])
    
    @unittest.skipIf(not os.path.isfile("test_data/4_3377853_146-301.hdf5"),
                     "Test file not present")
    def test_split_segments_146_300(self):
        hdf = hdf_file("test_data/4_3377853_146-301.hdf5")
        segment_tuples = split_segments(hdf)
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3168.0, None)),
                          ('START_AND_STOP', slice(3168.0, 6014.0, None)),
                          ('START_AND_STOP', slice(6014.0, 9504.0, None)),
                          ('START_AND_STOP', slice(9504.0, 12373.0, None)),
                          ('START_AND_STOP', slice(12373.0, 15410.0, None)),
                          ('START_AND_STOP', slice(15410.0, 18752.0, None))])    
    
    
class TestSegmentInfo(unittest.TestCase):
    def setUp(self):
        import analysis_engine.split_segments as splitseg
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
        seg = append_segment_info('fast', 'START_AND_STOP', slice(10,1000), 4) # TODO: Increase slice to be realitic for duration of data
        self.assertEqual(seg.path, 'fast')
        self.assertEqual(seg.part, 4)
        self.assertEqual(seg.type, 'START_AND_STOP')   
        self.assertEqual(seg.start_dt, datetime(2020,12,25,1,1,1))
        self.assertEqual(seg.go_fast_dt, datetime(2020,12,25,3,21,54)) # this is not right!
        self.assertEqual(seg.stop_dt, datetime(2020,12,26,17,52,45))
                         
    def test_append_segment_info_no_gofast(self):
        # example where it does not go fast
        seg = append_segment_info('slow', 'GROUND_ONLY', slice(10,110), 1)
        self.assertEqual(seg.path, 'slow')
        self.assertEqual(seg.go_fast_dt, None) # didn't go fast
        self.assertEqual(seg.start_dt, datetime(2020,12,25,1,1,1)) # still has a start
        self.assertEqual(seg.part, 1)
        self.assertEqual(seg.type, 'GROUND_ONLY')
        self.assertEqual(seg.hash, 'ABCDEFG') # taken from the "file"
        self.assertEqual(seg.stop_dt, datetime(2020,12,25,1,1,51)) # +50 seconds of airspeed
        
    def test_invalid_datetimes(self):
        seg = append_segment_info('invalid timestamps', '', slice(10,110), 2)
        self.assertEqual(seg.start_dt, datetime(1970,1,1,1,0)) # start of time!
        self.assertEqual(seg.go_fast_dt, datetime(1970, 1, 1, 3, 20, 53)) # went fast

    #TODO: Test using fallback_dt
    
    def test_calculate_start_datetime(self):
        """
        """
        hdf = {'Year':  P('Year',np.ma.array([2011])),
               'Month': P('Month',np.ma.array([11])),
               'Day':   P('Day',np.ma.array([11])),
               'Hour':  P('Hour',np.ma.array([11])),
               'Minute':P('Minute',np.ma.array([11])),
               'Second':P('Second',np.ma.array([11]))
               }
        dt = datetime(2012,12,12,12,12,12)
        # test with all params
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2011,11,11,11,11,11))
        # test without Year
        del hdf['Year']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,11,11,11,11,11))
        # test without Month
        del hdf['Month']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,12,11,11,11,11))
        # test without Day
        del hdf['Day']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,12,12,11,11,11))
        # test without Hour
        del hdf['Hour']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,12,12,12,11,11))
        # test without Minute
        del hdf['Minute']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,12,12,12,12,11))
        # test without Second
        del hdf['Second']
        res = _calculate_start_datetime(hdf, dt)
        self.assertEqual(res, datetime(2012,12,12,12,12,12))
        
    def test_empty_year_no_seconds(self):
        dt = datetime(2012,12,12,12,12,10)
        # Test only without second and empty year
        hdf = {
               'Month': P('Month',np.ma.array([11, 11, 11,11])),
               'Day':   P('Day',np.ma.array([])),
               'Hour':  P('Hour',np.ma.array([11,11,11,11], mask=[True, False, False, False])),
               'Minute':P('Minute',np.ma.array([11]), frequency=0.25),
               }
        res = _calculate_start_datetime(hdf, dt)
        # 9th second as the first sample (10th second) was masked
        self.assertEqual(res, datetime(2012,11,12,11,11,9))