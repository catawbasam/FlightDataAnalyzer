import mock
import numpy as np
import os.path
import unittest

from datetime import datetime

from analysis_engine.split_hdf_to_segments import (
    _calculate_start_datetime,
    append_segment_info,
    split_segments,
    TimebaseError)
from analysis_engine.node import P,  Parameter

from hdfaccess.file import hdf_file
from flightdatautilities.filesystem_tools import copy_file

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


class TestSplitSegments(unittest.TestCase):
    def test_split_segments(self): 
        # TODO: Test engine param splitting.
        # Mock hdf
        airspeed_array = np.ma.concatenate([np.ma.arange(200),
                                            np.ma.arange(200, 0, -1)])
        
        airspeed_frequency = 2
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        
        heading_array = np.ma.zeros(len(airspeed_array) / 2)
        heading_frequency = 1
        heading_array.mask = False
        
        eng_array = None
        eng_frequency = 1
        
        dfc_array = np.ma.arange(0, 200, 2)
        
        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.return_value = None
        hdf.reliable_frame_counter = False
        
        def hdf_getitem(self, key, **kwargs):
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
            elif key == 'Eng (1) N1' and eng_array is not None:
                return Parameter('Eng (1) N1', array=eng_array,
                                 frequency=eng_frequency)
            else:
                raise KeyError
        hdf.__getitem__ = hdf_getitem
        
        def hdf_get_param(key, valid_only=False):
            # Pretend that we recorded Heading True only on this aircraft
            if key == 'Heading':
                raise KeyError
            elif key == 'Heading True':
                return Parameter('Heading True', array=heading_array,
                                 frequency=heading_frequency)
        hdf.get_param = hdf_get_param
        
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
        
        
        # Split using engine params where DFC does not jump.
        eng_array = np.ma.concatenate([np.ma.arange(0, 100, 0.5),
                                       np.ma.arange(100, 0, -0.5),
                                       np.ma.arange(0, 100, 0.5),
                                       np.ma.arange(100, 0, -0.5),])
        segment_tuples = split_segments(hdf)
        segment_type, segment_slice = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 398.0)
        segment_type, segment_slice = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 398.0)
        self.assertEqual(segment_slice.stop, airspeed_secs)        
        
        # Split using Turning where DFC does not jump.
        dfc_array = np.ma.concatenate([np.ma.arange(4000, 4096),
                                       np.ma.arange(0, 105)])
        heading_array = np.ma.concatenate([np.ma.arange(390, 0, -1),
                                           np.ma.zeros(10),
                                           np.ma.arange(400, 800)])
        eng_array = None
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
    
    @unittest.skipIf(not os.path.isfile(os.path.join(test_data_path, 
                                                     "1_7295949_737-3C.hdf5")),
                     "Test file not present")
    def test_split_segments_737_3C(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf = hdf_file(os.path.join(test_data_path, "1_7295949_737-3C.hdf5"))
        segment_tuples = split_segments(hdf)
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3168.0, None)),
                          ('START_AND_STOP', slice(3168.0, 6260.0, None)),
                          ('START_AND_STOP', slice(6260.0, 9504.0, None)),
                          ('START_AND_STOP', slice(9504.0, 12680.0, None)),
                          ('START_AND_STOP', slice(12680.0, 15571.0, None)),
                          ('START_AND_STOP', slice(15571.0, 18752.0, None))])
    
    def test_split_segments_data_1(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf_path = os.path.join(test_data_path, "split_segments_1.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)
        
        segment_tuples = split_segments(hdf)
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 9952.0, None)),
                          ('START_AND_STOP', slice(9952.0, 21751.0, None)),
                          ('START_AND_STOP', slice(21751.0, 24665.0, None)),
                          ('START_AND_STOP', slice(24665.0, 27894.0, None)),
                          ('START_AND_STOP', slice(27894.0, 31424.0, None))])
    
    @unittest.skipIf(not os.path.isfile(os.path.join(test_data_path,
                                                     "4_3377853_146-301.hdf5")),
                     "Test file not present")
    def test_split_segments_146_300(self):
        hdf = hdf_file(os.path.join(test_data_path, "4_3377853_146-301.hdf5"))
        segment_tuples = split_segments(hdf)
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 24801.0, None)),
                          ('START_AND_STOP', slice(24801.0, 30000.0, None)),
                          ('START_AND_STOP', slice(30000.0, 49999.0, None)),
                          ('START_AND_STOP', slice(49999.0, 69999.0, None)),
                          ('START_AND_STOP', slice(69999.0, 73552.0, None))])


    def test_split_segments_multiple_types(self):
        '''
        Test data has multiple segments of differing segment types.
        Test data has already been validated
        '''
        hdf_path = os.path.join(test_data_path, "split_segments_multiple_types.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)
        self.maxDiff = None
        segment_tuples = split_segments(hdf)
        self.assertEqual(len(segment_tuples), 16, msg="Unexpected number of segments detected")
        segment_types = tuple(x[0] for x in segment_tuples)
        self.assertEqual(segment_types,
                         ('STOP_ONLY',
                          'START_ONLY',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'STOP_ONLY',
                          'START_AND_STOP',
                          'STOP_ONLY',
                          'START_ONLY',
                          'START_ONLY',
                          'START_AND_STOP',
                          'START_ONLY',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'START_ONLY'))


class mocked_hdf(object):
    def __init__(self, path=None):
        pass
    
    def __call__(self, path):
        self.path = path
        if path == 'slow':
            self.airspeed = np.ma.array(range(10,20)*5)
        else:
            self.airspeed = np.ma.array(
                np.load(os.path.join(test_data_path, 'airspeed_sample.npy')))
        self.duration = len(self.airspeed)
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def get(self, key, default=None):
        return self.__getitem__(key)
    
    def __contains__(self, key):
        return True
    
    def __getitem__(self, key):
        if key == 'Airspeed':
            data = self.airspeed
        else:
            if self.path == 'invalid timestamps':
                if key == 'Year':
                    data = np.ma.array([0] * 60)
                elif key == 'Month':
                    data = np.ma.array([13] * 60)
                elif key == 'Day':
                    data = np.ma.array([31] * 60)
                else:
                    data = np.ma.array(range(0,60))
            else:
                if key == 'Year':
                    if self.path == 'future timestamps':
                        data = np.ma.array([2020] * 60)
                    elif self.path == 'old timestamps':
                        data = np.ma.array([1999] * 60)
                    else:
                        data = np.ma.array([2012] * 60)
                elif key == 'Month':
                    data = np.ma.array([12] * 60)
                elif key == 'Day':
                    data = np.ma.array([25] * 60)
                else:
                    data = np.ma.array(range(0,60))
        return P(key, array=data)
    
    
class TestSegmentInfo(unittest.TestCase):
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file',
                new_callable=mocked_hdf)
    def test_timestamps_in_past(self, hdf_file_patch, sha_hash_file_patch):
        # Using Fallback Datetime is no longer recommended
        ### example where it goes fast
        ##seg = append_segment_info('old timestamps', 'START_AND_STOP', 
                                  ##slice(10,1000), 4,
                                  ##fallback_dt=datetime(2012,12,12,0,0,0))  
        ##self.assertEqual(seg.start_dt, datetime(2012,12,12,0,0,0))
        ##self.assertEqual(seg.go_fast_dt, datetime(2012,12,12,0,6,52))
        ##self.assertEqual(seg.stop_dt, datetime(2012,12,12,11,29,56))
        # Raising exception is more pythonic
        self.assertRaises(TimebaseError, append_segment_info,
                          'old timestamps', 'START_AND_STOP', slice(10,1000), 
                          4, fallback_dt=datetime(2012,12,12,0,0,0))  


    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file',
                new_callable=mocked_hdf)
    def test_timestamps_in_future_use_fallback(self, hdf_file_patch, sha_hash_file_patch):
        # Using fallback time is no longer recommended
        ### example where it goes fast
        ##seg = append_segment_info('future timestamps', 'START_AND_STOP', 
                                  ##slice(10,1000), 4,
                                  ##fallback_dt=datetime(2012,12,12,0,0,0))
        ##self.assertEqual(seg.start_dt, datetime(2012,12,12,0,0,0))
        ##self.assertEqual(seg.go_fast_dt, datetime(2012,12,12,0,6,52))
        ##self.assertEqual(seg.stop_dt, datetime(2012,12,12,11,29,56))
        # Raising exception is more pythonic
        self.assertRaises(TimebaseError, append_segment_info,
                          'future timestamps', 'START_AND_STOP', slice(10,1000), 
                          4, fallback_dt=datetime(2012,12,12,0,0,0))  
        
        
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file',
                new_callable=mocked_hdf)
    def test_append_segment_info(self, hdf_file_patch, sha_hash_file_patch):
        # example where it goes fast
        # TODO: Increase slice to be realitic for duration of data
        seg = append_segment_info('fast', 'START_AND_STOP', slice(10,1000), 4)
        self.assertEqual(seg.path, 'fast')
        self.assertEqual(seg.part, 4)
        self.assertEqual(seg.type, 'START_AND_STOP')   
        self.assertEqual(seg.start_dt, datetime(2012,12,25,0,0,0))
        self.assertEqual(seg.go_fast_dt, datetime(2012,12,25,0,6,52))
        self.assertEqual(seg.stop_dt, datetime(2012,12,25,11,29,56))
    
    
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file',
                new_callable=mocked_hdf)
    def test_append_segment_info_no_gofast(self, hdf_file_patch,
                                           sha_hash_file_patch):
        sha_hash_file_patch.return_value = 'ABCDEFG'
        # example where it does not go fast
        seg = append_segment_info('slow', 'GROUND_ONLY', slice(10,110), 1)
        self.assertEqual(seg.path, 'slow')
        self.assertEqual(seg.go_fast_dt, None) # didn't go fast
        self.assertEqual(seg.start_dt, datetime(2012,12,25,0,0,0)) # still has a start
        self.assertEqual(seg.part, 1)
        self.assertEqual(seg.type, 'GROUND_ONLY')
        self.assertEqual(seg.hash, 'ABCDEFG') # taken from the "file"
        self.assertEqual(seg.stop_dt, datetime(2012,12,25,0,0,50)) # +50 seconds of airspeed
    
    
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file',
                new_callable=mocked_hdf)  
    def test_invalid_datetimes(self, hdf_file_patch, sha_hash_file_patch):
        # No longer using epoch, raising exception instead
        ##seg = append_segment_info('invalid timestamps', '', slice(10,110), 2)
        ##self.assertEqual(seg.start_dt, datetime(1970,1,1,1,0)) # start of time!
        ##self.assertEqual(seg.go_fast_dt, datetime(1970, 1, 1, 1, 6, 52)) # went fast
        self.assertRaises(TimebaseError, append_segment_info,
                          'invalid timestamps', '', slice(10,110), 2)  
    
    def test_calculate_start_datetime(self):
        """
        """
        hdf = {
            'Year':  P('Year',np.ma.array([2011])),
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
        # NB: 12's are the fallback_dt, 11's are the recorded time parameters
        dt = datetime(2012,12,12,12,12,10)
        # Test only without second and empty year
        hdf = {
               'Month': P('Month',np.ma.array([11, 11, 11,11])),
               'Day':   P('Day',np.ma.array([])),
               'Hour':  P('Hour',np.ma.array([11,11,11,11], mask=[True, False, False, False])),
               'Minute':P('Minute',np.ma.array([11,11]), frequency=0.5),
               }
        res = _calculate_start_datetime(hdf, dt)
        # 9th second as the first sample (10th second) was masked
        self.assertEqual(res, datetime(2012,11,12,11,11,9))
        
        
    def test_no_year_with_a_very_recent_fallback(self):
        """When fallback is a year after the flight and Year is not recorded,
        the result would be in the future. Ensure that a year is taken off if
        required.
        
        Generally Day and Month are recorded together and Year is optional.
        
        Should only Time be recorded, using the fallback date part is as good
        as you get.
        """
        # ensure current datetime is very recent
        dt = datetime.now()
        # Year is not recorded, and the data is for the very end of the
        # previous year. NB: This test could fail if ran on the very last day
        # of the month!
        hdf = {
               'Month': P('Month', np.ma.array([12, 12])), # last month of year
               'Day':   P('Day',   np.ma.array([31, 31])), # last day
               'Hour':  P('Hour',  np.ma.array([23, 23])), # last hour
               'Minute':P('Minute',np.ma.array([59, 59])), # last minute
               'Second':P('Minute',np.ma.array([58, 59])), # last two seconds
               }
        res = _calculate_start_datetime(hdf, dt)
        # result is a year behind the fallback datetime, even though the
        # fallback Year was used.
        self.assertEqual(res.year, datetime.now().year -1)
        #self.assertEqual(res, datetime(2012,6,01,11,11,1))