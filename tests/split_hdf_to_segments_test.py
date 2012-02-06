import os.path
import unittest

from analysis_engine.split_hdf_to_segments import (validate_aircraft,
                                                   split_hdf_to_segments)


class TestSplitHDFToSegments(unittest.TestCase):
    
    
    @unittest.skipIf(not os.path.isfile("test_data/1_7295949_737-3C.hdf5"), 
                     'Test file not present')
    def test_1_7295949_737_3C(self):
        hdf_path = "test_data/1_7295949_737-3C.hdf5"
        segs = split_hdf_to_segments(hdf_path, draw=False)
        self.assertEqual(len(segs), 6)
        #Oil temp states breaks at index 392, 768-782, - - but recorded at a 0.125hz low frequency!
        # Eng (1) and (2) say 3167  - NOTE 3136 is at mins, but is masked!
        
        #todo: getwithmask, validate, align, normalise, stack, min, min_durations?!
        
        #
        
        # normalise?
    
    @unittest.skipIf(not os.path.isfile("test_data/4_3377853_146-301.hdf5"), 
                     'Test file not present')
    def test_146_301(self):
        hdf_path = "test_data/4_3377853_146-301.hdf5"
        segs = split_hdf_to_segments(hdf_path, draw=False)
        self.assertEqual(len(segs), 3)
        exp = ['START_AND_STOP'] * 5
        self.assertEqual([s.type for s in segs], exp)
        
    @unittest.skip('not implemented')
    def test_invalid_aircraft_ident(self):
        validate_aircraft() 
    
    @unittest.skip('not implemented')
    def test_split_hdf_to_segments(self):
        hdf_path = ''
        split_hdf_to_segments(hdf_path, draw=False)
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_split_segments(self):
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_datetime_for_segment(self):
        self.assertTrue(False)        
        
    @unittest.skip('not implemented')
    def test_store_segment(self):
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_deidentify_file(self):
        self.assertTrue(False)
        
    ##@unittest.skip('not implemented')
    ##def test_join_files(self):
        ##self.assertTrue(False)
        


