try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import os.path

from analysis.split_hdf_file_into_segments import (split_hdf_to_segments,
                                                   )


class TestSplitHDFToSegments(unittest.TestCase):
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
        pass 
    
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
        


