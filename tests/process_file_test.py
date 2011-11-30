try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.process_file import split_hdf_to_segments, split_segments, store_segment, deidentify_file, join_files


class TestProcessFile(unittest.TestCase):
    def test_split_hdf_to_segments(self):
        hdf_path = ''
        split_hdf_to_segments(hdf_path, draw=False)
        self.assertTrue(False)
        
    def test_split_segments(self):
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_store_segment(self):
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_deidentify_file(self):
        self.assertTrue(False)
        
    @unittest.skip('not implemented')
    def test_join_files(self):
        self.assertTrue(False)
        


