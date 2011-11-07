try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.process_flight import process_flight, validate_and_derive_parameters

class TestProcessFlight(unittest.TestCase):
    
    def test_get_required_params(self):
        self.assertTrue(False)
        
    def test_process_flight(self):
        self.assertTrue(False)