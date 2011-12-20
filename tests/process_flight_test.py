try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.process_flight import process_flight, derive_parameters, get_derived_nodes

class TestProcessFlight(unittest.TestCase):
    def test_146_301(self):
        hdf_path = "test_data/4_3377853_146-301.005.hdf5"
        ac_info = {'Frame': '737-3C',
                   'Identifier': '5',
                   'Main Gear To Altitude Radio': 10,
                   'Manufacturer': 'Boeing',
                   'Tail Number': 'G-ABCD',
                   }
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 3)
    
    @unittest.skip('Not Implemented')
    def test_get_required_params(self):
        self.assertTrue(False)
    
    @unittest.skip('Not Implemented')    
    def test_process_flight(self):
        self.assertTrue(False)
        
    def test_get_derived_nodes(self):
        nodes = get_derived_nodes(['sample_derived_parameters'])
        self.assertEqual(len(nodes), 13)
        self.assertEqual(sorted(nodes.keys())[0], 'Heading Rate')
        self.assertEqual(sorted(nodes.keys())[-1], 'Vertical g')
        