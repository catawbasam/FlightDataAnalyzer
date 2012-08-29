import unittest

from analysis_engine.process_flight import get_derived_nodes

class TestProcessFlight(unittest.TestCase):

    def test_get_derived_nodes(self):
        '''
        '''
        nodes = get_derived_nodes(['sample_derived_parameters'])
        self.assertEqual(len(nodes), 13)
        self.assertEqual(sorted(nodes.keys())[0], 'Heading Rate')
        self.assertEqual(sorted(nodes.keys())[-1], 'Vertical g')

    @unittest.skip('Not Implemented')
    def test_get_required_params(self):
        '''
        '''
        self.assertTrue(False)

