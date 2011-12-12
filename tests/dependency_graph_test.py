try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

import networkx as nx

from analysis.dependency_graph import (breadth_first_search_all_nodes,
    dependency_order, graph_nodes, process_order)
from analysis.node import (A, KPV, KTI, Node, NodeManager, Parameter, P, 
                           Section, S)

# mock function
f = lambda x: x

class TestDependencyGraph(unittest.TestCase):

    def setUp(self):
        class MockParam(Node):
            def __init__(self, dependencies=['a'], operational=True):
                self.dependencies = dependencies
                self.operational = operational
                
            def can_operate(self, avail):
                return self.operational
            
            def derive(self, a=P('a')):
                pass
            
            def get_derived(self, args):
                pass
            
            def get_dependency_names(self):
                return self.dependencies
            
        # nodes found on this aircraft's LFL
        self.lfl_params = [
            'Raw1',
            'Raw2',
            'Raw3',
            'Raw4',
            'Raw5',
        ]
        
        # nodes found from all the derived params code (top level, not their dependencies)
        #NOTE: For picturing it, it should show ALL raw params required.
        self.derived_nodes = {
            'P4' : MockParam(dependencies=['Raw1', 'Raw2']), 
            'P5' : MockParam(dependencies=['Raw3', 'Raw4']),
            'P6' : MockParam(dependencies=['Raw3']),
            'P7' : MockParam(dependencies=['P4', 'P5', 'P6']),
            'P8' : MockParam(dependencies=['Raw5']),
        }
        ##########################################################
    
    def tearDown(self):
        pass
    
    def test_graph_nodes(self):
        """ Tests a few of the colours
        """
        gr = graph_nodes(NodeManager([1, 2], [2], {}))
        self.assertEqual(len(gr), 3)
        self.assertEqual(gr.node, 
                         {1: {'color': 'forestgreen'}, 2: {'color': 'forestgreen'}, 
                          'root': {'color': 'red'}})
        required_nodes = ['P7', 'P8']
        mgr = NodeManager(self.lfl_params, required_nodes, self.derived_nodes)
        gr = graph_nodes(mgr)
        self.assertEqual(len(gr), 11)
        
        
    def test_dependency(self):
        required_nodes = ['P7', 'P8']
        mgr = NodeManager(self.lfl_params, required_nodes, self.derived_nodes)
        gr = graph_nodes(mgr)
        gr_all, gr_st, order = process_order(gr, mgr)
        
        self.assertEqual(len(gr_st), 11)
        pos = order.index
        self.assertTrue(pos('P8') > pos('Raw5'))
        self.assertTrue(pos('P7') > pos('P4'))
        self.assertTrue(pos('P7') > pos('P5'))
        self.assertTrue(pos('P7') > pos('P6'))
        self.assertTrue(pos('P5') > pos('Raw3'))
        self.assertTrue(pos('P6') > pos('Raw3'))
        self.assertFalse('root' in order) #don't include the root!
        
    def test_sample_parameter_module(self):
        """Tests many options:
        can_operate on SmoothedTrack works with 
        """
        module = 'tests.sample_derived_parameters'
        required_nodes = ['Smoothed Track', 'Moment Of Takeoff', 'Vertical Speed', 'Slip On Runway']
        lfl_params = ['Indicated Airspeed', 
              'Groundspeed', 
              'Pressure Altitude',
              'Heading', 'TAT', 
              'Latitude', 'Longitude',
              'Longitudinal g', 'Lateral g', 'Normal g', 
              'Pitch', 'Roll', 
              ]
        nodes, order = dependency_order(lfl_params, required_nodes, [module])
        pos = order.index
        #print nodes
        self.assertTrue(pos('Vertical Speed') > pos('Pressure Altitude'))
        self.assertTrue(pos('Slip On Runway') > pos('Groundspeed'))
        self.assertTrue(pos('Slip On Runway') > pos('Horizontal g Across Track'))
        self.assertTrue(pos('Horizontal g Across Track') > pos('Roll'))
        self.assertFalse('MACH' in order) # MACH wasn't requested!
        self.assertFalse('Radio Altimeter' in order)
        self.assertEqual(len(nodes.lfl), 12)
        self.assertEqual(len(nodes.requested), 4)
        self.assertEqual(len(nodes.derived_nodes), 13)
        # remove some lfl params to see inactive nodes
        
        
    def test_invalid_requirement_raises(self):
        module = 'tests.sample_derived_parameters'
        lfl_params = []
        required_nodes = ['Smoothed Track', 'Moment of Takeoff'] #it's called Moment Of Takeoff
        self.assertRaises(ValueError, dependency_order, lfl_params, required_nodes, [module])
        
        
    def test_kpv_dependency(self):
        #TODO?: Handle dependencies on one of the returns values!!
        
        # This may not be necessary as a parameter can still depend on a KPV, so long as it knows which class creates it.
        # But either way, adding some other types to the sample dependency tree can't do any harm!!
        
        #create a kpv
        #create a param that depends on one of the kpv return types
        self.assertTrue(False)
    
    def test_breadth_first_search_all_nodes_recursive(self):
        digraph = nx.DiGraph()
        digraph.add_node('root')
        digraph.add_edges_from([('Bounced Landing', 'Bounced Landing'),
                                ('root', 'Bounced Landing')])
        self.assertRaises(ValueError,
                          breadth_first_search_all_nodes, digraph, 'root')