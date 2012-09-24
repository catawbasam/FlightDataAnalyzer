import collections
import unittest
import networkx as nx

from datetime import datetime

from analysis_engine.node import (DerivedParameterNode, Node, NodeManager, P)
from analysis_engine.process_flight import get_derived_nodes
from analysis_engine.dependency_graph import (
    dependency_order, 
    graph_nodes, 
    graph_adjacencies, 
    process_order,
)
  
def flatten(l):
    "Flatten an iterable of many levels of depth (generator)"
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in sorted(flatten(el)):
                yield sub
        else:
            yield el

class TestDependencyGraph(unittest.TestCase):

    def setUp(self):
        class MockParam(Node):
            def __init__(self, dependencies=['a'], operational=True):
                self.dependencies = dependencies
                self.operational = operational
                # Hack to allow objects rather than classes to be added
                # to the tree.
                self.__base__ = DerivedParameterNode
                
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
    
    def test_graph_nodes_using_sample_tree(self): 
        required_nodes = ['P7', 'P8']
        mgr2 = NodeManager(datetime.now(), self.lfl_params, required_nodes, 
                           self.derived_nodes, {}, {})
        gr = graph_nodes(mgr2)
        self.assertEqual(len(gr), 11)
    
    def test_graph_nodes_with_duplicate_key_in_lfl_and_derived(self):
        """ Test that LFL nodes are used in place of Derived where available.
        Tests a few of the colours
        """
        class One(DerivedParameterNode):
            # Hack to allow objects rather than classes to be added to the tree.            
            __base__ = DerivedParameterNode
            def derive(self, dep=P('DepOne')):
                pass
        class Four(DerivedParameterNode):
            # Hack to allow objects rather than classes to be added to the tree. 
            __base__ = DerivedParameterNode
            def derive(self, dep=P('DepFour')):
                pass
        one = One('overridden')
        four = Four('used')
        mgr1 = NodeManager(datetime.now(), [1, 2], [2, 4], {1:one, 4:four},{}, {})
        gr = graph_nodes(mgr1)
        self.assertEqual(len(gr), 5)
        # LFL
        self.assertEqual(gr.edges(1), []) # as it's in LFL, it shouldn't have any edges
        self.assertEqual(gr.node[1], {'color': 'forestgreen'})
        # Derived
        self.assertEqual(gr.edges(4), [(4,'DepFour')])
        self.assertEqual(gr.node[4], {'color': 'yellow'})
        # Root
        from analysis_engine.dependency_graph import draw_graph
        draw_graph(gr, 'test_graph_nodes_with_duplicate_key_in_lfl_and_derived')
        self.assertEqual(gr.successors('root'), [2,4]) # only the two requested are linked
        self.assertEqual(gr.node['root'], {'color': 'red'})
        
    def test_dependency(self):
        required_nodes = ['P7', 'P8']
        mgr = NodeManager(datetime.now(), self.lfl_params, required_nodes, 
                          self.derived_nodes, {}, {})
        gr = graph_nodes(mgr)
        gr_all, gr_st, order = process_order(gr, mgr)
        
        self.assertEqual(len(gr_st), 11)
        pos = order.index
        self.assertTrue(pos('P8') > pos('Raw5'))
        self.assertTrue(pos('P7') > pos('P4'))
        self.assertTrue(pos('P7') > pos('P5'))
        self.assertTrue(pos('P7') > pos('P6'))
        self.assertTrue(pos('P6') > pos('Raw3'))
        self.assertTrue(pos('P5') > pos('Raw3'))
        self.assertTrue(pos('P5') > pos('Raw4'))
        self.assertTrue(pos('P4') > pos('Raw1'))
        self.assertTrue(pos('P4') > pos('Raw2'))
        self.assertFalse('root' in order) #don't include the root!
        
        """
# Sample demonstrating which nodes have predecessors, successors and so on:
for node in node_mgr.keys():
    print 'Node: %s \tPre: %s \tSucc: %s \tNeighbors: %s \tEdges: %s' % (node, gr_all.predecessors(node), gr_all.successors(node), gr_all.neighbors(node), gr_all.edges(node))

Node: P4 	Pre: ['P7'] 	Succ: ['Raw2', 'Raw1'] 	Neighbors: ['Raw2', 'Raw1'] 	Edges: [('P4', 'Raw2'), ('P4', 'Raw1')]
Node: P5 	Pre: ['P7'] 	Succ: ['Raw3', 'Raw4'] 	Neighbors: ['Raw3', 'Raw4'] 	Edges: [('P5', 'Raw3'), ('P5', 'Raw4')]
Node: P6 	Pre: ['P7'] 	Succ: ['Raw3'] 	Neighbors: ['Raw3'] 	Edges: [('P6', 'Raw3')]
Node: P7 	Pre: [] 	Succ: ['P6', 'P4', 'P5'] 	Neighbors: ['P6', 'P4', 'P5'] 	Edges: [('P7', 'P6'), ('P7', 'P4'), ('P7', 'P5')]
Node: P8 	Pre: [] 	Succ: ['Raw5'] 	Neighbors: ['Raw5'] 	Edges: [('P8', 'Raw5')]
Node: Raw1 	Pre: ['P4'] 	Succ: [] 	Neighbors: [] 	Edges: []
Node: Raw2 	Pre: ['P4'] 	Succ: [] 	Neighbors: [] 	Edges: []
Node: Raw3 	Pre: ['P6', 'P5'] 	Succ: [] 	Neighbors: [] 	Edges: []
Node: Raw4 	Pre: ['P5'] 	Succ: [] 	Neighbors: [] 	Edges: []
Node: Raw5 	Pre: ['P8'] 	Succ: [] 	Neighbors: [] 	Edges: []
Node: Start Datetime 	Pre: [] 	Succ: [] 	Neighbors: [] 	Edges: []
"""

    def test_dependency_with_lowlevel_dependencies_requested(self):
        """ Simulate requesting a Raw Parameter as a dependency. This requires
        the requested node to be removed when it is not at the top of the
        dependency tree.
        """
        required_nodes = ['P7', 'P8', # top level nodes
                          'P4', 'P5', 'P6', # middle level node
                          'Raw3', # bottom level node
                          ]
        mgr = NodeManager(datetime.now(), self.lfl_params + ['Floating'], required_nodes, 
                          self.derived_nodes, {}, {})
        gr = graph_nodes(mgr)
        gr_all, gr_st, order = process_order(gr, mgr)
        
        self.assertEqual(len(gr_st), 11)
        pos = order.index
        self.assertTrue(pos('P8') > pos('Raw5'))
        self.assertTrue(pos('P7') > pos('P4'))
        self.assertTrue(pos('P7') > pos('P5'))
        self.assertTrue(pos('P7') > pos('P6'))
        self.assertTrue(pos('P6') > pos('Raw3'))
        self.assertTrue(pos('P5') > pos('Raw3'))
        self.assertTrue(pos('P5') > pos('Raw4'))
        self.assertTrue(pos('P4') > pos('Raw1'))
        self.assertTrue(pos('P4') > pos('Raw2'))
        self.assertFalse('Floating' in order)
        self.assertFalse('root' in order) #don't include the root!

    def test_sample_parameter_module(self):
        """Tests many options:
        can_operate on SmoothedTrack works with 
        """
        required_nodes = ['Smoothed Track', 'Moment Of Takeoff', 'Vertical Speed', 'Slip On Runway']
        lfl_params = ['Indicated Airspeed', 
              'Groundspeed', 
              'Pressure Altitude',
              'Heading', 'TAT', 
              'Latitude', 'Longitude',
              'Longitudinal g', 'Lateral g', 'Normal g', 
              'Pitch', 'Roll', 
              ]
        try:
            # for test cmd line runners
            derived = get_derived_nodes(['tests.sample_derived_parameters'])
        except ImportError:
            # for IDE test runners
            derived = get_derived_nodes(['sample_derived_parameters'])
        nodes = NodeManager(datetime.now(), lfl_params, required_nodes, derived, {}, {})
        order, _ = dependency_order(nodes)
        pos = order.index
        self.assertTrue(len(order))
        self.assertTrue(pos('Vertical Speed') > pos('Pressure Altitude'))
        self.assertTrue(pos('Slip On Runway') > pos('Groundspeed'))
        self.assertTrue(pos('Slip On Runway') > pos('Horizontal g Across Track'))
        self.assertTrue(pos('Horizontal g Across Track') > pos('Roll'))
        self.assertFalse('Mach' in order) # Mach wasn't requested!
        self.assertFalse('Radio Altimeter' in order)
        self.assertEqual(len(nodes.lfl), 12)
        self.assertEqual(len(nodes.requested), 4)
        self.assertEqual(len(nodes.derived_nodes), 13)
        # remove some lfl params to see inactive nodes
        
    def test_invalid_requirement_raises(self):
        lfl_params = []
        required_nodes = ['Smoothed Track', 'Moment of Takeoff'] #it's called Moment Of Takeoff
        try:
            # for test cmd line runners
            derived = get_derived_nodes(['tests.sample_derived_parameters'])
        except ImportError:
            # for IDE test runners
            derived = get_derived_nodes(['sample_derived_parameters'])
        mgr = NodeManager(datetime.now(), lfl_params, required_nodes, derived, 
                          {}, {})
        self.assertRaises(nx.NetworkXError, dependency_order, mgr, draw=False)
        

class TestGraphAdjacencies(unittest.TestCase):
    def test_graph_adjacencies(self):
        g = nx.DiGraph()
        g.add_node('a', color='blue', label='a1')
        g.add_nodes_from(['b', 'c', 'd'])
        g.add_node('root', color='red')
        g.add_edge('root', 'a')
        g.add_edge('root', 'd')
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.add_edge('d', 'c')
        res = graph_adjacencies(g)
        exp = [
        {
            'id': 'root',
            'name': 'root',
            'data': {
                'color': 'red',
                },
            'adjacencies': [
                {
                    'nodeTo': 'a',
                    'data': {},
                    },
                {
                    'nodeTo': 'd',
                    'data': {},
                    },
                ],
            },
        {
            'id': 'a',
            'name': 'a1',
            'data': {
                'color': 'blue',
                },
            'adjacencies': [
                {
                    'nodeTo': 'b',
                    'data': {},
                    },
                {
                    'nodeTo': 'c',
                    'data': {},
                    },
                ],
            },
        {
            'id': 'b',
            'name': 'b',
            'data': {},
            'adjacencies': [
                ],
            },
        {
            'id': 'c',
            'name': 'c',
            'data': {},
            'adjacencies': [
                ],
            },
        {
            'id': 'd',
            'name': 'd',
            'data': {},
            'adjacencies': [
                {
                    'nodeTo': 'c',
                    'data': {},
                    }
                ],
            },
        ]
        self.assertEqual(list(flatten(exp)), list(flatten(res)))
