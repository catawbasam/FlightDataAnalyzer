try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.node import Node
from analysis.dependency_graph import dependency_order, process_order, graph_nodes

# mock function
f = lambda x: x

class TestDependencyGraph(unittest.TestCase):

    def setUp(self):
        ##########################################################
        ##                SETUP TEST 
        ##########################################################
        class MockParam(Node):
            self.dependencies = []
            def __init__(self, dependencies=[], operational=True):
                self.dependencies = dependencies
                self.operational = operational
                
            def can_operate(self, avail):
                return self.operational
            
            def derive(self):
                pass
            
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
            'P4' : type('P4', (Node,), dict(derive=f, dependencies=['Raw1', 'Raw2']))(), 
            'P5' : type('P5', (Node,), dict(derive=f, dependencies=['Raw3', 'Raw4']))(), 
            'P6' : type('P6', (Node,), dict(derive=f, dependencies=['Raw3']))(),
            'P7' : type('P7', (Node,), dict(derive=f, dependencies=['P4', 'P5', 'P6']))(),
            'P8' : type('P8', (Node,), dict(derive=f, dependencies=['Raw5']))(),
        }
        ##########################################################
    
    def tearDown(self):
        pass
    
    def test_graph_nodes(self):
        """ Tests a few of the colours
        """
        gr = graph_nodes([1, 2], [2], {})
        self.assertEqual(len(gr), 3)
        self.assertEqual(gr.node, 
                         {1: {'color': 'forestgreen'}, 2: {'color': 'forestgreen'}, 
                          'root': {'color': 'red'}})
        required_nodes = ['P7', 'P8']
        gr = graph_nodes(self.lfl_params, required_nodes, self.derived_nodes)
        self.assertEqual(len(gr), 11)
        
        
    def test_dependency(self):
        required_nodes = ['P7', 'P8']
        gr = graph_nodes(self.lfl_params, required_nodes, self.derived_nodes)
        gr_all, gr_st, order = process_order(gr, self.lfl_params, self.derived_nodes)
        
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
        """tests get_derived_nodes too
        """
        module = 'tests.sample_derived_parameters'
        required_nodes = ['Smoothed Track', 'Moment Of Takeoff', 'Vertical Speed', 'Slip On Runway']
        lfl_params = ['Indicated Airspeed', 
              'Groundspeed', 
              'Pressure Altitude',
              'Heading', 'TAT', 
              'Latitude', 'Longitude',
              ##'Inertial Latitude', #but no Inertial Logitude!
              'Longitudinal g', 'Lateral g', 'Normal g', 
              'Pitch', 'Roll', 
              ]
        order = dependency_order(lfl_params, required_nodes, [module])
        pos = order.index
        #print nodes
        self.assertTrue(pos('Vertical Speed') > pos('Pressure Altitude'))
        self.assertTrue(pos('Slip On Runway') > pos('Groundspeed'))
        self.assertTrue(pos('Slip On Runway') > pos('Horizontal g Across Track'))
        self.assertTrue(pos('Horizontal g Across Track') > pos('Roll'))
        self.assertFalse('MACH' in order) # MACH wasn't requested!
        self.assertFalse('Radio Altimeter' in order)
        # remove some lfl params to see inactive nodes
        
        
    def test_invalid_requirement_raises(self):
        module = 'tests.sample_derived_parameters'
        lfl_params = []
        required_nodes = ['Smoothed Track', 'Moment of Takeoff'] #it's called Moment Of Takeoff
        self.assertRaises(ValueError, dependency_order, lfl_params, required_nodes, [module])
        
        
    def test_missing_optional_accepted(self):
        """ Inactive param is optional so doesn't break the tree
        """
        # TEST taken from old dependency_test.py - only test not working so far
        self.assertTrue(False)
        
        #P4 = type('P4', (Node,), dict(derive=f, dependencies=['Raw1', 'Raw2']))
        #any_available = lambda s, avail: any([y in ['Raw1', 'Raw2'] for y in avail])
        #app = type('OptionalApp', (Node,), dict(derive=f, dependencies=[P4], 
                                                #can_operate=any_available))
        ## only one dep available
        #lfl_params = ['Raw1']
        #process_order = dependencies3(app, lfl_params)
        #self.assertEqual(process_order, ['Raw1', 'P4', 'Optional App'])
        
   