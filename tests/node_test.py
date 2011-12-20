try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import mock
from random import shuffle
from datetime import datetime

from analysis.node import (
    Attribute, DerivedParameterNode, get_verbose_name, KeyPointValue,
    KeyPointValueNode, KeyTimeInstance, KeyTimeInstanceNode, FormattedNameNode,
    Node, NodeManager, P, Parameter, powerset, Section, SectionNode)


class TestAbstractNode(unittest.TestCase):
    
    def test_node(self):
        pass
    

class TestNode(unittest.TestCase):
    
    def test_name(self):
        """ Splits on CamelCase and title cases
        """
        NewNode = type('Camel4CaseName', (Node,), dict(derive=lambda x:x))
        self.assertEqual(NewNode.get_name(), 'Camel 4 Case Name')
        NewNode = type('ThisIsANode', (Node,), dict(derive=lambda x:x))
        self.assertEqual(NewNode.get_name(), 'This Is A Node')
        NewNode = type('My2BNode', (Node,), dict(derive=lambda x:x))
        self.assertEqual(NewNode.get_name(), 'My 2B Node')
        NewNode = type('_1000Ft', (Node,), dict(derive=lambda x:x))
        self.assertEqual(NewNode.get_name(), '1000 Ft')
        NewNode = type('TouchdownV2Max', (Node,), dict(derive=lambda x:x))
        self.assertEqual(NewNode.get_name(), 'Touchdown V2 Max')
        NewNode.name = 'MACH'
        self.assertEqual(NewNode.get_name(), 'MACH')

    def test_get_dependency_names(self):
        """ Check class names or strings return strings
        """            
        class RateOfClimb(DerivedParameterNode):
            def derive(self):
                pass
        
        class RateOfDescentHigh(KeyPointValueNode):
            def derive(self, rod=P('Rate Of Descent'), roc=RateOfClimb):
                pass
            
        self.assertEqual(RateOfDescentHigh.get_dependency_names(), 
                         ['Rate Of Descent', 'Rate Of Climb'])
        
    def test_can_operate(self):
        deps = ['a', 'b', 'c']
        class NewNode(Node):
            def derive(self, aa=P('a'), bb=P('b'), cc=P('c')):
                pass
            def get_derived(self, deps):
                pass
        self.assertTrue(NewNode.can_operate(deps))
        extra_deps = deps + ['d', 'e', 'f']
        self.assertTrue(NewNode.can_operate(extra_deps))
        # shuffle them about
        shuffle(extra_deps)
        self.assertTrue(NewNode.can_operate(extra_deps))
        shuffle(extra_deps)
        self.assertTrue(NewNode.can_operate(extra_deps))
        not_enough_deps = ['b', 'c']
        self.assertFalse(NewNode.can_operate(not_enough_deps))
        
    def test_can_operate_with_objects_and_string_dependencies(self):
        def deriveparent(self, one=P('a')):
            pass
        Parent = type('Parent', (Node,), dict(derive=deriveparent, 
                                              get_derived=lambda x:x))
        def derivenew(self, one=P('b'), two=Parent):
            pass
        NewNode = type('NewNode', (Node,), dict(derive=derivenew,
                                                get_derived=lambda x:x))
        
        available = ['a', 'Parent', 'b']
        self.assertTrue(NewNode.can_operate(available))
        
    def test_get_operational_combinations(self):
        """ NOTE: This shows a REALLY neat way to test all combinations of a
        derived Node class!
        """
        class Combo(Node):
            def derive(self, aa=P('a'), bb=P('b'), cc=P('c')): 
                # we require 'a' and 'b' and 'c' is a bonus
                assert aa == 'A'
                assert bb == 'B'
                return aa, bb, cc

            def get_derived(self, params):
                pass
            
            @classmethod
            def can_operate(cls, available):
                if 'a' in available and 'b' in available:
                    return True
                else:
                    return False
        
        options = Combo.get_operational_combinations()
        self.assertEqual(options, [('a', 'b'), ('a', 'b', 'c')])
        
        # get all operational options to test its derive method under
        options = Combo.get_operational_combinations()
        c = Combo()
            
        for args in options:
            # build ordered dependencies
            deps = []
            for param in c.get_dependency_names():
                if param in args:  # in expected combo
                    deps.append(param.upper())
                else:  # dependency not available
                    deps.append(None)
            # test derive method with this combination
            res = c.derive(*deps)
            self.assertEqual(res[:2], ('A', 'B'))
            
    def test_get_derived(self):
        def get_mock_params():
            param1 = mock.Mock()
            param1.name = 'PARAM1'
            param1.frequency = 2
            param1.offset = 0.5
            param1.get_aligned = mock.Mock()
            param1.get_aligned.return_value = 1
            param2 = mock.Mock()
            param2.name = 'PARAM2'
            param2.frequency = 0.5
            param2.offset = 1
            param2.get_aligned = mock.Mock()
            param2.get_aligned.return_value = 2
            return param1, param2
        param1, param2 = get_mock_params()
        class TestNode(Node):
            def derive(self, kwarg1=param1, kwarg2=param2):
                pass
        node = TestNode()
        node.derive = mock.Mock()
        node.derive.return_value = None
        node.get_derived([param1, param2])
        self.assertEqual(param1.method_calls, [])
        self.assertEqual(param2.method_calls, [('get_aligned', (param1,), {})])
        # check param1 is returned unchanged and param2 get_aligned is called (returns '2')
        self.assertEqual(node.derive.call_args, ((param1, 2), {}))
        
        class NotImplementedNode(Node):
            def derive(self, kwarg1=param1, kwarg2=param2):
                return NotImplemented
        not_implemented_node = NotImplementedNode()
        self.assertRaises(NotImplementedError, not_implemented_node.get_derived,
                          [param1, param2])
        class UnalignedNode(Node):
            align_to_first_dependency = False
            def derive(self, kwarg1=param1, kwarg2=param2):
                pass
        node = UnalignedNode()
        param1, param2 = get_mock_params()
        node.get_derived([param1, param2])
        self.assertEqual(param1.method_calls, [])
        self.assertEqual(param2.method_calls, [])
        
                        
class TestNodeManager(unittest.TestCase):
    def test_operational(self):
        mock_node = mock.Mock('can_operate') # operable node
        mock_node.can_operate = mock.Mock(return_value=True)
        mock_inop = mock.Mock('can_operate') # inoperable node
        mock_inop.can_operate = mock.Mock(return_value=False)
        aci = {'n':1, 'o':2, 'p':3}
        afr = {'l':4, 'm':5}
        mgr = NodeManager(None, ['a', 'b', 'c'], ['a', 'x'], 
                          {'x': mock_node, 'y': mock_node, 'z': mock_inop},
                          aci, afr)
        self.assertTrue(mgr.operational('a', []))
        self.assertTrue(mgr.operational('b', []))
        self.assertTrue(mgr.operational('c', []))
        self.assertTrue(mgr.operational('x', []))
        self.assertTrue(mgr.operational('y', ['a']))
        self.assertTrue(mgr.operational('n', ['a'])) # achieved flight record
        self.assertTrue(mgr.operational('p', ['a'])) # aircraft info
        self.assertFalse(mgr.operational('z', ['a', 'b']))
        self.assertEqual(mgr.keys(), ['Start Datetime'] + list('abclmnopxyz'))
        
    def test_get_attribute(self):
        aci = {'a':'a_value', 'b':None}
        afr = {'x':'x_value', 'y':None}
        mgr = NodeManager(None, [],[],{},aci, afr)
        # test aircraft info
        a = mgr.get_attribute('a')
        self.assertEqual(a.__repr__(), Attribute('a', 'a_value').__repr__())
        b = mgr.get_attribute('b')
        self.assertEqual(b, None)
        c = mgr.get_attribute('c')
        self.assertEqual(c, None)
        # test afr
        x = mgr.get_attribute('x')
        self.assertEqual(x.__repr__(), Attribute('x', 'x_value').__repr__())
        y = mgr.get_attribute('y')
        self.assertEqual(y, None)
        z = mgr.get_attribute('z')
        self.assertEqual(z, None)
        
    def test_get_start_datetime(self):
        dt = datetime(2020,12,25)
        mgr = NodeManager(dt, [],[],{},{},{})
        self.assertTrue('Start Datetime' in mgr.keys())
        start_dt = mgr.get_attribute('Start Datetime')
        self.assertEqual(start_dt.name, 'Start Datetime')
        self.assertEqual(start_dt.value, dt)
        
        
class TestPowerset(unittest.TestCase):
    def test_powerset(self):
        deps = ['aaa',  'bbb', 'ccc']
        res = list(powerset(deps))
        expected = [(),
                    ('aaa',),
                    ('bbb',), 
                    ('ccc',), 
                    ('aaa', 'bbb'),
                    ('aaa', 'ccc'),
                    ('bbb', 'ccc'),
                    ('aaa', 'bbb', 'ccc')]
        self.assertEqual(res, expected)

class TestSectionNode(unittest.TestCase):
    def setUp(self):
        class ExampleSectionNode(SectionNode):
            def derive(self, a=P('a',[], 2, 0.4)):
                pass
            def get_derived(self):
                pass
        self.section_node_class = ExampleSectionNode
    
    def test_get_aligned(self):
        '''
        TODO: Test offset alignment.
        '''
        section_node = self.section_node_class(frequency=1, offset=0.5)
        section_node.create_section(slice(2,4))
        section_node.create_section(slice(5,7))
        param = Parameter('p', frequency=0.5, offset=0.1)
        aligned_node = section_node.get_aligned(param)
        self.assertEqual(aligned_node.frequency, param.frequency)
        self.assertEqual(aligned_node.offset, param.offset)
        self.assertEqual(list(aligned_node),
                         [Section(name='Example Section Node',
                                  slice=slice(1.2, 2.2, None)),
                          Section(name='Example Section Node',
                                  slice=slice(2.7, 3.7, None))])
    
    def test_items(self):
        items = [Section('a', slice(0,10))]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        self.assertEqual(section_node, items)
    
    def test_get(self):
        items = [Section('a', slice(4,10)),
                 Section('b', slice(14,17)),
                 Section('b', slice(19,21)),
                 Section('c', slice(30,34)),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        sections = section_node.get()
        self.assertEqual(items, sections)
        sections = section_node.get(name='b')
        self.assertEqual(items[1:3], sections)
        sections = section_node.get(name='c')
        self.assertEqual(items[-1:], sections)
        sections = section_node.get(within_slice=slice(12, 25))
        self.assertEqual(items[1:3], sections)
        sections = section_node.get(within_slice=slice(15, 40), name='b')
        self.assertEqual(items[2:3], sections)
    
    def test_get_first(self):
        items = [Section('a', slice(4,10)),
                 Section('b', slice(14,17)),
                 Section('b', slice(19,21)),
                 Section('c', slice(30,34)),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        first_section = section_node.get_first()
        self.assertEqual(items[0], first_section)
        first_b_section = section_node.get_first(name='b')
        self.assertEqual(items[1], first_b_section)
        first_c_section = section_node.get_first(name='c')
        self.assertEqual(items[3], first_c_section)
        first_section_within_slice = section_node.get_first(within_slice=
                                                            slice(12, 25))
        self.assertEqual(items[1], first_section_within_slice)
        first_b_section_within_slice = section_node.get_first(within_slice=
                                                              slice(15, 40),
                                                              name='b')
        self.assertEqual(items[2], first_b_section_within_slice)
    
    def test_get_last(self):
        items = [Section('a', slice(4,10)),
                 Section('b', slice(14,17)),
                 Section('b', slice(19,21)),
                 Section('c', slice(30,34)),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        last_section = section_node.get_last()
        self.assertEqual(items[3], last_section)
        last_b_section = section_node.get_last(name='b')
        self.assertEqual(items[2], last_b_section)
        last_c_section = section_node.get_last(name='c')
        self.assertEqual(items[3], last_c_section)
        last_section_within_slice = section_node.get_last(within_slice=
                                                          slice(12, 25))
        self.assertEqual(items[2], last_section_within_slice)
        last_b_section_within_slice = section_node.get_last(within_slice=
                                                            slice(15, 40),
                                                            name='b')
        self.assertEqual(items[2], last_section_within_slice)
    
    def test_get_ordered_by_index(self):
        items = [Section('a', slice(4,10)),
                 Section('b', slice(19,21)),
                 Section('b', slice(14,17)),
                 Section('c', slice(30,34)),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        sections = section_node.get_ordered_by_index()
        self.assertEqual([items[0], items[2], items[1], items[3]], sections)
        sections = section_node.get_ordered_by_index(name='b')
        self.assertEqual([items[2], items[1]], sections)
        sections = section_node.get_ordered_by_index(name='c')
        self.assertEqual([items[-1]], sections)
        sections = section_node.get_ordered_by_index(within_slice=slice(12, 25))
        self.assertEqual([items[2], items[1]], sections)
        sections = section_node.get_ordered_by_index(within_slice=slice(15, 40), name='b')
        self.assertEqual([items[1]], sections)


class TestFormattedNameNode(unittest.TestCase):
    def setUp(self):
        class ExampleNameFormatNode(FormattedNameNode):
            def derive(self, a=P('a',[], 2, 0.4)):
                pass
            def get_derived(self):
                pass
        self.formatted_name_node = ExampleNameFormatNode()
        
        
    def test_no_name_uses_node_name(self):
        names = self.formatted_name_node.names()
        self.assertEqual(names, ["Example Name Format Node"])
        #Q: Should this include name of node?
        name = self.formatted_name_node.format_name()
        self.assertEqual(name, "Example Name Format Node")
        
    
    def test_names(self):
        """ Using all RETURNS options, apply NAME_FORMAT to obtain a complete
        list of KPV names this class will create.
        """
        formatted_name_node = self.formatted_name_node
        formatted_name_node.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
        formatted_name_node.NAME_VALUES = {'altitude' : range(100, 701, 300),'phase' : ['ascent', 'descent']}
        names = formatted_name_node.names()
        
        self.assertEqual(names, ['Speed in ascent at 100 ft',
                                 'Speed in ascent at 400 ft',
                                 'Speed in ascent at 700 ft',
                                 'Speed in descent at 100 ft',
                                 'Speed in descent at 400 ft',
                                 'Speed in descent at 700 ft',
                                 ])
    
    def test__validate_name(self):
        """ Ensures that created names have a validated option
        """
        formatted_name_node = self.formatted_name_node
        formatted_name_node.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
        formatted_name_node.NAME_VALUES = {'altitude' : range(100,1000,100),
                                'phase' : ['ascent', 'descent']}
        self.assertTrue(formatted_name_node._validate_name('Speed in ascent at 500 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 900 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 100 ft'))
        self.assertRaises(ValueError, formatted_name_node._validate_name, 'Speed in ascent at -10 ft')       
        
    def test_get_first(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        
        # no slice
        kti1 = kti_node.get_first()
        self.assertEqual(kti1.index, 2)
        # within a slice
        kti2 = kti_node.get_first(slice(15,100))
        self.assertEqual(kti2.index, 50)
        # with a particular name
        kti3 = kti_node.get_first(name='Slowest')
        self.assertEqual(kti3.index, 2)
        kti4 = kti_node.get_first(name='Fast')
        self.assertEqual(kti4.index, 50)
        # named within a slice
        kti5 = kti_node.get_first(slice(10,400), 'Slowest')
        self.assertEqual(kti5.index, 12)
        # does not exist
        kti6 = kti_node.get_first(name='Not Here')
        self.assertEqual(kti6, None)
        kti7 = kti_node.get_first(slice(500,600))
        self.assertEqual(kti7, None)
    
    def test_get_last(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        
        # no slice
        kti1 = kti_node.get_last()
        self.assertEqual(kti1.index, 342)
        # within a slice
        kti2 = kti_node.get_last(slice(15,100))
        self.assertEqual(kti2.index, 50)
        # with a particular name
        kti3 = kti_node.get_last(name='Slowest')
        self.assertEqual(kti3.index, 342)
        kti4 = kti_node.get_last(name='Fast')
        self.assertEqual(kti4.index, 50)
        # named within a slice
        kti5 = kti_node.get_last(slice(10,400), 'Slowest')
        self.assertEqual(kti5.index, 342)
        # does not exist
        kti6 = kti_node.get_last(name='Not Here')
        self.assertEqual(kti6, None)
        kti7 = kti_node.get_last(slice(500,600))
        self.assertEqual(kti7, None)
    
    def test_get_named(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        kti_node_returned1 = kti_node.get_named('Slowest')
        self.assertEqual(kti_node_returned1,
                         [KeyTimeInstance(12, 'Slowest'),
                          KeyTimeInstance(342, 'Slowest'),
                          KeyTimeInstance(2, 'Slowest'),])
        kti_node_returned2 = kti_node.get_named('Fast')
        self.assertEqual(kti_node_returned2, [KeyTimeInstance(50, 'Fast')])
        # named within a slice
        kti_node_returned3 = kti_node.get_named('Slowest',
                                                slice(10,400))
        self.assertEqual(kti_node_returned3,
                         [KeyTimeInstance(12, 'Slowest'),
                          KeyTimeInstance(342, 'Slowest')])
        # does not exist
        kti_node_returned4 = kti_node.get_named('Not Here')
        self.assertEqual(kti_node_returned4, [])
        kti_node_returned5 = kti_node.get_named('Slowest', slice(500,600))
        self.assertEqual(kti_node_returned5, [])

    
    def test_get_ordered_by_index(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        
        # no slice
        kti_node_returned1 = kti_node.get_ordered_by_index()
        self.assertTrue(isinstance(kti_node_returned1, KeyTimeInstanceNode))
        self.assertEqual(kti_node.name, kti_node_returned1.name)
        self.assertEqual(kti_node.frequency, kti_node_returned1.frequency)
        self.assertEqual(kti_node.offset, kti_node_returned1.offset)
        self.assertEqual(kti_node_returned1,
                         [KeyTimeInstance(2, 'Slowest'), 
                          KeyTimeInstance(12, 'Slowest'), 
                          KeyTimeInstance(50, 'Fast'),
                          KeyTimeInstance(342, 'Slowest')])
        # within a slice
        kti_node_returned2 = kti_node.get_ordered_by_index(slice(15,100))
        self.assertEqual(kti_node_returned2, [KeyTimeInstance(50, 'Fast')])
        # with a particular name
        kti_node_returned3 = kti_node.get_ordered_by_index(name='Slowest')
        self.assertEqual(kti_node_returned3,
                         [KeyTimeInstance(2, 'Slowest'),
                          KeyTimeInstance(12, 'Slowest'), 
                          KeyTimeInstance(342, 'Slowest')])
        kti_node_returned4 = kti_node.get_ordered_by_index(name='Fast')
        self.assertEqual(kti_node_returned4, [KeyTimeInstance(50, 'Fast')])
        # named within a slice
        kti_node_returned5 = kti_node.get_ordered_by_index(slice(10,400),
                                                           'Slowest')
        self.assertEqual(kti_node_returned5,
                         [KeyTimeInstance(12, 'Slowest'),
                          KeyTimeInstance(342, 'Slowest')])
        # does not exist
        kti_node_returned6 = kti_node.get_ordered_by_index(name='Not Here')
        self.assertEqual(kti_node_returned6, [])
        kti_node_returned7 = kti_node.get_ordered_by_index(slice(500,600))
        self.assertEqual(kti_node_returned7, [])


class TestKeyPointValueNode(unittest.TestCase):
    
    def setUp(self):
        class KPV(KeyPointValueNode):
            def derive(self, a=P('a',[], 2, 0.4)):
                pass
        self.knode = KPV(frequency=2, offset=0.4)


    def test_create_kpv(self):
        """ Tests name format substitution and return type
        """
        knode = self.knode
        knode.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)dft'
        knode.NAME_VALUES = {'phase':['ascent', 'descent'],
                                'altitude':[1000,1500],
                                }
        
        self.assertEqual(knode.frequency, 2)
        self.assertEqual(knode.offset, 0.4)
        # use keyword arguments
        spd_kpv = knode.create_kpv(10, 12.5, phase='descent', altitude=1000.0)
        self.assertTrue(isinstance(knode[0], KeyPointValue))
        self.assertTrue(isinstance(spd_kpv, KeyPointValue))
        self.assertEqual(spd_kpv.index, 10)
        self.assertEqual(spd_kpv.value, 12.5)
        self.assertEqual(spd_kpv.name, 'Speed in descent at 1000ft')
        # use dictionary argument
        # and check interpolation value 'ignored' is indeed ignored
        spd_kpv2 = knode.create_kpv(10, 12.5, dict(phase='ascent', 
                                                   altitude=1500.0,
                                                   ignored='excuseme'))
        self.assertEqual(spd_kpv2.name, 'Speed in ascent at 1500ft')
        # missing interpolation value "altitude"
        self.assertRaises(KeyError, knode.create_kpv, 1, 2,
                          dict(phase='ascent'))
        # wrong type raises TypeError
        self.assertRaises(TypeError, knode.create_kpv, 2, '3', 
                          phase='', altitude='')
    
    def test_get_aligned(self):
        '''
        TODO: Test offset alignment.
        '''
        knode = self.knode
        knode.NAME_FORMAT = 'Speed at %(altitude)dft'
        knode.NAME_VALUES = {'altitude':[1000,1500]}
        param = Parameter('p', frequency=0.5, offset=1.5)
        knode.create_kpv(10, 12.5, altitude=1000.0)
        knode.create_kpv(24, 12.5, altitude=1000.0)
        aligned_node = self.knode.get_aligned(param)
        self.assertEqual(aligned_node.frequency, param.frequency)
        self.assertEqual(aligned_node.offset, param.offset)
        self.assertEqual(aligned_node,
                         [KeyPointValue(index=1.95, value=12.5, name='Speed at 1000ft'),
                          KeyPointValue(index=5.45, value=12.5, name='Speed at 1000ft')])
        
    def test_get_min(self):
        kpv_node = KeyPointValueNode(items=[KeyPointValue(12, 30, 'Slowest'), 
                                            KeyPointValue(342, 60, 'Slowest'), 
                                            KeyPointValue(2, 14, 'Slowest'), 
                                            KeyPointValue(50, 369, 'Fast')])
        
        # no slice
        kpv1 = kpv_node.get_min()
        self.assertEqual(kpv1.value, 14)
        # within a slice
        kpv2 = kpv_node.get_min(slice(15,100))
        self.assertEqual(kpv2.value, 369)
        # with a particular name
        kpv3 = kpv_node.get_min(name='Slowest')
        self.assertEqual(kpv3.value, 14)
        kpv4 = kpv_node.get_min(name='Fast')
        self.assertEqual(kpv4.value, 369)
        # named within a slice
        kpv5 = kpv_node.get_min(slice(10,400), 'Slowest')
        self.assertEqual(kpv5.value, 30)
        # does not exist
        kpv6 = kpv_node.get_min(name='Not Here')
        self.assertEqual(kpv6, None)
        kpv7 = kpv_node.get_min(slice(500,600))
        self.assertEqual(kpv7, None)
    
    def test_get_max(self):
        kpv_node = KeyPointValueNode(items=[KeyPointValue(12, 30, 'Slowest'), 
                                            KeyPointValue(342, 60, 'Slowest'), 
                                            KeyPointValue(2, 14, 'Slowest'), 
                                            KeyPointValue(50, 369, 'Fast')])
        
        # no slice
        kpv1 = kpv_node.get_max()
        self.assertEqual(kpv1.value, 369)
        # within a slice
        kpv2 = kpv_node.get_max(slice(15,100))
        self.assertEqual(kpv2.value, 369)
        # with a particular name
        kpv3 = kpv_node.get_max(name='Slowest')
        self.assertEqual(kpv3.value, 60)
        kpv4 = kpv_node.get_max(name='Fast')
        self.assertEqual(kpv4.value, 369)
        # named within a slice
        kpv5 = kpv_node.get_max(slice(10,400), 'Slowest')
        self.assertEqual(kpv5.value, 60)
        # does not exist
        kpv6 = kpv_node.get_max(name='Not Here')
        self.assertEqual(kpv6, None)
        kpv7 = kpv_node.get_max(slice(500,600))
        self.assertEqual(kpv7, None)
    
    def test_get_ordered_by_value(self):
        kpv_node = KeyPointValueNode(items=[KeyPointValue(12, 30, 'Slowest'), 
                                            KeyPointValue(342, 60, 'Slowest'), 
                                            KeyPointValue(2, 14, 'Slowest'), 
                                            KeyPointValue(50, 369, 'Fast')])
        
        # no slice
        kpv_node_returned1 = kpv_node.get_ordered_by_value()
        self.assertTrue(isinstance(kpv_node_returned1, KeyPointValueNode))
        self.assertEqual(kpv_node.name, kpv_node_returned1.name)
        self.assertEqual(kpv_node.frequency, kpv_node_returned1.frequency)
        self.assertEqual(kpv_node.offset, kpv_node_returned1.offset)
        self.assertEqual(kpv_node_returned1,
                         [KeyPointValue(2, 14, 'Slowest'), 
                          KeyPointValue(12, 30, 'Slowest'), 
                          KeyPointValue(342, 60, 'Slowest'), 
                          KeyPointValue(50, 369, 'Fast')])
        # within a slice
        kpv_node_returned2 = kpv_node.get_ordered_by_value(slice(15,100))
        self.assertEqual(kpv_node_returned2, [KeyPointValue(50, 369, 'Fast')])
        # with a particular name
        kpv_node_returned3 = kpv_node.get_ordered_by_value(name='Slowest')
        self.assertEqual(kpv_node_returned3,
                         [KeyPointValue(2, 14, 'Slowest'), 
                          KeyPointValue(12, 30, 'Slowest'), 
                          KeyPointValue(342, 60, 'Slowest')])
        kpv_node_returned4 = kpv_node.get_ordered_by_value(name='Fast')
        self.assertEqual(kpv_node_returned4, [KeyPointValue(50, 369, 'Fast')])
        # named within a slice
        kpv_node_returned5 = kpv_node.get_ordered_by_value(slice(10,400),
                                                           'Slowest')
        self.assertEqual(kpv_node_returned5,
                         [KeyPointValue(12, 30, 'Slowest'), 
                          KeyPointValue(342, 60, 'Slowest')])
        # does not exist
        kpv_node_returned6 = kpv_node.get_ordered_by_value(name='Not Here')
        self.assertEqual(kpv_node_returned6, [])
        kpv_node_returned7 = kpv_node.get_ordered_by_value(slice(500,600))
        self.assertEqual(kpv_node_returned7, [])

    
class TestKeyTimeInstanceNode(unittest.TestCase):
    def setUp(self):
        class KTI(KeyTimeInstanceNode):
            def derive(self, a=P('a')):
                pass
        self.kti = KTI(frequency=2, offset=0.4)
    
    def test_create_kti(self):
        kti = self.kti
        #KTI = type('MyKti', (KeyTimeInstanceNode,), dict(derive=lambda x:x,
                                                         #dependencies=['a']))
        #params = {'a':Parameter('a',[], 2, 0.4)}
        #kti = KTI(frequency=2, offset=0.4)
        kti.create_kti(12, 'fast')
        self.assertEqual(kti, [KeyTimeInstance(index=12, name='fast')])
    
    def test_get_aligned(self):
        '''
        TODO: Test offset alignment.
        '''
        kti = self.kti
        #KTI = type('MyKti', (KeyTimeInstanceNode,), dict(derive=lambda x:x,
                                                         #dependencies=['a']))
        #params = {'a':Parameter('a',[], 2, 0.4)}
        #kti = KTI(frequency=2, offset=0.4)
        kti.create_kti(16, 'fast')
        kti.create_kti(18, 'fast')
        param = Parameter('p', frequency=0.25, offset=2)
        aligned_kti = kti.get_aligned(param)
        self.assertEqual(aligned_kti,
                         [KeyTimeInstance(index=1.6, name='fast'),
                          KeyTimeInstance(index=1.85, name='fast')])
    
class TestDerivedParameterNode(unittest.TestCase):
    def setUp(self):
        class ExampleDerivedParameterNode(DerivedParameterNode):
            def derive(self, alt_std=P('Altitude STD'),
                       pitch=P('Pitch')):
                pass
        class UnalignedDerivedParameterNode(DerivedParameterNode):
            align_to_first_dependency = False
            def derive(self, alt_std=P('Altitude STD'),
                       pitch=P('Pitch')):
                pass
        self.derived_class = ExampleDerivedParameterNode
        self.unaligned_class = UnalignedDerivedParameterNode
    
    @unittest.skip('Not Implemented')
    def test_frequency(self):
        self.assertTrue(False)
        # assert that Frequency MUST be set
        
        # assert that Derived masked array is of the correct length given the frequency
        # given a sample set of data for 10 seconds, assert that a 2 Hz param has len 20
        
        
        # Q: Have an aceessor on the DerivedParameter.get_result() which asserts the correct length of data returned.
        
    @unittest.skip('Not Implemented')
    def test_offset(self):
        self.assertTrue(False)
        # assert that offset MUST be set
    
    def test_get_derived_aligns(self):
        '''
        Tests get_derived returns a Parameter object rather than a
        DerivedParameterNode aligned to the frequency and offset of the first
        parameter passed to get_derived()
        '''
        param1 = Parameter('Altitude STD', frequency=1, offset=0)
        param2 = Parameter('Pitch', frequency=0.5, offset=1)
        # use first available param's freq and offset.
        derive_param = self.derived_class(frequency=param1.frequency, 
                                          offset=param1.offset)
        result = derive_param.get_derived([param1, param2])
        self.assertIsInstance(result, DerivedParameterNode)
        self.assertEqual(result.frequency, param1.frequency)
        self.assertEqual(result.offset, param1.offset)
        
    def test_get_derived_unaligned(self):
        """
        Set the class attribute align_to_first_dependency = False
        """
        param1 = Parameter('Altitude STD', frequency=1, offset=0)
        param2 = Parameter('Pitch', frequency=0.5, offset=1)
        unaligned_param = self.unaligned_class(frequency=param1.frequency, 
                                               offset=param1.offset)
        result = unaligned_param.get_derived([param1, param2])
        self.assertIsInstance(result, DerivedParameterNode)
        self.assertEqual(result.frequency, unaligned_param.frequency)
        self.assertEqual(result.offset, unaligned_param.offset)