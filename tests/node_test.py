import mock
import numpy as np
import unittest

from random import shuffle
from datetime import datetime

from analysis_engine.node import (
    Attribute,
    DerivedParameterNode,
    get_verbose_name,
    KeyPointValueNode, KeyPointValue, 
    KeyTimeInstanceNode, KeyTimeInstance, KTI,
    FlightAttributeNode,
    FormattedNameNode, 
    Node, NodeManager, 
    Parameter, P,
    powerset,
    SectionNode, Section,
)


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
                return 'a' in available and 'b' in available
        
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
        self.assertFalse(param1.get_aligned.called)
        param2.get_aligned.assert_called_once_with(param1)
        # check param1 is returned unchanged and param2 get_aligned is called
        # (returns '2')
        node.derive.assert_called_once_with(param1, 2)
        param1, param2 = get_mock_params()
        param3 = FlightAttributeNode('Attr')
        node.derive = mock.Mock()
        node.derive.return_value = None
        node.get_derived([param3, param2])
        node.derive.assert_called_once_with(param3, param2)
        # NotImplemented
        class NotImplementedNode(Node):
            def derive(self, kwarg1=param1, kwarg2=param2):
                return NotImplemented
        not_implemented_node = NotImplementedNode()
        self.assertRaises(NotImplementedError, not_implemented_node.get_derived,
                          [param1, param2])
        # align_to_first_dependency = False
        class UnalignedNode(Node):
            align_to_first_dependency = False
            def derive(self, kwarg1=param1, kwarg2=param2):
                pass
        node = UnalignedNode()
        param1, param2 = get_mock_params()
        node.get_derived([param1, param2])
        self.assertEqual(param1.method_calls, [])
        self.assertEqual(param2.method_calls, [])
        
        
class TestFlightAttributeNode(unittest.TestCase):
    def test_nonzero_flight_attr_node(self):
        'If no value is set, object evaluates to False - else True'
        attr = FlightAttributeNode()
        self.assertFalse(bool(attr))
        attr.value = []
        self.assertFalse(bool(attr))
        attr.value = 'something'
        self.assertTrue(bool(attr))
        attr.value = True
        self.assertTrue(bool(attr))
        attr.value = False
        self.assertFalse(bool(attr))
        
    def test_nonzero_attribute(self):
        'If no value is set, object evaluates to False - else True'
        attr = Attribute('Attribute')
        self.assertFalse(bool(attr))
        attr.value = []
        self.assertFalse(bool(attr))
        attr.value = 'something'
        self.assertTrue(bool(attr))
        attr.value = True
        self.assertTrue(bool(attr))
        attr.value = False
        self.assertFalse(bool(attr))

class TestNodeManager(unittest.TestCase):
    def test_operational(self):
        mock_node = mock.Mock('can_operate') # operable node
        mock_node.can_operate = mock.Mock(return_value=True)
        mock_inop = mock.Mock('can_operate') # inoperable node
        mock_inop.can_operate = mock.Mock(return_value=False)
        aci = {'n':1, 'o':2, 'p':3}
        afr = {'l':4, 'm':5}
        mgr = NodeManager(None, ['a', 'b', 'c', 'x'], ['a', 'x'], 
                          {'x': mock_inop, # note: derived node is not operational, but is already available in LFL - so this should return true!
                           'y': mock_node, 'z': mock_inop},
                          aci, afr)
        self.assertTrue(mgr.operational('a', []))
        self.assertTrue(mgr.operational('b', []))
        self.assertTrue(mgr.operational('c', []))
        # to ensure that if an lfl param is available, it's can_operate
        # returns True rather than calling the Derived node which may not
        # have all it's dependencies set. 'x' should return from the LFL!
        self.assertTrue(mgr.operational('x', []))
        self.assertTrue(mgr.operational('y', ['a']))
        self.assertTrue(mgr.operational('n', ['a'])) # achieved flight record
        self.assertTrue(mgr.operational('p', ['a'])) # aircraft info
        self.assertFalse(mgr.operational('z', ['a', 'b']))
        self.assertEqual(mgr.keys(), ['Start Datetime'] + list('abclmnopxyz'))
        
    def test_get_attribute(self):
        aci = {'a': 'a_value', 'b': None}
        afr = {'x': 'x_value', 'y': None}
        mgr = NodeManager(None, [],[],{},aci, afr)
        # test aircraft info
        a = mgr.get_attribute('a')
        self.assertEqual(a.__repr__(), Attribute('a', 'a_value').__repr__())
        b = mgr.get_attribute('b')
        self.assertFalse(b)
        c = mgr.get_attribute('c')
        self.assertFalse(c)
        # test afr
        x = mgr.get_attribute('x')
        self.assertEqual(x.__repr__(), Attribute('x', 'x_value').__repr__())
        y = mgr.get_attribute('y')
        self.assertFalse(y)
        z = mgr.get_attribute('z')
        self.assertFalse(z)
        
    def test_get_start_datetime(self):
        dt = datetime(2020,12,25)
        mgr = NodeManager(dt, [],[],{},{},{})
        self.assertTrue('Start Datetime' in mgr.keys())
        start_dt = mgr.get_attribute('Start Datetime')
        self.assertEqual(start_dt.name, 'Start Datetime')
        self.assertEqual(start_dt.value, dt)
        self.assertTrue(mgr.operational('Start Datetime', []))
                
        
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
                                  slice=slice(2, 3, None),start_edge=1.2,stop_edge=2.2),
                          Section(name='Example Section Node',
                                  slice=slice(3, 4, None),start_edge=2.7,stop_edge=3.7)])
        
    def test_get_aligned_with_slice_start_as_none(self):
        section_node = self.section_node_class(frequency=1, offset=0.5)
        section_node.create_section(slice(None,4))
        section_node.create_section(slice(5,None))
        param = Parameter('p', frequency=0.5, offset=0.1)
        aligned_node = section_node.get_aligned(param)
        self.assertEqual(list(aligned_node),
                         [Section(name='Example Section Node',
                                  slice=slice(None, 3, None),start_edge=None,stop_edge=2.2),
                          Section(name='Example Section Node',
                                  slice=slice(3, None, None),start_edge=2.7,stop_edge=None)])
    
    def test_items(self):
        items = [Section('a', slice(0,10), start_edge=0, stop_edge=10)]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        self.assertEqual(section_node, items)
    
    def test_get(self):
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
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
        sections = section_node.get(within_slice=slice(15, 40), name='b',
                                    within_use='stop')
        self.assertEqual(items[1:3], sections)        
    
    def test_get_first(self):
        # First test empty node.
        empty_section_node = self.section_node_class()
        self.assertEqual(empty_section_node.get_first(), None)
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
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
        first_section_within_slice = section_node.get_first(within_slice=
                                                            slice(12, 25),
                                                            first_by='stop')
        self.assertEqual(items[2], first_section_within_slice)        
        first_b_section_within_slice = section_node.get_first(within_slice=
                                                              slice(15, 40),
                                                              name='b')
        self.assertEqual(items[2], first_b_section_within_slice)
        first_b_section_within_slice = section_node.get_first(within_slice=
                                                              slice(17, 40),
                                                              name='b',
                                                              within_use='start')
        self.assertEqual(items[2], first_b_section_within_slice)
        first_b_section_within_slice = section_node.get_first(within_slice=
                                                              slice(17, 40),
                                                              name='b',
                                                              within_use='stop')
        self.assertEqual(items[1], first_b_section_within_slice)        
    
    def test_get_last(self):
        # First test empty node.
        empty_section_node = self.section_node_class()
        self.assertEqual(empty_section_node.get_last(), None)
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        last_section = section_node.get_last()
        self.assertEqual(items[3], last_section)
        last_section = section_node.get_last(within_slice=slice(13,24),
                                             last_by='stop')
        self.assertEqual(items[1], last_section)     
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
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        sections = section_node.get_ordered_by_index()
        self.assertEqual([items[0], items[1], items[2], items[3]], sections)
        sections = section_node.get_ordered_by_index(order_by='stop')
        self.assertEqual([items[0], items[2], items[1], items[3]], sections)        
        sections = section_node.get_ordered_by_index(name='b')
        self.assertEqual([items[1], items[2]], sections)
        sections = section_node.get_ordered_by_index(name='c')
        self.assertEqual([items[-1]], sections)
        sections = section_node.get_ordered_by_index(within_slice=slice(12, 25))
        self.assertEqual([items[1], items[2]], sections)
        sections = section_node.get_ordered_by_index(within_slice=slice(15, 40), name='b')
        self.assertEqual([items[2]], sections)
    
    def test_get_next(self):
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        section = section_node.get_next(16)
        self.assertEqual(items[2], section)
        section = section_node.get_next(16, use='stop')
        self.assertEqual(items[1], section)
        section = section_node.get_next(16, name='c')
        self.assertEqual(items[3], section)
        section = section_node.get_next(16, within_slice=slice(25, 40))
        self.assertEqual(items[3], section)
        section = section_node.get_next(3, frequency=0.5)
        self.assertEqual(items[1], section)
    
    def test_get_previous(self):
        items = [Section('a', slice(4,10),4,10),
                 Section('b', slice(14,23),14,23),
                 Section('b', slice(19,21),19,21),
                 Section('c', slice(30,34),30,34),]
        section_node = self.section_node_class(frequency=1, offset=0.5,
                                               items=items)
        section = section_node.get_previous(16)
        self.assertEqual(items[0], section)
        section = section_node.get_previous(16, use='start')
        self.assertEqual(items[1], section)
        section = section_node.get_previous(30, name='a')
        self.assertEqual(items[0], section)
        section = section_node.get_previous(23, within_slice=slice(0, 12))
        self.assertEqual(items[0], section)
        section = section_node.get_previous(40, frequency=2)
        self.assertEqual(items[0], section)    

    def test_get_surrounding(self):
        node = SectionNode()
        self.assertEqual(node.get_surrounding(12), [])
        sect_1 = Section('ThisSection', slice(2,15), 2, 15)
        node.append(sect_1)
        self.assertEqual(node.get_surrounding(2), [sect_1])
        sect_2 = Section('ThisSection', slice(5,25), 5, 25)
        node.append(sect_2)
        self.assertEqual(node.get_surrounding(12), [sect_1, sect_2])
        self.assertEqual(node.get_surrounding(-3), [])
        self.assertEqual(node.get_surrounding(25), [sect_2])


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
        """
        Using all RETURNS options, apply NAME_FORMAT to obtain a complete
        list of KPV names this class will create.
        """
        class Speed(FormattedNameNode):
            NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
            NAME_VALUES = {'altitude': range(100, 701, 300),
                           'phase': ['ascent', 'descent']}            
            def derive(self, *args, **kwargs):
                pass
        formatted_name_node = Speed()
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
        class Speed(FormattedNameNode):
            NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
            NAME_VALUES = {'altitude' : range(100,1000,100),
                           'phase' : ['ascent', 'descent']}          
            def derive(self, *args, **kwargs):
                pass
        formatted_name_node = Speed()
        self.assertTrue(formatted_name_node._validate_name('Speed in ascent at 500 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 900 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 100 ft'))
        self.assertRaises(ValueError, formatted_name_node._validate_name, 'Speed in ascent at -10 ft')       
        
    def test_get_first(self):
        # Test empty Node first.
        empty_kti_node = KeyTimeInstanceNode()
        self.assertEqual(empty_kti_node.get_last(), None)
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
        # Test empty Node first.
        empty_kti_node = KeyTimeInstanceNode()
        self.assertEqual(empty_kti_node.get_last(), None)
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
        kti_node_returned1 = kti_node.get(name='Slowest')
        self.assertEqual(kti_node_returned1,
                         [KeyTimeInstance(12, 'Slowest'),
                          KeyTimeInstance(342, 'Slowest'),
                          KeyTimeInstance(2, 'Slowest'),])
        kti_node_returned2 = kti_node.get(name='Fast')
        self.assertEqual(kti_node_returned2, [KeyTimeInstance(50, 'Fast')])
        # named within a slice
        kti_node_returned3 = kti_node.get(name='Slowest',
                                          within_slice=slice(10,400))
        self.assertEqual(kti_node_returned3,
                         [KeyTimeInstance(12, 'Slowest'),
                          KeyTimeInstance(342, 'Slowest')])
        # does not exist
        kti_node_returned4 = kti_node.get(name='Not Here')
        self.assertEqual(kti_node_returned4, [])
        kti_node_returned5 = kti_node.get(name='Slowest',
                                          within_slice=slice(500,600))
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
    
    def test_get_next(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        next_kti = kti_node.get_next(35)
        self.assertEqual(next_kti, KeyTimeInstance(50, 'Fast'))
        next_kti = kti_node.get_next(35, name="Slowest")
        self.assertEqual(next_kti, KeyTimeInstance(342, 'Slowest'))
        next_kti = kti_node.get_next(1, within_slice=slice(10,20))
        self.assertEqual(next_kti, KeyTimeInstance(12, 'Slowest'))
        next_kti = kti_node.get_next(65, name="Fast")
        self.assertEqual(next_kti, None)
        next_kti = kti_node.get_next(1, frequency=0.25)
        self.assertEqual(next_kti, KeyTimeInstance(12, 'Slowest'))        
        
    def test_get_previous(self):
        kti_node = KeyTimeInstanceNode(items=[KeyTimeInstance(12, 'Slowest'), 
                                              KeyTimeInstance(342, 'Slowest'), 
                                              KeyTimeInstance(2, 'Slowest'), 
                                              KeyTimeInstance(50, 'Fast')])
        previous_kti = kti_node.get_previous(56)
        self.assertEqual(previous_kti, KeyTimeInstance(50, 'Fast'))
        previous_kti = kti_node.get_previous(410, name="Slowest")
        self.assertEqual(previous_kti, KeyTimeInstance(342, 'Slowest'))
        previous_kti = kti_node.get_previous(60, within_slice=slice(10,20))
        self.assertEqual(previous_kti, KeyTimeInstance(12, 'Slowest'))
        previous_kti = kti_node.get_previous(25, name="Fast")
        self.assertEqual(previous_kti, None)
        previous_kti = kti_node.get_previous(40, frequency=4)
        self.assertEqual(previous_kti, KeyTimeInstance(2, 'Slowest'))

    def test_initial_items_storage(self):
        node = FormattedNameNode(['a', 'b', 'c'])
        self.assertEqual(list(node), ['a', 'b', 'c'])
        node = FormattedNameNode(('a', 'b', 'c'))
        self.assertEqual(list(node), ['a', 'b', 'c'])
        node = FormattedNameNode(items=['a', 'b', 'c'])
        self.assertEqual(list(node), ['a', 'b', 'c'])


class TestKeyPointValueNode(unittest.TestCase):
    
    def setUp(self):
        class KPV(KeyPointValueNode):
            def derive(self, a=P('a',[], 2, 0.4)):
                pass
        self.knode = KPV(frequency=2, offset=0.4)

    def test_create_kpv(self):
        """ Tests name format substitution and return type
        """
        class Speed(KeyPointValueNode):
            NAME_FORMAT = 'Speed in %(phase)s at %(altitude)dft'
            NAME_VALUES = {'phase': ['ascent', 'descent'],
                           'altitude': [1000, 1500],}            
            def derive(self, *args, **kwargs):
                pass
        
        knode = Speed(frequency=2, offset=0.4)
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
        # None index
        self.assertEqual(knode.create_kpv(None, 'b'), None)        
        # missing interpolation value "altitude"
        self.assertRaises(KeyError, knode.create_kpv, 1, 2,
                          dict(phase='ascent'))
        # wrong type raises TypeError
        self.assertRaises(TypeError, knode.create_kpv, 2, '3', 
                          phase='', altitude='')
        # None index -- now logs a WARNING and does not raise an error
        knode.create_kpv(None, 'b')
        self.assertTrue('b' not in knode)  ## this test isn't quite right...!

        
    def test_create_kpvs_at_ktis(self):
        knode = self.knode
        param = P('Param', np.ma.arange(10))
        # value_at_index will interpolate masked values.
        param.array[3:7] = np.ma.masked
        ktis = KTI('KTI', items=[KeyTimeInstance(i, 'a') for i in range(0,10,2)])
        knode.create_kpvs_at_ktis(param.array, ktis)
        self.assertEqual(list(knode),
                         [KeyPointValue(index=0, value=0, name='Kpv'),
                          KeyPointValue(index=2, value=2, name='Kpv'),
                          KeyPointValue(index=8, value=8, name='Kpv')])
    
    def test_create_kpvs_at_ktis_suppressed_zeros(self):
        knode = self.knode
        param = P('Param', np.ma.array([0]*5+[7]*5))
        # value_at_index will interpolate masked values.
        param.array[3:7] = np.ma.masked
        ktis = KTI('KTI', items=[KeyTimeInstance(i, 'a') for i in range(0,10,2)])
        knode.create_kpvs_at_ktis(param.array, ktis, suppress_zeros=True)
        self.assertEqual(list(knode),
                         [KeyPointValue(index=8, value=7, name='Kpv')])
    
    def test_create_kpvs_within_slices(self):
        knode = self.knode
        function = mock.Mock()
        return_values = [(10, 15), (22, 27)]
        def side_effect(*args, **kwargs):
            return return_values.pop()
        function.side_effect = side_effect
        slices = [slice(1,10), slice(15, 25)]
        array = np.ma.arange(10)
        knode.create_kpvs_within_slices(array, slices, function)
        self.assertEqual(list(knode),
                         [KeyPointValue(index=22, value=27, name='Kpv'),
                          KeyPointValue(index=10, value=15, name='Kpv')])

    def test_create_kpvs_from_discretes(self):
        knode = self.knode
        param = P('Disc',np.ma.array([0.0]*20, dtype=float))
        param.array[5:8] = 1.0
        param.array[11:17] = 1.0
        knode.create_kpvs_from_discretes(param.array, param.hz)
        knode.create_kpvs_from_discretes(param.array, param.hz, sense='reverse')
        knode.create_kpvs_from_discretes(param.array, param.hz, min_duration=3)
        # Need to add result for min_duration case.
        self.assertEqual(list(knode),
                         [KeyPointValue(index=5, value=3, name='Kpv'),
                          KeyPointValue(index=11, value=6, name='Kpv'),
                          KeyPointValue(index=0, value=5, name='Kpv'),
                          KeyPointValue(index=8, value=3, name='Kpv'),
                          KeyPointValue(index=17, value=3, name='Kpv')])
    
    def test_get_aligned(self):
        '''
        TODO: Test offset alignment.
        '''
        class Speed(KeyPointValueNode):
            NAME_FORMAT = 'Speed at %(altitude)dft'
            NAME_VALUES = {'altitude': [1000, 1500],}            
            def derive(self, *args, **kwargs):
                pass        
        knode = Speed(frequency=2, offset=0.4)
        param = Parameter('p', frequency=0.5, offset=1.5)
        knode.create_kpv(10, 12.5, altitude=1000)
        knode.create_kpv(24, 12.5, altitude=1000)
        aligned_node = knode.get_aligned(param)
        self.assertEqual(aligned_node.frequency, param.frequency)
        self.assertEqual(aligned_node.offset, param.offset)
        self.assertEqual(aligned_node,
                         [KeyPointValue(index=1.95, value=12.5, name='Speed at 1000ft'),
                          KeyPointValue(index=5.45, value=12.5, name='Speed at 1000ft')])
        
    def test_get_min(self):
        # Test empty Node first.
        empty_kpv_node = KeyPointValueNode()
        self.assertEqual(empty_kpv_node.get_min(), None)
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
        # Test empty Node first.
        empty_kpv_node = KeyPointValueNode()
        self.assertEqual(empty_kpv_node.get_max(), None)
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
        kti.create_kti(12)
        self.assertEqual(list(kti), [KeyTimeInstance(index=12, name='Kti')])
        
        # NAME_FORMAT
        class Flap(KeyTimeInstanceNode):
            NAME_FORMAT = 'Flap %(setting)d'
            NAME_VALUES = {'setting': [10, 25, 35]}
            def derive(self, a=P('a')):
                pass              
        kti = Flap()
        kti.create_kti(24, setting=10)
        kti.create_kti(35, {'setting':25})
        self.assertEqual(list(kti), [KeyTimeInstance(index=24, name='Flap 10'),
                                     KeyTimeInstance(index=35, name='Flap 25'),])
        # None index
        self.assertRaises(ValueError, kti.create_kti, None)
    
    def test_create_ktis_at_edges_inside_phases(self):
        kti=self.kti
        test_array = np.ma.array([0,1,0,1,0,1,0,1,0])
        test_phase = SectionNode(items=[Section('try',slice(None,2,None),None,2),
                                        Section('try',slice(5,None,None),5,None)])
        kti.create_ktis_at_edges(test_array, phase=test_phase)
        self.assertEqual(kti,[KeyTimeInstance(index=0.5, name='Kti'),
                              KeyTimeInstance(index=6.5, name='Kti')])
    
    def test_create_ktis_at_edges_either(self):
        kti=self.kti
        test_param = np.ma.array([0,0,0,1,1,1,0,0,0])
        kti.create_ktis_at_edges(test_param, direction='all_edges')
        self.assertEqual(kti,[KeyTimeInstance(index=2.5, name='Kti'),
                          KeyTimeInstance(index=5.5, name='Kti')])
        
    def test_create_ktis_at_edges_rising(self):
        kti=self.kti
        test_param = np.ma.array([0,0,0,1,1,1,0,0,0])
        kti.create_ktis_at_edges(test_param)
        self.assertEqual(kti,[KeyTimeInstance(index=2.5, name='Kti')])
        
    def test_create_ktis_at_edges_falling(self):
        kti=self.kti
        test_param = np.ma.array([0,1,1,0,0,0,-1,-1,0])
        kti.create_ktis_at_edges(test_param, direction='falling_edges')
        self.assertEqual(kti,[KeyTimeInstance(index=2.5, name='Kti'),
                              KeyTimeInstance(index=5.5, name='Kti')])
        
    def test_create_ktis_at_edges_fails(self):
        kti=self.kti
        test_param = np.ma.array([0])
        self.assertRaises(ValueError, kti.create_ktis_at_edges, test_param, direction='sideways')
        
    def test_get_aligned(self):
        '''
        TODO: Test offset alignment.
        '''
        kti = self.kti
        #KTI = type('MyKti', (KeyTimeInstanceNode,), dict(derive=lambda x:x,
                                                         #dependencies=['a']))
        #params = {'a':Parameter('a',[], 2, 0.4)}
        #kti = KTI(frequency=2, offset=0.4)
        kti.create_kti(16)
        kti.create_kti(18)
        param = Parameter('p', frequency=0.25, offset=2)
        aligned_kti = kti.get_aligned(param)
        self.assertEqual(aligned_kti,
                         [KeyTimeInstance(index=1.6, name='Kti'),
                          KeyTimeInstance(index=1.85, name='Kti')])


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
        
    def test_get_derived_discrete_align(self):
        """
        Ensure that interpolations does not occur
        """
        class Flap(DerivedParameterNode):
            data_type = 'Multi-state'
            
        master_param = Parameter(array=np.ma.array([5,6,7]), 
                                 frequency=1, offset=0.4, data_type=None)
            
        slave_flap = Flap(array=np.ma.array([1,2,3]),frequency=1, offset=0)
        res = slave_flap.get_aligned(master_param)
        # note it has not interpolated to 1.4, 2.4, 3.4 forward
        self.assertEqual(list(res.array), [1,2,3]) 
        
    def test_parameter_at(self):
        # using a plain range as the parameter array, the results are equal to 
        # the index used to get the value (cool)
        spd = Parameter('Airspeed', np.ma.array(range(20)), 2, 0.75)
        self.assertEqual(spd.at(0.75), 0) # min val possible to return
        self.assertEqual(spd.at(1.75), 1*2) # one second in (*2Hz)
        self.assertEqual(spd.at(2.5), 1.75*2) # interpolates
        self.assertEqual(spd.at(9.75), 9*2) # max val without extrapolation
        self.assertEqual(spd.at(0), 0) # Extrapolation at bottom end
        self.assertEqual(spd.at(11), 19) # Extrapolation at top end
        
    @mock.patch('analysis_engine.node.slices_above')
    def test_slices_above(self, slices_above):
        '''
        Ensure slices_above is called with the expected arguments.
        '''
        array = np.ma.arange(10)
        slices_above.return_value = (array, [slice(0,10)])
        param = DerivedParameterNode('Param', array=array)
        slices = param.slices_above(5)
        slices_above.assert_called_once_with(array, 5)
        self.assertEqual(slices, slices_above.return_value[1])
        
    @mock.patch('analysis_engine.node.slices_below')
    def test_slices_below(self, slices_below):
        '''
        Ensure slices_below is called with the expected arguments.
        '''
        array = np.ma.arange(10)
        slices_below.return_value = (array, [slice(0,10)])
        param = DerivedParameterNode('Param', array=array)
        slices = param.slices_below(5)
        slices_below.assert_called_once_with(array, 5)
        self.assertEqual(slices, slices_below.return_value[1]) 
    
    @mock.patch('analysis_engine.node.slices_between')
    def test_slices_between(self, slices_between):
        '''
        Ensure slices_between is called with the expected arguments.
        '''
        array = np.ma.arange(10)
        slices_between.return_value = (array, [slice(0, 10)])
        param = DerivedParameterNode('Param', array=array)
        slices = param.slices_between(5, 15)
        slices_between.assert_called_once_with(array, 5, 15)
        self.assertEqual(slices, slices_between.return_value[1])
    
    @mock.patch('analysis_engine.node.slices_from_to')
    def test_slices_from_to(self, slices_from_to):
        '''
        Ensure slices_from_to is called with the expected arguments.
        '''
        array = np.ma.arange(10)
        slices_from_to.return_value = (array, [slice(0, 10)])
        param = DerivedParameterNode('Param', array=array)
        slices = param.slices_from_to(5, 15)
        slices_from_to.assert_called_once_with(array, 5, 15)
        self.assertEqual(slices, slices_from_to.return_value[1])
        
    def test_slices_to_touchdown_basic(self):
        heights = np.ma.arange(100,-10,-10)
        heights[:-1] -= 10
        alt_aal = P('Altitude AAL',heights)
        tdwns = KTI(items=[KeyTimeInstance(name='Touchdown', index=9.5)])
        result = alt_aal.slices_to_kti(75, tdwns)
        expected = [slice(2,9.5)]
        self.assertEqual(result, expected)

    def test_slices_to_touchdown_early(self):
        heights = np.ma.arange(100,-10,-10)
        heights[:-1] -= 10
        alt_aal = P('Altitude AAL',heights)
        tdwns = KTI(items=[KeyTimeInstance(name='Touchdown', index=6.7)])
        result = alt_aal.slices_to_kti(75, tdwns)
        expected = [slice(2,6.7)]
        self.assertEqual(result, expected)
        
    def test_slices_to_touchdown_outside_range(self):
        heights = np.ma.arange(100,-10,-10)
        heights[:-1] -= 10
        alt_aal = P('Altitude AAL',heights)
        tdwns = KTI(items=[KeyTimeInstance(name='Touchdown', index=-2)])
        result = alt_aal.slices_to_kti(75, tdwns)
        expected = []
        self.assertEqual(result, expected)


