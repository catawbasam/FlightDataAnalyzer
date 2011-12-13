try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

import mock

from random import shuffle

from analysis.node import (
    DerivedParameterNode, get_verbose_name, KeyPointValue, KeyPointValueNode,
    KeyTimeInstance, KeyTimeInstanceNode, FormattedNameNode, Node, NodeManager,
    P, Parameter, powerset, Section, SectionNode)


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
            param2 = mock.Mock()
            param2.name = 'PARAM2'
            param2.frequency = 0.5
            param2.offset = 1
            return param1, param2
        param1, param2 = get_mock_params()
        class TestNode(Node):
            def derive(self, kwarg1=param1, kwarg2=param2):
                pass
        node = TestNode()
        node.get_derived([param1, param2])
        self.assertEqual(param1.method_calls, [])
        self.assertEqual(param2.method_calls, [('get_aligned', (param1,), {})])
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
        mgr = NodeManager(['a', 'b', 'c'], ['a', 'x'], 
                          {'x': mock_node, 'y': mock_node, 'z': mock_inop})
        self.assertTrue(mgr.operational('a', []))
        self.assertTrue(mgr.operational('b', []))
        self.assertTrue(mgr.operational('c', []))
        self.assertTrue(mgr.operational('x', []))
        self.assertTrue(mgr.operational('y', ['a']))
        self.assertFalse(mgr.operational('z', ['a', 'b']))
        
        
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

class TestFormattedNameNode(unittest.TestCase):
    def setUp(self):
        class ExampleNameFormatNode(FormattedNameNode):
            def derive(self, a=P('a',[], 2, 0.4)):
                pass
            def get_derived(self):
                pass
        self.formatted_name_node = ExampleNameFormatNode()
    
    def test_names(self):
        """ Using all RETURNS options, apply NAME_FORMAT to obtain a complete
        list of KPV names this class will create.
        """
        formatted_name_node = self.formatted_name_node
        formatted_name_node.NAME_FORMAT = 'Speed in %(phase)s at %(altitude)d ft'
        formatted_name_node.RETURN_OPTIONS = {'altitude' : range(100, 701, 300),'phase' : ['ascent', 'descent']}
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
        formatted_name_node.RETURN_OPTIONS = {'altitude' : range(100,1000,100),
                                'phase' : ['ascent', 'descent']}
        self.assertTrue(formatted_name_node._validate_name('Speed in ascent at 500 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 900 ft'))
        self.assertTrue(formatted_name_node._validate_name('Speed in descent at 100 ft'))
        self.assertRaises(ValueError, formatted_name_node._validate_name, 'Speed in ascent at -10 ft')


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
        knode.RETURN_OPTIONS = {'phase':['ascent', 'descent'],
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
        knode.RETURN_OPTIONS = {'altitude':[1000,1500]}
        param = Parameter('p', frequency=0.5, offset=1.5)
        knode.create_kpv(10, 12.5, altitude=1000.0)
        knode.create_kpv(24, 12.5, altitude=1000.0)
        aligned_node = self.knode.get_aligned(param)
        self.assertEqual(aligned_node.frequency, param.frequency)
        self.assertEqual(aligned_node.offset, param.offset)
        self.assertEqual(aligned_node,
                         [KeyPointValue(index=1.95, value=12.5, name='Speed at 1000ft'),
                          KeyPointValue(index=5.45, value=12.5, name='Speed at 1000ft')])
    
    
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
        self.assertEqual(kti, [KeyTimeInstance(index=12, state='fast')])
    
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
                         [KeyTimeInstance(index=1.6, state='fast'),
                          KeyTimeInstance(index=1.85, state='fast')])
    
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
    
    def test_frequency(self):
        self.assertTrue(False)
        # assert that Frequency MUST be set
        
        # assert that Derived masked array is of the correct length given the frequency
        # given a sample set of data for 10 seconds, assert that a 2 Hz param has len 20
        
        
        # Q: Have an aceessor on the DerivedParameter.get_result() which asserts the correct length of data returned.
        
    def test_offset(self):
        self.assertTrue(False)
        # assert that offset MUST be set
    
    def test_get_derived(self):
        '''
        Tests get_derived returns a Parameter object rather than a
        DerivedParameterNode aligned to the frequency and offset of the first
        parameter passed to get_derived()
        '''
        derive_param = self.derived_class(frequency=2, offset=1)
        param1 = Parameter('Altitude STD', frequency=1, offset=0)
        param2 = Parameter('Pitch', frequency=0.5, offset=1)
        result = derive_param.get_derived([param1, param2])
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.frequency, param1.frequency)
        self.assertEqual(result.offset, param1.offset)
        unaligned_param = self.unaligned_class(frequency=2, offset=1)
        param1 = Parameter('Altitude STD', frequency=1, offset=0)
        param2 = Parameter('Pitch', frequency=0.5, offset=1)
        result = unaligned_param.get_derived([param1, param2])
        self.assertIsInstance(result, Parameter)
        self.assertEqual(result.frequency, unaligned_param.frequency)
        self.assertEqual(result.offset, unaligned_param.offset)