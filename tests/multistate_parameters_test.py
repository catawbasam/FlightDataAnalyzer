import numpy as np
import operator
import unittest

#from hdfaccess.parameter import MappedArray
from flightdatautilities import masked_array_testutils as ma_test

from analysis_engine.node import (
    #Attribute,
    A,
    #App,
    #ApproachItem,
    #KeyPointValue,
    #KPV,
    #KeyTimeInstance,
    #KTI,
    #load,
    #M,
    Parameter,
    P,
    #Section,
    #S,
)
from analysis_engine.multistate_parameters import (
    Configuration,
    Flap,
    FlapLever,
    )


class TestConfiguration(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Configuration
        self.operational_combinations = [
            ('Flap', 'Slat', 'Series', 'Family'),
            ('Flap', 'Slat', 'Flaperon', 'Series', 'Family'),
        ]
        # Note: The last state is invalid...
        s = [0] * 2 + [16] * 4 + [20] * 4 + [23] * 6 + [16]
        f = [0] * 4 + [8] * 4 + [14] * 4 + [22] * 2 + [32] * 2 + [14]
        a = [0] * 4 + [5] * 2 + [10] * 10 + [10]
        self.slat = P('Slat', np.tile(np.ma.array(s), 10000))
        self.flap = M('Flap', np.tile(np.ma.array(f), 10000),
                      values_mapping={x: str(x) for x in np.ma.unique(f)})
        self.ails = P('Flaperon', np.tile(np.ma.array(a), 10000))

    def test_conf_for_a330(self):
        # Note: The last state is invalid...
        expected = ['0', '1', '1+F', '1*', '2', '2*', '3', 'Full']
        expected = list(reduce(operator.add, zip(expected, expected)))
        expected += [np.ma.masked]
        series = A('Series', 'A330-301')
        family = A('Family', 'A330')
        node = self.node_class()
        node.derive(self.slat, self.flap, self.ails, series, family)
        self.assertEqual(list(node.array[:17]), expected)

    def test_conf_for_bombardier(self):
        # Note: Bombardier does not use configuration settings so should
        # return masked array
        series = A('Series', 'Global Express XRS')
        family = A('Family', 'Global')
        manuf = A('Manufacturer', 'Bombardier')
        node = self.node_class()
        node.derive(self.slat, self.flap, self.ails, series, family, manuf)
        
        self.assertEqual(np.ma.count_masked(node.array), 170000)
        self.assertEqual(np.ma.count(node.array), 0)

    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_conf_for_a330)
        time = min(timer.repeat(1, 1))
        self.assertLess(time, 0.3, msg='Took too long: %.3fs' % time)


class TestFlap(unittest.TestCase):
        
    def test_can_operate(self):
        opts = Flap.get_operational_combinations()
        self.assertEqual(opts, [
            ('Flap Angle', 'Series', 'Family'), # normal
        ])

    def test_flap_stepped_nearest_5(self):
        flap = P('Flap Angle', np.ma.arange(50))
        node = Flap()
        node.derive(flap, A('Series', None), A('Family', None))
        expected = [0] * 3 + [5] * 5 + [10] * 5 + [15] * 2
        self.assertEqual(node.array[:15].tolist(), expected)
        expected = [45] * 5 + [50] * 2
        self.assertEqual(node.array[-7:].tolist(), expected)
        self.assertEqual(
            node.values_mapping,
            {0: '0', 35: '35', 5: '5', 40: '40', 10: '10', 45: '45', 15: '15',
             50: '50', 20: '20', 25: '25', 30: '30'})

        flap = P('Flap Angle', np.ma.array(range(20), mask=[True] * 10 + [False] * 10))
        node.derive(flap, A('Series', None), A('Family', None))
        expected = [-1] * 10 + [10] * 3 + [15] * 5 + [20] * 2
        self.assertEqual(np.ma.filled(node.array, fill_value=-1).tolist(),
                         expected)
        self.assertEqual(node.values_mapping, {10: '10', 20: '20', 15: '15'})

    def test_flap_using_md82_settings(self):
        # Note: Using flap detents for MD-82 of (0, 13, 20, 25, 30, 40)
        # Note: Flap uses library.step_values(..., step_at='move_end')!
        indexes = (1, 57, 58)
        flap = P(
            name='Flap Angle',
            array=np.ma.array(range(50) + range(-5, 0) + [13.1, 1.3, 10, 10]),
        )
        for index in indexes:
            flap.array[index] = np.ma.masked

        node = self.node_class()
        node.derive(flap, A('Series', None), A('Family', 'DC-9'))

        expected = reduce(operator.add, [
            [0, -999] + [0] * 11,  #  0.0 -> 12.5 (one masked)
            [13] * 7,              # 13.0 -> 19.5
            [20] * 5,              # 20.0 -> 24.5
            [25] * 5,              # 25.0 -> 29.5
            [30] * 10,             # 30.0 -> 39.5
            [40] * 10,             # 40.0 -> 49.5
            [0] * 5,               # -5.0 -> -1.0
            [13, 0],               # odd float values
            [-999] * 2,            # masked values
        ])
        expected = np.ma.array(
            ([0] * 7) + ([13] * 10) + ([20] * 6) + ([25] * 5) + ([30] * 8) + 
            ([40] * 14) + ([0] * 5) + [13] + ([0] * 3),
            mask=[False, True] + ([False] * 55) + [True, True]
        )

        self.assertEqual(node.array.size, 59)
        self.assertEqual(node.array.tolist(), expected.tolist())
        self.assertEqual(
            node.values_mapping,
            {0: '0', 40: '40', 13: '13', 20: '20', 25: '25', 30: '30'})
        for index in indexes:
            self.assertTrue(np.ma.is_masked(node.array[index]))

    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_flap_using_md82_settings)
        time = min(timer.repeat(2, 100))
        self.assertLess(time, 1.5, msg='Took too long: %.3fs' % time)
        
    def test_decimal_flap_settings(self):
        # Beechcraft has a flap 17.5
        flap_param = Parameter('Flap Angle', array=np.ma.array(
            [0, 5, 7.2, 
             17, 17.4, 17.9, 20, 
             30]))
        flap = Flap()
        flap.derive(flap_param, A('Series', '1900D'), A('Family', 'Beechcraft'))
        self.assertEqual(flap.values_mapping,
                         {0: '0', 17: '17.5', 35: '35'})
        ma_test.assert_array_equal(
            flap.array, ['0', '0', '0',
                         '17.5', '17.5', '17.5', '17.5',
                         '35'])
        
    def test_flap_settings_for_hercules(self):
        # No flap recorded; ensure it converts exactly the same
        flap_param = Parameter('Altitude AAL', array=np.ma.array(
            [0, 0, 0, 
             50, 50, 50,
             100]))
        flap = Flap()
        flap.derive(flap_param, A('Series', 'L1011-100'), A('Family', 'L1011'))
        self.assertEqual(flap.values_mapping,
                         {0: '0', 50: '50', 100: '100'})
        ma_test.assert_array_equal(
            flap.array, ['0', '0', '0',
                         '50', '50', '50',
                         '100'])
        
        
class TestFlapLever(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = FlapLever
        self.operational_combinations = [
            ('Flap Lever Angle', 'Series', 'Family'),
        ]
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
