import datetime
import numpy as np
import operator
import os
import unittest

from hdfaccess.parameter import MappedArray
from flightdatautilities import masked_array_testutils as ma_test

from analysis_engine.node import (
    Attribute,
    A,
    #App,
    #ApproachItem,
    #KeyPointValue,
    #KPV,
    #KeyTimeInstance,
    #KTI,
    load,
    M,
    Parameter,
    P,
    Section,
    S,
)
from analysis_engine.multistate_parameters import (
    APEngaged,
    Autoland,
    Configuration,
    Daylight,
    EngThrustModeRequired,
    Eng_1_Fire,
    Eng_2_Fire,
    Eng_3_Fire,
    Eng_4_Fire,
    Eng_Fire,
    Flap,
    FlapExcludingTransition,
    FlapIncludingTransition,
    FlapLever,
    FuelQty_Low,
    GearDown,
    GearDownSelected,
    GearOnGround,
    GearUpSelected,
    KeyVHFCapt,
    KeyVHFFO,
    MasterWarning,
    PackValvesOpen,
    PitchAlternateLaw,
    SpeedbrakeSelected,
    StableApproach,
    StickShaker,
    TakeoffConfigurationWarning,
    TAWSAlert,
    TAWSDontSink,
    TAWSGlideslopeCancel,
    TAWSTooLowGear,
    TCASFailure,
    ThrustReversers,
)


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

class NodeTest(object):
    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            combinations = map(set, self.node_class.get_operational_combinations())
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)


class TestAPEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngaged
        self.operational_combinations = [
            ('AP (1) Engaged',),
            ('AP (2) Engaged',),
            ('AP (3) Engaged',),
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]
    def test_single_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        eng = APEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.1)        
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        # Two result in just "Engaged" state still
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')        
        ap3 = None
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,1,1,1,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.1)        
        
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged', 
                   frequency=1, 
                   offset=0.1)        
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged', 
                   frequency=1, 
                   offset=0.2)        
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged', 
                   frequency=1, 
                   offset=0.4)        
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,1,1,1,1,1]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged', 
                   frequency=1, 
                   offset=0.25)        
        
        ma_test.assert_array_equal(expected.array, eng.array)


class TestAutoland(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Autoland
        self.operational_combinations = [
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]

    def test_single_ap(self):
        # Cannot auto_land on one AP
        ap1 = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        eng = Autoland()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping={2: 'Dual', 3: 'Triple'},
                   name='Autoland', 
                   frequency=1, 
                   offset=0.1)        
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')        
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')        
        ap3 = None
        eng = Autoland()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,0,2,0,0]),
                   values_mapping={2: 'Dual', 3: 'Triple'},
                   name='Autoland', 
                   frequency=1, 
                   offset=0.1)        
        
        ma_test.assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged', 
                   frequency=1, 
                   offset=0.1)        
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged', 
                   frequency=1, 
                   offset=0.2)        
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged', 
                   frequency=1, 
                   offset=0.4)        
        eng = Autoland()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,2,3,2,0]),
                   values_mapping={2: 'Dual', 3: 'Triple'},
                   name='Autoland', 
                   frequency=1, 
                   offset=0.25)        
        
        ma_test.assert_array_equal(expected.array, eng.array)


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
    
    def test_can_operate_not_airbus(self):
        self.assertFalse(self.node_class.can_operate(
            ['Flap', 'Slat', 'Series', 'Family'],
            manu=Attribute('Manufacturer', 'Boeing')))
        self.assertTrue(self.node_class.can_operate(
            ['Flap', 'Slat', 'Series', 'Family'],
            manu=Attribute('Manufacturer', 'Airbus')))

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

    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_conf_for_a330)
        time = min(timer.repeat(1, 1))
        self.assertLess(time, 0.3, msg='Took too long: %.3fs' % time)


class TestDaylight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Latitude Smoothed', 'Longitude Smoothed', 
                     'Start Datetime', 'HDF Duration')]
        opts = Daylight.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_daylight_aligning(self):
        lat = P('Latitude', np.ma.array([51.1789]*128), offset=0.1)
        lon = P('Longitude', np.ma.array([-1.8264]*128))
        start_dt = A('Start Datetime', datetime.datetime(2012,6,20, 20,25))
        dur = A('HDF Duration', 128)
        
        don = Daylight()
        don.get_derived((lat, lon, start_dt, dur))
        self.assertEqual(list(don.array), [np.ma.masked] + ['Day']*31)
        self.assertEqual(don.frequency, 0.25)
        self.assertEqual(don.offset, 0)

    def test_father_christmas(self):
        # Starting on the far side of the world, he flies all round
        # delivering parcels mostly by night (in the northern lands).
        lat=P('Latitude', np.ma.arange(60,64,1/64.0))
        lon=P('Longitude', np.ma.arange(-180,180,90/64.0))
        start_dt = A('Start Datetime', datetime.datetime(2012,12,25,01,00))
        dur = A('HDF Duration', 256)
        
        don = Daylight()
        don.align_frequency = 1/64.0  # Force frequency to simplify test
        don.get_derived((lat, lon, start_dt, dur))
        expected = ['Day', 'Night', 'Night', 'Night']
        np.testing.assert_array_equal(don.array, expected)  # FIX required to test as no longer superframe samples


class TestEng_1_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_Fire
        self.operational_combinations = [('Eng (1) Fire On Ground', 'Eng (1) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_2_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_Fire
        self.operational_combinations = [('Eng (2) Fire On Ground', 'Eng (2) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_3_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_Fire
        self.operational_combinations = [('Eng (3) Fire On Ground', 'Eng (3) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_4_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_Fire
        self.operational_combinations = [('Eng (4) Fire On Ground', 'Eng (4) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_Fire
        self.operational_combinations = [
            ('Eng (1) Fire',), ('Eng (2) Fire',), ('Eng (3) Fire',), ('Eng (4) Fire',),
            ('Eng (1) Fire', 'Eng (2) Fire'), ('Eng (1) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (4) Fire'), ('Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (2) Fire', 'Eng (4) Fire'), ('Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngThrustModeRequired(unittest.TestCase):
    def test_can_operate(self):
        opts = EngThrustModeRequired.get_operational_combinations()
        self.assertTrue(('Eng (1) Thrust Mode Required',) in opts)
        self.assertTrue(('Eng (2) Thrust Mode Required',) in opts)
        self.assertTrue(('Eng (3) Thrust Mode Required',) in opts)
        self.assertTrue(('Eng (4) Thrust Mode Required',) in opts)
        self.assertTrue(('Eng (1) Thrust Mode Required',
                         'Eng (2) Thrust Mode Required',
                         'Eng (3) Thrust Mode Required',
                         'Eng (4) Thrust Mode Required',) in opts)
    
    def test_derive_one_param(self):
        thrust_array = np.ma.array([0, 0, 1, 0])
        thrust = M('Eng (2) Thrust Mode Required', array=thrust_array,
                   values_mapping=EngThrustModeRequired.values_mapping)
        node = EngThrustModeRequired()
        node.derive(None, thrust, None, None)
        self.assertEqual(thrust.array.raw.tolist(), thrust_array.tolist())
    
    def test_derive_four_params(self):
        thrust_array1 = np.ma.array([0, 0, 1, 0],
                                    mask=[False, False, True, False])
        thrust_array2 = np.ma.array([1, 0, 0, 0],
                                    mask=[True, False, False, False])
        thrust_array3 = np.ma.array([0, 1, 0, 0])
        thrust_array4 = np.ma.array([0, 0, 1, 0])
        thrust1 = M('Eng (1) Thrust Mode Required', array=thrust_array1,
                    values_mapping=EngThrustModeRequired.values_mapping)
        thrust2 = M('Eng (2) Thrust Mode Required', array=thrust_array2,
                    values_mapping=EngThrustModeRequired.values_mapping)
        thrust3 = M('Eng (3) Thrust Mode Required', array=thrust_array3,
                    values_mapping=EngThrustModeRequired.values_mapping)
        thrust4 = M('Eng (4) Thrust Mode Required', array=thrust_array4,
                    values_mapping=EngThrustModeRequired.values_mapping)
        node = EngThrustModeRequired()
        node.derive(thrust1, thrust2, thrust3, thrust4)
        
        self.assertEqual(
            node.array.tolist(),
            MappedArray([1, 1, 1, 0],
                        mask=[True, False, True, False],
                        values_mapping=EngThrustModeRequired.values_mapping).tolist())


class TestFlapExcludingTransition(unittest.TestCase):
        
    def test_can_operate(self):
        self.assertTrue(FlapExcludingTransition.can_operate(
            ('Flap Angle', 'Series', 'Family',)))


class TestFlapIncludingTransition(unittest.TestCase):
        
    def test_can_operate(self):
        self.assertTrue(FlapIncludingTransition.can_operate(
            ('Flap Angle', 'Series', 'Family',)))


class TestFlap(unittest.TestCase):
        
    def test_can_operate(self):
        self.assertTrue(Flap.can_operate(('Altitude AAL',),
                                         frame=Attribute('Frame', 'L382-Hercules')))
        self.assertTrue(Flap.can_operate(('Flap Angle', 'Series', 'Family')))

    def test_flap_stepped_nearest_5(self):
        flap = P('Flap Angle', np.ma.arange(50))
        node = Flap()
        node.derive(flap, A('Series', None), A('Family', None))
        expected = [0] * 3 + [5] * 2 + [10] * 5 + [15] * 5
        self.assertEqual(node.array.raw[:15].tolist(), expected)
        expected = [45] * 2 + [50] * 5
        self.assertEqual(node.array.raw[-7:].tolist(), expected)
        self.assertEqual(
            node.values_mapping,
            {0: '0', 35: '35', 5: '5', 40: '40', 10: '10', 45: '45', 15: '15',
             50: '50', 20: '20', 25: '25', 30: '30'})

        flap = P('Flap Angle', np.ma.array(range(20), mask=[True] * 10 + [False] * 10))
        node.derive(flap, A('Series', None), A('Family', None))
        expected = [-1] * 10 + [15] * 5 + [20] * 5
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

        node = Flap()
        node.derive(flap, A('Series', None), A('Family', 'DC-9'))

        #expected = reduce(operator.add, [
            #[0, -999] + [0] * 11,  #  0.0 -> 12.5 (one masked)
            #[13] * 7,              # 13.0 -> 19.5
            #[20] * 5,              # 20.0 -> 24.5
            #[25] * 5,              # 25.0 -> 29.5
            #[30] * 10,             # 30.0 -> 39.5
            #[40] * 10,             # 40.0 -> 49.5
            #[0] * 5,               # -5.0 -> -1.0
            #[13, 0],               # odd float values
            #[-999] * 2,            # masked values
        #])
        expected = np.ma.array(
            ([0] * 7) + ([13] * 6) + ([20] * 7) + ([25] * 5) + ([30] * 5) + 
            ([40] * 10) + ([0] * 15) + [13] + ([0] * 3),
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
        self.assertLess(time, 2, msg='Took too long: %.3fs' % time)
        
    def test_decimal_flap_settings(self):
        # Beechcraft has a flap 17.5
        flap_param = Parameter('Flap Angle', array=np.ma.array(
            [0, 5, 7.2, 
             17, 17.4, 17.9, 20, 
             30]))
        flap = Flap()
        flap.derive(flap_param, A('Series', '1900D'), A('Family', 'Beechcraft'))
        self.assertEqual(flap.values_mapping,
                         {0: '0', 17.5: '17.5', 35: '35'})
        ma_test.assert_array_equal(
            flap.array, ['0', '0', '0', '17.5', '17.5', '35', '35', '35'])
        
    def test_flap_settings_for_hercules(self):
        # No flap recorded; ensure it converts exactly the same
        flap_param = Parameter('Altitude AAL', array=np.ma.array(
            [0, 0, 0, 50, 50, 50, 100]))
        flap = Flap()
        flap.derive(flap_param, A('Series', ''), A('Family', 'C-130'))
        self.assertEqual(flap.values_mapping,
                         {0: '0', 50: '50', 100: '100'})
        ma_test.assert_array_equal(
            flap.array, ['0', '0', '0', '50', '50', '100', '100'])


class TestFlapLever(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = FlapLever
        self.operational_combinations = [
            ('Flap Lever Angle', 'Series', 'Family'),
        ]
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestFuelQtyLow(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQty_Low.get_operational_combinations()
        self.assertIn(('Fuel Qty Low',), opts)
        self.assertIn(('Fuel Qty (1) Low',), opts)
        self.assertIn(('Fuel Qty (2) Low',), opts)
        self.assertIn(('Fuel Qty (1) Low', 'Fuel Qty (2) Low'), opts)

    def test_derive_fuel_qty_low_warning(self):
        low = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(low, None, None)
        self.assertEqual(warn.array.sum(), 2)
        
    def test_derive_fuel_qty_low_warning_two_params(self):
        one = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        two = M(array=np.ma.array([0,0,1,1,0,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(None, one, two)
        self.assertEqual(warn.array.sum(), 3)


class TestGearDown(unittest.TestCase, NodeTest):
    
    def setUp(self):
        self.node_class = GearDown
        self.operational_combinations = [
            ('Gear (L) Down',),
            ('Gear (R) Down',),
            ('Gear (L) Down', 'Gear (R) Down'),
            ('Gear (L) Down', 'Gear (N) Down', 'Gear (R) Down'),
            ('Gear Down Selected',),
        ]
        
    def test_derive_from_select_down(self):
        sel_down = M(array=np.ma.array([1,0,0,1,1]), values_mapping={
            0: 'Up',
            1: 'Down',
        })
        down = GearDown()
        down.derive(None, None, None, sel_down)
        self.assertEqual(list(down.array),
                         ['Down', 'Up', 'Up', 'Down', 'Down'])


class TestGearDownSelected(unittest.TestCase):
    def test_can_operate(self):
        opts = GearDownSelected.get_operational_combinations()
        self.assertEqual(opts, [('Gear Up Selected',)])

    def test_gear_down_selected_from_recorded_up(self):
        gup_sel = M(array=np.ma.array(data=[1,1,0,0,1,1]),
                    values_mapping={0:'Down',1:'Up'},
                    name='Gear Up Selected', 
                    frequency=1, 
                    offset=0.1)
        dn_sel = GearDownSelected()
        dn_sel.derive(gup_sel)
        np.testing.assert_array_equal(dn_sel.array, [0,0,1,1,0,0])
        

class TestGearOnGround(unittest.TestCase):
    def test_can_operate(self):
        opts = GearOnGround.get_operational_combinations()
        self.assertEqual(opts, [
            ('Gear (L) On Ground',),
            ('Gear (R) On Ground',),
            ('Gear (L) On Ground', 'Gear (R) On Ground'),
            ])
        
    def test_gear_on_ground_basic(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.6)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1,1,1,1,1])
        self.assertEqual(wow.frequency, 2.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_common_word(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,1,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_left_only(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground', 
                   frequency=1, 
                   offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, None)
        np.testing.assert_array_equal(wow.array, [0,0,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_right_only(self):
        p_right = M(array=np.ma.array(data=[0,0,0,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground', 
                    frequency=1, 
                    offset=0.7)
        wow=GearOnGround()
        wow.derive(None, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.7)


class TestGearUpSelected(unittest.TestCase):
    def test_can_operate(self):
        opts = GearUpSelected.get_operational_combinations()
        self.assertEqual(opts[0], ('Gear Down',))
        self.assertEqual(opts[1], ('Gear Down', 'Gear (L) Red Warning'))
        self.assertEqual(opts[-1], ('Gear Down', 'Gear (L) Red Warning', 
                                    'Gear (N) Red Warning',
                                    'Gear (R) Red Warning'))
        
    def test_gear_up_selected_basic(self):
        gdn = M(array=np.ma.array(data=[1,1,1,0,0,0]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        up_sel = GearUpSelected()
        up_sel.derive(gdn, None, None, None)
        np.testing.assert_array_equal(up_sel.array, [0,0,0,1,1,1])

    def test_gear_up_selected_with_warnings(self):
        gdn = M(array=np.ma.array(data=[1,1,1,0,0,0,0,0,0,0,0,1,1,1]),
                   values_mapping={1:'Down',0:'Up'},
                   name='Gear Down', 
                   frequency=1, 
                   offset=0.1)
        redl = M(array=np.ma.array(data=[0,0,0,1,1,1,0,0,0,0,1,1,0,0]),
                values_mapping={0:'-',1:'Warning'},
                name='Gear (L) Red Warning', 
                frequency=1, 
                offset=0.6)
        redr = redl
        redn = M(array=np.ma.array(data=[0,0,0,1,1,1,0,0,0,1,1,0,0,0]),
                values_mapping={0:'-',1:'Warning'},
                name='Gear (N) Red Warning', 
                frequency=1, 
                offset=0.6)
        up_sel = GearUpSelected()
        up_sel.derive(gdn, redl, redr, redn)
        np.testing.assert_array_equal(up_sel.array,
                                        [0,0,0,1,1,1,1,1,1,0,0,0,0,0])


class TestKeyVHFCapt(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(KeyVHFCapt.get_operational_combinations(),
                         [('Key VHF (1) (Capt)',),
                          ('Key VHF (2) (Capt)',),
                          ('Key VHF (3) (Capt)',),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (2) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)', 'Key VHF (3) (Capt)')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestKeyVHFFO(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(KeyVHFFO.get_operational_combinations(),
                         [('Key VHF (1) (FO)',),
                          ('Key VHF (2) (FO)',),
                          ('Key VHF (3) (FO)',),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (2) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)', 'Key VHF (3) (FO)')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestMasterWarning(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarning
        self.operational_combinations = [
            ('Master Warning (Capt)',),
            ('Master Warning (FO)',),
            ('Master Warning (Capt)', 'Master Warning (FO)'),
        ]

    def test_derive(self):
        warn_capt = M(
            name='Master Warning (Capt)',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Warning (FO)',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestPackValvesOpen(unittest.TestCase):
    def test_can_operate(self):
        opts = PackValvesOpen.get_operational_combinations()
        self.assertEqual(opts, [
            ('ECS Pack (1) On', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (2) On', 'ECS Pack (2) High Flow'), 
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On', 'ECS Pack (2) High Flow')])
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitchAlternateLaw(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAlternateLaw
        self.operational_combinations = [
            ('Pitch Alternate (1) Law',),
            ('Pitch Alternate (2) Law',),
            ('Pitch Alternate (1) Law', 'Pitch Alternate (2) Law'),
        ]

    def test_derive(self):
        alt_law1 = M(
            name='Pitch Alternate (1) Law',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Alternate'},
            frequency=1,
            offset=0.1,
        )
        alt_law2 = M(
            name='Pitch Alternate (2) Law',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Alternate'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(alt_law1, alt_law2)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestSpeedbrakeSelected(unittest.TestCase):

    def test_can_operate(self):
        opts = SpeedbrakeSelected.get_operational_combinations()
        self.assertTrue(('Speedbrake Deployed',) in opts)
        self.assertTrue(('Speedbrake', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Speedbrake', 'Family') in opts)
        
    def test_derive(self):
        # test with deployed
        spd_sel = SpeedbrakeSelected()
        spd_sel.derive(
            deployed=M(array=np.ma.array(
                [0, 0, 0, 1, 1, 0]), values_mapping={1:'Deployed'}),
            armed=M(array=np.ma.array(
                [0, 0, 1, 1, 0, 0]), values_mapping={1:'Armed'})
        )
        self.assertEqual(list(spd_sel.array),
            ['Stowed', 'Stowed', 'Armed/Cmd Dn', 'Deployed/Cmd Up', 'Deployed/Cmd Up', 'Stowed'])
        
    def test_b737_speedbrake(self):
        self.maxDiff = None
        spd_sel = SpeedbrakeSelected()
        spdbrk = P(array=np.ma.array([0]*10 + [1.3]*20 + [0.2]*10))
        handle = P(array=np.ma.arange(40))
        # Follow the spdbrk only
        res = spd_sel.b737_speedbrake(spdbrk, None)
        self.assertEqual(list(res),
                        ['Stowed']*10 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)
        # Follow the handle only
        res = spd_sel.b737_speedbrake(None, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*32 + ['Deployed/Cmd Up']*5)
        # Follow the combination
        res = spd_sel.b737_speedbrake(spdbrk, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*7 + ['Deployed/Cmd Up']*20 + ['Armed/Cmd Dn']*5 + ['Deployed/Cmd Up']*5)


class TestStableApproach(unittest.TestCase):
    def test_can_operate(self):
        opts = StableApproach.get_operational_combinations()
        combinations = [
            # all
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp and Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Localizer', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and ILS Localizer and Vapp
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'Eng (*) N1 Min For 5 Sec', 'Altitude AAL'),
            # using EPR and exc. Airspeed Relative
            ('Approach And Landing', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) EPR Min For 5 Sec', 'Altitude AAL', 'Vapp'),
        ]
        for combo in combinations:
            self.assertIn(combo, opts)

    def test_stable_approach(self):
        stable = StableApproach()
        
        # Arrays will be 20 seconds long, index 4, 13,14,15 are stable
        #0. first and last values are not in approach slice
        apps = S()
        apps.create_section(slice(1,20))
        #1. gear up for index 0-2
        g = [ 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        gear = M(array=np.ma.array(g), values_mapping={1:'Down'})
        #2. landing flap invalid index 0, 5
        f = [ 5, 15, 15, 15, 15,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        flap = P(array=np.ma.array(f))
        #3. Heading stays within limits except for index 11-12, although we weren't on the heading sample 15 (masked out)
        h = [20, 20,  2,  3,  4,  8,  0,  0,  0,  0,  2, 20, 20,  8,  2,  0,  1,  1,  1,  1,  1]
        hm= [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        head = P(array=np.ma.array(h, mask=hm))
        #4. airspeed relative within limits for periods except 0-3
        a = [50, 50, 50, 45,  9,  8,  3, 7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        aspd = P(array=np.ma.array(a))
        #5. glideslope deviation is out for index 8, index 10-11 ignored as under 200ft, last 4 values ignored due to alt cutoff
        g = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,0.9,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm= [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide = P(array=np.ma.array(g, mask=gm))
        #6. localizer deviation is out for index 7, index 10 ignored as just under 200ft, last 4 values ignored due to alt cutoff
        l = [ 0,  0,  0,  0,  0,  0,  0,  2,  0.8, 0.1, -3,  0,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        loc = P(array=np.ma.array(l))
        #7. Vertical Speed too great at index 8, but change is smoothed out and at 17 (59ft)
        v = [-500] * 20
        v[6] = -2000
        v[18:19] = [-2000]*1
        vert_spd = P(array=np.ma.array(v))
        
        #TODO: engine cycling at index 12?
        
        #8. Engine power too low at index 5-12
        e = [80, 80, 80, 80, 80, 30, 20, 30, 20, 30, 20, 30, 44, 40, 80, 80, 80, 50, 50, 50, 50]
        eng = P(array=np.ma.array(e))
        
        # Altitude for cutoff heights, 9th element is 200 below, last 4 values are below 100ft last 2 below 50ft
        al = range(2000,219,-200) + range(219,18, -20) + [0]
        # == [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 219, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19]
        alt = P(array=np.ma.array(al))
        # DERIVE without using Vapp (using Vref limits)
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide, loc, eng, None, alt, None)
        self.assertEqual(len(stable.array), len(alt.array))
        self.assertEqual(len(stable.array), len(head.array))
        
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 8, 8, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])
        self.assertEqual(list(stable.array.mask),
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
        #========== NO GLIDESLOPE ==========
        # Test without the use of Glideslope (not on it at 1000ft) therefore
        # instability for index 7-10 is now due to low Engine Power
        glide2 = P(array=np.ma.array([3.5]*20))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 8, 8, 8, 8, 3, 3, 8, 9, 9, 9, 9, 9, 9, 0])
        
        #========== VERTICAL SPEED ==========
        # Test with a lot of vertical speed (rather than just gusts above)
        v2 = [-1800] * 20
        vert_spd2 = P(array=np.ma.array(v2))
        stable.derive(apps, gear, flap, head, aspd, vert_spd2, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 7, 7, 7, 7, 7, 3, 3, 7, 7, 7, 9, 9, 9, 9, 0])

        #========== UNSTABLE GLIDESLOPE JUST ABOVE 200ft ==========
        # Test that with unstable glideslope just before 200ft, this stability 
        # reason is continued to touchdown. Higher level checks (Heading at 3) 
        # still take priority at indexes 11-12
        #                                        219ft == 1.5 dots
        g3 = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,1.5,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide3 = P(array=np.ma.array(g3, mask=gm))
        stable.derive(apps, gear, flap, head, aspd, vert_spd, glide3, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 0])
        
        
    def test_with_real_data(self):
        
        apps = S(items=[Section(name='Approach And Landing', 
                                slice=slice(2702, 2993, None), 
                                start_edge=2702.0, stop_edge=2993.0)])
        
        #def save_test_node(param):
            #param.save('../tests/test_data/Stable Approach - '+param.name+'.nod')

        def test_node(name):
            return load(os.path.join(test_data_path, 'Stable Approach - '+name+'.nod'))
        stable = StableApproach()
        
        gear = test_node('Gear Down')
        flap = test_node('Flap')
        tdev = test_node('Track Deviation From Runway')
        vspd = test_node('Vertical Speed')
        gdev = test_node('ILS Glideslope')
        ldev = test_node('ILS Localizer')
        eng = test_node('Eng (star) N1 Min For 5 Sec')
        alt = test_node('Altitude AAL')

        stable.derive(
            apps=apps,
            gear=gear,
            flap=flap,
            tdev=tdev,
            aspd=None,
            vspd=vspd,
            gdev=gdev,
            ldev=ldev,
            eng_n1=eng,
            eng_epr=None,
            alt=alt,
            vapp=None)
        
        self.assertEqual(len(stable.array), len(alt.array))
        analysed = np.ma.clump_unmasked(stable.array)
        self.assertEqual(len(analysed), 1)
        # valid for the approach slice
        self.assertEqual(analysed[0].start, apps[0].slice.start)
        # stop is 10 secs after touchdown
        self.assertEqual(analysed[0].stop, 2946)
        
        sect = stable.array[analysed[0]]
        # assert that last few values are correct (masked in gear and flap params should not influence end result)
        self.assertEqual(list(sect[-4:]), ['Stable']*4)
        self.assertEqual(list(sect[0:73]), ['Gear Not Down']*73)
        self.assertEqual(list(sect[74:117]), ['Not Landing Flap']*43)
        self.assertTrue(np.all(sect[117:] == ['Stable']))


class TestStickShaker(unittest.TestCase):

    def test_can_operate(self):
        opts = StickShaker.get_operational_combinations()
        self.assertEqual(opts, [
            ('Stick Shaker (L)',),
            ('Shaker Activation',), 
            ('Stick Shaker (L)', 'Stick Shaker (R)'),
            ('Stick Shaker (L)', 'Shaker Activation'),
            ('Stick Shaker (R)', 'Shaker Activation'), 
            ('Stick Shaker (L)', 'Stick Shaker (R)', 'Shaker Activation')])
        
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustReversers(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def setUp(self):
        eng_1_unlocked_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_1_deployed_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_2_unlocked_array = [ 1,  1,  1,  1,  0,  1,  0,  0,  0,  0]
        eng_2_deployed_array = [ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0]

        self.eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        self.eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})
        self.thrust_reversers = ThrustReversers()

    def test_derive(self):
        result = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_masked_value(self):
        self.eng_1_unlocked.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.eng_1_deployed.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.eng_2_unlocked.array.mask = [ 0,  0,  0,  1,  0,  0,  0,  1,  1,  0]
        self.eng_2_deployed.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result_array)
        np.testing.assert_equal(self.thrust_reversers.array.mask, result_mask)

    def test_derive_in_transit_avaliable(self):
        result = [ 2,  2,  1,  1,  1,  1,  0,  0,  0,  0]
        transit_array = [ 0,  0,  1,  1,  1,  1,  0,  0,  0,  0]
        eng_1_in_transit = M(name='Eng (1) Thrust Reverser In Transit', array=np.ma.array(transit_array), values_mapping={1:'In Transit'})
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                eng_1_in_transit,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_unlock_at_edges(self):
        '''
        test for aircraft which only record Thrust Reverser Unlocked during
        transition, not whilst deployed
        '''
        result =               [ 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]

        eng_1_unlocked_array = [ 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
        eng_1_deployed_array = [ 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        eng_2_unlocked_array = [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        eng_2_deployed_array = [ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

        eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})

        self.thrust_reversers.get_derived([eng_1_deployed,
                                None,
                                None,
                                eng_1_unlocked,
                                None,
                                None,
                                None,
                                eng_2_deployed,
                                None,
                                None,
                                eng_2_unlocked] + [None] * 17)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)


class TestTakeoffConfigurationWarning(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Parking Brake Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Flap Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Gear Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Rudder Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Spoiler Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',
             'Takeoff Configuration Parking Brake Warning',
             'Takeoff Configuration Flap Warning',
             'Takeoff Configuration Gear Warning',
             'Takeoff Configuration Rudder Warning',
             'Takeoff Configuration Spoiler Warning',)))
    
    @unittest.skip('Test Not Implemented')
    def test_derive_basic(self):
        pass


class TestTAWSAlert(unittest.TestCase):
    def test_can_operate(self):
        parameters = ['TAWS Caution Terrain',
                       'TAWS Caution',
                       'TAWS Dont Sink',
                       'TAWS Glideslope'
                       'TAWS Predictive Windshear',
                       'TAWS Pull Up',
                       'TAWS Sink Rate',
                       'TAWS Terrain',
                       'TAWS Terrain Warning Amber',
                       'TAWS Terrain Pull Up',
                       'TAWS Terrain Warning Red',
                       'TAWS Too Low Flap',
                       'TAWS Too Low Gear',
                       'TAWS Too Low Terrain',
                       'TAWS Windshear Warning',
                       ]
        for p in parameters:
            self.assertTrue(TAWSAlert.can_operate(p))

    def setUp(self):
        terrain_array = [ 1,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        pull_up_array = [ 0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0]

        self.airs = S(name='Airborne')
        self.airs.create_section(slice(5,15))
        self.terrain = M(name='TAWS Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        self.pull_up = M(name='TAWS Pull Up', array=np.ma.array(pull_up_array), values_mapping={1:'Warning'})
        self.taws_alert = TAWSAlert()

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_masked_values(self):
        result = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]
        self.terrain.array[8] = np.ma.masked
        self.terrain.array[10] = np.ma.masked

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_zeros(self):
        result = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0]
        
        terrain_array = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        
        caution = M(name='TAWS Caution Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        caution.array.mask = True

        self.taws_alert.get_derived((self.airs,
                                caution,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)


class TestTAWSDontSink(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(TAWSDontSink.get_operational_combinations(),
                         [('TAWS (L) Dont Sink',),
                          ('TAWS (R) Dont Sink',),
                          ('TAWS (L) Dont Sink', 'TAWS (R) Dont Sink')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSGlideslopeCancel(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(TAWSGlideslopeCancel.get_operational_combinations(),
                         [('TAWS (L) Glideslope Cancel',),
                          ('TAWS (R) Glideslope Cancel',),
                          ('TAWS (L) Glideslope Cancel', 'TAWS (R) Glideslope Cancel')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSTooLowGear(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(TAWSTooLowGear.get_operational_combinations(),
                         [('TAWS (L) Too Low Gear',),
                          ('TAWS (R) Too Low Gear',),
                          ('TAWS (L) Too Low Gear', 'TAWS (R) Too Low Gear')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTCASFailure(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(TCASFailure.get_operational_combinations(),
                         [('TCAS (L) Failure',),
                          ('TCAS (R) Failure',),
                          ('TCAS (L) Failure', 'TCAS (R) Failure')])
    
    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass

