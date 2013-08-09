import numpy as np
import unittest


from analysis_engine.fmis import (
    FMISCruise,
    FMISHWindCompCruise,
    FMISMachCruise,
    FMISMinutesCruise,
    FMISPaltCruise,
    FMISTimeFirst2LastEngineStart,
    FMISTimeFirst2LastEngineStop,
    FMISTimeFlap,
    FMISTimeLastEngStart2Takeoff,
    FMISTimeTouchDown2FirstEngShutDown,
    FMISWeightCruise,
)
from analysis_engine.node import (KeyPointValue, KeyTimeInstance,
                                  KeyTimeInstanceNode, M, P, Section,
                                  SectionNode)


def builditem(name, begin, end):
    '''
    This code more accurately represents the aligned section values, but is
    not suitable for test cases where the data does not get aligned.

    if begin is None:
        ib = None
    else:
        ib = int(begin)
        if ib < begin:
            ib += 1
    if end is None:
        ie = None
    else:
        ie = int(end)
        if ie < end:
            ie += 1
            '''
    return Section(name, slice(begin, end, None), begin, end)


def buildsection(name, begin, end):
    '''
    A little routine to make building Sections for testing easier.

    :param name: name for a test Section
    :param begin: index at start of section
    :param end: index at end of section

    :returns: a SectionNode populated correctly.

    Example: land = buildsection('Landing', 100, 120)
    '''
    result = builditem(name, begin, end)
    return SectionNode(name, items=[result])


class TestFMISCruise(unittest.TestCase):
    
    def test_can_operate(self):
        expected = [('Cruise',)]
        opts = FMISCruise.get_operational_combinations()
        self.assertEqual(opts, expected)
    
    def test_derive_under_5_minutes(self):
        node = FMISCruise()
        node.derive(buildsection('Cruise', 200, 210))
        self.assertEqual(node, [])
    
    def test_derive_under_10_minutes(self):
        node = FMISCruise()
        node.derive(buildsection('Cruise', 200, 600))
        self.assertEqual(list(node), [Section('FMIS Cruise 1', slice(385, 415), 385, 415)])
    
    def test_derive_under_15_minutes(self):
        node = FMISCruise()
        node.derive(buildsection('Cruise', 200, 1000))
        self.assertEqual(node, [Section('FMIS Cruise 1', slice(500, 530), 500, 530),
                                Section('FMIS Cruise 2', slice(670, 700), 670, 700)])
    
    def test_derive_over_15_minutes(self):
        node = FMISCruise()
        node.derive(buildsection('Cruise', 200, 1500))
        self.assertEqual(node, [Section('FMIS Cruise 1', slice(500, 530), 500, 530),
                                Section('FMIS Cruise 2', slice(835, 865), 835, 865),
                                Section('FMIS Cruise 3', slice(1170, 1200), 1170, 1200)])


class TestFMISHWindCompCruise(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISHWindCompCruise.get_operational_combinations(),
                         [('FMIS Cruise', 'Headwind')])


class TestFMISMachCruise(unittest.TestCase):
    '''
    Also tests FMISCruiseKPV superclass.
    '''
    
    def test_can_operate(self):
        self.assertEqual(FMISMachCruise.get_operational_combinations(),
                         [('FMIS Cruise', 'Mach')])
    
    def test_derive_empty_fmis_cruises(self):
        fmis_cruises = SectionNode('FMIS Cruises')
        mach = P('Mach', np.ma.arange(100))
        node = FMISMachCruise()
        node.derive(fmis_cruises, mach)
        self.assertEqual(node, [])
    
    def test_derive_under_10_minutes(self):
        fmis_cruises = buildsection('FMIS Cruise', 200, 230)
        mach = P('Mach', np.ma.arange(500))
        node = FMISMachCruise()
        node.derive(fmis_cruises, mach)
        self.assertEqual(list(node), [KeyPointValue(215, 214.5, 'FMIS MachCruise 1')])
        
    def test_derive_under_15_minutes(self):
        fmis_cruises = SectionNode('FMIS Cruise', items=[
            Section('FMIS Cruise 1', slice(200, 230), 200, 230),
            Section('FMIS Cruise 2', slice(570, 600), 570, 600)])
        mach = P('Mach', np.ma.arange(1000))
        node = FMISMachCruise()
        node.derive(fmis_cruises, mach)
        self.assertEqual(list(node), [
            KeyPointValue(215, 214.5, 'FMIS MachCruise 1'),
            KeyPointValue(585, 584.5, 'FMIS MachCruise 2'),
        ])
    
    def test_derive_over_15_minutes(self):
        fmis_cruises = SectionNode('FMIS Cruise', items=[
            Section('FMIS Cruise 1', slice(200, 230), 200, 230),
            Section('FMIS Cruise 2', slice(570, 600), 570, 600),
            Section('FMIS Cruise 3', slice(1070, 1100), 1070, 1100)])
        mach = P('Mach', np.ma.arange(2000))
        node = FMISMachCruise()
        node.derive(fmis_cruises, mach)
        self.assertEqual(list(node), [
            KeyPointValue(215, 214.5, 'FMIS MachCruise 1'),
            KeyPointValue(585, 584.5, 'FMIS MachCruise 2'),
            KeyPointValue(1085, 1084.5, 'FMIS MachCruise 3'),
        ])


class TestFMISMinutesCruise(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISMinutesCruise.get_operational_combinations(),
                         [('Cruise', 'FMIS Cruise')])
    
    def test_derive_empty_cruises(self):
        cruises = SectionNode('Cruise')
        fmis_cruises = SectionNode('FMIS Cruise')
        node = FMISMinutesCruise()
        node.derive(cruises, fmis_cruises)
        self.assertEqual(list(node), [])
        
    
    def test_derive_empty_fmis_cruises(self):
        cruises = buildsection('Cruise', 400, 500)
        fmis_cruises = SectionNode('FMIS Cruise')
        node = FMISMinutesCruise()
        node.derive(cruises, fmis_cruises)
        self.assertEqual(list(node), [])
    
    def test_derive_under_15_minutes(self):
        cruises = buildsection('Cruise', 200, 1100)
        fmis_cruises = SectionNode('FMIS Cruise 1', items=[
            Section('FMIS Cruise 1', slice(500, 530), 500, 530),
            Section('FMIS Cruise 2', slice(770, 800), 770, 800)])
        node = FMISMinutesCruise()
        node.derive(cruises, fmis_cruises)
        self.assertEqual(list(node), [
            KeyPointValue(515, 5, 'FMIS MinutesCruise 1'),
            KeyPointValue(785, 10, 'FMIS MinutesCruise 2'),
        ])
    
    def test_derive_over_15_minutes(self):
        cruises = buildsection('Cruise', 200, 1550)
        fmis_cruises = SectionNode('FMIS Cruise 1', items=[
            Section('FMIS Cruise 1', slice(500, 530), 500, 530),
            Section('FMIS Cruise 2', slice(810, 840), 810, 840),
            Section('FMIS Cruise 3', slice(1220, 1250), 1220, 1250)])
        node = FMISMinutesCruise()
        node.derive(cruises, fmis_cruises)
        self.assertEqual(list(node), [
            KeyPointValue(515, 5, 'FMIS MinutesCruise 1'),
            KeyPointValue(825, 11.25, 'FMIS MinutesCruise 2'),
            KeyPointValue(1235, 17.5, 'FMIS MinutesCruise 3'),
        ])
    
    def test_derive_under_10_minutes(self):
        cruises = buildsection('Cruise', 200, 650)
        fmis_cruises = buildsection('FMIS Cruise 1', 410, 440)
        node = FMISMinutesCruise()
        node.derive(cruises, fmis_cruises)
        self.assertEqual(list(node), [KeyPointValue(425, 3.75,
                                                    'FMIS MinutesCruise 1')])


class TestFMISPaltCruise(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISPaltCruise.get_operational_combinations(),
                         [('FMIS Cruise', 'Altitude STD')])


class TestFMISWeightCruise(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISWeightCruise.get_operational_combinations(),
                         [('FMIS Cruise', 'Gross Weight Smoothed')])


class TestFMISTimeFirst2LastEngineStart(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFirst2LastEngineStart.get_operational_combinations(),
                         [('Eng Start',)])
    
    def test_derive(self):
        eng_start = KeyTimeInstanceNode('Eng Start', items=[KeyTimeInstance(10),
                                                            KeyTimeInstance(50),
                                                            KeyTimeInstance(100)])
        node = FMISTimeFirst2LastEngineStart()
        node.derive(eng_start)
        self.assertEqual(node, [
            KeyPointValue(55, 90, 'FMIS TimeFirst2LastEngineStart')])


class TestFMISTimeLastEngStart2Takeoff(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeLastEngStart2Takeoff.get_operational_combinations(),
                         [('Eng Start', 'Acceleration Start')])
    
    def test_derive(self):
        eng_start = KeyTimeInstanceNode('Eng Start', items=[KeyTimeInstance(10)])
        eng_stop = KeyTimeInstanceNode('Acceleration Start', items=[KeyTimeInstance(25)])
        node = FMISTimeLastEngStart2Takeoff()
        node.derive(eng_start, eng_stop)
        self.assertEqual(node, [
            KeyPointValue(18, 15, 'FMIS TimeLastEngStart2Takeoff')])


class TestFMISTimeTouchDown2FirstEngShutDown(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeTouchDown2FirstEngShutDown.get_operational_combinations(),
                         [('Touchdown', 'Eng Stop')])
    
    def test_derive(self):
        eng_stop = KeyTimeInstanceNode('Eng Stop', items=[KeyTimeInstance(35)])
        touchdowns = KeyTimeInstanceNode('Touchdown', items=[KeyTimeInstance(25)])
        node = FMISTimeTouchDown2FirstEngShutDown()
        node.derive(touchdowns, eng_stop)
        self.assertEqual(node, [
            KeyPointValue(30, 10, 'FMIS TimeTouchDown2FirstEngShutDown')])


class TestFMISTimeFirst2LastEngineStart(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFirst2LastEngineStop.get_operational_combinations(),
                         [('Eng Stop',)])
    
    def test_derive(self):
        eng_start = KeyTimeInstanceNode('Eng Stop', items=[KeyTimeInstance(10),
                                                           KeyTimeInstance(50),
                                                           KeyTimeInstance(100)])
        node = FMISTimeFirst2LastEngineStop()
        node.derive(eng_start)
        self.assertEqual(node, [
            KeyPointValue(55, 90, 'FMIS TimeFirst2LastEngineStop')])


class TestFMISTimeFlap(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFlap.get_operational_combinations(),
                         [('Flap', 'Approach And Landing')])
    
    def test_derive(self):
        values_mapping = {x: x for x in [0, 1, 2, 5, 10, 15, 20, 25, 30, 40]}
        flap = P('Flap', values_mapping=values_mapping, array=np.ma.array(
            ([1] * 10) + ([2] * 5) + ([5] * 15) + ([10] * 5) + ([20] * 20) +
            ([25] * 5) + ([30] * 10) + ([40] * 40)))
        apps = buildsection('Approach And Landing', 5, 100)
        node = FMISTimeFlap()
        node.derive(flap, apps)
        self.assertEqual(node, [
            KeyPointValue(index=53, value=5.0, name='FMIS TimeFlap1'),
            KeyPointValue(index=53, value=5.0, name='FMIS TimeFlap2'),
            KeyPointValue(index=53, value=15.0, name='FMIS TimeFlap5'),
            KeyPointValue(index=53, value=5.0, name='FMIS TimeFlap10'),
            KeyPointValue(index=53, value=0.0, name='FMIS TimeFlap15'),
            KeyPointValue(index=53, value=5.0, name='FMIS TimeFlap25'),
            KeyPointValue(index=53, value=10.0, name='FMIS TimeFlap30'),
            KeyPointValue(index=53, value=30.0, name='FMIS TimeFlap40')])