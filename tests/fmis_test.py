import numpy as np
import unittest


from analysis_engine.fmis import (
    FMISAALFirstSlatFlapSelApp,
    FMISApuTime,
    FMISCruise,
    FMISHWindCompCruise,
    FMISLevelFlight,
    FMISMachCruise,
    FMISMinutesCruise,
    FMISPaltCruise,
    FMISTimeFirst2LastEngineStart,
    FMISTimeFirst2LastEngineStop,
    FMISTimeFirst2LastEngShutDown,
    FMISTimeFL250ToTDown,
    FMISTimeFL1000ToTDown,
    FMISTimeFlap,
    FMISTimeFlap_Airbus,
    FMISTimeLvlFltBelowFL250Climb,
    FMISTimeLvlFltFL250toFL70Desc,
    FMISGearDownAALapp,
    FMISTimeLastEngStart2Takeoff,
    FMISTimeSpdBrakeExt,
    FMISTimeTouchDown2FirstEngShutDown,
    FMISWeightCruise,
)
from analysis_engine.key_time_instances import (
    AltitudeWhenClimbing,
    AltitudeWhenDescending,
    FlapSet,
    SlatSet,
)
from analysis_engine.node import (KeyPointValue, KeyPointValueNode,
                                  KeyTimeInstance, KeyTimeInstanceNode, M, P,
                                  Section, SectionNode)


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


class TestFMISTimeFlap_Airbus(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFlap_Airbus.get_operational_combinations(),
                         [('Configuration', 'Approach And Landing')])
    
    def test_derive(self):
        values_mapping = {x: x for x in [0, 1, 2, 5, 10, 15, 20, 25, 30, 40]}
        conf = P('Configuration', values_mapping=values_mapping, array=np.ma.array(
            (['1'] * 10) + (['1+F'] * 5) + (['2'] * 15) + (['3'] * 5) + (['4'] * 20)))
        apps = buildsection('Approach And Landing', 5, 45)
        node = FMISTimeFlap_Airbus()
        node.derive(conf, apps)
        self.assertEqual(node, [
            KeyPointValue(index=25, value=5.0, name='FMIS TimeFlap1Airbus'),
            KeyPointValue(index=25, value=5.0, name='FMIS TimeFlap1FAirbus'),
            KeyPointValue(index=25, value=15.0, name='FMIS TimeFlap2Airbus'),
            KeyPointValue(index=25, value=5.0, name='FMIS TimeFlap3Airbus'),
            KeyPointValue(index=25, value=10.0, name='FMIS TimeFlap4Airbus')])


class TestFMISTimeFL250ToTDown(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFL250ToTDown.get_operational_combinations(),
                         [('Altitude When Descending', 'Touchdown')])
    
    def test_derive(self):
        touchdowns = KeyTimeInstanceNode('Touchdowns',
                                         items=[KeyTimeInstance(25, 'Touchdown'),
                                                KeyTimeInstance(50, 'Touchdown')])
        alt_descs = AltitudeWhenDescending(
            'Altitude When Descending',
            items=[KeyTimeInstance(10, '2500 Ft Descending'),
                   KeyTimeInstance(15, '1000 Ft Descending'),
                   KeyTimeInstance(40, '2500 Ft Descending'),
                   KeyTimeInstance(45, '1000 Ft Descending')])
        node = FMISTimeFL250ToTDown()
        node.derive(alt_descs, touchdowns)
        self.assertEqual(node, [KeyPointValue(30, 40, 'FMIS TimeFL250ToTDown')])


class TestFMISTimeFL1000ToTDown(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFL1000ToTDown.get_operational_combinations(),
                         [('Altitude When Descending', 'Touchdown')])
    
    def test_derive(self):
        touchdowns = KeyTimeInstanceNode('Touchdowns',
                                         items=[KeyTimeInstance(25, 'Touchdown'),
                                                KeyTimeInstance(50, 'Touchdown')])
        alt_descs = AltitudeWhenDescending(
            'Altitude When Descending',
            items=[KeyTimeInstance(10, '10000 Ft Descending'),
                   KeyTimeInstance(15, '5000 Ft Descending'),
                   KeyTimeInstance(40, '10000 Ft Descending'),
                   KeyTimeInstance(45, '5000 Ft Descending')])
        node = FMISTimeFL1000ToTDown()
        node.derive(alt_descs, touchdowns)
        self.assertEqual(node, [KeyPointValue(30, 40, 'FMIS TimeFL1000ToTDown')])


class TestFMISAPUTime(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISApuTime.get_operational_combinations(),
                         [('Eng Stop',)])
    
    def test_derive(self):
        eng_stops = KeyTimeInstanceNode('Eng Stop',
                                        items=[KeyTimeInstance(25, 'Eng Stop'),
                                               KeyTimeInstance(50, 'Eng Stop')])
        node = FMISApuTime()
        node.derive(eng_stops)
        self.assertEqual(node, [KeyPointValue(50, 50, 'FMIS ApuTime')])


class TestFMISTimeSpdBrakeExt(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeSpdBrakeExt.get_operational_combinations(),
                         [('Speedbrake', 'Altitude When Descending', 'Touchdown')])
    
    def test_derive(self):
        speedbrake = P('Speedbrake', array=np.ma.array([0] * 25 + [10] * 25))
        touchdowns = KeyTimeInstanceNode('Touchdowns',
                                         items=[KeyTimeInstance(25, 'Touchdown'),
                                                KeyTimeInstance(50, 'Touchdown')])
        alt_descs = AltitudeWhenDescending(
            'Altitude When Descending',
            items=[KeyTimeInstance(10, '2500 Ft Descending'),
                   KeyTimeInstance(15, '1000 Ft Descending'),
                   KeyTimeInstance(40, '2500 Ft Descending'),
                   KeyTimeInstance(45, '1000 Ft Descending')])
        node = FMISTimeSpdBrakeExt()
        node.derive(speedbrake, alt_descs, touchdowns)
        self.assertEqual(node, [KeyPointValue(30, 25, 'FMIS TimeSpdBrakeExt')])


class TestFMISGearDownAALapp(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISGearDownAALapp.get_operational_combinations(),
                         [('Altitude At Gear Down Selection',)])
    
    def test_derive(self):
        alt_gear_downs = KeyPointValueNode(
            'Altitude At Gear Down Selection',
            items=[KeyPointValue(10, 500, 'Altitude At Gear Down'),
                   KeyPointValue(15, 250, 'Altitude At Gear Down')])
        node = FMISGearDownAALapp()
        node.derive(alt_gear_downs)
        self.assertEqual(node, [KeyPointValue(15, 1000, 'FMIS GearDownAALapp')])


class TestFMISGearDownAALapp(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISGearDownAALapp.get_operational_combinations(),
                         [('Altitude At Gear Down Selection',)])
    
    def test_derive(self):
        alt_gear_downs = KeyPointValueNode(
            'Altitude At Gear Down Selection',
            items=[KeyPointValue(10, 500, 'Altitude At Gear Down'),
                   KeyPointValue(15, 250, 'Altitude At Gear Down')])
        node = FMISGearDownAALapp()
        node.derive(alt_gear_downs)
        self.assertEqual(node, [KeyPointValue(15, 1000, 'FMIS GearDownAALapp')])


class TestFMISAALFirstSlatFlapSelApp(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISAALFirstSlatFlapSelApp.get_operational_combinations(),
                         [('Altitude AAL', 'Descending', 'Flap Set'),
                          ('Altitude AAL', 'Descending', 'Slat Set'),
                          ('Altitude AAL', 'Descending', 'Flap Set', 'Slat Set')])
    
    def test_derive_basic(self):
        alt_aal = P('Altitude AAL', np.ma.arange(0, 1000, 10))
        flap_sets = FlapSet('Flap Set', items=[KeyTimeInstance(10, 'Flap 10'),
                                              KeyTimeInstance(20, 'Flap 20')])
        slat_sets = SlatSet('Slat Set', items=[KeyTimeInstance(15, 'Slat 15'),
                                              KeyTimeInstance(30, 'Slat 30')])
        desc = buildsection('Descending', 12, 50)
        alt_gear_downs = KeyPointValueNode(
            'Altitude At Gear Down Selection',
            items=[KeyPointValue(10, 500, 'Altitude At Gear Down'),
                   KeyPointValue(15, 250, 'Altitude At Gear Down')])
        node = FMISAALFirstSlatFlapSelApp()
        node.derive(alt_aal, desc, flap_sets, slat_sets)
        self.assertEqual(node, [KeyPointValue(15, 150, 'FMIS AALFirstSlatFlapSelApp')])
        node = FMISAALFirstSlatFlapSelApp()
        node.derive(alt_aal, desc, flap_sets, None)
        self.assertEqual(node, [KeyPointValue(20, 200, 'FMIS AALFirstSlatFlapSelApp')])
        node = FMISAALFirstSlatFlapSelApp()
        node.derive(alt_aal, desc, None, slat_sets)
        self.assertEqual(node, [KeyPointValue(15, 150, 'FMIS AALFirstSlatFlapSelApp')])


class TestFMISTimeFirst2LastEngShutDown(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeFirst2LastEngShutDown.get_operational_combinations(),
                         [('Eng Stop',)])
    
    def test_derive(self):
        eng_stops = KeyTimeInstanceNode('Eng Stop', items=[])
        node = FMISTimeFirst2LastEngShutDown()
        node.derive(eng_stops)
        self.assertEqual(node, [])
        
        eng_stops = KeyTimeInstanceNode('Eng Stop', items=[KeyTimeInstance(5, 'Eng Stop')])
        node = FMISTimeFirst2LastEngShutDown()
        node.derive(eng_stops)
        self.assertEqual(node, [KeyPointValue(5, 0, 'FMIS TimeFirst2LastEngShutDown')])
        
        eng_stops = KeyTimeInstanceNode('Eng Stop', items=[KeyTimeInstance(5, 'Eng Stop'),
                                                           KeyTimeInstance(15, 'Eng Stop'),])
        node = FMISTimeFirst2LastEngShutDown()
        node.derive(eng_stops)
        self.assertEqual(node, [KeyPointValue(10, 10, 'FMIS TimeFirst2LastEngShutDown')])


class TestFMISLevelFlight(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISLevelFlight.get_operational_combinations(),
                         [('Vertical Speed Inertial', 'Airborne')])
    
    def test_derive(self):
        vert_spd_array = np.ma.array([500] * 10 + [100] * 10 + [500] * 10)
        airborne = SectionNode('Airborne', items=[Section('Airborne', slice(0, 14), 0, 14),
                                                  Section('Airborne', slice(16, 30), 16, 30)])
        vert_spd = P('Vertical Speed Inertial', array=vert_spd_array)
        node = FMISLevelFlight()
        node.derive(vert_spd, airborne)
        self.assertEqual(node, [Section('FMIS LevelFlight', slice(10, 14), 10, 14),
                                Section('FMIS LevelFlight', slice(16, 20), 16, 20)])


class TestFMISTimeLvlFltBelowFL250Climb(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeLvlFltBelowFL250Climb.get_operational_combinations(),
                         [('Altitude When Climbing', 'FMIS LevelFlight')])
    
    def test_derive(self):
        alt_climbs = AltitudeWhenClimbing(
            'Altitude When Climbing',
            items=[KeyTimeInstance(15, '1000 Ft Climbing'),
                   KeyTimeInstance(35, '2500 Ft Climbing'),
                   KeyTimeInstance(55, '2500 Ft Climbing')])
        level_flights = SectionNode('FMIS LevelFlight', items=[
            Section('FMIS LevelFlight', slice(10, 20), 10, 20),
            Section('FMIS LevelFlight', slice(30, 40), 30, 40),
            Section('FMIS LevelFlight', slice(50, 60), 50, 60)])
        
        node = FMISTimeLvlFltBelowFL250Climb()
        node.derive(alt_climbs, level_flights)
        self.assertEqual(
            node,
            [KeyPointValue(index=35, value=15.0, name='FMIS TimeLvlFltBelowFL250Climb')])


class TestFMISTimeLvlFltFL250toFL70Desc(unittest.TestCase):
    
    def test_can_operate(self):
        self.assertEqual(FMISTimeLvlFltFL250toFL70Desc.get_operational_combinations(),
                         [('Altitude AAL', 'Descending', 'FMIS LevelFlight')])
        
    def test_derive(self):
        
        alt_aal = P('Altitude AAL', )
        level_flights = SectionNode('FMIS LevelFlight')
        
        node = FMISTimeLvlFltFL250toFL70Desc()
        node.derive(alt_aal, descs, level_flights)
        self.assertEqual(node, [])

