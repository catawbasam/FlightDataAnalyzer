import numpy as np
import os
import unittest

from utilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import (Airborne,
                                          Approach,
                                          ApproachAndLanding,
                                          BouncedLanding,
                                          ClimbCruiseDescent,
                                          Climbing,
                                          Cruise,
                                          Descending,
                                          DescentLowClimb,
                                          DescentToFlare,
                                          Fast,
                                          FinalApproach,
                                          GearExtending,
                                          GearRetracting,
                                          GoAroundAndClimbout,
                                          GoAround5MinRating,
                                          Grounded,
                                          Holding,
                                          InAir,
                                          ILSGlideslopeEstablished,
                                          ILSLocalizerEstablished,
                                          Landing,
                                          LevelFlight,
                                          Mobile,
                                          Takeoff,
                                          Takeoff5MinRating,
                                          TakeoffRoll,
                                          TakeoffRotation,
                                          Taxiing,
                                          TaxiIn,
                                          TaxiOut,
                                          TurningInAir,
                                          TurningOnGround,
                                          TwoDegPitchTo35Ft,
                                          )
from analysis_engine.key_time_instances import TopOfClimb, TopOfDescent
from analysis_engine.library import integrate
from analysis_engine.node import (A, App, ApproachItem, 
                                  FlightPhaseNode,
                                  M, Parameter, P,
                                  Section, SectionNode, load)
from analysis_engine.process_flight import process_flight

from analysis_engine.settings import AIRSPEED_THRESHOLD


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

'''
Three little routines to make building Sections for testing easier.
'''
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
    raise NotImplementedError('use phase()!')
    return Section(begin, end, name=name)


def buildsection(name, begin, end):
    '''
    A little routine to make building Sections for testing easier.

    :param name: name for a test Section
    :param begin: index at start of section
    :param end: index at end of section

    :returns: a SectionNode populated correctly.

    Example: land = buildsection('Landing', 100, 120)
    '''
    raise NotImplementedError('use phase()!')
    result = builditem(name, begin, end)
    return SectionNode(items=[result], name=name)


def buildsections(name, *slice_periods):
    '''
    Like buildsection, this is used to build SectionNodes for test purposes.

    lands = buildsections('name',[from1,to1],[from2,to2])

    Example of use:
    approach = buildsections('Approach', [80,90], [100,110])
    '''
    built_list=[]
    for (start, stop) in slice_periods:
        new_section = builditem(name, start, stop)
        built_list.append(new_section)
    return SectionNode(items=built_list, name=name)



def phase(interval_notation):
    '''
    e.g. phase('[2..5],(6..8]')
    '''
    return FlightPhaseNode.from_string(interval_notation)

##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(),
            self.operational_combinations,
        )


##############################################################################


class TestInAir(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases', 'Fast')]
        opts = InAir.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_inair_phase_basic(self):
        vert_spd_data = np.ma.array([0] * 5 + range(0,400,20)+
                                    range(400,-400,-20)+
                                    range(-400,50,20))
        altitude = Parameter('Altitude AAL For Flight Phases', integrate(vert_spd_data, 1, 0, 1.0/60.0))
        fast = SectionNode(items=[Section(3, 80)])
        air = InAir()
        air.derive(altitude, fast)
        expected = [Section(7.5, 80.5, name='Airborne')]
        self.assertEqual(list(air), expected)

    def test_inair_phase_not_fast(self):
        altitude_data = np.ma.array(range(0,10))
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = []
        air = InAir()
        air.derive(alt_aal, fast)
        self.assertEqual(air, SectionNode([]))
        self.assertEqual(air, '<Empty>')

    def test_inair_phase_started_midflight(self):
        altitude_data = np.ma.array([100]*20+[60,30,10]+[0]*4)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = phase('(...25]')
        air = InAir()
        air.derive(alt_aal, fast)
        self.assertEqual(air, '(...22.5]')

    def test_inair_phase_ends_in_midflight(self):
        altitude_data = np.ma.array([0]*5+[30,80]+[100]*20)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = phase('(2...)')
        air = InAir()
        air.derive(alt_aal, fast)
        self.assertEqual(air, '[4.5...)')
        
    def test_inair_training_flight(self):
        inair = InAir()
        aal = load(os.path.join(test_data_path, 'alt_aal_training.nod'))
        airs = (
          '[723.5..1001.5], [1026.5..1321.5], [1346.5..1981.5], [2002.5..2419.5], '
          '[2439.5..2790.5], [2813.5..3211.5], [3235.5..3615.5], [3644.5..4031.5], '
          '[4060.5..4670.5], [4701.5..5050.5], [5078.5..5414.5], [5439.5..5767.5], '
          '[5791.5..6137.5], [6165.5..6537.5], [6561.5..7196.5], [7221.5..7569.5], '
          '[7593.5..7930.5], [7956.5..8288.5], [8313.5..8603.5], [8630.5..10113.5]')
        
        fast = phase('[378..10129]')
        inair.derive(aal, fast)
        self.assertEqual(inair, airs)
        
        
class TestAirborne(unittest.TestCase):
    def test_airborne_keeps_long_inairs(self):
        inair = phase('[10..200],[201..203],[300..400]')
        airborne = Airborne()
        airborne.derive(inair)
        self.assertEqual(airborne, '[10..200],[300..400]')


class TestApproachAndLanding(unittest.TestCase):
    def test_approach_and_landing_basic(self):
        alt = np.ma.array(range(5000, 500, -500) + [0] * 10)
        land = buildsection('Landing', 11, 20)
        # Go-around above 3000ft will be ignored.
        ga = buildsection('Go Around And Climbout', 8, 13)
        app = ApproachAndLanding()
        app.derive(
            Parameter('Altitude AAL For Flight Phases', alt), 
            lands=phase('[11..20]'), 
            go_arounds=phase('[8..13]'))
        self.assertEqual(app, '[4.0..20]')

    def test_approach_and_landing_landing_and_go_around_overlap(self):
        alt = np.ma.array([3500, 2500, 2000, 2500, 3500, 3500, 3500])
        app = ApproachAndLanding()
        app.derive(
            Parameter('Altitude AAL For Flight Phases', alt),
            lands=phase('[5..6]'), 
            go_arounds=phase('[2.5..3.5]'))
        # 3000 feet starts half a sample into data, data ends after 6 samples
        self.assertEqual(app, '[0.5..6]')

    def test_approach_and_landing_separate_landing_phase_go_around(self):
        alt = np.ma.array([3500, 2500, 2000, 2500, 3500, 3500])
        land = buildsection('Landing', 5, 6)
        ga = buildsection('Go Around And Climbout', 1.5, 2.0)
        app = ApproachAndLanding()
        app.derive(
            Parameter('Altitude AAL For Flight Phases', alt),
            lands=phase('[5..6]'), 
            go_arounds=phase('[1.5..2]'))
        self.assertEqual(app, '[0.5..2], [3.5..6]')


class TestApproach(unittest.TestCase):
    def test_approach_basic(self):
        app = Approach()
        app.derive(apps=phase('[5..15]'), lands=phase('[9.5..15]'))
        self.assertEqual(app, '[5..9.5)')

    def test_approach_complex(self):
        app = Approach()
        app.derive(apps=phase('[25..35),[5..15],(3..4)'), lands=phase('[12..27]'))
        self.assertEqual(app, '(3..4),[5..12),(27..35)')
        


class TestBouncedLanding(unittest.TestCase):
    def test_bounce_basic(self):
        air = phase('[3..10]')
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,0,0,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL', alt), air)
        self.assertEqual(bl, '[3..10]')
        
    def test_no_bounce(self):
        # tests first section is skipped as too long, second too high, third 
        air = phase('[10..100],[110..120],(200..230]')
        alt = np.ma.arange(250)
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL', alt), air)
        self.assertEqual(bl, '<Empty>')

    def test_bounce_with_bounce(self):
        air = phase('[9..11]')
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,3,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL', alt), air)
        self.assertEqual(bl, '[9..11]')

    def test_bounce_with_double_bounce(self):
        air = phase('[3..8],(8..12]')
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,0,5,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL', alt), air)
        self.assertEqual(bl, '[3..8],(8..12]')

    def test_bounce_not_detected_with_multiple_touch_and_go(self):
        # test data is a training flight with many touch and go
        inair = InAir()
        aal = load(os.path.join(test_data_path, 'alt_aal_training.nod'))
        ##airs = phase('[724..1002], [1027..1322], [1347..1982], [2003..2420], '
                     ##'[2440..2791], [2814..3212], [3236..3616], [3645..4032], '
                     ##'[4061..4671], [4702..5051], [5079..5415], [5440..5768], '
                     ##'[5792..6138], [6166..6538], [6562..7197], [7222..7570], '
                     ##'[7594..7931], [7957..8289], [8314..8604], [8631..10114]')
        exp = phase('[345.5..624.5],[648.5..944.5],  [968.5..1604.5],'
                  '[1624.5..2042.5],[2061.5..2413.5],[2435.5..2834.5],'
                  '[2857.5..3238.5],[3266.5..3654.5],[3682.5..4293.5],'
                  '[4323.5..4673.5],[4700.5..5037.5],[5061.5..5390.5],'
                  '[5413.5..5760.5],[5787.5..6160.5],[6183.5..6819.5],'
                  '[6843.5..7192.5],[7215.5..7553.5],[7578.5..7911.5],'
                  '[7935.5..8226.5],[8252.5..9736.5]')

        fast = phase('[378..10129]')
        inair.derive(aal, fast)
        # should not create any bounced landings (used to create 20 at 8000ft)
        bl = BouncedLanding()
        bl.derive(aal, inair)
        self.assertEqual(len(bl), 0)
        self.assertEqual(bl, '<Empty>')


class TestILSGlideslopeEstablished(unittest.TestCase):
    def test_can_operate(self):
        expected=[('ILS Glideslope', 'ILS Localizer Established',
                   'Altitude AAL')]
        opts = ILSGlideslopeEstablished.get_operational_combinations()
        self.assertEqual(opts, expected)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        hdf_copy = copy_file(os.path.join(test_data_path, 'coreg.hdf5'),
                             postfix='_test_copy')
        result = process_flight(hdf_copy, {
            'Engine': {'classification': 'JET',
                       'quantity': 2},
            'Frame': {'doubled': False, 'name': '737-3C'},
            'id': 1,
            'Identifier': '1000',
            'Model': {'family': 'B737 NG',
                      'interpolate_vspeeds': True,
                      'manufacturer': 'Boeing',
                      'model': 'B737-86N',
                      'precise_positioning': True,
                      'series': 'B737-800'},
            'Recorder': {'name': 'SAGEM', 'serial': '123456'},
            'Tail Number': 'G-DEMA'})
        phases = result['phases']
        sections = phases.get(name='ILS Glideslope Established')
        sections
        self.assertTrue(False, msg='Test not implemented.')


class TestILSLocalizerEstablished(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ILSLocalizerEstablished.get_operational_combinations(),
                         [('ILS Localizer', 'Altitude AAL For Flight Phases',
                           'Approach And Landing')])

    def test_ils_localizer_established_basic(self):
        ils = P('ILS Localizer',np.ma.arange(-3, 0, 0.3))
        alt_aal = P('Alttiude AAL For Flight Phases',
                    np.ma.arange(1000, 0, -100))
        app = App(items=[ApproachItem('LANDING', slice(0, 10))])
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 10*2.0/3.0, 10)
        # Slightly daft choice of ils array makes exact equality impossible!
        self.assertAlmostEqual(establish.get_first().start_edge,
                               expected.get_first().start_edge)

    def test_ils_localizer_established_never_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([3]*10))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = App(items=[ApproachItem('LANDING', slice(2, 9))])
        establish = ILSLocalizerEstablished()
        self.assertEqual(establish.derive(ils, alt_aal, app), None)

    def test_ils_localizer_established_always_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([-0.2]*10))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = App(items=[ApproachItem('LANDING', slice(2, 9))])
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established',2, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_only_last_segment(self):
        app = App(items=[ApproachItem('LANDING', slice(2, 9))])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,3,3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 7, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_stays_established_with_large_visible_deviations(self):
        app = App(items=[ApproachItem('LANDING', slice(1, 9))])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,2.3,2.3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_insensitive_to_few_masked_values(self):
        app = App(items=[ApproachItem('LANDING', slice(1, 9))])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0,0,0,1,2.3,2.3,2,1,0,0],
                                            mask=[0,0,0,0,0,1,1,0,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_skips_too_many_masked_values(self):
        app = App(items=[ApproachItem('LANDING', slice(1, 9))])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0.0]*20,
                                            mask=[0,1]*10))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        self.assertEqual(establish, [])

    def test_ils_localizer_skips_too_few_values(self):
        app = App(items=[ApproachItem('LANDING', slice(2, 9))])
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0.0]*5,
                                            mask=[0]*5))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        self.assertEqual(establish, [])

"""
class TestInitialApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach')]
        opts = InitialApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_approach_phase_basic(self):
        alt = np.ma.array(range(4000,0,-500)+range(0,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 8, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(2, 6, None))]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_over_high_ground(self):
        alt_aal = np.ma.array(range(0,4000,500)+range(4000,0,-500))
        # Raising the ground makes the radio altitude trigger one sample sooner.
        alt_rad = alt_aal - 600
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt_aal)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(10, 16, None))])
        app.derive(alt_aal, app_land)
        expected = [Section(name='Initial Approach', slice=slice(10, 14, None))]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_with_go_around(self):
        alt = np.ma.array(range(4000,2000,-500)+range(2000,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 5, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section(name='Initial Approach', slice=slice(2, 4, None))]
        self.assertEqual(app, expected)
"""

class TestClimbCruiseDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases','Airborne')]
        opts = ClimbCruiseDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_cruise_descent_start_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array([15000] * 5 + range(15000, 1000, -1000))
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        air=buildsection('Airborne', None, 19)
        camel.derive(alt_aal, air)
        expected = buildsection('Climb Cruise Descent', None, 18)
        self.assertEqual(list(camel), list(expected))

    def test_climb_cruise_descent_end_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array(range(1000,15000,1000)+[15000]*5)
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        air=buildsection('Airborne',0, None)
        camel.derive(alt_aal, air)
        expected = buildsection('Climb Cruise Descent', 0, None)
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_all_high(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array([15000]*5)
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,5)
        camel.derive(Parameter('Altitude AAL For Flight Phases',
                               np.ma.array(testwave)),air)
        expected = []
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_one_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air=buildsection('Airborne', 0, 62)
        camel.derive(Parameter('Altitude AAL For Flight Phases',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 1)

    def test_climb_cruise_descent_two_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 4, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,122)
        camel.derive(Parameter('Altitude AAL For Flight Phases',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 2)

    def test_climb_cruise_descent_three_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,186)
        camel.derive(Parameter('Altitude AAL For Flight Phases',
                               np.ma.array(testwave)), air)
        self.assertEqual(len(camel), 3)



'''
# ClimbFromBottomOfDescent is commented out in flight_phase.py
class TestClimbFromBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Climb', 'Climb Start', 'Bottom Of Descent')]
        opts = ClimbFromBottomOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_to_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        dlc = DescentLowClimb()
        dlc.derive(alt)
        bod = BottomOfDescent()
        bod.derive(alt, dlc)

        descent_phase = ClimbFromBottomOfDescent()
        descent_phase.derive(toc, [], bod) # TODO: include start of climb instance
        expected = [Section(name='Climb From Bottom Of Descent',slice=slice(63, 94, None))]
        self.assertEqual(descent_phase, expected)
'''


class TestClimbing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        vert_spd_data = np.ma.array(range(500,1200,100) +
                                    range(1200,-1200,-200) +
                                    range(-1200,500,100))
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.array(vert_spd_data))
        air = buildsection('Airborne', 2, 8)
        up = Climbing()
        up.derive(vert_spd, air)
        expected = buildsection('Climbing', 3, 8)
        self.assertEqual(up, expected)


class TestCruise(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Climb Cruise Descent',
                     'Top Of Climb', 'Top Of Descent')]
        opts = Cruise.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_cruise_phase_basic(self):
        alt_data = np.ma.array(
            np.cos(np.arange(0, 12.6, 0.1)) * -3000 + 12500)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt_p = Parameter('Altitude STD', alt_data)
        # Transform the "recorded" altitude into the CCD input data.
        ccd = ClimbCruiseDescent()
        ccd.derive(alt_p, buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt_p, ccd)
        tod = TopOfDescent()
        tod.derive(alt_p, ccd)

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================

        # With this test waveform, the peak at 31:32 is just flat enough
        # for the climb and descent to be a second apart, whereas the peak
        # at 94 genuinely has no interval with a level cruise.
        expected = buildsections('Cruise',[31, 32],[94, 95])
        self.assertEqual(list(test_phase), list(expected))

    def test_cruise_truncated_start(self):
        alt_data = np.ma.array([15000]*5+range(15000,2000,-4000))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = buildsection('Cruise', None, 5)
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 0)
        self.assertEqual(len(tod), 1)

    def test_cruise_truncated_end(self):
        alt_data = np.ma.array(range(300,36000,6000)+[36000]*4)
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = Cruise()
        expected.create_section(slice(6, 7), 'Cruise')
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 1)
        self.assertEqual(len(tod), 0)


class TestDescentLowClimb(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentLowClimb.get_operational_combinations(),
                         [('Altitude AAL For Flight Phases',)])

    def test_descent_low_climb_basic(self):
        # Wave is 5000ft to 0 ft and back up, with climb of 5000ft.
        testwave = np.cos(np.arange(0, 6.3, 0.1)) * (2500) + 2500
        dsc = testwave - testwave[0]
        dsc[32:] = 0.0
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        #descend = Parameter('Descend For Flight Phases', np.ma.array(dsc))
        #climb = Parameter('Climb For Flight Phases', np.ma.array(clb))
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        expected = buildsection('Descent Low Climb', 14, 49)
        self.assertEqual(list(dlc), list(expected))

    def test_descent_low_climb_inadequate_climb(self):
        testwave = np.cos(np.arange(0, 6.3, 0.1)) * (240) + 2500 # 480ft climb
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases',
                            np.ma.array(testwave))
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        self.assertEqual(len(dlc), 0)


class TestDescending(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Descending.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descending_basic(self):
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.array([0, 1000, -600, -800, 0]))
        air = buildsection('Airborne',2,8)
        phase = Descending()
        phase.derive(vert_spd, air)
        expected = buildsection('Descending', 2, 4)
        self.assertEqual(phase, expected)


"""
class TestDescentToBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Descent', 'Bottom Of Descent')]
        opts = DescentToBottomOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_to_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        dlc = DescentLowClimb()
        dlc.derive(alt)
        bod = BottomOfDescent()
        bod.derive(alt, dlc)

        descent_phase = DescentToBottomOfDescent()
        descent_phase.derive(tod, bod)
        expected = [Section(name='Descent To Bottom Of Descent',slice=slice(32,63,None))]
        self.assertEqual(descent_phase, expected)
"""

class TestFast(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Fast.get_operational_combinations(),
                         [('Airspeed For Flight Phases',)])

    def test_fast_phase_basic(self):
        slow_and_fast_data = np.ma.array(range(60, 120, 10) + [120] * 300 + \
                                         range(120, 50, -10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = buildsection('Fast', 2, 311)
        if AIRSPEED_THRESHOLD == 70:
            expected = buildsection('Fast', 1, 312)
        self.assertEqual(phase_fast, expected)

    def test_fast_all_fast(self):
        fast_data = np.ma.array([120] * 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast', None, None)
        self.assertEqual(phase_fast, expected)

    def test_fast_all_slow(self):
        fast_data = np.ma.array([12] * 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        self.assertEqual(phase_fast, [])

    def test_fast_slowing_only(self):
        fast_data = np.ma.arange(110, 60, -10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast', None, 4)
        self.assertEqual(phase_fast, expected)

    def test_fast_speeding_only(self):
        fast_data = np.ma.arange(60, 120, 10)
        ias = Parameter('Airspeed For Flight Phases', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast', 2, None)
        self.assertEqual(phase_fast, expected)

    #def test_fast_phase_with_masked_data(self): # These tests were removed.
    #We now use Airspeed For Flight Phases which has a repair mask function,
    #so this is not applicable.

class TestGrounded(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Grounded.get_operational_combinations(),
                         [('Airborne', 'Airspeed For Flight Phases')])

    def test_grounded_phase_basic(self):
        slow_and_fast_data = \
            np.ma.array(range(60, 120, 10) + [120] * 300 + range(120, 50, -10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data, 1, 0)
        air = buildsection('Airborne', 2, 311)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias)
        expected = buildsections('Grounded', [0, 2], [311, 313])
        self.assertEqual(phase_grounded, expected)

    def test_grounded_all_fast(self):
        grounded_data = np.ma.array([120] * 10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data, 1, 0)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias)
        expected = buildsection('Grounded', None, None)
        self.assertEqual(phase_grounded, expected)

    def test_grounded_all_slow(self):
        grounded_data = np.ma.array([12]*10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data, 1, 0)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias)
        expected = buildsection('Grounded', 0, 10)
        self.assertEqual(phase_grounded.get_first(), expected[0])

    def test_grounded_landing_only(self):
        grounded_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data,1,0)
        air = buildsection('Airborne',None,4)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias)
        expected = buildsection('Grounded',4,5)
        self.assertEqual(phase_grounded.get_first(), expected[0])

    def test_grounded_speeding_only(self):
        grounded_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed For Flight Phases', grounded_data,1,0)
        air = buildsection('Airborne',2,None)
        phase_grounded = Grounded()
        phase_grounded.derive(air, ias)
        expected = buildsection('Grounded',0,2)
        self.assertEqual(phase_grounded.get_first(), expected[0])


class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',)]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40)+[0,0,0])
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        expected = buildsection('Final Approach', 18, 31)
        fapp=FinalApproach()
        fapp.derive(alt_aal)
        self.assertEqual(fapp, expected)


class TestGearRetracting(unittest.TestCase):
    '''
    The Gear Extending and Gear Retracting flight phases were written when
    these were integer arrays, but now they are multistate arrays the flight
    phases themselves need to be rewritten before tests are created.

    As a result of this change, the KPVs AirspeedAsGearRetractingMax,
    AirspeedAsGearExtendingMax, MachAsGearRetractingMax &
    MachAsGearExtendingMax are currently inoperative.
    '''

    def test_can_operate(self):
        opts = GearRetracting.get_operational_combinations()
        self.assertTrue(all(['Gear Down' for o in opts]))
        expected = [('Gear Down', 'Gear (L) Red Warning',),
                    ('Gear Down', 'Gear (N) Red Warning',),
                    ('Gear Down', 'Gear (R) Red Warning',),
                    ('Gear Down', 'Frame',),
                    ('Gear Down', 'Airborne',),
                    ('Gear Down', 'Gear (L) Red Warning',
                     'Gear (N) Red Warning', 'Gear (R) Red Warning', 'Frame',
                     'Airborne'),]
        self.assertTrue([e in opts for e in expected])

    def test_737_3C(self):
        gear_down = M('Gear Down', np.ma.array([1,1,1,0,0,0,0,0,0,0,0,1,1]),
                      values_mapping={0:'Up', 1:'Down'})
        gear_warn_l = M('Gear (L) Red Warning',
                        np.ma.array([0,0,0,1,0,0,0,0,0,1,0,0]),
                        values_mapping={1:'Warning', 0:'False'})
        gear_warn_n = M('Gear (N) Red Warning',
                        np.ma.array([0,0,0,0,1,0,0,0,1,0,0,0]),
                        values_mapping={1:'Warning', 0:'False'})
        gear_warn_r = M('Gear (R) Red Warning',
                        np.ma.array([0,0,0,0,0,1,0,1,0,0,0,0]),
                        values_mapping={1:'Warning', 0:'False'})
        frame = A('Frame', value='737-3C')
        airs=buildsection('Airborne', 1, 11)
        gr = GearRetracting()
        gr.derive(gear_down, gear_warn_l, gear_warn_n, gear_warn_r, frame, airs)
        expected=buildsection('Gear Retracting', 3, 6)
        self.assertEqual(list(gr), list(expected))


class TestGoAroundAndClimbout(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(GoAroundAndClimbout.get_operational_combinations(),
                         [('Altitude AAL','Go Around')])

    def test_go_around_and_climbout_phase_basic(self):
        '''
        down = np.ma.array(range(4000,1000,-490)+[1000]*7) - 4000
        up = np.ma.array([1000]*7+range(1000,4500,490)) - 1500
        ga_kti = KTI('Go Around', items=[KeyTimeInstance(index=7, name='Go Around')])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(Parameter('Descend For Flight Phases',down),
                        Parameter('Climb For Flight Phases',up),
                        ga_kti)
        expected = buildsection('Go Around And Climbout', 4.9795918367346941,
                                12.102040816326531)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)
        '''
        alt_aal = load(os.path.join(test_data_path, 'alt_aal_goaround.nod'))
        gas = load(os.path.join(test_data_path, 'go_around_kti_goaround.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, gas)
        expected = buildsections('Go Around And Climbout',
                                 [3586.0, 3729], [4895, 5141], [7124, 7265])
        for n in range(3):
            self.assertAlmostEqual(ga_phase[n].start_edge,
                                   expected[n].start_edge,
                                   places=0)
            self.assertAlmostEqual(ga_phase[n].stop_edge,
                                   expected[n].stop_edge,
                                   places=0)

    def test_go_around_and_climbout_real_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_alt_aal.nod'))
        gas = load(os.path.join(test_data_path,
                                'GoAroundAndClimbout_gas.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, gas)
        self.assertEqual(
            ga_phase,
            [Section(name='Go Around And Climbout',
                     slice=slice(1057.4680851063829, 1170),
                     start_edge=1057.4680851063829,
                     stop_edge=1170),
             Section(name='Go Around And Climbout',
                     slice=slice(1393.7412587412587, 1505, None),
                     start_edge=1393.7412587412587,
                     stop_edge=1505),
             Section(name='Go Around And Climbout',
                     slice=slice(1722.2450331125829, 1837, None),
                     start_edge=1722.2450331125829,
                     stop_edge=1837),
             Section(name='Go Around And Climbout',
                     slice=slice(2071.2133891213389, 2206, None),
                     start_edge=2071.2133891213389,
                     stop_edge=2206),
             Section(name='Go Around And Climbout',
                     slice=slice(2391.4565217391305, 2506, None),
                     start_edge=2391.4565217391305,
                     stop_edge=2506)])


class TestHolding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Holding.get_operational_combinations(),
                         [('Altitude AAL For Flight Phases',
                          'Heading Increasing','Latitude Smoothed',
                          'Longitude Smoothed')])

    def test_straightish_not_detected(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*0.45)
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)

    def test_bent_detected(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*(1.1))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=buildsection('Holding',0,3000)
        self.assertEqual(hold, expected)

    def test_rejected_outside_height_range(self):
        hdg=P('Heading Increasing', np.ma.arange(3000)*(1.1))
        alt=P('Altitude AAL For Flight Phases', np.ma.arange(3000,0,-1)*10)
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        # OK - I cheated. Who cares about the odd one sample passing 5000ft :o)
        expected=buildsection('Holding', 1001, 2700)
        self.assertEqual(list(hold), list(expected))

    def test_hold_detected(self):
        rot=[0]*600+([3]*60+[0]*60)*6+[0]*180+([3]*60+[0]*90)*6+[0]*600
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=buildsections('Holding',[570,1290],[1470,2340])
        self.assertEqual(hold, expected)

    def test_hold_rejected_if_travelling(self):
        rot=[0]*600+([3]*60+[0]*60)*6+[0]*180+([3]*60+[0]*90)*6+[0]*600
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.arange(0, 30, 0.01))
        lon=P('Longitude Smoothed', np.ma.arange(0, 30, 0.01))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)

    def test_single_turn_rejected(self):
        rot=np.ma.array([0]*500+[3]*60+[0]*600+[6]*60+[0]*600+[0]*600+[3]*90+[0]*490, dtype=float)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt=P('Altitude AAL For Flight Phases', np.ma.array([10000]*3000))
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        expected=[]
        self.assertEqual(hold, expected)



class TestLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Landing.get_operational_combinations(),
                         [('Heading Continuous', 'Altitude AAL', 'Fast')])

    def test_landing_basic(self):
        head = np.ma.array([20]*8+[10,0])
        alt_aal = np.ma.array([80,40,20,5]+[0]*6)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 9)
        self.assertEqual(landing, expected)

    def test_landing_turnoff(self):
        head = np.ma.array([20]*15+range(20,0,-2))
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 24)
        self.assertEqual(landing, expected)

    def test_landing_turnoff_left(self):
        head = np.ma.array([20]*15+range(20,0,-2))*-1.0
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast', 0, 5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 24)
        self.assertEqual(landing, expected)


class TestMobile(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn',),('Rate Of Turn','Groundspeed')]
        opts = Mobile.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_rot_only(self):
        rot = np.ma.array([0,0,5,5,5,0,0])
        move = Mobile()
        move.derive(P('Rate Of Turn',rot), None)
        expected = buildsection('Mobile', 2, 4)
        self.assertEqual(move, expected)

    def test_gspd_first(self):
        rot = np.ma.array([0,0,0,5,5,0,0])
        gspd= np.ma.array([0,6,6,6,0,0,0])
        move = Mobile()
        move.derive(P('Rate Of Turn',rot),
                    P('Groundspeed',gspd))
        expected = buildsection('Mobile', 1, 4)
        self.assertEqual(move, expected)

    def test_gspd_last(self):
        rot = np.ma.array([0,0,5,5,0,0,0])
        gspd= np.ma.array([0,0,0,6,6,6,0])
        move = Mobile()
        move.derive(P('Rate Of Turn',rot),
                    P('Groundspeed',gspd))
        expected = buildsection('Mobile', 2, 5)
        self.assertEqual(move, expected)


class TestLevelFlight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LevelFlight
        self.operational_combinations = [('Airborne', 'Vertical Speed For Flight Phases')]

    def test_level_flight_phase_basic(self):
        data = range(0, 400, 50) + range(400, -450, -50) + range(-450, 50, 50)
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(0, 36, None), 0, 36),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd)
        self.assertEqual(level, [
            Section('Level Flight', slice(0, 7, None), 0, 7),
            Section('Level Flight', slice(10, 23, None), 10, 23),
            Section('Level Flight', slice(28, 35, None), 28, 35),
        ])

    def test_level_flight_phase_not_airborne_basic(self):
        data = range(0, 400, 50) + range(400, -450, -50) + range(-450, 50, 50)
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(8, 30, None), 8, 30),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd)
        self.assertEqual(level, [
            Section('Level Flight', slice(10, 23, None), 10, 23),
            Section('Level Flight', slice(28, 30, None), 28, 30),
        ])


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Takeoff.get_operational_combinations(),
                         [('Heading Continuous', 'Altitude AAL', 'Fast')])

    def test_takeoff_basic(self):
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', 6.5, 10)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Takeoff', 1.5, 9.125)
        self.assertEqual(takeoff, expected)

    def test_takeoff_with_zero_slices(self):
        '''
        A zero slice was causing the derive method to raise an exception.
        This test aims to replicate the problem, and shows that with a None
        slice an empty takeoff phase is produced.
        '''
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', None, None)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = []
        self.assertEqual(takeoff, expected)


class TestTaxiOut(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Grounded', 'Takeoff')]
        self.assertEqual(TaxiOut.get_operational_combinations(), expected)

    def test_taxi_out(self):
        gnd = buildsection('Grounded',3, 8)
        toff = buildsection('Takeoff', 6, 12)
        tout = TaxiOut()
        tout.derive(gnd, toff)
        expected = buildsection('Taxi Out',4, 5)
        self.assertEqual(tout, expected)

class TestTaxiIn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Grounded', 'Landing')]
        self.assertEqual(TaxiIn.get_operational_combinations(), expected)

    def test_taxi_in(self):
        gnd = buildsection('Grounded',7,14)
        toff = buildsection('Landing', 5,12)
        t_in = TaxiIn()
        t_in.derive(gnd, toff)
        expected = buildsection('Taxi In',12,14)
        self.assertEqual(t_in,expected)


class TestTaxiing(unittest.TestCase):
    def test_can_operate(self):
        expected=[('Taxi Out', 'Taxi In')]
        self.assertEqual(Taxiing.get_operational_combinations(), expected)

    def test_taxiing(self):
        tout = buildsection('Taxi Out', 2,  5)
        t_in = buildsection('Taxi In',  8, 11)
        ting = Taxiing()
        ting.derive(tout, t_in)
        expected = buildsections('Taxiing', [2,5],[8,11])
        self.assertEqual(ting, expected)


class TestTurningInAir(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn', 'Airborne')]
        self.assertEqual(TurningInAir.get_operational_combinations(), expected)

    def test_turning_in_air_phase_basic(self):
        rate_of_turn_data = np.arange(-4, 4.4, 0.4)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(list(turning_in_air), list(expected))

    def test_turning_in_air_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[6] = np.ma.masked
        rate_of_turn_data[16] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(turning_in_air, expected)


class TestTurningOnGround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn', 'Grounded')]
        self.assertEqual(TurningOnGround.get_operational_combinations(), expected)

    def test_turning_on_ground_phase_basic(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground, expected)

    def test_turning_on_ground_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        # Masked inside is exclusive of the range outer limits, this behaviour
        # is not consistent with TurningInAir test which is inclusive of the
        # start of the range.
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground, expected)

    def test_turning_on_ground_after_takeoff_inhibited(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0,10)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7])
        self.assertEqual(turning_on_ground, expected)


class TestDescentToFlare(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentToFlare.get_operational_combinations(),
                         [('Descent', 'Altitude AAL For Flight Phases')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGearExtending(unittest.TestCase):
    '''
    The Gear Extending and Gear Retracting flight phases were written when
    these were integer arrays, but now they are multistate arrays the flight
    phases themselves need to be rewritten before tests are created.

    As a result of this change, the KPVs AirspeedAsGearRetractingMax,
    AirspeedAsGearExtendingMax, MachAsGearRetractingMax &
    MachAsGearExtendingMax are currently inoperative.
    '''
    def test_can_operate(self):
        combinations = GearExtending.get_operational_combinations()
        self.assertTrue(
            all('Gear Down' in c and 'Airborne' in c for c in combinations))
        self.assertTrue(('Gear Down', 'Airborne') in combinations)
        self.assertTrue((
            'Gear Down', 'Gear (L) Red Warning', 'Gear (N) Red Warning',
            'Gear (R) Red Warning', 'Frame', 'Airborne') in combinations)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestGoAround5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GoAround5MinRating.get_operational_combinations(),
                         [('Go Around And Climbout',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoff5MinRating(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Takeoff5MinRating.get_operational_combinations(),
                         [('Takeoff',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRoll.get_operational_combinations(),
                         [('Takeoff', 'Takeoff Acceleration Start', 'Pitch',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTakeoffRotation(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRotation.get_operational_combinations(),
                         [('Liftoff',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTwoDegPitchTo35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TwoDegPitchTo35Ft.get_operational_combinations(),
                         [('Pitch', 'Takeoff',)])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')
