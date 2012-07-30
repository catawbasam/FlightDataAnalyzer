import unittest
import numpy as np

from analysis_engine.node import (A, KPV, KTI, KeyTimeInstance, Parameter, P,
                                  Section, SectionNode, S)

from analysis_engine.key_time_instances import (BottomOfDescent,
                                         TopOfClimb, 
                                         TopOfDescent
                                         )
from analysis_engine.plot_flight import plot_parameter
from analysis_engine.flight_phase import (Airborne,
                                          Approach,
                                          ClimbCruiseDescent,
                                          Climbing,
                                          Cruise,
                                          Descending,
                                          DescentLowClimb,
                                          Fast,
                                          FinalApproach,
                                          GoAroundAndClimbout,
                                          Holding,
                                          ILSLocalizerEstablished,
                                          Landing,
                                          LevelFlight,
                                          OnGround,
                                          Takeoff,
                                          Taxiing,
                                          TaxiIn,
                                          TaxiOut,
                                          TurningInAir,
                                          TurningOnGround
                                          )

from analysis_engine.library import hysteresis, integrate, repair_mask

from analysis_engine.settings import AIRSPEED_THRESHOLD, HYSTERESIS_FPALT_CCD

'''
Three little routines to make building Sections for testing easier.
'''
def builditem(name, begin, end):
    if begin==None:
        ib = None
    else:
        ib = begin
    if end==None:
        ie = None
    else:
        ie = end
    '''
    This code more accurately represents the aligned section values, but is
    not suitable for test cases where the data does not get aligned.
    
    if begin==None:
        ib = None
    else:
        ib = int(begin)
        if ib < begin:
            ib += 1
    if end==None:
        ie = None
    else:
        ie = int(end)
        if ie < end:
            ie += 1
            '''
    return Section(name, slice(ib, ie, None), begin, end)

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

def buildsections(*args):
    '''
    Like buildsection, this is used to build SectionNodes for test purposes.
    
    lands = buildsections('name',[from1,to1],[from2,to2])

    Example of use:
    approach = buildsections('Approach', [80,90], [100,110])
    '''
    built_list=[]
    name = args[0]
    for a in args[1:]:
        new_section = builditem(name, a[0], a[1])
        built_list.append(new_section)
    return SectionNode(name, items=built_list)

class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases', 'Fast')]
        opts = Airborne.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_airborne_phase_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+
                                         range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        altitude = Parameter('Altitude AAL For Flight Phases', integrate(rate_of_climb_data, 1, 0, 1.0/60.0))
        fast = SectionNode('Fast', items=[Section('Fast',slice(1,29,None),1,29)])
        air = Airborne()
        air.derive(altitude, fast)
        expected = Section(name='Airborne', slice=slice(2, 28, None), start_edge=2, stop_edge=28)
        self.assertEqual(air.get_first(), expected)

    def test_airborne_phase_not_fast(self):
        altitude_data = np.ma.array(range(0,10))
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = []
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(air, [])

    def test_airborne_phase_started_midflight(self):
        altitude_data = np.ma.array([100]*20+[60,30,10]+[0]*4)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', None, 25)
        air = Airborne()
        air.derive(alt_aal, fast)
        expected = buildsection('Airborne', None, 23)
        self.assertEqual(air, expected)
        
    def test_airborne_phase_ends_in_midflight(self):
        altitude_data = np.ma.array([0]*5+[30,80]+[100]*20)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', 2, None)
        air = Airborne()
        air.derive(alt_aal, fast)
        expected = buildsection('Airborne', 5, None)
        self.assertEqual(air, expected)



class TestApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Approach.get_operational_combinations(),
                         [('Altitude AAL For Flight Phases', 'Landing',
                           'Go Around And Climbout')])

    def test_approach_and_landing_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=buildsection('Landing',11,20)
        # Go-around above 3000ft will be ignored.
        ga=buildsection('Go Around And Climbout',8,13)
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   land, ga)
        expected = buildsection('Approach', 4.0, 20)
        self.assertEqual(app, expected)

    def test_approach_landing_and_go_around_overlap(self):
        alt = np.ma.array([3500,2500,2000,2500,3500,3500])
        land=buildsection('Landing',5,6)
        ga=buildsection('Go Around And Climbout',2.5,3.5)
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   land, ga)
        expected = buildsection('Approach', 0, 6)
        self.assertEqual(app, expected)

    def test_approach_separate_landing_phase_go_around(self):
        alt = np.ma.array([3500,2500,2000,2500,3500,3500])
        land=buildsection('Landing',5,6)
        ga=buildsection('Go Around And Climbout',1.5,2.0)
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   land, ga)
        expected = buildsections('Approach', [0, 2], [3, 6])
        self.assertEqual(app, expected)


class TestILSLocalizerEstablished(unittest.TestCase):
    def test_can_operate(self):
        expected=[('ILS Localizer','Altitude AAL For Flight Phases','Approach')]
        opts = ILSLocalizerEstablished.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_ils_localizer_established_basic(self):
        ils = P('ILS Localizer',np.ma.arange(-3,0,0.3))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app=buildsection('Approach',0,10)
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 10*2.0/3.0, 10)
        # Slightly daft choice of ils array makes exact equality impossible!
        self.assertAlmostEqual(establish.get_first().start_edge, expected.get_first().start_edge)

    def test_ils_localizer_established_never_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([3]*10))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = buildsection('Approach',2, 9)
        establish = ILSLocalizerEstablished()
        self.assertEqual(establish.derive(ils, alt_aal, app), None)

    def test_ils_localizer_established_always_on_loc(self):
        ils = P('ILS Localizer',np.ma.array([-0.2]*10))
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        app = buildsection('Approach',2, 9)
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established',2, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_only_last_segment(self):
        app = buildsection('Approach',2, 9)
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,3,3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 7, 9)
        self.assertEqual(establish, expected)

    def test_ils_localizer_stays_established_with_large_visible_deviations(self):
        app = buildsection('Approach',1, 9)
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,2.3,2.3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish, expected)


    def test_ils_localizer_insensitive_to_few_masked_values(self):
        app = buildsection('Approach',1, 9)
        alt_aal = P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-100))
        ils = P('ILS Localizer',np.ma.array(data=[0,0,0,1,2.3,2.3,2,1,0,0],
                                            mask=[0,0,0,0,0,1,1,0,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(ils, alt_aal, app)
        expected = buildsection('ILS Localizer Established', 1, 9)
        self.assertEqual(establish, expected)

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
        testwave = np.ma.array([15000]*5+range(15000,1000,-1000))
        # plot_parameter (testwave)
        air=buildsection('Airborne',None,19)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
        expected = buildsection('Climb Cruise Descent', None, 19)
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_end_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array(range(1000,15000,1000)+[15000]*5)
        # plot_parameter (testwave)
        air=buildsection('Airborne',0, None)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
        expected = buildsection('Climb Cruise Descent', 0, None)
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_all_high(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.array([15000]*5)
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,5)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
        expected = []
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_one_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0,3.14*2,0.1))*(-3000)+12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,62)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
        self.assertEqual(len(camel), 1)
        
    def test_climb_cruise_descent_two_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0,3.14*4,0.1))*(-3000)+12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,122)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
        self.assertEqual(len(camel), 2)
        
    def test_climb_cruise_descent_three_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0,3.14*6,0.1))*(-3000)+12500
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,186)
        camel.derive(Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave)),air)
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
        expected = [('Rate Of Climb For Flight Phases', 'Airborne')]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        rate_of_climb_data = np.ma.array(range(500,1200,100)+
                                         range(1200,-1200,-200)+
                                         range(-1200,500,100))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        air = buildsection('Airborne',2,8)
        up = Climbing()
        up.derive(rate_of_climb, air)
        expected = buildsection('Climbing', 3, 8)
        self.assertEqual(up, expected)


class TestCruise(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Climb Cruise Descent',
                     'Top Of Climb', 'Top Of Descent')]
        opts = Cruise.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_cruise_phase_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of 
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt_p = Parameter('Altitude STD', alt_data)
        # Transform the "recorded" altitude into the CCD input data.
        ccd_p = Parameter('Altitude For Climb Cruise Descent', 
                          hysteresis(repair_mask(alt_data), HYSTERESIS_FPALT_CCD))
        ccd = ClimbCruiseDescent()
        ccd.derive(alt_p, buildsection('Airborne',0,len(alt_data)))
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
        expected = buildsections('Cruise',[31, 32],[94, 94])
        self.assertEqual(test_phase, expected)

    def test_cruise_truncated_start(self):
        alt_data = np.ma.array([15000]*5+range(15000,2000,-4000))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne',0,len(alt_data)))
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
                   buildsection('Airborne',0,len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = buildsection('Cruise', 6, None)
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 1)
        self.assertEqual(len(tod), 0)


class TestDescentLowClimb(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentLowClimb.get_operational_combinations(),
            [('Altitude AAL For Flight Phases',
              'Descend For Flight Phases',
              'Climb For Flight Phases',
              'Fast')])
        
    def test_descent_low_climb_basic(self):
        # Wave is 5000ft to 0 ft and back up, with climb of 5000ft.
        testwave = np.cos(np.arange(0,6.3,0.1))*(2500)+2500
        dsc = testwave - testwave[0]
        dsc [ 32:] = 0.0
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        descend = Parameter('Descend For Flight Phases', np.ma.array(dsc))
        climb = Parameter('Climb For Flight Phases', np.ma.array(clb))
        fast = buildsection('Fast', 0, 126)
        dlc = DescentLowClimb()
        dlc.derive(alt_aal, descend, climb, fast)
        expected = buildsection('Descent Low Climb', 14, 50)    
        self.assertEqual(dlc, expected)

    def test_descent_low_climb_inadequate_climb(self):
        testwave = np.cos(np.arange(0,6.3,0.1))*(240)+2500 # 480ft climb
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        climb = Parameter('Climb For Flight Phases', np.ma.array(clb))
        fast = buildsection('Fast', 0, 126)
        dlc = DescentLowClimb()
        dlc.derive(alt_aal, climb, fast)
        self.assertEqual(len(dlc), 0)


class TestDescending(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb For Flight Phases', 'Airborne')]
        opts = Descending.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descending_basic(self):
        roc = Parameter('Rate Of Climb For Flight Phases',np.ma.array([0,1000,-600,-800,0]))
        air = buildsection('Airborne',2,8)
        phase = Descending()
        phase.derive(roc, air)
        expected = buildsection('Descending',2,4)
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
        slow_and_fast_data = np.ma.array(range(60,120,10)+[120]*300+range(120,50,-10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = buildsection('Fast',2,311)
        if AIRSPEED_THRESHOLD == 70:
            expected = buildsection('Fast',1,312)
        self.assertEqual(phase_fast, expected)
        
    def test_fast_all_fast(self):
        fast_data = np.ma.array([120]*10)
        ias = Parameter('Airspeed For Flight Phases', fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast',None,None)
        self.assertEqual(phase_fast, expected)

    def test_fast_all_slow(self):
        fast_data = np.ma.array([12]*10)
        ias = Parameter('Airspeed For Flight Phases', fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast',None,None)
        self.assertEqual(phase_fast, expected)

    def test_fast_slowing_only(self):
        fast_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed For Flight Phases', fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast',None,4)
        self.assertEqual(phase_fast, expected)
        
    def test_fast_speeding_only(self):
        fast_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed For Flight Phases', fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = buildsection('Fast',2,None)
        self.assertEqual(phase_fast, expected)

    #def test_fast_phase_with_masked_data(self): # These tests were removed.
    #We now use Airspeed For Flight Phases which has a repair mask function,
    #so this is not applicable.
        

class TestOnGround(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(OnGround.get_operational_combinations(), 
                         [('Airspeed For Flight Phases','Airborne')])

    def test_on_ground_phase_basic(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+[120]*300+range(120,50,-10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data,1,0)
        air = buildsection('Airborne',2,311)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias, air)
        expected = buildsections('On Ground',[0,2],[311,313])
        self.assertEqual(phase_on_ground, expected)
        
    def test_on_ground_all_fast(self):
        on_ground_data = np.ma.array([120]*10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        air = buildsection('Airborne',None,None)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias, air)
        expected = buildsection('On Ground',None,None)
        self.assertEqual(phase_on_ground, expected)

    def test_on_ground_all_slow(self):
        on_ground_data = np.ma.array([12]*10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        air = buildsection('Airborne',None,None)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias, air)
        expected = buildsection('On Ground',0,10)
        self.assertEqual(phase_on_ground.get_first(), expected[0])

    def test_on_ground_landing_only(self):
        on_ground_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        air = buildsection('Airborne',None,4)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias, air)
        expected = buildsection('On Ground',4,5)
        self.assertEqual(phase_on_ground.get_first(), expected[0])
        
    def test_on_ground_speeding_only(self):
        on_ground_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        air = buildsection('Airborne',2,None)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias, air)
        expected = buildsection('On Ground',0,2)
        self.assertEqual(phase_on_ground.get_first(), expected[0])





class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',)]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40)+[0,0,0])
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        expected = buildsection('Final Approach', 22, 32)
        fapp=FinalApproach()
        fapp.derive(alt_aal)
        self.assertEqual(fapp, expected)
        

class TestGoAroundAndClimbout(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GoAroundAndClimbout.get_operational_combinations(),
                         [('Descend For Flight Phases', 'Climb For Flight Phases',
                           'Go Around')])

    def test_go_around_and_climbout_phase_basic(self):
        down = np.ma.array(range(4000,1000,-490)+[1000]*7) - 4000
        up = np.ma.array([1000]*7+range(1000,4500,490)) - 1500
        ga_kti = KTI('Go Around', items=[KeyTimeInstance(index=7, name='Go Around')])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(Parameter('Descend For Flight Phases',down),
                        Parameter('Climb For Flight Phases',up), 
                        ga_kti)
        expected = buildsection('Go Around And Climbout', 5.1020408163265305,
                                9.0408163265306118)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)

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
        alt=P('Altitude AAL For Flight Phases', np.ma.arange(3000)*10)
        lat=P('Latitude Smoothed', np.ma.array([24.0]*3000))
        lon=P('Longitude Smoothed', np.ma.array([24.0]*3000))
        hold=Holding()
        hold.derive(alt, hdg, lat, lon)
        # OK - I cheated. Who cares about the odd one sample passing 5000ft :o)
        expected=buildsection('Holding',501,2000)
        self.assertEqual(hold, expected)

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
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast')])

    def test_landing_basic(self):
        head = np.ma.array([20]*8+[10,0])
        ias  = np.ma.array([110]*4+[80,50,30,20,10,10])
        alt_aal = np.ma.array([80,40,20,5]+[0]*6)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 8.5)
        self.assertEqual(landing, expected)
        
    def test_landing_turnoff(self):
        head = np.ma.array([20]*15+range(20,0,-2))
        ias  = np.ma.array([110]*4+[80,50,40,30,20]+[10]*21)
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 22.5)
        self.assertEqual(landing, expected)
        
    def test_landing_turnoff_left(self):
        head = np.ma.array([20]*15+range(20,0,-2))*-1.0
        ias  = np.ma.array([110]*4+[80,50,40,30,20]+[10]*21)
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast)
        expected = buildsection('Landing', 0.75, 22.5)
        self.assertEqual(landing, expected)
        
        
"""
class TestLevelFlight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb For Flight Phases','Airborne')]
        opts = LevelFlight.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_level_flight_phase_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        airborne = SectionNode('Airborne',
                               items=[Section('Airborne',slice(0,36,None))])
        level = LevelFlight()
        level.derive(rate_of_climb, airborne)
        expected = [Section(name='Level Flight', slice=slice(0, 7, None)),
                    Section(name='Level Flight', slice=slice(10, 23, None)), 
                    Section(name='Level Flight', slice=slice(28, 35, None))]
        self.assertEqual(level, expected)
        
    def test_level_flight_phase_not_airborne_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        airborne = SectionNode('Airborne',
                               items=[Section('Airborne',slice(8,30,None))])
        level = LevelFlight()
        level.derive(rate_of_climb, airborne)
        expected = [Section(name='Level Flight', slice=slice(10, 23, None)), 
                    Section(name='Level Flight', slice=slice(28, 30, None))]
        self.assertEqual(level, expected)
        """


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Takeoff.get_operational_combinations(),
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast')])

    def test_takeoff_basic(self):
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        ias  = np.ma.array([10,10,10,10,10,40,70,90,105,110,110])
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
        ias  = np.ma.array([10,10,10,10,10,40,70,90,105,110,110])
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
        expected = [('On Ground', 'Takeoff')]
        self.assertEqual(TaxiOut.get_operational_combinations(), expected)
        
    def test_taxi_out(self):
        gnd = buildsection('On Ground',3,7)
        toff = buildsection('Takeoff', 5,12)
        tout = TaxiOut()
        tout.derive(gnd, toff)
        expected = buildsection('Taxi Out',3,5)
        self.assertEqual(tout,expected)
        
class TestTaxiIn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('On Ground', 'Landing')]
        self.assertEqual(TaxiIn.get_operational_combinations(), expected)
        
    def test_taxi_in(self):
        gnd = buildsection('On Ground',7,14)
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
        expected = buildsections('Turning In Air',[0, 4],[17,21])
        self.assertEqual(turning_in_air, expected)
        
    def test_turning_in_air_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 4],[17,21])
        self.assertEqual(turning_in_air, expected)
        
class TestTurningOnGround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn', 'On Ground')]
        self.assertEqual(TurningOnGround.get_operational_combinations(), expected)
        
    def test_turning_on_ground_phase_basic(self):
        rate_of_turn_data = np.arange(-12, 12, 1)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, airborne)
        expected = buildsections('Turning On Ground',[0, 4],[21,24])
        self.assertEqual(turning_on_ground, expected)
        
    def test_turning_on_ground_phase_with_mask(self):
        rate_of_turn_data = np.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, airborne)
        expected = buildsections('Turning On Ground',[0, 4],[21,24])
        self.assertEqual(turning_on_ground, expected)

    def test_turning_on_ground_after_takeoff_inhibited(self):
        rate_of_turn_data = np.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,10)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, airborne)
        expected = buildsections('Turning On Ground',[0, 4])
        self.assertEqual(turning_on_ground, expected)
