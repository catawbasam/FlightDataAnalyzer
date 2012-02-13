import unittest
import numpy as np

from analysis_engine.node import (A, KPV, KTI, KeyTimeInstance, Parameter, P,
                                  Section, SectionNode, S)

from analysis_engine.key_time_instances import (BottomOfDescent,
                                         TopOfClimb, 
                                         TopOfDescent
                                         )
from analysis_engine.plot_flight import plot_parameter
from analysis_engine.flight_phase import (
    Airborne,
    ApproachAndGoAround,
    ApproachAndLanding,
    ClimbCruiseDescent,
    Climbing,
    Cruise,
    Descending,
    DescentLowClimb,
    Fast,
    FinalApproach,
    ILSLocalizerEstablished,
    InitialApproach,
    Landing,
    LevelFlight,
    Takeoff,
    Turning
    )

from analysis_engine.settings import AIRSPEED_THRESHOLD

class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Rate Of Climb For Flight Phases', 'Fast')]
        opts = Airborne.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_airborne_phase_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+
                                         range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        fast = SectionNode('Fast', items=[Section('Fast',slice(2,25,None))])
        air = Airborne()
        air.derive(rate_of_climb, fast)
        expected = Section(name='Airborne', slice=slice(6, 25, None))
        self.assertEqual(air.get_first(), expected)

    def test_airborne_phase_not_airborne(self):
        rate_of_climb_data = np.ma.array(range(0,10))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        fast = []
        air = Airborne()
        air.derive(rate_of_climb, fast)
        self.assertEqual(air, [])


"""
class TestApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach And Landing')]
        opts = Approach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+range(500,3000,500))
        aal = S('Approach And Landing', items=[Section('Approach And Landing', slice(4, 14, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt), aal)
        expected = [Section(name='Approach', slice=slice(4, 9, None))]
        self.assertEqual(app, expected)
"""


class TestApproachAndGoAround(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ApproachAndGoAround.get_operational_combinations(),
                         [('Altitude AAL', 'Altitude Radio',
                           'Climb For Flight Phases', 'Go Around', 'Fast')])

    def test_approach_and_go_around_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*11)
        ga = KTI('Go Around', items=[KeyTimeInstance(index=11, name='Go Around')])
        climb=S(items=[Section('Climb For Flight Phases',slice=slice(11,20))])
        fast=S(items=[Section('Fast',slice=slice(0,19))])
        app = ApproachAndGoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL',alt),
                   Parameter('Altitude Radio',alt),
                   climb, ga, fast)
        expected = [Section(name='Approach And Go Around', slice=slice(4, 19, None))]
        self.assertEqual(app, expected)

    def test_approach_and_go_around_phase_no_ralt(self):
        alt = np.ma.array(range(4000,400,-400)+[0]*11)
        ga = KTI('Go Around', items=[KeyTimeInstance(index=11, name='Go Around')])
        climb=S(items=[Section('Climb For Flight Phases',slice=slice(11,20))])
        fast=S(items=[Section('Fast',slice=slice(0,19))])
        app = ApproachAndGoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL',alt),
                   None,
                   climb, ga, fast)
        expected = [Section(name='Approach And Go Around', slice=slice(2.5, 19, None))]
        self.assertEqual(app, expected)

    def test_approach_and_go_around_over_high_ground(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*12)
        ga = KTI('Go Around', items=[KeyTimeInstance(index=11, name='Go Around')])
        climb=S(items=[Section('Climb For Flight Phases',slice=slice(11,20))])
        fast=S(items=[Section('Fast',slice=slice(0,19))])
        app = ApproachAndGoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL',alt),
                   Parameter('Altitude Radio',alt-500),
                   climb, ga, fast)
        expected = [Section(name='Approach And Go Around', slice=slice(3, 19, None))]
        self.assertEqual(app, expected)
    
    def test_approach_and_go_around_levels_out(self):
        # The height does not reach 3000ft, and drops away before that.
        alt = np.ma.array([2500,2600]+range(2500,500,-500)+[0]*14)
        ga = KTI('Go Around', items=[KeyTimeInstance(index=11, name='Go Around')])
        climb=S(items=[Section('Climb For Flight Phases',slice=slice(11,17))])
        fast=S(items=[Section('Fast',slice=slice(0,17))])
        app = ApproachAndGoAround()
        app.derive(Parameter('Altitude AAL',alt),
                   Parameter('Altitude Radio',alt),
                   climb, ga, fast)
        expected = [Section(name='Approach And Go Around', slice=slice(1, 17, None))]
        self.assertEqual(app, expected)
    

class TestApproachAndLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ApproachAndLanding.get_operational_combinations(),
            [('Altitude AAL For Flight Phases', 'Landing'),
             ('Altitude AAL For Flight Phases', 'Altitude Radio', 'Landing')])

    def test_approach_and_landing_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        app = ApproachAndLanding()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   land)
        expected = [Section(name='Approach And Landing', slice=slice(4, 20, None))]
        self.assertEqual(app, expected)

    def test_approach_and_landing_phase_no_ralt(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        alt_param = Parameter('Altitude AAL For Flight Phases',alt)
        app = ApproachAndLanding()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_param, None, land)
        expected = [Section(name='Approach And Landing', slice=slice(4, 20, None))]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_over_high_ground(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        app = ApproachAndLanding()
        # Raising the ground makes the radio altitude trigger one sample sooner.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt-750),
                   land)
        expected = [Section(name='Approach And Landing', slice=slice(2.5, 20, None))]
        self.assertEqual(app, expected)
    

class TestILSLocalizerEstablished(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Approach And Landing',
                     'Approach And Landing Lowest Point',
                     'ILS Localizer')]
        opts = ILSLocalizerEstablished.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_ils_localizer_established_basic(self):
        aal = S('Approach And Landing', items=[Section('Approach And Landing', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.arange(-3,0,0.3))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = [Section('ILS Localizer Established', slice(2, 10, None))]
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_not_on_loc_at_minimum(self):
        aal = S('Approach And Landing', items=[Section('Approach And Landing', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.array([3]*10))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = []
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_only_last_segment(self):
        aal = S('Approach And Landing', items=[Section('Approach And Landing', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,3,3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = [Section('ILS Localizer Established', slice(6, 10, None))]
        self.assertEqual(establish, expected)


"""
class TestInitialApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach And Landing')]
        opts = InitialApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_approach_phase_basic(self):
        alt = np.ma.array(range(4000,0,-500)+range(0,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach And Landing',
            items=[Section('Approach And Landing', slice(2, 8, None))])
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
        app_land = SectionNode('Approach And Landing',
            items=[Section('Approach And Landing', slice(10, 16, None))])
        app.derive(alt_aal, app_land)
        expected = [Section(name='Initial Approach', slice=slice(10, 14, None))]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_with_go_around(self):
        alt = np.ma.array(range(4000,2000,-500)+range(2000,4000,500))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach And Landing', 
            items=[Section('Approach And Landing', slice(2, 5, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section(name='Initial Approach', slice=slice(2, 4, None))]
        self.assertEqual(app, expected)
"""

class TestClimbCruiseDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude For Climb Cruise Descent',)]
        opts = ClimbCruiseDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_cruise_descent_basic(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        # plot_parameter (testwave)
        camel.derive(Parameter('Altitude For Climb Cruise Descent', np.ma.array(testwave)))
        self.assertEqual(len(camel), 2)


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
        expected = [('Rate Of Climb For Flight Phases', 'Fast')]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        rate_of_climb_data = np.ma.array(range(500,1200,100)+
                                         range(1200,-1200,-200)+
                                         range(-1200,500,100))
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array(rate_of_climb_data))
        fast = SectionNode('Fast', item=[Section('Fast',slice(2,8,None))])
        up = Climbing()
        up.derive(rate_of_climb, fast)
        expected = [Section(name='Climbing', slice=slice(3, 8, None))]
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
        alt = Parameter('Altitude STD', alt_data)
        
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        
        # With this test waveform, the peak at 31:32 is just flat enough
        # for the climb and descent to be a second apart, whereas the peak
        # at 94 genuinely has no interval with a level cruise.
        expected = SectionNode('Cruise',
            items=[Section(name='Cruise', slice=slice(31, 32, None)),
                   Section(name='Cruise', slice=slice(94, 94, None))])
        self.assertEqual(test_phase, expected)

    def test_cruise_truncated_start(self):
        alt_data = np.ma.array([15000]*5+range(15000,12000,-1000))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = [Section(name='Cruise', slice=slice(0, 5, None))]
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 0)
        self.assertEqual(len(tod), 1)

    def test_cruise_truncated_end(self):
        alt_data = np.ma.array(range(35000,36000,100)+[36000]*4)
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod)
        #===========================================================
        expected = [Section(name='Cruise', slice=slice(10, 14, None))]
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 1)
        self.assertEqual(len(tod), 0)


class TestDescentLowClimb(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentLowClimb.get_operational_combinations(),
            [('Altitude AAL For Flight Phases', 'Climb For Flight Phases',
              'Landing', 'Fast')])

    def test_descent_low_climb_inadequate_climb(self):
        # This test will find out if we can separate the two humps on this camel
        dlc = DescentLowClimb()
        testwave = np.cos(np.arange(0,12.6,0.1))*(100)+3000
        alt = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        clb = Parameter('Climb For Flight Phases', np.ma.array([0]*len(alt.array)))
        land = SectionNode('Landing') # items=[Section(name='Landing',slice=None)])
        fast = SectionNode('Fast',
                           items=[Section(name='Fast',slice=slice(0,len(testwave)))])        
        dlc.derive(alt, clb, land, fast)
        self.assertEqual(len(dlc), 0)

    def test_descent_low_climb_with_climbs(self):
        # This test will find out if we can separate the two humps on this camel
        dlc = DescentLowClimb()
        testwave = np.cos(np.arange(0,12.6,0.1))*(1000)+3000
        alt = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        clb = Parameter('Climb For Flight Phases', np.ma.array([1000]*len(alt.array)))
        land = SectionNode('Landing') # [Section(name='Landing',slice=None)])
        fast = SectionNode('Fast',
                           items=[Section(name='Fast',slice=slice(0,len(testwave)))])
        dlc.derive(alt, clb, land, fast)
        self.assertEqual(len(dlc), 2)

    def test_descent_low_climb_with_one_climb(self):
        # This test will find out if we can separate the two humps on this camel
        dlc = DescentLowClimb()
        testwave = np.cos(np.arange(0,12.6,0.1))*(1000)+3000
        alt = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        clb = Parameter('Climb For Flight Phases', np.ma.array([0]*63+[600]*63))
        land = SectionNode('Landing') # items=[Section(name='Landing',slice=None)])
        fast = SectionNode('Fast',
                           items=[Section(name='Fast',slice=slice(0,len(testwave)))])        
        dlc.derive(alt, clb, land, fast)
        self.assertEqual(len(dlc), 1)


class TestDescending(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb For Flight Phases', 'Fast')]
        opts = Descending.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descending_basic(self):
        roc = Parameter('Rate Of Climb For Flight Phases',np.ma.array([0,1000,-600,-800,0]))
        fast = SectionNode('Fast', items=[Section('Fast',slice(1,5,None))])
        phase = Descending()
        phase.derive(roc,fast)
        expected = [Section('Descending',slice(2,4))]
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
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = [Section(name='Fast',slice=slice(2,11,None))]
        if AIRSPEED_THRESHOLD == 70:
            expected = [Section(name='Fast', slice=slice(1, 12, None))]
        self.assertEqual(phase_fast, expected)
        
    def test_fast_phase_with_small_mask(self):
        slow_and_fast_data = np.ma.concatenate([np.ma.arange(60,120,10),
                                                np.ma.arange(120,50,-10)])
        slow_and_fast_data[5:8] = np.ma.masked
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = [Section(name='Fast',slice=slice(2,11,None))]
        if AIRSPEED_THRESHOLD == 70:
            expected = [Section(name='Fast', slice=slice(1, 12, None))]
        self.assertEqual(phase_fast, expected)


    def test_fast_phase_with_large_mask(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+[120]*8+range(120,50,-10))
        slow_and_fast_data[5:17] = np.ma.masked
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = [Section(name='Fast',slice=slice(2,5,None)),
                        Section(name='Fast',slice=slice(17,19,None))]
        if AIRSPEED_THRESHOLD == 70:
            expected = [Section(name='Fast', slice=slice(1, 5, None)), 
                        Section(name='Fast', slice=slice(17, 20, None))]
        self.assertEqual(phase_fast, expected)


class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases',
                     'Approach And Landing')]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40))
        app = FinalApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        alt_radio = Parameter('Altitude Radio For Flight Phases', array=alt)
        app_land = SectionNode('Approach And Landing',
            items=[Section('Approach And Landing', slice(0, -1, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, alt_radio, app_land)
        expected = [Section('Final Approach', slice(17, 30, None))]
        self.assertEqual(app, expected)

    def test_approach_phase_starting_inside_phase_and_with_go_around(self):
        alt = np.ma.array(range(400,300,-50)+range(300,500,50))
        app = FinalApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        alt_radio = Parameter('Altitude Radio For Flight Phases', array=alt)
        app_land = SectionNode('Approach And Landing',
            items=[Section('Approach And Landing', slice(0, 3, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, alt_radio, app_land)
        expected = [Section(name='Final Approach', slice=slice(0, 2, None))]
        self.assertEqual(app, expected)


class TestLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Landing.get_operational_combinations(),
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast'),
             ('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',
              'Altitude Radio For Phases')])

    def test_landing_basic(self):
        head = np.ma.array([20]*8+[10,0])
        ias  = np.ma.array([110]*4+[80,50,30,20,10,10])
        alt_aal = np.ma.array([80,40,20,5]+[0]*6)
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast, None)
        expected = [Section(name='Landing', slice=slice(0.75, 8.5, None))]
        self.assertEqual(landing, expected)
        
    def test_landing_with_rad_alt(self):
        head = np.ma.array([20]*8+[10,0])
        ias  = np.ma.array([110]*4+[80,50,30,20,10,10])
        alt_aal = np.ma.array([80,40,20,5]+[0]*6)
        alt_rad = alt_aal - 4
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast,
                       P('Altitude Radio For Phases',alt_rad))
        expected = [Section(name='Landing', slice=slice(0.65, 8.5, None))]
        self.assertEqual(landing, expected)
        
    def test_landing_turnoff(self):
        head = np.ma.array([20]*15+[10]*13)
        ias  = np.ma.array([110]*4+[80,50,40,30,20]+[10]*21)
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast, None)
        expected = [Section(name='Landing', slice=slice(0.75, 14, None))]
        self.assertEqual(list(landing), expected)
        
    def test_landing_turnoff_left(self):
        head = np.ma.array([4]*15+[-5]*13)
        ias  = np.ma.array([110]*4+[80,50,40,30,20]+[10]*21)
        alt_aal = np.ma.array([80,40,20,5]+[0]*26)
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        landing = Landing()
        landing.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast, None)
        expected = [Section(name='Landing', slice=slice(0.75, 14, None))]
        self.assertEqual(list(landing), expected)
        
        
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


'''
# OnGround has been commented out in flight_phase.py
class TestOnGround(unittest.TestCase):
    # Based simply on moving too slowly to be airborne.
    # Keeping to minimum number of validated sensors makes this robust logic.
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = OnGround.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_onground_basic(self):
        slow_and_fast_data = np.ma.concatenate([np.ma.arange(60,120,10),
                                        np.ma.arange(120,50,-10)])
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_onground = OnGround()
        phase_onground.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = [Section(name='On Ground',slice=slice(2,10,None))]
        if AIRSPEED_THRESHOLD == 70:
            expected = [Section(name='On Ground',slice=slice(1,11,None))]
        self.assertEqual(phase_onground, expected)
'''


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Takeoff.get_operational_combinations(),
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast'),
             ('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',
              'Altitude Radio For Phases')])

    def test_takeoff_basic(self):
        head = np.ma.array([ 0,10,20,20,20,20,20,20,20,20])
        ias  = np.ma.array([10,10,10,10,40,70,100,105,110,110])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,10,30,70])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal),
                       phase_fast, None) #  No Rad Alt in this basic case
        expected = Section(name='Takeoff', slice=slice(0.5, 8.125, None))
        self.assertEqual(takeoff[0], expected)
        
    def test_takeoff_with_rad_alt(self):
        head = np.ma.array([ 0,10,20,20,20,20,20,20,20,20])
        ias  = np.ma.array([10,10,10,10,40,70,100,105,110,110])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,10,30,70])
        alt_rad = alt_aal - 5
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed',ias))
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases',alt_aal), phase_fast, 
                       P('Altitude Radio',alt_rad))
        expected = Section(name='Takeoff', slice=slice(0.5, 8.25, None))
        self.assertEqual(takeoff[0], expected)
    
    def test_takeoff_with_zero_slices(self):
        '''
        A zero slice is causing the derive method to raise an exception.
        '''
        self.assertFalse(True)


class TestTurning(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn', 'Airborne')]
        opts = Turning.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_turning_phase_basic(self):
        rate_of_turn_data = np.arange(-4, 4.4, 0.4)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = S('Airborne')
        turning = Turning()
        turning.derive(rate_of_turn, airborne)
        expected = [Section(name='Turning On Ground', slice=slice(0, 7, None)),
                    Section(name='Turning On Ground', slice=slice(14, 21, None))]
        self.assertEqual(turning, expected)
        
    def test_turning_phase_basic_masked_not_turning(self):
        airborne = S('Airborne', items=[Section('Airborne', slice(13, 22))])
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = Turning()
        turning.derive(rate_of_turn, airborne)
        expected = [Section(name='Turning On Ground', slice=slice(0, 7, None)),
                    Section(name='Turning In Air', slice=slice(14, 21, None))]
        self.assertEqual(turning, expected)
        
    def test_turning_phase_basic_masked_while_turning(self):
        airborne = S('Airborne',
                     items=[Section(name='Airborne', slice=slice(0, 8))])
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[1] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = Turning()
        turning.derive(rate_of_turn, airborne)
        expected = [Section(name='Turning In Air', slice=slice(0, 7, None)),
                    Section(name='Turning On Ground', slice=slice(14, 21, None))]
        self.assertEqual(turning, expected)
