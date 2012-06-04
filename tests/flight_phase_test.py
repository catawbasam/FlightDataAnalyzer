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
                                          ILSLocalizerEstablished,
                                          Landing,
                                          LevelFlight,
                                          OnGround,
                                          Takeoff,
                                          TurningInAir,
                                          TurningOnGround
                                          )

from analysis_engine.library import integrate

from analysis_engine.settings import AIRSPEED_THRESHOLD


'''
Two little routines to make building Sections for testing easier.
'''
def buildsection(name, begin, end):
    return [Section(name, slice(begin, end), begin, end)]

def buildsections(*args):
    # buildsections('name',[from1,to1],[from2,to2])
    built_list=[]
    name = args[0]
    for a in args[1:]:
        new_section = buildsection(name, a[0], a[1])
        built_list.append(new_section)
    return built_list


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

    def test_airborne_phase_not_airborne(self):
        altitude_data = np.ma.array(range(0,10))
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = []
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(air, [])


"""
class TestApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach')]
        opts = Approach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+range(500,3000,500))
        aal = S('Approach', items=[Section('Approach', slice(4, 14, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt), aal)
        expected = [Section(name='Approach', slice=slice(4, 9, None))]
        self.assertEqual(app, expected)
"""


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
        """
    

class TestApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Approach.get_operational_combinations(),
                         [('Altitude AAL For Flight Phases', 'Altitude Radio',
                           'Landing')])

    def test_approach_and_landing_phase_basic(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        app = Approach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   land)
        expected = [Section(name='Approach', slice=slice(4, 20, None))]
        self.assertEqual(app, expected)

    def test_approach_and_landing_phase_no_ralt(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        alt_param = Parameter('Altitude AAL For Flight Phases',alt)
        app = Approach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_param, None, land)
        expected = [Section(name='Approach', slice=slice(4, 20, None))]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_over_high_ground(self):
        alt = np.ma.array(range(5000,500,-500)+[0]*10)
        land=S(items=[Section('Landing',slice=slice(11,20))])
        app = Approach()
        # Raising the ground makes the radio altitude trigger one sample sooner.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt-750),
                   land)
        expected = [Section(name='Approach', slice=slice(2.5, 20, None))]
        self.assertEqual(app, expected)
    

class TestILSLocalizerEstablished(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Approach',
                     'Approach And Go Around',
                     'ILS Localizer')]
        opts = ILSLocalizerEstablished.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_ils_localizer_established_basic(self):
        # TODO: Either fix test by passing in Approach And Go Around instead of None or remove.
        aal = S('Approach', items=[Section('Approach', slice(2, 9, None))])
        ils = P('ILS Localizer',np.ma.arange(-3,0,0.3))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, None, ils)
        expected = [Section('ILS Localizer Established', slice(2, 10, None))]
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_not_on_loc_at_minimum(self):
        # TODO: Fix test by passing in Approach And Go Around SectionNode instead of Approach And Landing Lowest Point KTI.
        aal = S('Approach',
                items=[Section('Approach', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point',
                  items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.array([3]*10))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = []
        self.assertEqual(establish, expected)

    def test_ils_localizer_established_only_last_segment(self):
        # TODO: Fix test by passing in Approach And Go Around SectionNode instead of Approach And Landing Lowest Point KTI.
        aal = S('Approach', items=[Section('Approach', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.array([0,0,0,1,3,3,2,1,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = [Section('ILS Localizer Established', slice(6, 10, None))]
        self.assertEqual(establish, expected)

    def test_ils_localizer_insensitive_to_few_masked_values(self):
        # TODO: Fix test by passing in Approach And Go Around SectionNode instead of Approach And Landing Lowest Point KTI.
        aal = S('Approach', items=[Section('Approach', slice(2, 9, None))])
        low = KTI('Approach And Landing Lowest Point', items=[KeyTimeInstance(index=8, name='Approach And Landing Lowest Point')])
        ils = P('ILS Localizer',np.ma.array(data=[0,0,0,1,3,3,2,1,0,0],
                                            mask=[0,0,0,0,0,1,1,0,0,0]))
        establish = ILSLocalizerEstablished()
        establish.derive(aal, low, ils)
        expected = [Section('ILS Localizer Established', slice(6, 10, None))]
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
        expected = [('Altitude For Climb Cruise Descent',)]
        opts = ClimbCruiseDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_cruise_descent_basic(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0,12.6,0.1))*(-3000)+12500
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
                         [('Airspeed For Flight Phases',)])

    def test_on_ground_phase_basic(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+[120]*300+range(120,50,-10))
        ias = Parameter('Airspeed For Flight Phases', slow_and_fast_data,1,0)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias)
        if AIRSPEED_THRESHOLD == 80:
            expected = buildsections('On Ground',[0,3],[310,313])
        if AIRSPEED_THRESHOLD == 70:
            expected = buildsections('On Ground',[0,3],[56,0]) # Not set up.
        self.assertEqual(phase_on_ground.get_first(), expected[0])
        self.assertEqual(phase_on_ground.get_last(), expected[1])
        
    def test_on_ground_all_fast(self):
        on_ground_data = np.ma.array([120]*10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias)
        self.assertEqual(phase_on_ground.get_first(), None)

    def test_on_ground_all_slow(self):
        on_ground_data = np.ma.array([12]*10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias)
        expected = buildsection('On Ground',0,10)
        self.assertEqual(phase_on_ground.get_first(), expected)

    def test_on_ground_slowing_only(self):
        on_ground_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias)
        expected = buildsection('On Ground',3,5)
        self.assertEqual(phase_on_ground.get_first(), expected)
        
    def test_on_ground_speeding_only(self):
        on_ground_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed For Flight Phases', on_ground_data,1,0)
        phase_on_ground = OnGround()
        phase_on_ground.derive(ias)
        expected = buildsection('On Ground',0,3)
        self.assertEqual(phase_on_ground.get_first(), expected)





class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases',
                     'Approach')]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40))
        app = FinalApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        alt_radio = Parameter('Altitude Radio For Flight Phases', array=alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(0, -1, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, alt_radio, app_land)
        expected = [Section('Final Approach', slice(17, 30, None))]
        self.assertEqual(app, expected)

    def test_approach_phase_starting_inside_phase_and_with_go_around(self):
        alt = np.ma.array(range(400,300,-50)+range(300,500,50))
        app = FinalApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        alt_radio = Parameter('Altitude Radio For Flight Phases', array=alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(0, 3, None))])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, alt_radio, app_land)
        expected = [Section(name='Final Approach', slice=slice(0, 2, None))]
        self.assertEqual(app, expected)


class TestLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Landing.get_operational_combinations(),
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast'),
             ('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',
              'Altitude Radio For Flight Phases')])

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
                       P('Altitude Radio For Flight Phases',alt_rad))
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
            [('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast'),
             ('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',
              'Altitude Radio For Flight Phases')])

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
        self.assertEqual([turning_in_air.get_first()], expected[0])
        self.assertEqual([turning_in_air.get_last()], expected[1])
        
    def test_turning_in_air_phase_with_mask(self):
        rate_of_turn_data = np.arange(-4, 4.4, 0.4)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 4],[17,21])
        self.assertEqual([turning_in_air.get_first()], expected[0])
        self.assertEqual([turning_in_air.get_last()], expected[1])

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
        self.assertEqual([turning_on_ground.get_first()], expected[0])
        self.assertEqual([turning_on_ground.get_last()], expected[1])
        
    def test_turning_on_ground_phase_with_mask(self):
        rate_of_turn_data = np.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, airborne)
        expected = buildsections('Turning On Ground',[0, 4],[21,24])
        self.assertEqual([turning_on_ground.get_first()], expected[0])
        self.assertEqual([turning_on_ground.get_last()], expected[1])

    def test_turning_on_ground_after_takeoff_inhibited(self):
        rate_of_turn_data = np.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,10)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, airborne)
        expected = buildsections('Turning On Ground',[0, 4])
        self.assertEqual([turning_on_ground.get_first()], expected[0])
        self.assertEqual([turning_on_ground.get_last()], expected[0])
