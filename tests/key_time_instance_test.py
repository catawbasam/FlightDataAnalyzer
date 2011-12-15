import unittest
import numpy as np

from analysis.library import rate_of_change
from analysis.plot_flight import plot_parameter
from analysis.node import A, KPV, KeyTimeInstance, KTI, Parameter, P, Section, S
from analysis.flight_phase import (Airborne,
                                   ClimbCruiseDescent,
                                   Climbing,
                                   DescentLowClimb,
                                   Fast
                                   )
from analysis.derived_parameters import (ClimbForFlightPhases,
                                         )
from analysis.key_time_instances import (BottomOfDescent,
                                         ClimbStart,
                                         GoAround,
                                         InitialClimbStart,
                                         LandingStart,
                                         LandingTurnOffRunway,
                                         Liftoff,
                                         TakeoffStartAcceleration,
                                         TakeoffTurnOntoRunway,
                                         TopOfClimb,
                                         TopOfDescent,
                                         Touchdown
                                         )

import sys
debug = sys.gettrace() is not None

class TestBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-2500)+12500
        alt_ph = Parameter('Altitude For Flight Phases', np.ma.array(testwave))
        alt_std = Parameter('Altitude STD', np.ma.array(testwave))
        dlc = DescentLowClimb()
        dlc.derive(alt_ph)
        bod = BottomOfDescent()
        bod.derive(dlc, alt_std)    
        expected = [KeyTimeInstance(index=63, state='Bottom Of Descent')]        
        self.assertEqual(bod, expected)
        
        
class TestClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_climb_start_basic(self):
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc)
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        # These values give an result with an index of 4.5454 recurring.
        expected = [KeyTimeInstance(index=5/1.1, state='Climb Start')]
        self.assertEqual(kpi, expected)


class TestGoAround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases',
                     'Fast','Climb For Flight Phases')]
        opts = GoAround.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_go_around_basic(self):
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,1000,501))
        ias = Parameter('Airspeed', np.ma.ones(len(alt))*100)
        phase_fast = Fast()
        phase_fast.derive(ias)
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), phase_fast)

        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio',alt),
                   phase_fast, climb)
        expected = [KeyTimeInstance(index=16, state='Go Around')]
        self.assertEqual(goa, expected)

    def test_multiple_go_arounds(self):
        alt = np.ma.array(np.cos(np.arange(0,20,0.02))*(1000)+2500)
        if debug:
            plot_parameter(alt)
        # rate_of_change takes a complete parameter, but only returns the 
        # differentiated array.
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.ones(len(alt))*100))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), phase_fast)
        
        goa = GoAround()
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   phase_fast, climb)
                   
        expected = [KeyTimeInstance(index=157, state='Go Around'), 
                    KeyTimeInstance(index=471, state='Go Around'), 
                    KeyTimeInstance(index=785, state='Go Around')]
        self.assertEqual(goa, expected)

    def test_go_around_insufficient_climb(self):
        # 500 ft climb is not enough to trigger the go-around. 
        # Compare to 501 ft for the "basic" test.
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,700,499))
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.ones(len(alt))*100))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), phase_fast)
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio',alt),
                   phase_fast, climb)
        expected = []
        self.assertEqual(goa, expected)


class TestInitialClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = InitialClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_climb_start_basic(self):
        instance = InitialClimbStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Takeoff',slice(0,3.5,None))])
        expected = [KeyTimeInstance(index=3.5, state='Initial Climb Start')]
        self.assertEqual(instance, expected)


class TestLiftoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Rate Of Climb')]
        opts = Liftoff.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between 
        # the 5th and 6th sample points.
        rate_of_climb = Parameter('Rate Of Climb', np.ma.arange(10)-0.5)
        # Takeoff section encloses the test point.
        takeoff = [Section('Takeoff',slice(0,9,None))]
        lift = Liftoff()
        lift.derive(takeoff, rate_of_climb)
        expected = [KeyTimeInstance(index=5.5, state='Liftoff')]
        self.assertEqual(lift, expected)
    

class TestTakeoffTurnOntoRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = TakeoffTurnOntoRunway.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_turn_onto_runway_basic(self):
        instance = TakeoffTurnOntoRunway()
        # This just needs the takeoff slice startpoint, so trivial to test
        takeoff = [Section('Takeoff',slice(1.7,3.5,None))]
        instance.derive(takeoff)
        expected = [KeyTimeInstance(index=1.7, state='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)


class TestTakeoffStartAcceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Acceleration Longitudinal')]
        opts = TakeoffStartAcceleration.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_start_acceleration(self):
        instance = TakeoffStartAcceleration()
        # This just needs the takeoff slice startpoint, so trivial to test
        takeoff = [Section('Takeoff',slice(1,5,None))]
        accel = P('AccelerationLongitudinal',np.ma.array([0,0,0,0.2,0.3,0.3,0.3]))
        instance.derive(takeoff,accel)
        expected = [KeyTimeInstance(index=2.5, state='Takeoff Start Acceleration')]
        self.assertEqual(instance, expected)

    
class TestTopOfClimb(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD','Climb Cruise Descent')]
        opts = TopOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_climb_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, state='Top Of Climb')]
        self.assertEqual(phase, expected)

    def test_top_of_climb_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)

    def test_top_of_climb_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, state='Top Of Climb')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)


class TestTopOfDescent(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD','Climb Cruise Descent')]
        opts = TopOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_descent_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=13, state='Top Of Descent')]
        self.assertEqual(phase, expected)

    def test_top_of_descent_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, state='Top Of Descent')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)

    def test_top_of_descent_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)
    
        
class TestTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing','Rate Of Climb')]
        opts = Touchdown.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_touchdown_basic(self):
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array([-30,-20,-11,-1,0,0,0]))
        land = [Section('Landing',slice(1,5))]                        
        tdown = Touchdown()
        tdown.derive(land, rate_of_climb)
        # and the real answer is this KTI
        expected = [KeyTimeInstance(index=2.1, state='Touchdown')]
        self.assertEqual(tdown, expected)
    
    
class TestLandingStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_start_basic(self):
        instance = LandingStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None))])
        expected = [KeyTimeInstance(index=66, state='Landing Start')]
        self.assertEqual(instance, expected)


class TestLandingTurnOffRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingTurnOffRunway.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_turn_off_runway_basic(self):
        instance = LandingTurnOffRunway()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None))])
        expected = [KeyTimeInstance(index=77, state='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)


class TestLandingStartDeceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Acceleration Longitudinal')]
        opts = LandingStartDeceleration.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_start_acceleration(self):
        instance = LandingStartDeceleration()
        # This just needs the takeoff slice startpoint, so trivial to test
        takeoff = [Section('Takeoff',slice(1,5,None))]
        accel = P('AccelerationLongitudinal',np.ma.array([0,0,0,0.2,0.3,0.3,0.3]))
        instance.derive(takeoff,accel)
        expected = [KeyTimeInstance(index=2.5, state='Landing Start Deceleration')]
        self.assertEqual(instance, expected)
