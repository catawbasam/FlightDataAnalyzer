import mock
import numpy as np
import sys
import unittest

from analysis.derived_parameters import ClimbForFlightPhases
from analysis.node import KeyTimeInstance, Parameter, P, Section, S
from analysis.flight_phase import (ApproachAndLanding,
                                   ClimbCruiseDescent,
                                   Climbing,
                                   DescentLowClimb,
                                   )
from analysis.key_time_instances import (
    AltitudeInApproach,
    AltitudeInFinalApproach,
    AltitudeWhenClimbing,
    AltitudeWhenDescending,
    BottomOfDescent,
    ClimbStart,
    GoAround,
    InitialClimbStart,
    LandingDecelerationEnd,
    LandingPeakDeceleration,
    LandingStart,
    LandingTurnOffRunway,
    Liftoff,
    TakeoffAccelerationStart,
    TakeoffTurnOntoRunway,
    TopOfClimb,
    TopOfDescent,
    Touchdown,
)

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
        bod.derive(alt_std, dlc)    
        expected = [KeyTimeInstance(index=63, name='Bottom Of Descent')]        
        self.assertEqual(bod, expected)
        
        
class TestClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_climb_start_basic(self):
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc, [Section('Fast',slice(0,8,None))])
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        # These values give an result with an index of 4.5454 recurring.
        expected = [KeyTimeInstance(index=5/1.1, name='Climb Start')]
        self.assertEqual(kpi, expected)


    def test_climb_start_cant_climb_when_slow(self):
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc, []) #  No Fast phase found in this data
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        expected = [] #  Even though the altitude climbed, the a/c can't have
        self.assertEqual(kpi, expected)


class TestGoAround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach And Landing',
                     'Climb For Flight Phases'),
                    ('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases',
                     'Approach And Landing',
                     'Climb For Flight Phases'),
                    ]
        opts = GoAround.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_go_around_basic(self):
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,1000,501))
        aal = [Section('Approach And Landing',slice(10,18))]
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), aal)
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   aal, climb)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_multiple_go_arounds(self):
        # This tests for three go-arounds, but the fourth part of the curve
        # does not produce a go-around as it ends in mid-descent.
        alt = np.ma.array(np.cos(np.arange(0,21,0.02))*(1000)+2500)

        if debug:
            from analysis.plot_flight import plot_parameter
            plot_parameter(alt)
            
        aal = ApproachAndLanding()
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
            
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), 
                     [Section('Fast',slice(0,len(alt),None))])
        
        goa = GoAround()
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   aal, climb)
                   
        expected = [KeyTimeInstance(index=157, name='Go Around'), 
                    KeyTimeInstance(index=471, name='Go Around'), 
                    KeyTimeInstance(index=785, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_go_around_insufficient_climb(self):
        # 500 ft climb is not enough to trigger the go-around. 
        # Compare to 501 ft for the "basic" test.
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,700,499))
        aal = ApproachAndLanding()
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
            
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt),  
                     [Section('Fast',slice(0,len(alt),None))])
        
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio',alt),
                   aal, climb)
        expected = []
        self.assertEqual(goa, expected)

    def test_go_around_no_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,1000,501))
        aal = ApproachAndLanding()

        # Call derive method. Note: "None" required to replace rad alt argument.
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),None) 
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), aal)
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        alt_aal=Parameter('Altitude AAL For Flight Phases',alt)
        
        # !!! None is positional argument in place of alt_rad !!!
        goa.derive(alt_aal, None, aal, climb)
        
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)


class TestAltitudeInApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInApproach.get_operational_combinations(),
                         [('Approach And Landing', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach And Landing', items=[Section('a', slice(4, 7)),
                                                      Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(1950, 0, -200) + \
                                       range(1950, 0, -200)))
        altitude_in_approach = AltitudeInApproach()
        altitude_in_approach.derive(approaches, alt_aal)
        self.assertEqual(list(altitude_in_approach),
          [KeyTimeInstance(index=4.75, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.75, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=12.25, name='1500 Ft In Approach',
                           datetime=None, latitude=None, longitude=None)])


class TestAltitudeInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInFinalApproach.get_operational_combinations(),
                         [('Approach And Landing', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach And Landing',
                       items=[Section('a', slice(2, 7)),
                              Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(950, 0, -100) + \
                                       range(950, 0, -100)))
        altitude_in_approach = AltitudeInFinalApproach()
        altitude_in_approach.derive(approaches, alt_aal)
        
        self.assertEqual(list(altitude_in_approach),
          [KeyTimeInstance(index=4.5,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=18.512820512820511,
                           name='100 Ft In Final Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.5,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None)])


class TestAltitudeWhenClimbing(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeWhenClimbing.get_operational_combinations(),
                         [('Climbing', 'Altitude AAL')])
    
    @mock.patch('analysis.key_time_instances.hysteresis')
    def test_derive(self, hysteresis):
        climbing = S('Climbing', items=[Section('a', slice(4, 10)),
                                        Section('b', slice(12, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(0, 200, 20) + \
                                       range(0, 200, 20),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        # Do not apply hysteresis to simplify testing.
        hysteresis.return_value = alt_aal.array
        altitude_when_climbing = AltitudeWhenClimbing()
        altitude_when_climbing.derive(climbing, alt_aal)
        self.assertEqual(hysteresis.call_args,
            ((alt_aal.array, altitude_when_climbing.HYSTERESIS), {}))
        self.assertEqual(list(altitude_when_climbing),
          [KeyTimeInstance(index=5.0, name='100 Ft Climbing'),
           KeyTimeInstance(index=12.5, name='50 Ft Climbing'),
           KeyTimeInstance(index=13.75, name='75 Ft Climbing'),
           KeyTimeInstance(index=15.0, name='100 Ft Climbing'),
           KeyTimeInstance(index=17.5, name='150 Ft Climbing')])


class TestAltitudeWhenDescending(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeWhenDescending.get_operational_combinations(),
                         [('Descending', 'Altitude AAL')])
    
    @mock.patch('analysis.key_time_instances.hysteresis')
    def test_derive(self, hysteresis):
        descending = S('Descending', items=[Section('a', slice(0, 10)),
                                            Section('b', slice(12, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(100, 0, -10) + \
                                       range(100, 0, -10),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        # Do not apply hysteresis to simplify testing.
        hysteresis.return_value = alt_aal.array
        altitude_when_descending = AltitudeWhenDescending()
        altitude_when_descending.derive(descending, alt_aal)
        self.assertEqual(hysteresis.call_args,
            ((alt_aal.array, altitude_when_descending.HYSTERESIS), {}))
        self.assertEqual(list(altitude_when_descending),
          [KeyTimeInstance(index=5.0, name='50 Ft Descending'),
           KeyTimeInstance(index=2.5, name='75 Ft Descending'), 
           KeyTimeInstance(index=19.0, name='10 Ft Descending'),
           KeyTimeInstance(index=18.0, name='20 Ft Descending'),
           KeyTimeInstance(index=16.5, name='35 Ft Descending'),
           KeyTimeInstance(index=15.0, name='50 Ft Descending'),
           KeyTimeInstance(index=12.5, name='75 Ft Descending')])


class TestInitialClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = InitialClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_climb_start_basic(self):
        instance = InitialClimbStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Takeoff',slice(0,3.5,None))])
        expected = [KeyTimeInstance(index=3.5, name='Initial Climb Start')]
        self.assertEqual(instance, expected)

class TestLandingDecelerationEnd(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Landing')]
        opts = LandingDecelerationEnd.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_start_deceleration(self):
        landing = [Section('Landing',slice(2,40,None))]
        speed = np.ma.array([79,77.5,76,73.9,73,70.3,68.8,67.6,66.4,63.4,62.8,
                             61.6,61.9,61,60.1,56.8,53.8,49.6,47.5,46,44.5,43.6,
                             42.7,42.4,41.8,41.5,40.6,39.7,39.4,38.5,37.9,38.5,
                             38.5,38.8,38.5,37.9,37.9,37.9,37.9,37.9])
        aspd = P('Airspeed',speed)
        kpv = LandingDecelerationEnd()
        kpv.derive(aspd, landing)
        expected = [KeyTimeInstance(index=24, name='Landing Deceleration End')]
        self.assertEqual(kpv, expected)


class TestLandingPeakDeceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous', 'Landing',
                     'Acceleration Longitudinal')]
        opts = LandingPeakDeceleration.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_peak_deceleration_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = P('Acceleration Longitudinal',
                np.ma.array([0,0,-.1,-.1,-.2,-.1,0,0]))
        landing = [Section('Landing',slice(2,5,None))]
        kti = LandingPeakDeceleration()
        kti.derive(head, landing, acc)
        expected = [KeyTimeInstance(index=4, name='Landing Peak Deceleration')]
        self.assertEqual(kti, expected)


class TestLandingStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_start_basic(self):
        instance = LandingStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None))])
        expected = [KeyTimeInstance(index=66, name='Landing Start')]
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
        expected = [KeyTimeInstance(index=77, name='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)


class TestLiftoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb For Flight Phases','Takeoff')]
        opts = Liftoff.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between 
        # the 5th and 6th sample points.
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.arange(10)-0.5)
        # Takeoff section encloses the test point.
        takeoff = [Section('Takeoff',slice(0,9,None))]
        lift = Liftoff()
        lift.derive(rate_of_climb, takeoff)
        expected = [KeyTimeInstance(index=5.5, name='Liftoff')]
        self.assertEqual(lift, expected)
    

class TestTakeoffAccelerationStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed', 'Takeoff')]
        opts = TakeoffAccelerationStart.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_acceleration_start(self):
        # This test uses the same airspeed data as the library routine test,
        # so should give the same answer!
        airspeed_data = np.ma.array([37.9,37.9,37.9,37.9,37.9,38.2,38.2,38.2,
                                     38.2,38.8,38.2,38.8,39.1,39.7,40.6,41.5,
                                     42.7,43.6,44.5,46,47.5,49.6,52,53.2,54.7,
                                     57.4,60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                     74.8])
        takeoff = [Section('Takeoff',slice(3,len(airspeed_data),None))]
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff)
        expected = [KeyTimeInstance(index=16, name='Takeoff Acceleration Start')]
        self.assertEqual(instance, expected)

    
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
        expected = [KeyTimeInstance(index=1.7, name='Takeoff Turn Onto Runway')]
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
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
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
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
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
        expected = [KeyTimeInstance(index=13, name='Top Of Descent')]
        self.assertEqual(phase, expected)

    def test_top_of_descent_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, name='Top Of Descent')]
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
        expected = [('Rate Of Climb For Flight Phases', 'Landing')]
        opts = Touchdown.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_touchdown_basic(self):
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.array([-30,-20,-11,-1,0,0,0]))
        land = [Section('Landing',slice(1,5))]                        
        tdown = Touchdown()
        tdown.derive(rate_of_climb, land)
        # and the real answer is this KTI
        expected = [KeyTimeInstance(index=2.1, name='Touchdown')]
        self.assertEqual(tdown, expected)
    
    
