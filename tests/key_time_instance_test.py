import mock
import numpy as np
import sys
import unittest

from analysis_engine.node import (KeyTimeInstance, Parameter, P, Section, S, M)

from analysis_engine.flight_phase import Climbing

from analysis_engine.key_time_instances import (
    AltitudePeak,
    AltitudeWhenClimbing,
    AltitudeWhenDescending,
    AutopilotDisengagedSelection,
    AutopilotEngagedSelection,
    AutothrottleDisengagedSelection,
    AutothrottleEngagedSelection,
    BottomOfDescent,
    ClimbStart,
    EnterHold,
    ExitHold,
    GearDownSelection,
    GearUpSelection,
    GoAround,
    GoAroundFlapRetracted,
    FlapStateChanges,
    InitialClimbStart,
    LandingDecelerationEnd,
    ##LandingPeakDeceleration,
    LandingStart,
    LandingTurnOffRunway,
    Liftoff,
    LocalizerEstablishedEnd,
    LocalizerEstablishedStart,
    LowestPointOnApproach,
    MinsToTouchdown,
    SecsToTouchdown,
    TakeoffAccelerationStart,
    TakeoffPeakAcceleration,
    TakeoffTurnOntoRunway,
    TopOfClimb,
    TopOfDescent,
    TouchAndGo,
    Touchdown,
    Transmit,
)

from flight_phase_test import buildsection, buildsections

debug = sys.gettrace() is not None


class TestAltitudePeak(unittest.TestCase):
    def setUp(self):
        self.alt_aal = P(name='Altitude AAL', array=np.ma.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
        ]))

    def test_can_operate(self):
        expected = [('Altitude AAL',)]
        self.assertEqual(AltitudePeak.get_operational_combinations(), expected)

    def test_derive(self):
        alt_peak = AltitudePeak()
        alt_peak.derive(self.alt_aal)
        expected = [KeyTimeInstance(name='Altitude Peak', index=9)]
        self.assertEqual(alt_peak, expected)


class TestBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases', 
                     'Descent Low Climb', 'Airborne')]
        opts = BottomOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,6.3,0.1))*(2500)+2560
        alt_aal = Parameter('Altitude AAL For Flight Phases', np.ma.array(testwave))
        dlc = buildsection('Descent Low Climb', 14, 50) # See dlc flight phase test.
        air = buildsection('Airborne', 0,50)
        bod = BottomOfDescent()
        bod.derive(alt_aal, dlc, air)    
        expected = [KeyTimeInstance(index=31, name='Bottom Of Descent'),
                    KeyTimeInstance(index=50, name='Bottom Of Descent')]        
        self.assertEqual(bod, expected)
        
        
class TestClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_climb_start_basic(self):
        vert_spd = Parameter('Vertical Speed', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(vert_spd, [Section('Fast',slice(0,8,None),0,8)])
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        # These values give an result with an index of 4.5454 recurring.
        expected = [KeyTimeInstance(index=5/1.1, name='Climb Start')]
        self.assertEqual(kpi, expected)


    def test_climb_start_cant_climb_when_slow(self):
        vert_spd = Parameter('Vertical Speed', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(vert_spd, []) #  No Fast phase found in this data
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        expected = [] #  Even though the altitude climbed, the a/c can't have
        self.assertEqual(kpi, expected)


class TestGoAround(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GoAround.get_operational_combinations(),
                    [('Descent Low Climb', 'Altitude AAL For Flight Phases'),
                     ('Descent Low Climb', 'Altitude AAL For Flight Phases', 'Altitude Radio')])

    def test_go_around_basic(self):
        dlc = [Section('Descent Low Climb',slice(10,18),10,18)]
        alt = Parameter('Altitude AAL',\
                        np.ma.array(range(0,4000,500)+\
                                    range(4000,0,-500)+\
                                    range(0,1000,501)))
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(dlc,alt,alt)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_multiple_go_arounds(self):
        # This tests for three go-arounds, but the fourth part of the curve
        # does not produce a go-around as it ends in mid-descent.
        alt = Parameter('Altitude AAL',
                        np.ma.array(np.cos(
                            np.arange(0,21,0.02))*(1000)+2500))
        if debug:
            from analysis_engine.plot_flight import plot_parameter
            plot_parameter(alt.array)
            
        dlc = buildsections('Descent Low Climb',[50,260],[360,570],[670,890])
        
        ## Merge with analysis_engine refactoring
            #from analysis_engine.plot_flight import plot_parameter
            #plot_parameter(alt)
            
        #aal = ApproachAndLanding()
        #aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   #Parameter('Altitude Radio For Flight Phases',alt))
            
        #climb = ClimbForFlightPhases()
        #climb.derive(Parameter('Altitude STD Smoothed', alt), 
                     #[Section('Fast',slice(0,len(alt),None))])
        
        goa = GoAround()
        goa.derive(dlc,alt,alt)
                   
        expected = [KeyTimeInstance(index=157, name='Go Around'), 
                    KeyTimeInstance(index=471, name='Go Around'), 
                    KeyTimeInstance(index=785, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_go_around_no_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        dlc = [Section('Descent Low Climb',slice(10,18),10,18)]
        alt = Parameter('Altitude AAL',\
                        np.ma.array(range(0,4000,500)+\
                                    range(4000,0,-500)+\
                                    range(0,1000,501)))
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(dlc,alt,None)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)


    def test_go_around_with_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        alt = Parameter('Altitude AAL',
                        np.ma.array(np.cos(
                            np.arange(0,21,0.02))*(1000)+2500))
        alt_rad = Parameter('Altitude Radio',\
                        alt.array-range(len(alt.array)))
        if debug:
            from analysis_engine.plot_flight import plot_parameter
            plot_parameter(alt_rad.array)
        # The sloping graph has shifted minima. We only need to check one to
        # show it's using the rad alt signal.
        dlc = [Section('Descent Low Climb',slice( 50,260),50,260)]

        goa = GoAround()
        goa.derive(dlc,alt,alt_rad)
        expected = [KeyTimeInstance(index=160, name='Go Around')]
        self.assertEqual(goa, expected)



"""
class TestAltitudeInApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInApproach.get_operational_combinations(),
                         [('Approach', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach', items=[Section('a', slice(4, 7)),
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
"""

"""
class TestAltitudeInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInFinalApproach.get_operational_combinations(),
                         [('Approach', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach',
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
"""

class TestAltitudeWhenClimbing(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeWhenClimbing.get_operational_combinations(),
                         [('Climbing', 'Altitude AAL')])
    
    @mock.patch('analysis_engine.key_time_instances.hysteresis')
    def test_derive(self, hysteresis):
        climbing = S('Climbing', items=[Section('a', slice(4, 10), 4, 10),
                                        Section('b', slice(12, 20), 12, 20)])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(0, 200, 20) + \
                                       range(0, 200, 20),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        # Do not apply hysteresis to simplify testing.
        hysteresis.return_value = alt_aal.array
        altitude_when_climbing = AltitudeWhenClimbing()
        altitude_when_climbing.derive(climbing, alt_aal)
        hysteresis.assert_called_once_with(alt_aal.array,
                                           altitude_when_climbing.HYSTERESIS)
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

    def test_derive(self):
        descending = buildsections('Descending', [0, 10], [11, 20])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(100, 0, -10),
                                       mask=[False] * 6 + [True] * 3 + [False]))
        altitude_when_descending = AltitudeWhenDescending()
        altitude_when_descending.derive(descending, alt_aal)
        self.assertEqual(list(altitude_when_descending),
          [KeyTimeInstance(index=2.5, name='75 Ft Descending'),
           KeyTimeInstance(index=5.0, name='50 Ft Descending'),
        ])


class TestInitialClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = InitialClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_climb_start_basic(self):
        instance = InitialClimbStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Takeoff',slice(0,4,None),0,3.5)])
        expected = [KeyTimeInstance(index=3.5, name='Initial Climb Start')]
        self.assertEqual(instance, expected)

class TestLandingDecelerationEnd(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Landing')]
        opts = LandingDecelerationEnd.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_end_deceleration(self):
        landing = [Section('Landing',slice(2,40,None),2,40)]
        speed = np.ma.array([79,77.5,76,73.9,73,70.3,68.8,67.6,66.4,63.4,62.8,
                             61.6,61.9,61,60.1,56.8,53.8,49.6,47.5,46,44.5,43.6,
                             42.7,42.4,41.8,41.5,40.6,39.7,39.4,38.5,37.9,38.5,
                             38.5,38.8,38.5,37.9,37.9,37.9,37.9,37.9])
        aspd = P('Airspeed',speed)
        kpv = LandingDecelerationEnd()
        kpv.derive(aspd, landing)
        expected = [KeyTimeInstance(index=21.0, name='Landing Deceleration End')]
        self.assertEqual(kpv, expected)


class TestTakeoffPeakAcceleration(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffPeakAcceleration.get_operational_combinations(),
                         [('Takeoff', 'Acceleration Longitudinal')])
        
    def test_takeoff_peak_acceleration_basic(self):
        acc = P('Acceleration Longitudinal',
                np.ma.array([0,0,.1,.1,.2,.1,0,0]))
        landing = [Section('Takeoff',slice(2,5,None),2,5)]
        kti = TakeoffPeakAcceleration()
        kti.derive(landing, acc)
        expected = [KeyTimeInstance(index=4, name='Takeoff Peak Acceleration')]
        self.assertEqual(kti, expected)


class TestLandingStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_start_basic(self):
        instance = LandingStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None),66,77)])
        expected = [KeyTimeInstance(index=66, name='Landing Start')]
        self.assertEqual(instance, expected)


class TestLandingTurnOffRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous','Landing','Fast')]
        opts = LandingTurnOffRunway.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_landing_turn_off_runway_basic(self):
        instance = LandingTurnOffRunway()
        head = P('Heading Continuous', np.ma.array([0]*30))
        fast = buildsection('Fast', 0, 20)
        land = buildsection('Landing', 10, 26)
        instance.derive(head, land, fast)
        expected = [KeyTimeInstance(index=26, name='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)

    def test_landing_turn_off_runway_curved(self):
        instance = LandingTurnOffRunway()
        head = P('Heading Continuous',np.ma.array([0]*70+range(20)))
        fast = buildsection('Fast',0,65)
        land = buildsection('Landing',60,87)
        instance.derive(head, land, fast)
        expected = [KeyTimeInstance(index=73, name='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)

    def test_landing_turn_off_runway_curved_left(self):
        instance = LandingTurnOffRunway()
        head = P('Heading Continuous',np.ma.array([0]*70+range(20))*-1.0)
        fast = buildsection('Fast',0,65)
        land = buildsection('Landing',60,87)
        instance.derive(head, land, fast)
        expected = [KeyTimeInstance(index=73, name='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)


class TestLiftoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Liftoff.get_operational_combinations(),
                         [('Airborne',),
                          ('Vertical Speed Inertial', 'Airborne')])

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between 
        # the 5th and 6th sample points.
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.arange(10) - 0.5) * 40)
        # Airborne section encloses the test point.
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(vert_spd, airs)
        expected = [KeyTimeInstance(index=5.5, name='Liftoff')]
        self.assertEqual(lift, expected)
    
    def test_liftoff_no_vert_spd_detected(self):
        # Check the backstop setting.
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.array([0] * 40)))
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(vert_spd, airs)
        expected = [KeyTimeInstance(index=6, name='Liftoff')]
        self.assertEqual(lift, expected)
    
    def test_liftoff_already_airborne(self):
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.array([0] * 40)))
        airs = buildsection('Airborne', None, 10)
        lift=Liftoff()
        lift.derive(vert_spd, airs)
        expected = []
        self.assertEqual(lift, expected)
        
    
class TestTakeoffAccelerationStart(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(\
            TakeoffAccelerationStart.get_operational_combinations(),
            [('Airspeed', 'Takeoff'),
             ('Airspeed', 'Takeoff', 'Acceleration Longitudinal')])

    def test_takeoff_acceleration_start(self):
        # This test uses the same airspeed data as the library routine test,
        # so should give the same answer!
        airspeed_data = np.ma.array(data=[37.9,37.9,37.9,37.9,37.9,37.9,37.9,
                                          37.9,38.2,38.2,38.2,38.2,38.8,38.2,
                                          38.8,39.1,39.7,40.6,41.5,42.7,43.6,
                                          44.5,46,47.5,49.6,52,53.2,54.7,57.4,
                                          60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                          74.8],
                                    mask=[1]*22+[0]*15
                                    )
        takeoff = buildsection('Takeoff',3,len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        self.assertLess(instance[0].index, 1.0)
        self.assertGreater(instance[0].index, 0.5)

    def test_takeoff_acceleration_start_truncated(self):
        # This test uses the same airspeed data as the library routine test,
        # so should give the same answer!
        airspeed_data = np.ma.array(data=[37.9,37.9,37.9,37.9,37.9,
                                          37.9,38.2,38.2,38.2,38.2,38.8,38.2,
                                          38.8,39.1,39.7,40.6,41.5,42.7,43.6,
                                          44.5,46,47.5,49.6,52,53.2,54.7,57.4,
                                          60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                          74.8],
                                    mask=[1]*20+[0]*15
                                    )
        takeoff = buildsection('Takeoff',3,len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        self.assertEqual(instance[0].index, 0.0)

    
class TestTakeoffTurnOntoRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous','Takeoff','Fast')]
        opts = TakeoffTurnOntoRunway.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_turn_onto_runway_basic(self):
        instance = TakeoffTurnOntoRunway()
        head = P('Heading Continuous',np.ma.arange(5))
        takeoff = buildsection('Takeoff',1.7,5.5)
        fast = buildsection('Fast',3.7,7)
        instance.derive(head, takeoff, fast)
        expected = [KeyTimeInstance(index=1.7, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)

    def test_takeoff_turn_onto_runway_curved(self):
        instance = TakeoffTurnOntoRunway()
        head = P('Heading Continuous',np.ma.array(range(20)+[20]*70))
        fast = buildsection('Fast',40,90)
        takeoff = buildsection('Takeoff',4,75)
        instance.derive(head, takeoff, fast)
        expected = [KeyTimeInstance(index=21.5, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)


class TestTopOfClimb(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Climb Cruise Descent')]
        opts = TopOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_climb_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)

    def test_top_of_climb_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)

    def test_top_of_climb_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)


class TestTopOfDescent(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed', 'Climb Cruise Descent')]
        opts = TopOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_descent_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=13, name='Top Of Descent')]
        self.assertEqual(phase, expected)

    def test_top_of_descent_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, name='Top Of Descent')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)

    def test_top_of_descent_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)


class TestTouchdown(unittest.TestCase):
    def test_can_operate(self):
        opts = Touchdown.get_operational_combinations()
        self.assertTrue(('Gear On Ground',) in opts)
        self.assertTrue(('Vertical Speed Inertial',
                         'Altitude AAL',
                         'Airborne',
                         'Landing',) in opts)

                         #[('Vertical Speed', 'Altitude AAL', 'Airborne',
                           #'Landing')])

    def test_touchdown_basic(self):
        vert_spd = Parameter('Vertical Speed', np.ma.arange(10)*40 - 380.0)
        altitude = Parameter('Altitude AAL',
                             np.ma.array(data=[28.0, 21, 15, 10, 6, 3, 1, 0, 0,  0],
                                         mask = False))
        airs = buildsection('Airborne', 1, 8)
        lands = buildsection('Landing', 2, 9)
        tdwn=Touchdown()
        tdwn.derive(None, vert_spd, altitude, airs, lands)
        expected = [KeyTimeInstance(index=6.7490996398559435, name='Touchdown')]
        self.assertEqual(tdwn, expected)

    def test_touchdown_doesnt_land(self):
        vert_spd = Parameter('Vertical Speed', np.ma.arange(10)*40)
        altitude = Parameter('Altitude AAL',
                             np.ma.array(data=[28, 21, 15, 10, 6, 3, 1, 0, 0,  0],
                                         mask = False))
        airs = buildsection('Airborne', 10, None)
        lands = buildsection('Landing', 2, 9)
        tdwn=Touchdown()
        tdwn.derive(None, vert_spd, altitude, airs, lands)
        expected = []
        self.assertEqual(tdwn, expected)


class TestAutopilotDisengagedSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('AP Engaged', 'Airborne')]
        self.assertEqual(
            AutopilotDisengagedSelection.get_operational_combinations(),
            expected)

    def test_derive(self):
        ap = M('AP Engaged',
               ['Off', 'Off', 'Off', 'Engaged', 'Off', 'Off', 'Off'],
               values_mapping={0: 'Off', 1: 'Engaged'})
        ads = AutopilotDisengagedSelection()
        air = buildsection('Airborne', 2, 5)
        ads.derive(ap, air)
        expected = [KeyTimeInstance(index=3.5, name='AP Disengaged Selection')]
        self.assertEqual(ads, expected)


class TestAutopilotEngagedSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('AP Engaged', 'Airborne')]
        self.assertEqual(
            AutopilotEngagedSelection.get_operational_combinations(),
            expected)

    def test_derive(self):
        ap = M('AP Engaged',
               ['Off', 'Off', 'Off', 'Engaged', 'Off', 'Off', 'Off'],
               values_mapping={0: 'Off', 1: 'Engaged'})
        ads = AutopilotEngagedSelection()
        air = buildsection('Airborne', 2, 5)
        ads.derive(ap, air)
        expected = [KeyTimeInstance(index=2.5, name='AP Engaged Selection')]
        self.assertEqual(ads, expected)


class TestAutothrottleDisengagedSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('AT Engaged', 'Airborne')]
        self.assertEqual(
            AutothrottleDisengagedSelection.get_operational_combinations(),
            expected)

    def test_derive(self):
        ap = M('AT Engaged',
               ['Off', 'Off', 'Off', 'Engaged', 'Off', 'Off', 'Off'],
               values_mapping={0: 'Off', 1: 'Engaged'})
        ads = AutothrottleDisengagedSelection()
        air = buildsection('Airborne', 2, 5)
        ads.derive(ap, air)
        expected = [KeyTimeInstance(index=3.5, name='AT Disengaged Selection')]
        self.assertEqual(ads, expected)


class TestAutothrottleEngagedSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('AT Engaged', 'Airborne')]
        self.assertEqual(
            AutothrottleEngagedSelection.get_operational_combinations(),
            expected)

    def test_derive(self):
        ap = M('AT Engaged',
               ['Off', 'Off', 'Off', 'Engaged', 'Off', 'Off', 'Off'],
               values_mapping={0: 'Off', 1: 'Engaged'})
        ads = AutothrottleEngagedSelection()
        air = buildsection('Airborne', 2, 5)
        ads.derive(ap, air)
        expected = [KeyTimeInstance(index=2.5, name='AT Engaged Selection')]
        self.assertEqual(ads, expected)


##### Is this KTI working at all?
####class TestEng_Stop(unittest.TestCase):
####    def test_can_operate(self):
####        self.assertTrue(False, msg='Test not implemented.')
####
####    def test_derive(self):
####        self.assertTrue(False, msg='Test not implemented.')


class TestEnterHold(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Holding',)]
        self.assertEqual(expected, EnterHold.get_operational_combinations())

    def test_derive(self):
        hold = buildsection('Holding', 2, 5)
        expected = [KeyTimeInstance(index=2, name='Enter Hold')]
        eh = EnterHold()
        eh.derive(hold)
        self.assertEqual(eh, expected)


class TestExitHold(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Holding',)]
        self.assertEqual(expected, ExitHold.get_operational_combinations())

    def test_derive(self):
        hold = buildsection('Holding', 2, 5)
        expected = [KeyTimeInstance(index=2, name='Enter Hold')]
        eh = EnterHold()
        eh.derive(hold)
        self.assertEqual(eh, expected)


class TestFlapStateChanges(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Flap',)]
        self.assertEqual(
            expected,
            FlapStateChanges.get_operational_combinations())

    def test_derive(self):
        f = P('Flap', [0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0])
        fsc = FlapStateChanges()
        expected = [
            KeyTimeInstance(index=1.5, name='Flap 5 Set'),
            KeyTimeInstance(index=3.5, name='Flap 10 Set'),
            KeyTimeInstance(index=5.5, name='Flap 15 Set'),
            KeyTimeInstance(index=6.5, name='Flap 10 Set'),
            KeyTimeInstance(index=8.5, name='Flap 5 Set'),
            KeyTimeInstance(index=10.5, name='Flap 0 Set'),
        ]
        fsc.derive(f)
        self.assertEqual(fsc, expected)


class TestGearDownSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Gear Down Selected', 'Airborne')]
        self.assertEqual(
            expected,
            GearDownSelection.get_operational_combinations())

    def test_derive(self):
        gup = M('Gear Down Selected',
                ['Down', 'Down', 'Down', 'Up', 'Up', 'Down', 'Down'],
                values_mapping={0: 'Down', 1: 'Up'})
        airs = buildsection('Airborne', 0, 7)
        gear_up = GearUpSelection()
        gear_up.derive(gup, airs)
        self.assertTrue(gear_up[0].index, 4.5)


class TestGearUpSelection(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Gear Up Selected', 'Airborne', 'Go Around And Climbout')]
        self.assertEqual(
            expected,
            GearUpSelection.get_operational_combinations())

    def test_normal_operation(self):
        gup = M('Gear Up Selected', ['Down','Down','Down','Up','Up','Down','Down'],
                values_mapping={0: 'Down', 1: 'Up'})
        airs = buildsection('Airborne', 0, 7)
        gas = buildsection('Go Around', 6, 7)
        gear_up = GearUpSelection()
        gear_up.derive(gup, airs, gas)
        self.assertTrue(gear_up[0].index, 2.5)

    def test_during_ga(self):
        gup = M('Gear Up Selected', ['Down','Down','Down','Up','Up','Down','Down'],
                values_mapping={0: 'Down', 1: 'Up'})
        airs = buildsection('Airborne', 0, 7)
        gas = buildsection('Go Around', 2, 4)
        gear_up = GearUpSelection()
        gear_up.derive(gup, airs, gas)
        if gear_up == []:
            self.assertTrue(True)
        else:
            self.assertTrue(False)


class TestGoAroundFlapRetracted(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Flap', 'Go Around And Climbout')]
        self.assertEqual(
            expected,
            GoAroundFlapRetracted.get_operational_combinations())

    def test_derive(self):
        f = P('Flap', [0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0])
        goaround = buildsection('Go Around', 2, 12)
        fsc = GoAroundFlapRetracted()
        expected = [
            KeyTimeInstance(index=6.5, name='Go Around Flap Retracted'),
            KeyTimeInstance(index=8.5, name='Go Around Flap Retracted'),
            KeyTimeInstance(index=10.5, name='Go Around Flap Retracted'),
        ]
        fsc.derive(f, goaround)
        self.assertEqual(fsc, expected)


#### class TestGoAroundGearRetracted(unittest.TestCase):
####     def test_can_operate(self):
####         self.assertTrue(False, msg='Test not implemented.')
####
####     def test_derive(self):
####         self.assertTrue(False, msg='Test not implemented.')


class TestLocalizerEstablishedEnd(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer Established',)]
        self.assertEqual(
            expected,
            LocalizerEstablishedEnd.get_operational_combinations())

    def test_derive(self):
        ils = buildsection('ILS Localizer Established', 10, 20)
        expected = [
            KeyTimeInstance(index=20, name='Localizer Established End')]
        les = LocalizerEstablishedEnd()
        les.derive(ils)
        self.assertEqual(les, expected)


class TestLocalizerEstablishedStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer Established',)]
        self.assertEqual(
            expected,
            LocalizerEstablishedStart.get_operational_combinations())

    def test_derive(self):
        ils = buildsection('ILS Localizer Established', 10, 20)
        expected = [
            KeyTimeInstance(index=10, name='Localizer Established Start')]
        les = LocalizerEstablishedStart()
        les.derive(ils)
        self.assertEqual(les, expected)


class TestLowestPointOnApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Altitude Radio', 'Approach', 'Landing')]
        self.assertEqual(
            expected,
            LowestPointOnApproach.get_operational_combinations())

    def test_derive(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
            1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ]))
        alt_rad = P(name='Altitude Radio', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 1, 1, 1,
            1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ]))
        appr = buildsection('Approach', 2, 15)
        lpa = LowestPointOnApproach()
        lpa.derive(alt_aal, alt_rad, appr)
        expected = [KeyTimeInstance(index=7, name='Lowest Point On Approach')]
        self.assertEqual(lpa, expected)


class TestMinsToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Touchdown',)]
        self.assertEqual(
            expected,
            MinsToTouchdown.get_operational_combinations())

    def test_derive(self):
        td = [KeyTimeInstance(index=500, name='Touchdown')]
        sttd = MinsToTouchdown()
        sttd.derive(td)
        self.assertEqual(
            sttd,
            [
                KeyTimeInstance(index=200, name='5 Mins To Touchdown'),
                KeyTimeInstance(index=260, name='4 Mins To Touchdown'),
                KeyTimeInstance(index=320, name='3 Mins To Touchdown'),
                KeyTimeInstance(index=380, name='2 Mins To Touchdown'),
                KeyTimeInstance(index=440, name='1 Mins To Touchdown'),
            ]
        )


class TestSecsToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Touchdown',)]
        self.assertEqual(
            expected,
            SecsToTouchdown.get_operational_combinations())

    def test_derive(self):
        td = [KeyTimeInstance(index=100, name='Touchdown')]
        sttd = SecsToTouchdown()
        sttd.derive(td)
        self.assertEqual(
            sttd,
            [
                KeyTimeInstance(index=10, name='90 Secs To Touchdown'),
                KeyTimeInstance(index=70, name='30 Secs To Touchdown'),
            ]
        )


####class TestTAWSTooLowTerrainWarning(unittest.TestCase):
####     def test_can_operate(self):
####         self.assertTrue(False, msg='Test not implemented.')
####
####    def test_derive(self):
####        self.assertTrue(False, msg='Test not implemented.')


class TestTouchAndGo(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Go Around')]
        self.assertEqual(expected, TouchAndGo.get_operational_combinations())

    def test_derive(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ]))
        go_around = [KeyTimeInstance(index=10, name='Go Around')]
        t_a_g = TouchAndGo()
        t_a_g.derive(alt_aal, go_around)
        expected = [KeyTimeInstance(index=10, name='Touch And Go')]
        self.assertEqual(t_a_g, expected)


class TestTransmit(unittest.TestCase):
    def test_can_operate(self):
        expected = set([
            ('Key HF',),
            ('Key HF (1)',),
            ('Key HF (2)',),
            ('Key HF (3)',),
            ('Key Satcom',),
            ('Key Satcom (1)',),
            ('Key Satcom (2)',),
            ('Key VHF',),
            ('Key VHF (1)',),
            ('Key VHF (2)',),
            ('Key VHF (3)',),
        ])
        # All possible combinations of `expected` are allowed, so we only check
        # if all of them are included in the result
        self.assertTrue(expected.issubset(
            set(Transmit.get_operational_combinations())))

    def test_derive(self):
        hf = M('Key HF', ['Off', 'Off', 'Off', 'Keyed', 'Off', 'Off', 'Off'],
               values_mapping={0: 'Off', 1: 'Keyed'})
        tr = Transmit()
        tr.derive(hf)
        expected = [KeyTimeInstance(index=2.5, name='Transmit')]
        self.assertEqual(tr, expected)
