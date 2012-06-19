import mock
import numpy as np
import sys
import unittest

from analysis_engine.node import (KeyTimeInstance, Parameter, P, Section, S)

from analysis_engine.flight_phase import Climbing

from analysis_engine.key_time_instances import (AltitudeWhenClimbing,
                                                AltitudeWhenDescending,
                                                BottomOfDescent,
                                                ClimbStart,
                                                GoAround,
                                                InitialClimbStart,
                                                LandingDecelerationEnd,
                                                ##LandingPeakDeceleration,
                                                LandingStart,
                                                LandingTurnOffRunway,
                                                Liftoff,
                                                TakeoffAccelerationStart,
                                                TakeoffPeakAcceleration,
                                                TakeoffTurnOntoRunway,
                                                TopOfClimb,
                                                TopOfDescent,
                                                Touchdown,
                                                )

from flight_phase_test import buildsection, buildsections

debug = sys.gettrace() is not None

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
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc, [Section('Fast',slice(0,8,None),0,8)])
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
        #climb.derive(Parameter('Altitude STD', alt), 
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
                    np.ma.masked_array(range(100, 0, -10) + \
                                       range(100, 0, -10),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        altitude_when_descending = AltitudeWhenDescending()
        altitude_when_descending.derive(descending, alt_aal)
        self.assertEqual(list(altitude_when_descending),
          [KeyTimeInstance(index=2.5, name='75 Ft Descending'), 
           KeyTimeInstance(index=5.0, name='50 Ft Descending'),
           KeyTimeInstance(index=12.5, name='75 Ft Descending'), 
           KeyTimeInstance(index=15.0, name='50 Ft Descending'),
           KeyTimeInstance(index=16.5, name='35 Ft Descending'),
           KeyTimeInstance(index=18.0, name='20 Ft Descending'),
           KeyTimeInstance(index=19.0, name='10 Ft Descending')])


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
        head = P('Heading Continuous',np.ma.array([0]*30))
        fast = buildsection('Fast',0,20)
        land = buildsection('Landing',10,26)
        instance.derive(head,land,fast)
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
                         [('Rate Of Climb For Flight Phases', 'Airborne')])

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between 
        # the 5th and 6th sample points.
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', (np.ma.arange(10)-0.5)*40)
        # Airborne section encloses the test point.
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(rate_of_climb, airs)
        expected = [KeyTimeInstance(index=5.5, name='Liftoff')]
        self.assertEqual(lift, expected)
    
    def test_liftoff_no_roc_detected(self):
        # Check the backstop setting.
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', (np.ma.array([0]*40)))
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(rate_of_climb, airs)
        expected = [KeyTimeInstance(index=6, name='Liftoff')]
        self.assertEqual(lift, expected)
    
    def test_liftoff_already_airborne(self):
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', (np.ma.array([0]*40)))
        airs = buildsection('Airborne', None, 10)
        lift=Liftoff()
        lift.derive(rate_of_climb, airs)
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
        airspeed_data = np.ma.array([37.9,37.9,37.9,37.9,37.9,38.2,38.2,38.2,
                                     38.2,38.8,38.2,38.8,39.1,39.7,40.6,41.5,
                                     42.7,43.6,44.5,46,47.5,49.6,52,53.2,54.7,
                                     57.4,60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                     74.8])
        takeoff = buildsection('Takeoff',3,len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        expected = [KeyTimeInstance(index=15.083333333333361, name='Takeoff Acceleration Start')]
        self.assertEqual(instance, expected)

    
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
        expected = [KeyTimeInstance(index=1, name='Takeoff Turn Onto Runway')]
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
        expected = [('Altitude STD','Climb Cruise Descent')]
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
        expected = [('Altitude STD','Climb Cruise Descent')]
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
        self.assertEqual(Touchdown.get_operational_combinations(),
                         [('Rate Of Climb', 'Airborne')])

    def test_touchdown_basic(self):
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.arange(10)*40 - 320)
        airs = buildsection('Airborne', 1, 4.2)
        tdwn=Touchdown()
        tdwn.derive(rate_of_climb, airs)
        expected = [KeyTimeInstance(index=5.5, name='Touchdown')]
        self.assertEqual(tdwn, expected)
    
    def test_touchdown_no_roc_detected(self):
        # Check the backstop setting.
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.arange(10)*40)
        airs = buildsection('Airborne', 1, 5)
        tdwn=Touchdown()
        tdwn.derive(rate_of_climb, airs)
        expected = [KeyTimeInstance(index=5, name='Touchdown')]
        self.assertEqual(tdwn, expected)
    
    def test_touchdown_doesnt_land(self):
        rate_of_climb = Parameter('Rate Of Climb For Flight Phases', np.ma.arange(10)*40)
        airs = buildsection('Airborne', 10, None)
        tdwn=Touchdown()
        tdwn.derive(rate_of_climb, airs)
        expected = []
        self.assertEqual(tdwn, expected)
        