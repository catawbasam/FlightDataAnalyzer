import unittest
import numpy as np

from analysis.node import A, KPV, KTI, Parameter, P, Section, S

from analysis.key_time_instances import (BottomOfDescent,
                                         TopOfClimb, 
                                         TopOfDescent
                                         )
from analysis.plot_flight import plot_parameter
from analysis.flight_phase import (Airborne,
                                   Approach,
                                   ClimbCruiseDescent,
                                   ClimbFromBottomOfDescent,
                                   Climbing,
                                   Cruise,
                                   DescentLowClimb,
                                   DescentToBottomOfDescent,
                                   Fast,
                                   FinalApproach,
                                   InGroundEffect,
                                   LevelFlight,
                                   OnGround,
                                   Turning
                                   )


class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Rate Of Climb',)]
        opts = Airborne.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_airborne_phase_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+
                                         range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array(rate_of_climb_data))
        air = Airborne()
        air.derive(rate_of_climb)
        expected = [Section(name='Airborne', slice=slice(7, 27, None))]
        self.assertEqual(air, expected)


    def test_airborne_phase_not_airborne(self):
        rate_of_climb_data = np.ma.array(range(0,10))
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array(rate_of_climb_data))
        air = Airborne()
        air.derive(rate_of_climb)
        expected = [Section(name='Airborne', slice=slice(7, 27, None))]
        self.assertEqual(air, [])


class TestApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases')]
        opts = Approach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500))
        app = Approach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
        expected = [Section(name='Approach', slice=slice(10, 14, None))]
        self.assertEqual(app, expected)

    def test_approach_phase_over_high_ground(self):
        alt_aal = np.ma.array(range(0,4000,500)+range(4000,0,-500))
        # Raising the ground makes the radio altitude trigger one sample sooner.
        alt_rad = alt_aal - 600
        app = Approach()
        app.derive(Parameter('Altitude AAL For Flight Phases',alt_aal),
                   Parameter('Altitude Radio For Flight Phases',alt_rad))
        expected = [Section(name='Approach', slice=slice(9, 14, None))]
        self.assertEqual(app, expected)

    def test_approach_phase_with_go_around(self):
        alt = np.ma.array(range(4000,2000,-500)+range(2000,4000,500))
        app = Approach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
        expected = [Section(name='Approach', slice=slice(2, 4, None))]
        self.assertEqual(app, expected)


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
        bod.derive(dlc, alt)
                
        descent_phase = ClimbFromBottomOfDescent()
        descent_phase.derive(toc, [], bod) # TODO: include start of climb instance
        expected = [Section(name='Climb From Bottom Of Descent',slice=slice(63, 94, None))]
        self.assertEqual(descent_phase, expected)
                

class TestClimbing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb',)]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        rate_of_climb_data = np.ma.array(range(500,1200,100)+
                                         range(1200,-1200,-200)+
                                         range(-1200,500,100))
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array(rate_of_climb_data))
        up = Climbing()
        up.derive(rate_of_climb)
        expected = [Section(name='Climbing', slice=slice(3, 10, None))]
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
        expected = [Section(name='Cruise', slice=slice(31, 32, None)),
                    Section(name='Cruise', slice=slice(94, 94, None))]
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
        expected = [('Altitude For Flight Phases',)]
        opts = DescentLowClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_low_climb_basic(self):
        # This test will find out if we can separate the two humps on this camel
        dlc = DescentLowClimb()
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt = Parameter('Altitude For Flight Phases', np.ma.array(testwave))
        dlc.derive(alt)
        self.assertEqual(len(dlc), 1)


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
        bod.derive(dlc, alt)
                
        descent_phase = DescentToBottomOfDescent()
        descent_phase.derive(tod, bod)
        expected = [Section(name='Descent To Bottom Of Descent',slice=slice(32,63,None))]
        self.assertEqual(descent_phase, expected)
                

class TestFast(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = Fast.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_fast_phase_basic(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = [Section(name='Fast',slice=slice(2,11,None))]
        self.assertEqual(phase_fast, expected)
        
    def test_fast_phase_with_mask(self):
        slow_and_fast_data = np.ma.concatenate([np.ma.arange(60,120,10),
                                                np.ma.arange(120,50,-10)])
        slow_and_fast_data[5:8] = np.ma.masked
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias)
        expected = [Section(name='Fast',slice=slice(2,5,None)),
                  Section(name='Fast',slice=slice(8,11,None))]
        self.assertEqual(phase_fast, expected)


class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases')]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.array(range(0,1200,100)+range(1500,500,-100)+range(400,0,-40))
        app = FinalApproach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
        expected = [Section(name='Final Approach', slice=slice(17, 30, None))]
        self.assertEqual(app, expected)

    def test_approach_phase_starting_inside_phase_and_with_go_around(self):
        alt = np.ma.array(range(400,300,-50)+range(300,500,50))
        app = FinalApproach()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
        expected = [Section(name='Final Approach', slice=slice(0, 2, None))]
        self.assertEqual(app, expected)


class TestInGroundEffect(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio For Flight Phases',)]
        opts = InGroundEffect.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_onground_basic(self):
        alt_rad = Parameter('Altitude Radio For Flight Phases',
                            np.ma.array([range(0,200,10)+range(200,0,-10)]))
        ige = InGroundEffect()
        ige.derive(alt_rad)
        expected = [Section(name='In Ground Effect',slice=slice(0,8,None)),
                    Section(name='In Ground Effect',slice=slice(33,40,None))]
        self.assertEqual(ige, expected)
 

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
        expected = [Section(name='On Ground',slice=slice(2,10,None))]
        self.assertEqual(phase_onground, expected)
 
        
class TestTurning(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Turn',)]
        opts = Turning.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_turning_phase_basic(self):
        rate_of_turn_data = np.arange(-2, 2.2, 0.2)
        rate_of_turn = Parameter('Rate Of Turn', np.ma.array(rate_of_turn_data))
        turning = Turning()
        turning.derive(rate_of_turn)
        expected = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning, expected)
        
    def test_turning_phase_basic_masked_not_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = Turning()
        turning.derive(rate_of_turn)
        expected = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning, expected)
        
    def test_turning_phase_basic_masked_while_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[1] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = Turning()
        turning.derive(rate_of_turn)
        expected = [Section(name='Turning', slice=slice(0, 1, None)),
                  Section(name='Turning', slice=slice(2, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]

        self.assertEqual(turning, expected)
        
class TestLevelFlight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Rate Of Climb',)]
        opts = LevelFlight.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_level_flight_phase_basic(self):
        rate_of_climb_data = np.ma.array(range(0,400,50)+range(400,-450,-50)+
                                         range(-450,50,50))
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array(rate_of_climb_data))
        level = LevelFlight()
        level.derive(rate_of_climb)
        expected = [Section(name='Level Flight', slice=slice(0, 7, None)),
                  Section(name='Level Flight', slice=slice(10, 23, None)), 
                  Section(name='Level Flight', slice=slice(28, 35, None))]
        self.assertEqual(level, expected)
        
    def test_turning_phase_basic_masked_not_turning(self):
        rate_of_turn_data = np.ma.arange(-2, 2.2, 0.2)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn = Parameter('Rate Of Turn', rate_of_turn_data)
        turning = Turning()
        turning.derive(rate_of_turn)
        expected = [Section(name='Turning', slice=slice(0, 3, None)),
                  Section(name='Turning', slice=slice(18, 21, None))]
        self.assertEqual(turning, expected)
        
