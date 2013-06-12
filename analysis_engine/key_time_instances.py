import numpy as np
from math import ceil, copysign, floor
import os

from analysis_engine.library import (all_of,
                                     any_of,
                                     coreg,
                                     find_edges_on_state_change,
                                     find_toc_tod,
                                     first_valid_sample,
                                     hysteresis,
                                     index_at_value,
                                     max_value,
                                     minimum_unmasked,
                                     np_ma_masked_zeros_like,
                                     peak_curvature,
                                     slices_and,
                                     slices_not,
                                     touchdown_inertial)

from analysis_engine.node import A, M, P, S, KTI, KeyTimeInstanceNode

from settings import (CLIMB_THRESHOLD,
                      MIN_CORE_SUSTAINABLE,
                      NAME_VALUES_CLIMB,
                      NAME_VALUES_DESCENT,
                      NAME_VALUES_ENGINE,
                      NAME_VALUES_FLAP,
                      TAKEOFF_ACCELERATION_THRESHOLD,
                      VERTICAL_SPEED_FOR_LIFTOFF,
                      WORKING_DIR
                      )


class BottomOfDescent(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude AAL For Flight Phases'),
               dlc=S('Descent Low Climb'),
               airs=S('Airborne')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for this_dlc in dlc:
            kti = np.ma.argmin(alt_std.array[this_dlc.slice])
            self.create_kti(kti + this_dlc.start_edge)
        # For descents to landing, end where the aircraft is no longer airborne.
        for air in airs:
            if air.slice.stop:
                self.create_kti(air.stop_edge)


# TODO: Determine an altitude peak per climb.
class AltitudePeak(KeyTimeInstanceNode):
    '''
    Determines the peak value of altitude above airfield level which is used to
    correctly determine the splitting point when deriving the Altitude QNH
    parameter.
    '''

    def derive(self, alt_aal=P('Altitude AAL')):
        '''
        '''
        self.create_kti(np.ma.argmax(np.ma.abs(np.ma.diff(alt_aal.array))))


##############################################################################
# Automated Systems


class APEngagedSelection(KeyTimeInstanceNode):
    '''
    AP Engaged is defined as leaving the disengaged state, so that it works
    for simplex, duplex or triplex engagement options.
    '''

    name = 'AP Engaged Selection'

    def derive(self, ap=M('AP Engaged'), phase=S('Airborne')):
        # FIXME: Allow for other states?
        self.create_ktis_on_state_change(
            'Engaged',
            ap.array,
            change='entering',
            phase=phase
        )


class APDisengagedSelection(KeyTimeInstanceNode):
    '''
    AP Disengaged is defined as entering the disengaged state, so that it works
    for simplex, duplex or triplex engagement options.
    '''

    name = 'AP Disengaged Selection'

    def derive(self, ap=M('AP Engaged'), phase=S('Airborne')):
        # FIXME: Allow for other states?
        self.create_ktis_on_state_change(
            'Engaged',
            ap.array,
            change='leaving',
            phase=phase
        )


class ATEngagedSelection(KeyTimeInstanceNode):
    '''
    '''

    name = 'AT Engaged Selection'

    def derive(self, at=M('AT Engaged'), phase=S('Airborne')):

        self.create_ktis_on_state_change(
            'Engaged',
            at.array,
            change='entering',
            phase=phase
        )


class ATDisengagedSelection(KeyTimeInstanceNode):
    '''
    '''

    name = 'AT Disengaged Selection'

    def derive(self, at=P('AT Engaged'), phase=S('Airborne')):

        self.create_ktis_on_state_change(
            'Engaged',
            at.array,
            change='leaving',
            phase=phase
        )


##############################################################################


class Transmit(KeyTimeInstanceNode):

    @classmethod
    def can_operate(cls, available):
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
            hf=M('Key HF'),
            hf1=M('Key HF (1)'),
            hf2=M('Key HF (2)'),
            hf3=M('Key HF (3)'),
            sc=M('Key Satcom'),
            sc1=M('Key Satcom (1)'),
            sc2=M('Key Satcom (2)'),
            vhf=M('Key VHF'),
            vhf1=M('Key VHF (1)'),
            vhf2=M('Key VHF (2)'),
            vhf3=M('Key VHF (3)')):
        for p in [hf, hf1, hf2, hf3, sc, sc1, sc2, vhf, vhf1, vhf2, vhf3]:
            if p:
                self.create_ktis_on_state_change(
                    'Keyed',
                    p.array,
                    change='entering'
                )


class ClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = index_at_value(alt_aal.array,
                                                 CLIMB_THRESHOLD, climb.slice)
            # The aircraft may be climbing, but starting from an altitude
            # above CLIMB_THRESHOLD. In this case no kti is created.
            if initial_climb_index:
                self.create_kti(initial_climb_index)


class EngStart(KeyTimeInstanceNode):
    '''
    Records the moment of engine start for each engine in turn.
    
    Engines running at the start of the valid data are assumed to start when
    the data starts.
    
    For the purist, we should look at N3 for three-shaft engines, but
    checking N2 appears to work fine.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Start'
    NAME_VALUES = NAME_VALUES_ENGINE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Eng (%d) N2' % n for n in range(1, 5)), available)

    def derive(self,
               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2')):

        eng_n2_list = (eng_1_n2, eng_2_n2, eng_3_n2, eng_4_n2)
        for number, eng_n2 in enumerate(eng_n2_list, start=1):
            if not eng_n2:
                continue

            running = np.ma.where(eng_n2.array > MIN_CORE_SUSTAINABLE, 1, 0)
            first_speed = first_valid_sample(running)

            if first_speed.value:
                # The first valid sample shows the engine running when the
                # recording started.
                self.create_kti(first_speed.index,
                                replace_values={'number': number})

            else:
                # The engine stopped before the end of the data.
                self.create_ktis_at_edges(
                    running,
                    direction='rising_edges',
                    replace_values={'number': number},
                )


class EngStop(KeyTimeInstanceNode):
    '''
    Monitors the engine stop time. Engines still running at the end of the
    data are assumed to stop at the end of the data recording.
    
    We use MIN_CORE_SUSTAINABLE/2 to make sure the engine truly is stopping,
    and not just running freakishly slow.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Stop'
    NAME_VALUES = NAME_VALUES_ENGINE

    @classmethod
    def can_operate(cls, available):
        return any_of(('Eng (%d) N2' % n for n in range(1, 5)), available)

    def derive(self,
               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2')):

        eng_n2_list = (eng_1_n2, eng_2_n2, eng_3_n2, eng_4_n2)
        for number, eng_n2 in enumerate(eng_n2_list, start=1):
            if not eng_n2:
                continue

            running = np.ma.where(eng_n2.array > MIN_CORE_SUSTAINABLE/2, 1, 0)
            last_speed = first_valid_sample(running[::-1])

            if last_speed.value:
                # The last valid sample shows the engine running when the
                # recording stopped.
                self.create_kti(len(eng_n2.array)-last_speed.index-1,
                                replace_values={'number': number})

            else:
                # The engine stopped before the end of the data.
                self.create_ktis_at_edges(
                    running,
                    direction='falling_edges',
                    replace_values={'number': number},
                )


class EnterHold(KeyTimeInstanceNode):
    def derive(self, holds=S('Holding')):
        for hold in holds:
            self.create_kti(hold.slice.start)


class ExitHold(KeyTimeInstanceNode):
    def derive(self, holds=S('Holding')):
        for hold in holds:
            self.create_kti(hold.slice.stop)


class GoAround(KeyTimeInstanceNode):
    """
    In POLARIS we define a Go-Around as any descent below 3000ft followed by
    an increase of 500ft. This wide definition will identify more events than
    a tighter definition, however experience tells us that it is worth
    checking all these cases. For example, we have identified attemnpts to
    land on roads or at the wrong airport, EGPWS database errors etc from
    checking these cases.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters. If 'Altitude Radio For Flight
        # Phases' is available, that's a bonus and we will use it, but it is
        # not required.
        return ('Descent Low Climb' in available and
                'Altitude AAL For Flight Phases' in available)

    # List the optimal parameter set here

    def derive(self, dlcs=S('Descent Low Climb'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio')):
        for dlc in dlcs:
            # The try:except structure is used to cater for both cases where
            # a radio altimeter is not fitted, and where the altimeter data
            # is out of range, hence masked, at the lowest point of the
            # go-around.
            try:
                pit = np.ma.argmin(alt_rad.array[dlc.slice])  # if masked, argmin returns 0  - valid?
            except:  # FIXME: Catch expected exception
                pit = np.ma.argmin(alt_aal.array[dlc.slice])
            self.create_kti(pit + dlc.start_edge)


class TopOfClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               ccd=S('Climb Cruise Descent')):
        # This checks for the top of climb in each
        # Climb/Cruise/Descent period of the flight.
        for ccd_phase in ccd:
            ccd_slice = ccd_phase.slice
            try:
                n_toc = find_toc_tod(alt_std.array, ccd_slice, 'Climb')
            except:
                # altitude data does not have an increasing section, so quit.
                break
            # If the data started in mid-flight the ccd slice will start with None
            if ccd_slice.start is None:
                break
            # if this is the first point in the slice, it's come from
            # data that is already in the cruise, so we'll ignore this as well
            if n_toc==0:
                break
            # Record the moment (with respect to this section of data)
            self.create_kti(n_toc)


class TopOfDescent(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               ccd=S('Climb Cruise Descent')):
        # This checks for the top of descent in each
        # Climb/Cruise/Descent period of the flight.
        for ccd_phase in ccd:
            ccd_slice = ccd_phase.slice
            try:
                n_tod = find_toc_tod(alt_std.array, ccd_slice, 'Descent')
            except:
                # altitude data does not have a decreasing section, so quit.
                break
            # If this slice ended in mid-cruise, the ccd slice will end in None.
            if ccd_slice.stop is None:
                break
            # if this is the last point in the slice, it's come from
            # data that ends in the cruise, so we'll ignore this too.
            if n_tod==ccd_slice.stop - 1:
                break
            # Record the moment (with respect to this section of data)
            self.create_kti(n_tod)


##############################################################################
# Flap


class FlapSet(KeyTimeInstanceNode):
    '''
    '''

    # Note: We must use %s not %d as we've encountered a flap of 17.5 degrees.
    NAME_FORMAT = 'Flap %(flap)s Set'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self,
               flap=P('Flap')):

        # Mark all flap changes, and annotate with the new flap position.
        # Could include "phase=airborne" if we want to eliminate ground flap
        # changes.
        self.create_ktis_at_edges(flap.array, direction='all_edges',
                                  name='flap')


class FirstFlapExtensionWhileAirborne(KeyTimeInstanceNode):
    '''
    Records each flap extension from clean configuration.
    '''
    def derive(self,
               flap=P('Flap'),
               airborne=S('Airborne')):

        for air in airborne:
            cleans = np.ma.flatnotmasked_contiguous(
                np.ma.masked_not_equal(flap.array[air.slice],0.0))
            if not isinstance(cleans, list):
                continue
            for clean in cleans:
                self.create_kti(clean.stop + air.slice.start)



class FlapExtensionWhileAirborne(KeyTimeInstanceNode):
    '''
    Records every flap extension in flight.
    '''
    def derive(self,
               flap=P('Flap'),
               airborne=S('Airborne')):

        self.create_ktis_at_edges(flap.array, 
                                  phase = airborne)


class FlapRetractionWhileAirborne(KeyTimeInstanceNode):
    '''
    '''

    def derive(self,
               flap=P('Flap'),
               airborne=S('Airborne')):

        self.create_ktis_at_edges(
            flap.array,
            direction='falling_edges',
            phase=airborne,
        )


class FlapRetractionDuringGoAround(KeyTimeInstanceNode):
    '''
    '''

    def derive(self,
               flap=P('Flap'),
               go_arounds=S('Go Around And Climbout')):

        self.create_ktis_at_edges(
            flap.array,
            direction='falling_edges',
            phase=go_arounds,
        )


##############################################################################
# Gear


class GearDownSelection(KeyTimeInstanceNode):
    '''
    Instants at which gear down was selected while airborne.
    '''

    def derive(self,
               gear_dn_sel=M('Gear Down Selected'),
               airborne=S('Airborne')):

        self.create_ktis_on_state_change('Down', gear_dn_sel.array,
                                         change='entering', phase=airborne)


class GearUpSelection(KeyTimeInstanceNode):
    '''
    Instants at which gear up was selected while airborne excluding go-arounds.
    '''

    def derive(self,
               gear_up_sel=M('Gear Up Selected'),
               airborne=S('Airborne'),
               go_arounds=S('Go Around And Climbout')):

        air_slices = airborne.get_slices()
        ga_slices = go_arounds.get_slices()
        if not air_slices:
            return
        air_not_ga = slices_and(air_slices, slices_not(ga_slices,
            begin_at=air_slices[0].start,
            end_at=air_slices[-1].stop,
        ))
        good_phases = S(name='Airborne Not During Go Around',
                        frequency=gear_up_sel.frequency,
                        offset=gear_up_sel.offset)
        good_phases.create_sections(air_not_ga)
        self.create_ktis_on_state_change('Up', gear_up_sel.array,
                                         change='entering', phase=good_phases)


class GearUpSelectionDuringGoAround(KeyTimeInstanceNode):
    '''
    Instants at which gear up was selected while airborne including go-arounds.
    '''

    def derive(self,
               gear_up_sel=M('Gear Up Selected'),
               go_arounds=S('Go Around And Climbout')):

        self.create_ktis_on_state_change('Up', gear_up_sel.array,
                                         change='entering', phase=go_arounds)


##############################################################################
# Flight Sequence
    
class TakeoffTurnOntoRunway(KeyTimeInstanceNode):
    '''
    The Takeoff flight phase is computed to start when the aircraft turns
    onto the runway, so at worst this KTI is just the start of that phase.
    Where possible we compute the sharp point of the turn onto the runway.
    '''
    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff'),
               fast=S('Fast')):
        for toff in toffs:
            # Ideally we'd like to work from the start of the Fast phase
            # backwards, but in case there is a problem with the phases,
            # use the midpoint. This avoids identifying the heading
            # change immediately after liftoff as a turn onto the runway.
            start_search = fast.get_next(toff.slice.start).slice.start
            if (start_search is None) or (start_search > toff.slice.stop):
                start_search = (toff.slice.start + toff.slice.stop) / 2
            peak_bend = peak_curvature(head.array,slice(
                start_search, toff.slice.start, -1), curve_sense='Bipolar')
            if peak_bend:
                takeoff_turn = peak_bend
            else:
                takeoff_turn = toff.slice.start
            self.create_kti(takeoff_turn)


class TakeoffAccelerationStart(KeyTimeInstanceNode):
    '''
    The start of the takeoff roll is ideally computed from the forwards
    acceleration down the runway, but a quite respectable "backstop" is
    available from the point where the airspeed starts to increase (providing
    this is from an analogue source). This allows for aircraft either with a
    faulty sensor, or no longitudinal accelerometer.
    '''
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters. If 'Altitude Radio For Flight
        # Phases' is available, that's a bonus and we will use it, but it is
        # not required.
        return 'Airspeed' in available and 'Takeoff' in available

    # List the optimal parameter set here
    def derive(self, speed=P('Airspeed'), takeoffs=S('Takeoff'),
               accel=P('Acceleration Longitudinal')):
        for takeoff in takeoffs:
            start_accel = None
            if accel:
                # Ideally compute this from the forwards acceleration.
                # If they turn onto the runway already accelerating, take that as the start point.
                first_accel = accel.array[takeoff.slice.start]
                if first_accel > TAKEOFF_ACCELERATION_THRESHOLD:
                    start_accel = takeoff.slice.start
                else:
                    start_accel=index_at_value(accel.array,
                                               TAKEOFF_ACCELERATION_THRESHOLD,
                                               takeoff.slice)

            if start_accel is None:
                '''
                A quite respectable "backstop" is from the rate of change of
                airspeed. We use this if the acceleration is not available or
                if, for any reason, the previous computation failed.
                Originally we used the peak_curvature algorithm to identify
                where the airspeed started to increase, but when values lower
                than a threshold were masked this ceased to work (the "knee"
                is masked out) and so the extrapolated airspeed was adopted.
                '''
                #pc = peak_curvature(speed.array[takeoff.slice])
                p,m,c = coreg(speed.array[takeoff.slice])
                start_accel = max(takeoff.slice.start-c/m, 0.0)

            if start_accel is not None:
                self.create_kti(start_accel)


class TakeoffPeakAcceleration(KeyTimeInstanceNode):
    """
    As for landing, the point of maximum acceleration, is used to identify the
    location and heading of the takeoff.
    """
    def derive(self, toffs=S('Takeoff'),
               accel=P('Acceleration Longitudinal')):
        for toff in toffs:
            index, value = max_value(accel.array, _slice=toff.slice)
            if index: # In case all the Ay data is invalid.
                self.create_kti(index)


class Liftoff(KeyTimeInstanceNode):
    '''
    This checks for the moment when the inertial rate of climb increases
    through 200fpm, within 2 seconds of the nominal liftoff point.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Airborne' in available

    def derive(self, vert_spd=P('Vertical Speed Inertial'),
               ax=P('Acceleration Longitudinal'),
               alt_rad = P('Altitude Radio'),
               gog = P('Gear On Ground'),
               airs=S('Airborne'),
               pitch=P('Pitch'),
               frame = A('Frame')):
        
        for air in airs:
            index_vs = index_rad = index_gog = None
            index_air = air.slice.start
            if index_air == None:
                continue
            back_3 = (index_air - 3.0*self.frequency)
            on_3 = (index_air + 3.0*self.frequency) + 1 # For indexing
            to_scan = slice(back_3, on_3)

            if vert_spd:
                index_vs = index_at_value(vert_spd.array,
                                       VERTICAL_SPEED_FOR_LIFTOFF,
                                       to_scan)
            
            if alt_rad:
                index_rad = index_at_value(alt_rad.array, 0.0, to_scan)
                
            if gog:
                # Try using Gear On Ground switch
                edges = find_edges_on_state_change(
                    'Ground', gog.array[to_scan], change='leaving')
                if edges:
                    # use the last liftoff point
                    index = edges[-1] + back_3
                    # Check we were within 5ft of the ground when the switch triggered.
                    if not alt_rad or alt_rad.array[index] < 5.0:
                        index_gog = index


            # We pick the median recorded indication for the point of liftoff.
            lifts = [index_air, index_vs, index_gog, index_rad]
            index_list = [l for l in lifts if l is not None]
            if len(index_list)>1:
                index_lift = sorted(index_list)[1]
            else:
                index_lift = index_list[0]
            # but in any case, if we have a gear on ground signal which goes
            # off first, adopt that.
            if index_gog and index_gog<index_lift:
                index_lift = index_gog
            
            self.create_kti(index_lift)

            # Plotting process to view the results in an easy manner.
            import matplotlib.pyplot as plt
            name = 'Liftoff Plot %s, %d' %(frame.value, index_air)
            print name
            dt_pre = 5
            hz = self.frequency
            timebase=np.linspace(-dt_pre*hz, dt_pre*hz, 2*dt_pre*hz+1)
            plot_period = slice(floor(index_air-dt_pre*hz), floor(index_air-dt_pre*hz+len(timebase)))
            plt.figure()
            if vert_spd:
                plt.plot(timebase, np.ma.masked_greater(vert_spd.array[plot_period],400.0)/10.0, 'o-g')
            if index_vs:
                plt.plot(index_vs-index_air, 15.0,'dg', markersize=8)
                
            if alt_rad:
                plt.plot(timebase, np.ma.masked_greater(alt_rad.array[plot_period],40.0), 'o-r')
            if index_rad:
                plt.plot(index_rad-index_air, 25.0,'dr', markersize=8)
                
            if gog:
                plt.plot(timebase, gog.array[plot_period]*10, 'o-k')
            if index_gog:
                plt.plot(index_gog-index_air, 20.0,'dk', markersize=8)
                
            # Pitch plotted for guidance only - not used as a paramter for liftoff.
            if pitch:
                plt.plot(timebase, pitch.array[plot_period], 'o-m')

            if ax:
                plt.plot(timebase, ax.array[plot_period]*100.0, 'o-c')

            if index_lift:
                plt.plot(index_lift-index_air, -5.0,'db', markersize=14)

            plt.title(name)
            plt.grid()
            filename = name
            print name
            output_dir = os.path.join(WORKING_DIR, 'Liftoff_graphs')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            plt.savefig(os.path.join(output_dir, filename + '.png'))
            #plt.show()
            plt.clf()
            plt.close()
            

class LowestAltitudeDuringApproach(KeyTimeInstanceNode):
    '''
    For any approach phase that did not result in a landing, the lowest point
    is taken as key, from which the position, heading and height will be
    taken as KPVs.

    This KTI is essential to collect the related KPVs which inform the
    approach attribute, and thereafter compute the smoothed track.
    '''

    def derive(self,
               alt_aal=P('Altitude AAL'),
               alt_rad=P('Altitude Radio'),
               approaches=S('Approach And Landing')):

        height = minimum_unmasked(alt_aal.array, alt_rad.array)
        for approach in approaches:
            index = np.ma.argmin(height[approach.slice])
            self.create_kti(approach.start_edge + index)


class InitialClimbStart(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to run up to the start of the
    # initial climb, so this KTI is just at the end of that phase.
    def derive(self, toffs=S('Takeoff')):
        for toff in toffs:
            if toff.stop_edge:
                self.create_kti(toff.stop_edge)


class LandingStart(KeyTimeInstanceNode):
    # The Landing flight phase is computed to start passing through 50ft
    # (nominally), so this KTI is just at the end of that phase.
    def derive(self, landings=S('Landing')):
        for landing in landings:
            if landing.start_edge:
                self.create_kti(landing.start_edge)


class TouchAndGo(KeyTimeInstanceNode):
    #TODO: TESTS
    """
    In POLARIS we define a Touch and Go as a Go-Around that contacted the ground.
    """
    def derive(self, alt_AAL=P('Altitude AAL'), go_arounds=KTI('Go Around')):
        for ga in go_arounds:
            if alt_AAL.array[ga.index] == 0.0:
                # wheels on ground
                self.create_kti(ga.index)


class Touchdown(KeyTimeInstanceNode):
    '''
    Touchdown is notoriously difficult to identify precisely, and a
    suggestion from a Boeing engineer was to add a longitudinal acceleration
    term as there is always an instantaneous drag when the mainwheels touch.
    
    This was added in the form of two triggers, one detecting the short dip
    in Ax and another for larger changes which are less common.
    
    So, we look for the weight on wheels switch if this is the first indication,
    or the second indication of:
    * Zero feet AAL (normally derived from the radio altimeter)
    * Sudden rise in normal acceleration bumping the ground
    * Significant product of two samples of normal acceleration (correlating to a sudden drop in descent rate)
    * A transient reduction in longitudinal acceleration as the wheels first spin up
    * A large reduction in longitudinal acceleration when braking action starts
    
    http://www.flightdatacommunity.com/when-does-the-aircraft-land/
    '''
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return all_of(('Altitude AAL', 'Landing'), available)

    def derive(self, acc_norm=P('Acceleration Normal'),
               acc_long=P('Acceleration Longitudinal'),
               alt=P('Altitude AAL'), 
               gog=M('Gear On Ground'), 
               lands=S('Landing')):
        # The preamble here checks that the landing we are looking at is
        # genuine, it's not just because the data stopped in mid-flight. We
        # reduce the scope of the search for touchdown to avoid triggering in
        # mid-cruise, and it avoids problems for aircraft where the gear
        # signal changes state on raising the gear (OK, if they do a gear-up
        # landing it won't work, but this will be the least of the problems).

        dt_pre = 4.0 # Seconds to scan before estimate.
        dt_post = 3.0 # Seconds to scan after estimate.
        hz = alt.frequency
        index_gog = index_ax = index_az = index_daz = index_dax = index_z = None
        peak_ax = peak_az = delta = 0.0
        
        for land in lands:
            # We have to have an altitude signal, so this forms an initial
            # estimate of the touchdown point.
            index_alt = index_at_value(alt.array, 0.0, land.slice)
            
            if gog:
                # Try using Gear On Ground switch
                edges = find_edges_on_state_change(
                    'Ground', gog.array[land.slice])
                if edges:
                    # use the first contact with ground as touchdown point 
                    # (i.e. we ignore bounces)
                    index = edges[0] + land.slice.start
                    # Check we were within 5ft of the ground when the switch triggered.
                    if not alt or alt.array[index] < 5.0:
                        index_gog = index

            index_ref = min([x for x in index_alt, index_gog if x is not None])
            
            # With an estimate from the height and perhaps gear switch, set
            # up a period to scan across for accelerometer based
            # indications...
            period = slice(max(floor(index_ref-dt_pre*hz), 0), ceil(index_ref+dt_post*hz))
            
            if acc_long:
                drag = acc_long.array[period]
                touch = np_ma_masked_zeros_like(drag)
                
                # Look for inital wheel contact
                for i in range(2, len(touch)-2):
                    # Looking for a downward pointing "V" shape over half the
                    # Az sample rate. This is a common feature at the point
                    # of wheel touch.
                    touch[i-2] = max(0.0,drag[i-2]-drag[i]) * max(0.0,drag[i+2]-drag[i])
                peak_ax = np.max(touch)

                # Only use this if the value was significant.
                if peak_ax>0.0005:
                    ix_ax2 = np.argmax(touch)
                    ix_ax = ix_ax2
                    # See if this was the second of a pair, with the first a little smaller.
                    if np.ma.count(touch[:ix_ax2]) > 0:
                        # I have some valid data to scan
                        ix_ax1 = np.argmax(touch[:ix_ax2])
                        if touch[ix_ax1] > peak_ax*0.2:
                            # This earlier touch was a better guess.
                            peak_ax = touch[ix_ax1]
                            ix_ax = ix_ax1 
                                
                    index_ax = ix_ax+1+index_ref-dt_pre*hz

                # Trap landings with immediate braking where there is no skip effect.
                for i in range(0, len(drag)-4):
                    if drag[i] and drag[i+4]:
                        delta=drag[i]-drag[i+4]
                        if delta > 0.1:
                            index_dax = i+2+index_ref-dt_pre*hz
                            break
            
            if acc_norm:
                lift = acc_norm.array[period]
                mean = np.mean(lift)
                lift = np.ma.masked_less(lift-mean, 0.0)
                bump = np_ma_masked_zeros_like(lift)

                # A firm touchdown is typified by at least two large Az samples.
                for i in range(1, len(bump)-1):
                    bump[i-1]=lift[i]*lift[i+1]
                peak_az = np.max(bump)
                index_az = np.argmax(bump)+index_ref-dt_pre*hz
            
                # The first real contact is indicated by an increase in g of
                # more than 0.075, but this must be positive (hence the
                # masking above the local mean).
                for i in range(0, len(lift)-1):
                    if lift[i] and lift[i+1]:
                        delta=lift[i+1]-lift[i]
                        if delta > 0.1:
                            index_daz = i+1+index_ref-dt_pre*hz
                            break
                
            # Pick the first of the two normal accelerometer measures to
            # avoid triggering a touchdown from a single faulty sensor:
            index_z_list = [x for x in index_az, index_daz if x is not None]
            if index_z_list:
                index_z = min(index_z_list)
            # ...then collect up to four estimates of the touchdown point...
            index_list = [x for x in index_alt, 
                          index_gog, 
                          index_ax,
                          index_dax,
                          index_z, \
                          if x is not None]
            
            # ...and use the second where possible, as this has been found to
            # be more reliable than the first which may be erroneous.
            if len(index_list)>1:
                index_tdn = sorted(index_list)[1]
            else:
                index_tdn = index_list[0]
            # but in any case, if we have a gear on ground signal which goes
            # off first, adopt that.
            if index_gog and index_gog<index_tdn:
                index_tdn = index_gog
                

            # Plotting process to view the results in an easy manner.
            import matplotlib.pyplot as plt
            name = 'Touchdown with values Ax=%.4f, Az=%.4f and dAz=%.4f' %(peak_ax, peak_az, delta)
            self.info(name)
            timebase=np.linspace(-dt_pre*hz, dt_pre*hz, 2*dt_pre*hz+1)
            plot_period = slice(floor(index_ref-dt_pre*hz), floor(index_ref-dt_pre*hz+len(timebase)))
            plt.figure()
            if alt:
                plt.plot(timebase, alt.array[plot_period], 'o-r')
            if acc_long:
                plt.plot(timebase, acc_long.array[plot_period]*200, 'o-m')
            if acc_norm:
                plt.plot(timebase, acc_norm.array[plot_period]*100, 'o-g')
            if gog:
                plt.plot(timebase, gog.array[plot_period]*100, 'o-k')
            if index_gog:
                plt.plot(index_gog-index_ref, 20.0,'ok', markersize=8)
            if index_ax:
                plt.plot(index_ax-index_ref, 30.0,'om', markersize=8)
            if index_az:
                plt.plot(index_az-index_ref, 40.0,'og', markersize=8)
            if index_dax:
                plt.plot(index_dax-index_ref, 55.0,'dm', markersize=8)
            if index_daz:
                plt.plot(index_daz-index_ref, 50.0,'dg', markersize=8)
            if index_alt:
                plt.plot(index_alt-index_ref, 10.0,'or', markersize=8)
            if index_tdn:
                plt.plot(index_tdn-index_ref, -20.0,'db', markersize=10)
            plt.title(name)
            plt.grid()
            filename = name
            print name
            output_dir = os.path.join(WORKING_DIR, 'Touchdown_graphs')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            plt.savefig(os.path.join(output_dir, filename + '.png'))
            #plt.show()
            plt.clf()
            plt.close()
            
            self.create_kti(index_tdn)

            
class LandingDecelerationEnd(KeyTimeInstanceNode):
    '''
    Whereas peak acceleration at takeoff is a good measure of the start of
    the takeoff roll, the peak deceleration on landing often occurs very late
    in the landing when the brakes are applied harshly for a moment, for
    example when stopping to make a particular turnoff. For this reason we
    prefer to use the end of the steep reduction in airspeed as a measure of
    the end of the landing roll.
    '''
    def derive(self, speed=P('Airspeed'), landings=S('Landing')):
        for landing in landings:
            end_decel = peak_curvature(speed.array, landing.slice, curve_sense='Concave')
            # Create the KTI if we have found one, otherwise point to the end
            # of the data, as sometimes recordings stop in mid-landing phase
            if end_decel:
                self.create_kti(end_decel)
            else:
                self.create_kti(landing.stop_edge)


class LandingTurnOffRunway(KeyTimeInstanceNode):
    # See Takeoff Turn Onto Runway for description.
    def derive(self, head=P('Heading Continuous'),
               landings=S('Landing'),
               fast=S('Fast')):
        for landing in landings:
            # Check the landing slice is robust.
            if landing.slice.start and landing.slice.stop:
                start_search = fast.get_previous(landing.slice.stop)
                if start_search:
                    start_search = start_search.slice.stop

                if (start_search is None) or (start_search < landing.slice.start):
                    start_search = (landing.slice.start + landing.slice.stop) / 2

                head_landing = head.array[start_search:landing.slice.stop]

                peak_bend = peak_curvature(head_landing, curve_sense='Bipolar')

                fifteen_deg = index_at_value(
                    np.ma.abs(head_landing - head_landing[0]), 15.0)

                if peak_bend:
                    landing_turn = start_search + peak_bend
                else:
                    if fifteen_deg and fifteen_deg < peak_bend:
                        landing_turn = start_search + landing_turn
                    else:
                        # No turn, so just use end of landing run.
                        landing_turn = landing.slice.stop

                self.create_kti(landing_turn)


################################################################################        


class AltitudeWhenClimbing(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain altitudes when the aircraft is climbing.
    '''
    NAME_FORMAT = '%(altitude)d Ft Climbing'
    NAME_VALUES = NAME_VALUES_CLIMB

    HYSTERESIS = 0 # Was 10 Q: Better as setting? A: Remove this as we want the true altitudes - DJ

    def derive(self, climbing=S('Climbing'), alt_aal=P('Altitude AAL')):
        alt_array = hysteresis(alt_aal.array, self.HYSTERESIS)
        for climb in climbing:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is crossed)
                # per climbing phase.
                index = index_at_value(alt_array, alt_threshold, climb.slice)
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain heights when the aircraft is descending.
    '''
    NAME_FORMAT = '%(altitude)d Ft Descending'
    NAME_VALUES = NAME_VALUES_DESCENT

    HYSTERESIS = 0 # Was 10 Q: Better as a constant in the settings?

    def derive(self, descending=S('Descending'), alt_aal=P('Altitude AAL')):
        alt_array = alt_aal.array
        for descend in descending:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is
                # crossed) per descending phase. The altitude array is
                # scanned backwards to make sure we trap the last instance at
                # each height.
                index = index_at_value(alt_array, alt_threshold,
                                       slice(descend.slice.stop,
                                             descend.slice.start, -1))
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeSTDWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain Altitude STD heights when the aircraft is
    descending.
    '''
    name = 'Altitude STD When Descending'
    NAME_FORMAT = '%(altitude)d Ft Descending'
    NAME_VALUES = NAME_VALUES_DESCENT

    HYSTERESIS = 0 # Was 10 Q: Better as a constant in the settings?

    def derive(self, descending=S('Descending'), alt_std=P('Altitude STD')):
        alt_array = alt_std.array
        for descend in descending:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is
                # crossed) per descending phase. The altitude array is
                # scanned backwards to make sure we trap the last instance at
                # each height.
                index = index_at_value(alt_array, alt_threshold,
                                       slice(descend.slice.stop,
                                             descend.slice.start, -1))
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class MinsToTouchdown(KeyTimeInstanceNode):
    #TODO: TESTS
    NAME_FORMAT = "%(time)d Mins To Touchdown"
    NAME_VALUES = {'time': [5, 4, 3, 2, 1]}

    def derive(self, touchdowns=KTI('Touchdown')):
        #Q: is it sensible to create KTIs that overlap with a previous touchdown?
        for touchdown in touchdowns:
            for t in self.NAME_VALUES['time']:
                index = touchdown.index - (t * 60 * self.frequency)
                if index > 0:
                    # May happen when data starts mid-flight.
                    self.create_kti(index, time=t)


class SecsToTouchdown(KeyTimeInstanceNode):
    #TODO: TESTS
    NAME_FORMAT = "%(time)d Secs To Touchdown"
    NAME_VALUES = {'time': [90, 30]}

    def derive(self, touchdowns=KTI('Touchdown')):
        #Q: is it sensible to create KTIs that overlap with a previous touchdown?
        for touchdown in touchdowns:
            for t in self.NAME_VALUES['time']:
                index = touchdown.index - (t * self.frequency)
                if index >= 0:
                    self.create_kti(index, time=t)


#################################################################
# ILS Established Markers (primarily for development)

class LocalizerEstablishedStart(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Localizer Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)

class LocalizerEstablishedEnd(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Localizer Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)

class GlideslopeEstablishedStart(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Glideslope Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)


class GlideslopeEstablishedEnd(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Glideslope Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)


class VNAVModeAndEngThrustModeRequired(KeyTimeInstanceNode):
    '''
    Will create a KTI at the point where both discretes are enabled.
    '''
    
    name = 'VNAV Mode And Eng Thrust Mode Required'
    
    def derive(self,
               vnav_mode=P('VNAV Mode'),
               thrust=P('Eng Thrust Mode Required')):
        
        combined = ((thrust.array == 'Required') &
                    (vnav_mode.array == 'Engaged'))
        slices = np.ma.clump_unmasked(np.ma.masked_where(combined == False,
                                                         combined))
        for slice_ in slices:
            self.create_kti(slice_.start)
