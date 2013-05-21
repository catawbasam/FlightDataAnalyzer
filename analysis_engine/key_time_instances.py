import numpy as np

from analysis_engine.library import (all_of,
                                     any_of,
                                     coreg,
                                     find_edges_on_state_change,
                                     find_toc_tod,
                                     hysteresis,
                                     index_at_value,
                                     max_value,
                                     minimum_unmasked,
                                     peak_curvature,
                                     slices_and,
                                     slices_not,
                                     touchdown_inertial)

from analysis_engine.node import M, P, S, KTI, KeyTimeInstanceNode

from settings import (CLIMB_THRESHOLD,
                      NAME_VALUES_CLIMB,
                      NAME_VALUES_DESCENT,
                      NAME_VALUES_ENGINE,
                      NAME_VALUES_FLAP,
                      TAKEOFF_ACCELERATION_THRESHOLD,
                      VERTICAL_SPEED_FOR_LIFTOFF,
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


class EngStop(KeyTimeInstanceNode):
    '''
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
            self.create_ktis_at_edges(
                np.ma.where(eng_n2.array > 30.0, 1, 0),
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

    def derive(self, vert_spd=P('Vertical Speed Inertial'), airs=S('Airborne')):
        for air in airs:
            t0 = air.slice.start
            if t0 == None:
                continue

            if vert_spd:
                back_2 = (t0 - 2.0*vert_spd.frequency)
                on_2 = (t0 + 2.0*vert_spd.frequency) + 1 # For indexing
                index = index_at_value(vert_spd.array,
                                       VERTICAL_SPEED_FOR_LIFTOFF,
                                       slice(back_2,on_2))
                if index:
                    self.create_kti(index)
                else:
                    # An improved index was not identified.
                    self.create_kti(t0)
            else:
                # No vertical speed parameter available
                self.create_kti(t0)


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
    Just more development still to do!
    '''
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return all_of(('Altitude AAL', 'Landing'), available)

    def derive(self, gog=M('Gear On Ground'), roc=P('Vertical Speed Inertial'),
               alt=P('Altitude AAL'), lands=S('Landing')):
        # The preamble here checks that the landing we are looking at is
        # genuine, it's not just because the data stopped in mid-flight. We
        # reduce the scope of the search for touchdown to avoid triggering in
        # mid-cruise, and it avoids problems for aircraft where the gear
        # signal changes state on raising the gear (OK, if they do a gear-up
        # landing it won't work, but this will be the least of the problems).

        for land in lands:
            if gog:
                # try using Gear On Ground switch
                edges = find_edges_on_state_change(
                    'Ground', gog.array[land.slice])
                if edges:
                    # use the first contact with ground as touchdown point 
                    # (ignore bounces)
                    index = edges[0] + land.slice.start
                    if not alt:
                        self.create_kti(index)
                        continue
                    elif alt.array[index] < 5.0:
                        # Check computation is OK - we've seen 747 "Gear On
                        # Ground" at 21ft
                        self.create_kti(index)
                        continue
                    else:
                        # Did not find a realistic Gear On Ground trigger.
                        pass
                        # trigger at silly height > work it out from height only
                else:
                    # no gear on ground switch found > work it out from height only
                    pass

            alt_index = index_at_value(alt.array, 0.0, land.slice)
            # no touchdown found by Gear On Ground or it was not available
            if roc:
                # Beware, Q-200 roc caused invalid touchdown results.
                inertial_index = index_at_value(roc.array, 0.0, _slice=land.slice)
                index = min(alt_index, inertial_index)
                if index:
                    # found an intertial touchdown point
                    self.create_kti(index)
                    continue

            # no Gear On Ground or Intertial estimate, use altitude
            index = alt_index
            if index:
                self.create_kti(index)
            else:
                # Altitude did not get to 0 ft!
                continue


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
    NAME = 'Altitude STD When Descending'
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
