import numpy as np

from analysis_engine.library import (hysteresis, 
                                     index_at_value,
                                     is_index_within_slice,
                                     minimum_unmasked,
                                     slices_above,
                                     slices_not,
                                     slices_and,
                                     max_abs_value, 
                                     max_value, 
                                     peak_curvature,
                                     touchdown_inertial)

from analysis_engine.node import (M, P, S, KTI, KeyTimeInstanceNode)

from settings import (CLIMB_THRESHOLD,
                      NAME_VALUES_CLIMB,
                      NAME_VALUES_DESCENT,
                      NAME_VALUES_ENGINE,
                      NAME_VALUES_FLAP,
                      SLOPE_FOR_TOC_TOD,
                      TAKEOFF_ACCELERATION_THRESHOLD,
                      VERTICAL_SPEED_FOR_LIFTOFF,)

def find_toc_tod(alt_data, ccd_slice, mode):
    '''
    :alt_data : numpy masked array of pressure altitude data
    : ccd_slice : slice of a climb/cruise/descent phase above FL100
    : mode : Either 'Climb' or 'Descent' to define which to select.
    '''
    
    # Find the maximum altitude in this slice to reduce the effort later
    peak_index = np.ma.argmax(alt_data[ccd_slice])
    
    if mode == 'Climb':
        section = slice(ccd_slice.start, ccd_slice.start + peak_index + 1,
                        None)
        slope = SLOPE_FOR_TOC_TOD
    else:
        section = slice((ccd_slice.start or 0) + peak_index, ccd_slice.stop,
                        None)
        slope = -SLOPE_FOR_TOC_TOD
        
    # Quit if there is nothing to do here.
    if section.start == section.stop:
        raise ValueError('No range of data for top of climb or descent check')
        
    # Establish a simple monotonic timebase
    timebase = np.arange(len(alt_data[section]))
    # Then scale this to the required altitude data slope
    ramp = timebase * slope
    # For airborne data only, subtract the slope from the climb, then
    # the peak is at the top of climb or descent.
    return np.ma.argmax(alt_data[section] - ramp) + section.start


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


'''
Redundant, as either a go-around, or landing

class ApproachLowestPoint(KeyTimeInstanceNode):
    def derive(self, apps=S('Approach'), alt_aal=P('Altitude AAL')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for app in apps:
            index = np.ma.argmin(alt_aal.array[app.slice]) + app.slice.start
            value = alt_aal.array[index]
            if value:
                self.create_kti(index)
                '''
    

class AutopilotEngagedSelection(KeyTimeInstanceNode):
    name = 'AP Engaged Selection'

    def derive(self, autopilot=P('AP Engaged'), phase=S('Airborne')):
        self.create_ktis_on_state_change(
            'Engaged',
            autopilot.array,
            change='entering',
            phase=phase
        )


class AutopilotDisengagedSelection(KeyTimeInstanceNode):
    name = 'AP Disengaged Selection'

    def derive(self, autopilot=P('AP Engaged'), phase=S('Airborne')):
        self.create_ktis_on_state_change(
            'Engaged',
            autopilot.array,
            change='leaving',
            phase=phase
        )


class AutothrottleEngagedSelection(KeyTimeInstanceNode):
    name = 'AT Engaged Selection'

    def derive(self, autothrottle=P('AT Engaged'), phase=S('Airborne')):
        self.create_ktis_on_state_change(
            'Engaged',
            autothrottle.array,
            change='entering',
            phase=phase
        )


class AutothrottleDisengagedSelection(KeyTimeInstanceNode):
    name = 'AT Disengaged Selection'

    def derive(self, autothrottle=P('AT Engaged'), phase=S('Airborne')):
        self.create_ktis_on_state_change(
            'Engaged',
            autothrottle.array,
            change='leaving',
            phase=phase
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


class Eng_Stop(KeyTimeInstanceNode):
    NAME_FORMAT = 'Eng (%(number)d) Stop'
    NAME_VALUES = NAME_VALUES_ENGINE
    
    @classmethod
    def can_operate(cls, available):
        return 'Eng (*) N2 Min' in available and \
               any(x in available for x in ('Eng (1) N2',
                                            'Eng (2) N2',
                                            'Eng (3) N2',
                                            'Eng (4) N2',))
    
    name = 'Eng (*) Stop'
    def derive(self,
               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2')):
        for number, eng_n2 in enumerate([eng_1_n2, eng_2_n2, eng_3_n2,
                                         eng_4_n2,], start=1):
            power = np.ma.where(eng_n2.array > 30.0, 1, 0)
            self.create_ktis_at_edges(power, direction='falling_edges',
                                      replace_values={'number': number})


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
        return 'Descent Low Climb' in available and 'Altitude AAL For Flight Phases' in available
        
    # List the optimal parameter set here
    
    def derive(self, dlcs=S('Descent Low Climb'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio')):
        for dlc in dlcs:
            if alt_rad:
                pit = np.ma.argmin(alt_rad.array[dlc.slice])
            else:
                pit = np.ma.argmin(alt_aal.array[dlc.slice])
            self.create_kti(pit+dlc.start_edge)


class GoAroundFlapRetracted(KeyTimeInstanceNode):
    def derive(self, flap=P('Flap'), gas=S('Go Around And Climbout')):
        self.create_ktis_at_edges(flap.array, direction='falling_edges', phase=gas)
        

class GoAroundGearSelectedUp(KeyTimeInstanceNode):
    def derive(self, gear=M('Gear Down'), gas=S('Go Around And Climbout')):
        self.create_ktis_on_state_change('Up', gear.array, change='entering', phase=gas)
        

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


class FlapStateChanges(KeyTimeInstanceNode):
    NAME_FORMAT = 'Flap %(flap)d Set'
    NAME_VALUES = NAME_VALUES_FLAP
    
    def derive(self, flap=P('Flap')):
        # Mark all flap changes, and annotate with the new flap position.
        # Could include "phase=airborne" if we want to eliminate ground flap
        # changes.
        self.create_ktis_at_edges(flap.array, direction='all_edges',
                                  name='flap') 


class GearDownSelection(KeyTimeInstanceNode):
    def derive(self, gear_sel_down=M('Gear Down Selected'), phase=S('Airborne')):
        self.create_ktis_on_state_change('Down', gear_sel_down.array, change='entering', phase=phase)    
        

class GearUpSelection(KeyTimeInstanceNode):
    '''
    This covers normal gear up selections, not during a go-around. 
    See "Go Around Gear Retracted" for Go-Around case.
    '''
    def derive(self, gear_sel_up=M('Gear Up Selected'), airs=S('Airborne'),
               gas=S('Go Around And Climbout')):
        air_slices = airs.get_slices()
        ga_slices = gas.get_slices()
        if not air_slices or not ga_slices:
            return
        air_not_ga = slices_and(air_slices, slices_not(ga_slices,
            begin_at=air_slices[0].start,
            end_at=air_slices[-1].stop,
        ))
        good_phases = S() # s = SectionNode()
        good_phases.create_sections(air_not_ga)
        self.create_ktis_on_state_change('Up', gear_sel_up.array, change='entering', phase=airs)    


class TakeoffTurnOntoRunway(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to start when the aircraft turns
    # onto the runway, so at worst this KTI is just the start of that phase.
    # Where possible we compute the sharp point of the turn onto the runway.
    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff'),
               fast=S('Fast')):
        for toff in toffs:
            # Ideally we'd like to work from the start of the Fast phase
            # backwards, but in case there is a problem with the phases,
            # use the midpoint. This avoids identifying the heading
            # change immediately after liftoff as a turn onto the runway.
            start_search=fast.get_next(toff.slice.start).slice.start
            if (start_search is None) or (start_search > toff.slice.stop):
                start_search = (toff.slice.start+toff.slice.stop)/2
            peak_bend = peak_curvature(head.array,slice(
                start_search,toff.slice.start,-1),curve_sense='Bipolar')
            if peak_bend:
                takeoff_turn = peak_bend 
            else:
                takeoff_turn = toff.slice.start
            self.create_kti(takeoff_turn)


class TakeoffAccelerationStart(KeyTimeInstanceNode):
    '''
    The start of the takeoff roll is ideally computed from the forwards
    acceleration down the runway, but a quite respectable "backstop" is
    available from the point where the airspeed starts to increase. This
    allows for aircraft either with a faulty sensor, or no longitudinal
    accelerometer.
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
                if accel.array[takeoff.slice][0]>TAKEOFF_ACCELERATION_THRESHOLD:
                    start_accel = takeoff.slice.start
                else:
                    start_accel=index_at_value(accel.array,
                                               TAKEOFF_ACCELERATION_THRESHOLD,
                                               takeoff.slice)
            
            if start_accel is None:
                # A quite respectable "backstop" is from the rate of change
                # of airspeed. We use this if the acceleration is not
                # available or if, for any reason, the previous computation
                # failed.
                pc = peak_curvature(speed.array[takeoff.slice])
                if pc:
                    start_accel = pc + takeoff.slice.start
                else:
                    pass

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
                index = index_at_value(vert_spd.array, VERTICAL_SPEED_FOR_LIFTOFF, slice(back_2,on_2))
                if index:
                    self.create_kti(index)
                else:
                    # An improved index was not identified.
                    self.create_kti(t0)
            else:
                # No vertical speed parameter available
                self.create_kti(t0)
                


class LowestPointOnApproach(KeyTimeInstanceNode):
    '''
    For any approach phase that did not result in a landing, the lowest point
    is taken as key, from which the position, heading and height will be
    taken as KPVs.
    
    This KTI is essential to collect the related KPVs which inform the
    approach attribute, and thereafter compute the smoothed track.
    '''
    def derive(self, alt_aal=P('Altitude AAL'), alt_rad=P('Altitude Radio'),
               apps=S('Approach'), lands=S('Landing')):
        height = minimum_unmasked(alt_aal.array, alt_rad.array)
        for app in apps:
            index = np.ma.argmin(height[app.slice])
            self.create_kti(index+app.start_edge)
            

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


def find_edges_on_state_change(state, array, change='entering', 
                                phase=None):
    '''
    Derived from original create_ktis for cases where we don't want to create a KTI directly.
    '''
    def state_changes(state, array, edge_list, change, _slice=slice(0, -1)):
        state_periods = np.ma.clump_unmasked(
            np.ma.masked_not_equal(array[_slice], array.state[state]))
        for period in state_periods:
            if change == 'entering':
                edge_list.append(period.start - 0.5
                                 if period.start > 0 else 0.)
            elif change == 'leaving':
                edge_list.append(period.stop - 0.5)
            elif change == 'entering_and_leaving':
                edge_list.append(period.start - 0.5
                                 if period.start > 0 else 0.)
                edge_list.append(period.stop - 0.5)
        return edge_list

    # High level function scans phase blocks or complete array and
    # presents appropriate arguments for analysis. We test for phase.name
    # as phase returns False.
    edge_list = []
    if phase is None:
        state_changes(state, array, edge_list, change)
    else:
        for each_period in phase:
            state_changes(state, array, edge_list, change, each_period.slice)
    return edge_list


class Touchdown(KeyTimeInstanceNode):
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return 'Gear On Ground' in available or \
               all(x in available for x in ('Altitude AAL',
                                            'Airborne',
                                            'Landing',))

    def derive(self, wow=M('Gear On Ground'), roc=P('Vertical Speed Inertial'),
               alt=P('Altitude AAL'), airs=S('Airborne'), lands=S('Landing')):
        # The preamble here checks that the landing we are looking at is
        # genuine, it's not just because the data stopped in mid-flight. We
        # reduce the scope of the search for touchdown to avoid triggering in
        # mid-cruise, and it avoids problems for aircraft where the gear
        # signal changes state on raising the gear (OK, if they do a gear-up
        # landing it won't work, but this will be the least of the problems).
        for air in airs:
            t0 = air.slice.stop
            for land in lands:
                if t0 and is_index_within_slice(t0, land.slice):
                    """
                    # Let's scan from 30ft to 10 seconds after the approximate touchdown moment.
                    startpoint = index_at_value(alt.array, 30.0, slice(t0, t0-200,-1))
                    endpoint = min(t0+10.0*roc.hz, len(roc.array))
                    """

                    # If we have a wheel sensor, use this. It is often a derived
                    # parameter created by ORing the left and right main gear
                    # signals.
                    if wow:
                        edges = find_edges_on_state_change(
                            'Ground', wow.array[land.slice])
                        if edges != []:
                            self.create_kti(edges[0] + (land.slice.start or 0))
                            return
                    
                    if not wow or edges == []:
                        if roc:
                            index, _ = touchdown_inertial(land, roc, alt)
                            self.create_kti(index + land.start_edge)
                        else:
                            self.create_kti(index_at_value(alt.array, 0.0, land.slice))
                        '''
                        # Plot for ease of inspection during development.
                        from analysis_engine.plot_flight import plot_parameter
                        plot_parameter(alt.array[startpoint:endpoint], show=False)
                        plot_parameter(roc.array[startpoint:endpoint]/100.0, show=False)
                        #plot_parameter(on_gnd.array[startpoint:endpoint], show=False)
                        plot_parameter(sm_ht)
                        '''


class LandingTurnOffRunway(KeyTimeInstanceNode):
    # See Takeoff Turn Onto Runway for description.
    def derive(self, head=P('Heading Continuous'),
               landings=S('Landing'),
               fast=P('Fast')):
        for landing in landings:
            # Check the landing slice is robust.
            if landing.slice.start and landing.slice.stop:
                start_search=fast.get_previous(landing.slice.stop).slice.stop
    
                if (start_search is None) or (start_search < landing.slice.start):
                    start_search = (landing.slice.start+landing.slice.stop)/2
                peak_bend = peak_curvature(head.array[slice(
                    start_search,landing.slice.stop)], curve_sense='Bipolar')
                
                if peak_bend:
                    landing_turn = start_search + peak_bend
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


'''
Landing stopping distance limit points.

The two points computed here are the places on the runway where, using brakes
only on a surface with medium or poor braking action the aircraft would stop
furthest down the runway.

class LandingStopLimitPointPoorBraking(KeyTimeInstanceNode):
    def derive(self, gspd=P('Groundspeed'), landings=S('Landing')):
        for landing in landings:
            limit_point, _ = braking_action(gspd, landing, 0.05)
            self.create_kti(landing.slice.start + limit_point)

class LandingStopLimitPointMediumBraking(KeyTimeInstanceNode):
    def derive(self, gspd=P('Groundspeed'), landings=S('Landing')):
        for landing in landings:
            limit_point, _ = braking_action(gspd, landing, MU_MEDIUM)
            self.create_kti(landing.slice.start + limit_point)

class LandingStopLimitPointGoodBraking(KeyTimeInstanceNode):
    def derive(self, gspd=P('Groundspeed'), landings=S('Landing')):
        for landing in landings:
            limit_point, _ = braking_action(gspd, landing, MU_GOOD)
            self.create_kti(landing.slice.start + limit_point)
'''


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

    HYSTERESIS = 0 # Was 10 Q: Better as setting?
    
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
                self.create_kti(index, time=t)


class TAWSTooLowTerrainWarning(KeyTimeInstanceNode):
    name = 'TAWS Too Low Terrain Warning'
    def derive(self, taws_too_low_terrain=P('TAWS Too Low Terrain')):
        slices = slices_above(taws_too_low_terrain.array, 1)[1]
        for too_low_terrain_slice in slices:
            index = too_low_terrain_slice.start
            #value = taws_too_low_terrain.array[too_low_terrain_slice.start]
            self.create_kti(index)
            
            
class LocalizerEstablishedStart(KeyTimeInstanceNode):
    def derive(self, locs=S('ILS Localizer Established')):
        for loc in locs:
            self.create_kti(loc.slice.start)

class LocalizerEstablishedEnd(KeyTimeInstanceNode):
    def derive(self, locs=S('ILS Localizer Established')):
        for loc in locs:
            self.create_kti(loc.slice.stop)
