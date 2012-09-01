import numpy as np

from analysis_engine.library import (find_edges,
                                     hysteresis, 
                                     index_at_value,
                                     index_closest_value,
                                     integrate,
                                     is_index_within_slice,
                                     is_index_within_sections,
                                     minimum_unmasked,
                                     np_ma_zeros_like,
                                     repair_mask,
                                     slices_above,
                                     slices_overlap,
                                     max_value, 
                                     peak_curvature)

from analysis_engine.node import (M, P, S, KTI, KeyTimeInstanceNode)

from settings import (CLIMB_THRESHOLD,
                      VERTICAL_SPEED_FOR_LIFTOFF,
                      VERTICAL_SPEED_FOR_TOUCHDOWN,
                      SLOPE_FOR_TOC_TOD,
                      TAKEOFF_ACCELERATION_THRESHOLD
                      )
from analysis_engine.plot_flight import plot_parameter

def find_toc_tod(alt_data, ccd_slice, mode):
    '''
    :alt_data : numpy masked array of pressure altitude data
    : ccd_slice : slice of a climb/cruise/descent phase above FL100
    : mode : Either 'Climb' or 'Descent' to define which to select.
    '''
    
    # Find the maximum altitude in this slice to reduce the effort later
    peak_index = np.ma.argmax(alt_data[ccd_slice])
    
    if mode == 'Climb':
        section = slice(ccd_slice.start, ccd_slice.start+peak_index+1, None)
        slope = SLOPE_FOR_TOC_TOD
    else:
        section = slice((ccd_slice.start or 0)+peak_index, ccd_slice.stop, None)
        slope = -SLOPE_FOR_TOC_TOD
        
    # Quit if there is nothing to do here.
    if section.start == section.stop:
        raise ValueError, 'No range of data for top of climb or descent check'
        
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
            self.create_kti(kti + this_dlc.slice.start)
        # For descents to landing, end where the aircraft is no longer airborne.
        for air in airs:
            if air.slice.stop:
                self.create_kti(air.slice.stop)
        
           
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
            'On',
            autopilot.array,
            change='entering',
            phase=phase
        )


class AutopilotDisengagedSelection(KeyTimeInstanceNode):
    name = 'AP Disengaged Selection'

    def derive(self, autopilot=P('AP Engaged'), phase=S('Airborne')):
        self.create_ktis_on_state_change(
            'On',
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


class EngAllStop(KeyTimeInstanceNode):
    name = 'Eng (*) Stop'
    def derive(self, eng_n2=P('Eng (*) N2 Min')):
        power = np.ma.where(eng_n2.array > 30.0,1,0)
        self.create_ktis_at_edges(power, direction='falling_edges')


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
            self.create_kti(pit+dlc.slice.start)


class GoAroundFlapRetracted(KeyTimeInstanceNode):
    def derive(self, flap=P('Flap'), gas=S('Go Around And Climbout')):
        self.create_ktis_at_edges(flap.array, direction='falling_edges', phase=gas)
        

class GoAroundGearRetracted(KeyTimeInstanceNode):
    def derive(self, gear=P('Gear Down'), gas=S('Go Around And Climbout')):
        self.create_ktis_at_edges(gear.array, direction='falling_edges', phase=gas)
        

class TopOfClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD'), 
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
            if ccd_slice.start == None:
                break
            # if this is the first point in the slice, it's come from
            # data that is already in the cruise, so we'll ignore this as well
            if n_toc==0:
                break
            # Record the moment (with respect to this section of data)
            self.create_kti(n_toc)


class TopOfDescent(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD'), 
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
            if ccd_slice.stop == None:
                break
            # if this is the last point in the slice, it's come from
            # data that ends in the cruise, so we'll ignore this too.
            if n_tod==ccd_slice.stop - 1:
                break
            # Record the moment (with respect to this section of data)
            self.create_kti(n_tod)


class FlapStateChanges(KeyTimeInstanceNode):
    NAME_FORMAT = 'Flap %(flap)d Set'
    NAME_VALUES = {'flap': range(0, 101, 1)}
    
    def derive(self, flap=P('Flap')):
        # Mark all flap changes, and annotate with the new flap position.
        # Could include "phase=airborne" if we want to eliminate ground flap changes.
        self.create_ktis_at_edges(flap.array, direction='all_edges', name='flap') 


class GearSelectionDown(KeyTimeInstanceNode):
    def derive(self, gear_sel_down=M('Gear Selected Down'), phase=S('Airborne')):
        self.create_ktis_on_state_change('Down', gear_sel_down.array, change='entering', phase=phase)    
        

class GearSelectionUp(KeyTimeInstanceNode):
    def derive(self, gear_sel_up=M('Gear Selected Up'), phase=S('Airborne')):
        self.create_ktis_on_state_change('Up', gear_sel_up.array, change='entering', phase=phase)    


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
            if (start_search == None) or (start_search > toff.slice.stop):
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
            
            if start_accel == None:
                # A quite respectable "backstop" is from the rate of change
                # of airspeed. We use this if the acceleration is not
                # available or if, for any reason, the previous computation
                # failed.
                pc = peak_curvature(speed.array[takeoff.slice])
                if pc:
                    start_accel = pc + takeoff.slice.start
                else:
                    pass

            if start_accel != None:
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
    def derive(self, vert_spd=P('Vertical Speed Inertial'), airs=S('Airborne')):
        for air in airs:
            t0 = air.slice.start
            if t0:
                back_2 = (t0 - 2.0*vert_spd.frequency)
                on_2 = (t0 + 2.0*vert_spd.frequency) + 1 # For indexing
                index = index_at_value(vert_spd.array, VERTICAL_SPEED_FOR_LIFTOFF, slice(back_2,on_2))
                if index:
                    self.create_kti(index)
                else:
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
               apps = S('Approach'), lands=S('Landing')):
        height = minimum_unmasked(alt_aal.array, alt_rad.array)
        for app in apps:
            index = np.ma.argmin(height[app.slice])
            self.create_kti(index+app.slice.start)
            

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
            if landing.slice.start:
                self.create_kti(landing.slice.start)


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
    if phase == None:
        state_changes(state, array, edge_list, change)
    else:
        for each_period in phase:
            state_changes(state, array, edge_list, change, each_period.slice)
    return edge_list


class Touchdown(KeyTimeInstanceNode):
<<<<<<< TREE
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return 'Gear On Ground Discrete' in available or \
               ('Rate Of Climb Inertial'  in available and\
                'Altitude AAL'  in available and\
                'Airborne'  in available and\
                'Landing' in available)
     
    def derive(self, wow = P('Gear On Ground'), 
               roc=P('Rate Of Climb Inertial'), alt=P('Altitude AAL'), 
               airs=S('Airborne'), lands=S('Landing')
=======
    def derive(self, vert_spd=P('Vertical Speed Inertial'), alt=P('Altitude AAL'), 
               airs=S('Airborne'), lands=S('Landing'), on_gnd=P('Gear On Ground')
>>>>>>> MERGE-SOURCE
               ):
<<<<<<< TREE
        # The preamble here checks that the landing we are looking at is
        # genuine, that it, it's not just becasue the data stopped in
        # mid-flight. We reduce the scope of the search for touchdown to
        # avoid triggering in mid-cruise, and it avoids problems for aircraft
        # where the gear signal changes state on raising the gear (OK, if
        # they do a gear-up landing it won't work, but this will be the least
        # of the problems).
=======
        # We do a local integration of the inertial vertical speed to
        # estimate the actual point of landing. This is referenced to the
        # available altitude signal, altitude AAL, which will have been
        # derived from the best available source. This technique
        # leads on to the rate of descent at touchdown KPV which can then
        # make the best calculation of the landing ROD as we know more accurately the time 
        # where the mainwheels touched.
        
        # This technique works well with firm landings, where the ROD at
        # landing is important, but can be inaccurate with very gentle
        # landings. The Gear On Ground signal is included as a sanity check
        # to cover these cases, but for aircraft with no weight on wheels
        # switches, this is ignored..
        
        # Time constant
        tau = 0.3
>>>>>>> MERGE-SOURCE
        for air in airs:
            t0 = air.slice.stop
            if t0 and is_index_within_sections(t0, lands):
                # Let's scan from 30ft to 10 seconds after the approximate touchdown moment.
                startpoint = index_at_value(alt.array, 30.0, slice(t0, t0-200,-1))
<<<<<<< TREE
                endpoint = min(t0+10.0*roc.hz, len(roc.array))

                # If we have a wheel sensor, use this. It is often a derived
                # parameter created by ORing the left and right main gear
                # signals.
                if wow:
                    edges = find_edges_on_state_change('Ground', wow.array[startpoint:endpoint])
                    if edges != []:
                        self.create_kti(edges[0] + startpoint)
                    return
                
                if not wow or edges == []:
                    #For aircraft without weight on wheels swiches, or if
                    #there is a problem with the switch for this landing, we
                    #do a local integration of the inertial rate of climb to
                    #estimate the actual point of landing. This is referenced
                    #to the # available altitude signal, altitude AAL, which
                    #will have been # derived from the best available source.
                    #This technique # leads on to the rate of descent at
                    #landing KPV which can then # make the best calculation
                    #of the landing ROD as we know more accurately the time #
                    #where the mainwheels touched.
        
                    # Time constant of 3 seconds.
                    tau = 1/3.0
                    # Make space for the integrand
                    sm_ht = np_ma_zeros_like(roc.array[startpoint:endpoint])
                    # Repair the source data (otherwise we propogate masked data)
                    my_roc = repair_mask(roc.array[startpoint:endpoint])
                    my_alt = repair_mask(alt.array[startpoint:endpoint])
    
                    # Start at the beginning...
                    sm_ht[0] = alt.array[startpoint]
                    #...and calculate each with a weighted correction factor.
                    for i in range(1, len(sm_ht)):
                        sm_ht[i] = (1.0-tau)*sm_ht[i-1] + tau*my_alt[i-1] + my_roc[i]/60.0/roc.hz
    
                    t1 = index_at_value(sm_ht, 0.0)+startpoint
                    if t1:
                        self.create_kti(t1)
                    
                    '''
                    # Plot for ease of inspection during development.
                    plot_parameter(alt.array[startpoint:endpoint], show=False)
                    plot_parameter(roc.array[startpoint:endpoint]/100.0, show=False)
                    #plot_parameter(on_gnd.array[startpoint:endpoint], show=False)
                    plot_parameter(sm_ht)
                    '''
=======
                endpoint = min(t0+10.0*vert_spd.hz, len(vert_spd.array))
                # Make space for the integrand
                sm_ht = np_ma_zeros_like(vert_spd.array[startpoint:endpoint])
                # Repair the source data (otherwise we propogate masked data)
                my_vert_spd = repair_mask(vert_spd.array[startpoint:endpoint])
                my_alt = repair_mask(alt.array[startpoint:endpoint])

                # Start at the beginning...
                sm_ht[0] = alt.array[startpoint]
                #...and calculate each with a weighted correction factor.
                for i in range(1, len(sm_ht)):
                    sm_ht[i] = (1.0-tau)*sm_ht[i-1] + tau*my_alt[i-1] + my_vert_spd[i]/60.0/vert_spd.hz

                # Plot for ease of inspection during development.
                plot_parameter(alt.array[startpoint:endpoint], show=False)
                plot_parameter(vert_spd.array[startpoint:endpoint]/100.0, show=False)
                plot_parameter(sm_ht)
                
                # The final step is trivial.
                t1 = index_at_value(sm_ht, 0.0)+startpoint
                
                t2 = find_edges(on_gnd.array, slice(startpoint,endpoint), direction='falling_edges')
                
                if t1:
                    self.create_kti(t1)

>>>>>>> MERGE-SOURCE

class LandingTurnOffRunway(KeyTimeInstanceNode):
    # See Takeoff Turn Onto Runway for description.
    def derive(self, head=P('Heading Continuous'),
               landings=S('Landing'),
               fast=P('Fast')):
        for landing in landings:
            # Check the landing slice is robust.
            if landing.slice.start and landing.slice.stop:
                start_search=fast.get_previous(landing.slice.stop).slice.stop
    
                if (start_search == None) or (start_search < landing.slice.start):
                    start_search = (landing.slice.start+landing.slice.stop)/2
                peak_bend = peak_curvature(head.array[slice(
                    start_search,landing.slice.stop)],curve_sense='Bipolar')
                
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
                self.create_kti(landing.slice.stop)


class AltitudeWhenClimbing(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain altitudes when the aircraft is climbing.
    '''
    NAME_FORMAT = '%(altitude)d Ft Climbing'
    ALTITUDES = [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
                 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 
                 9000, 10000]
    NAME_VALUES = {'altitude': ALTITUDES}
    HYSTERESIS = 0 # Was 10 Q: Better as setting? A: Remove this as we want the true altitudes - DJ
    
    def derive(self, climbing=S('Climbing'), alt_aal=P('Altitude AAL')):
        alt_array = hysteresis(alt_aal.array, self.HYSTERESIS)
        for climb in climbing:
            for alt_threshold in self.ALTITUDES:
                # Will trigger a single KTI per height (if threshold is crossed)
                # per climbing phase.
                index = index_at_value(alt_array, alt_threshold, climb.slice)
                if index:
                    pass
                    #self.create_kti(index, altitude=alt_threshold)


class AltitudeWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain heights when the aircraft is descending.
    '''
    NAME_FORMAT = '%(altitude)d Ft Descending'
    ALTITUDES = [10000,9000,8000,7000,6000,5000,4000,3500,3000,2500,2000,1500,\
                 1000,750,500,400,300,200,150,100,75,50,35,20,10]
    NAME_VALUES = {'altitude': ALTITUDES}
    HYSTERESIS = 0 # Was 10 Q: Better as setting?
    
    def derive(self, descending=S('Descending'), alt_aal=P('Altitude AAL')):
        alt_array = alt_aal.array
        for descend in descending:
            for alt_threshold in self.ALTITUDES:
                # Will trigger a single KTI per height (if threshold is
                # crossed) per descending phase. The altitude array is
                # scanned backwards to make sure we trap the last instance at
                # each height.
                index = index_at_value(alt_array, alt_threshold, 
                                       slice(descend.slice.stop,
                                             descend.slice.start,-1))
                if index:
                    pass
                    #self.create_kti(index, altitude=alt_threshold)


class MinsToTouchdown(KeyTimeInstanceNode):
    #TODO: TESTS
    NAME_FORMAT = "%(time)d Mins To Touchdown"
    NAME_VALUES = {'time': [5,4,3,2,1]}
    
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
    NAME_VALUES = {'time': [90,30]}
    
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





class TestKTI(KeyTimeInstanceNode):
    '''
    Test function to operate on "real" hdf data rather than assumed data sets.
    '''
    align_to_first_dependency = False
    
    values_mapping = { 0: 'KTI_Air',
                       1: 'KTI_Ground',}

    def derive(self, gn = M('Gear (N) On Ground'), gear = M('Gear On Ground'), gd = M('Gear On Ground Discrete')):
        print 'KTI Test'
        for i in [5,14,1000, 7220, -1]:
            print
            print 'Nose > ',i, gn.array[i], gn.array.raw[i]
            print 'Main > ',i, gear.array[i], gear.array.raw[i]
            print 'Discrete > ', i, gd.array[i], gd.array.raw[i]
            print gd.array[i] == 'Air'
            '''
            '''
        pass
