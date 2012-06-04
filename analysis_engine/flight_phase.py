import logging
import numpy as np

from analysis_engine.library import (find_edges,
                                     first_valid_sample,
                                     hysteresis, 
                                     index_at_value,
                                     index_closest_value,
                                     is_slice_within_slice,
                                     last_valid_sample,
                                     minimum_unmasked,
                                     peak_curvature, 
                                     repair_mask, 
                                     shift_slice, 
                                     shift_slices, 
                                     slice_duration,
                                     slices_overlap,
                                     slices_and,
                                     slices_or,
                                     slice_samples)

from analysis_engine.node import FlightPhaseNode, A, P, S, KTI

from analysis_engine.settings import (AIRSPEED_THRESHOLD,
                               ALTITUDE_FOR_CLB_CRU_DSC,
                               APPROACH_MIN_DESCENT,
                               FLIGHT_WORTH_ANALYSING_SEC,
                               HEADING_TURN_OFF_RUNWAY,
                               HEADING_TURN_ONTO_RUNWAY,
                               HYSTERESIS_FPROT,
                               HYSTERESIS_FP_RAD_ALT,
                               ILS_MAX_SCALE,
                               INITIAL_CLIMB_THRESHOLD,
                               INITIAL_APPROACH_THRESHOLD,
                               LANDING_THRESHOLD_HEIGHT,
                               RATE_OF_CLIMB_FOR_CLIMB_PHASE,
                               RATE_OF_CLIMB_FOR_DESCENT_PHASE,
                               RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                               RATE_OF_TURN_FOR_FLIGHT_PHASES,
                               RATE_OF_TURN_FOR_TAXI_TURNS
                               )

    
class Airborne(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),fast=S('Fast')):
        # Just find out when altitude above airfield is non-zero.
        for speedy in fast:
            start_point = speedy.slice.start or 0
            stop_point = speedy.slice.stop or len(roc.array)
            self.create_phases(shift_slices(np.ma.clump_unmasked(\
                np.ma.masked_less_equal(\
                    repair_mask(alt_aal.array[start_point:stop_point]),0.0)),
                                            start_point))


class GoAroundAndClimbout(FlightPhaseNode):
    '''
    We already know that the Key Time Instance has been identified at the
    lowest point of the go-around, and that it lies below the 3000ft
    approach thresholds. The function here is to expand the phase 500ft in
    either direction.
    '''
    def derive(self, descend=P('Descend For Flight Phases'),
               climb = P('Climb For Flight Phases'),
               gas=KTI('Go Around')):

        # Prepare a home for multiple go-arounds. (Not a good day, eh?)
        ga_slice = []
        for ga in gas:
            back_up = descend.array - descend.array[ga.index]
            ga_start = index_closest_value(back_up,500,slice(ga.index,None,-1))
            ga_stop = index_closest_value(climb.array,500,slice(ga.index,None))
            ga_slice.append(slice(ga_start,ga_stop))
        self.create_phases(ga_slice)


class Approach(FlightPhaseNode):
    """
    This phase is used to identify an approach which may or may not include
    in a landing. It includes go-arounds, touch-and-go's and of course
    successful landings.
    """
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               lands=S('Landing'), go_arounds=S('Go Around And Climbout')):

        # Prepare a home for the approach slices
        app_slices = []    

        for land in lands:
            app_start = index_closest_value(alt_aal.array,
                                       INITIAL_APPROACH_THRESHOLD,
                                       slice(land.slice.start,0, -1))
            app_slices.append(slice(app_start,land.slice.stop,None))
            
        for ga in go_arounds:
            # When was this go-around below the approach threshold?
            gapp_slice = np.ma.clump_unmasked(\
                np.ma.masked_greater(alt_aal.array[ga.slice],
                                     INITIAL_APPROACH_THRESHOLD))
            # Only make an approach phase if it did go below the approach
            # threshold. Notice there can only be one slice returned from
            # this single go-around.
            if gapp_slice[0]:
                gapp_start = index_closest_value(alt_aal.array,
                                                 INITIAL_APPROACH_THRESHOLD,
                                                 slice(gapp_slice[0].start,0, -1))
                
            app_slices.append(shift_slice(slice(gapp_start, gapp_slice[0].stop),
                                          ga.slice.start))
        
        self.create_phases(app_slices)


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt_ccd=P('Altitude For Climb Cruise Descent'), 
               alt_aal=P('Altitude AAL For Flight Phases')):
        above_1000_ft = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array, 1000.0))
        low_flight = np.ma.clump_unmasked(np.ma.masked_greater(alt_ccd.array, ALTITUDE_FOR_CLB_CRU_DSC))
        low_slices = slices_and(above_1000_ft, low_flight)
        if len(low_slices)==0:
            return
        elif len(low_slices)==1:
            self.create_phase(above_1000_ft[0])
        else:
            # We have descended and climbed again, so split the flights at minimum height points.
            first_climb_start = above_1000_ft[0].start
            climb_stop = np.ma.argmin(alt_aal.array[low_slices[1]])+ \
                low_slices[1].start
            self.create_phase(slice(first_climb_start, climb_stop))
            for i in range(2,len(low_slices)):
                this_climb_start = climb_stop
                climb_stop = np.ma.argmin(alt_aal.array[low_slices[i]])+ \
                    low_slices[i].start
                self.create_phase(slice(this_climb_start, climb_stop))
            this_climb_start = climb_stop
            last_climb_stop = above_1000_ft[0].stop
            self.create_phase(slice(this_climb_start, last_climb_stop))


class Climb(FlightPhaseNode):
    def derive(self, 
               toc=P('Top Of Climb'),
               eot=P('Climb Start'), # AKA End Of Takeoff
               bod=P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)
            
        # Now see which follows a takeoff
        for this_eot in eot:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc == None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            self.create_phase(slice(eot, closest_toc))        
        

        # Now see which follows this minimum
        for this_bod in bod:
            bod = this_bod.index
            # Scan the TODs
            closest_toc = None
            for this_toc in toc_list:
                if (bod < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc == None)):
                    closest_toc = this_toc
                    # Build the slice from what we have found.
                    self.create_phase(slice(bod, closest_toc))        
        return 

        
class Climbing(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), airs=S('Airborne')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(roc.array[air.slice],
                                         RATE_OF_CLIMB_FOR_CLIMB_PHASE)
            climbing_slices = np.ma.clump_unmasked(climbing)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))

      
class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=P('Climb Cruise Descent'),
               tocs=P('Top Of Climb'),
               tods=P('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent at this time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.
        for ccd in ccds:
            toc = tocs.get_first(within_slice=ccd.slice)
            if toc:
                begin = toc.index
            else:
                begin = ccd.slice.start

            tod = tods.get_last(within_slice=ccd.slice)
            if tod:
                end = tod.index
            else:
                end = ccd.slice.stop
                
            self.create_phase(slice(begin,end))


class Descending(FlightPhaseNode):
    """ Descending faster than 500fpm towards the ground
    """
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), airs=S('Airborne')):
        # Rate of climb and descent limits of 500fpm gives good distinction
        # with level flight.
        for air in airs:
            descending = np.ma.masked_greater(roc.array[air.slice],
                                              RATE_OF_CLIMB_FOR_DESCENT_PHASE)
            desc_slices = np.ma.clump_unmasked(descending)
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self, 
               tod_set=P('Top Of Descent'), 
               bod_set=P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod_set:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod_set:
            bod = this_bod.index
            # Scan the TODs
            closest_tod = None
            for this_tod in tod_list:
                if (bod > this_tod and
                    this_tod > closest_tod):
                    closest_tod = this_tod

            # Build the slice from what we have found.
            self.create_phase(slice(closest_tod, bod))        
        return 


class DescentLowClimb(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               climb=P('Climb For Flight Phases'),
               lands=S('Landing'), fast=S('Fast')):
        my_list=[]

        # Select periods below the initial approach threshold, restricted to
        # periods fast enough to be airborne.
        for speedy in fast:
            dlc = np.ma.masked_outside(alt_aal.array[speedy.slice], 0.0, INITIAL_APPROACH_THRESHOLD)
            dlc_list = shift_slices(np.ma.clump_unmasked(dlc), speedy.slice.start)

        for this_dlc in dlc_list:
            this_alt = alt_aal.array[this_dlc]
            this_climb = climb.array[this_dlc]
            # OK, we want a descent followed by a climb that exceeds 500ft.
            # Note: Testing "this_alt[0]" is acceptable as this must be valid
            # as a result of the mask and clump process above.
            if this_alt[0]-np.ma.min(this_alt) > 500 and \
               np.ma.max(this_climb) > 500:
                my_list.append(this_dlc)
        self.create_phases(my_list)

        
class Fast(FlightPhaseNode):

    '''
    Data will have been sliced into single flights before entering the
    analysis engine, so we can be sure that there will be only one fast
    phase. This may have masked data within the phase, but by taking the
    notmasked edges we enclose all the data worth analysing.
    
    Therefore len(Fast) in [0,1]
    
    TODO: Discuss whether this assertion is reliable in the presence of air data corruption.
    '''
    
    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        # Did the aircraft go fast enough to possibly become airborne?
        
        """
        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        """
        
        fast_samples = np.ma.clump_unmasked(np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD))
        
        if fast_samples == []:
            # Did not go fast enough, so no phase created.
            new_list = [slice(None, None)]
                
        else:
            new_list = []
            for fast_sample in fast_samples:
                if abs(airspeed.array[fast_sample.start]-AIRSPEED_THRESHOLD) > 20:
                    fast_sample = slice(None, fast_sample.stop)
                if abs(airspeed.array[fast_sample.stop-1]-AIRSPEED_THRESHOLD) > 20:
                    fast_sample = slice(fast_sample.start, None)
                new_list.append(fast_sample)
        self.create_phases(new_list)
 

class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # From the lower of pressure or radio altitude reading 1000ft to
            # the first showing zero ft (bear in mind Altitude AAL is
            # computed from Altitude Radio below 50ft, so this is not
            # affected by pressure altitude variations at landing).
            alt = repair_mask(minimum_unmasked(alt_aal.array[app_land.slice],
                                               alt_rad.array[app_land.slice]))
            app = np.ma.clump_unmasked(np.ma.masked_outside(alt,0,1000))
            if len(app)>1:
                logging.debug('More than one final approach during a single approach and landing phase')
            if alt[app[0].start] > alt[app[0].stop-1]:  # Trap descents only
                self.create_phase(shift_slice(app[0],app_land.slice.start))


class GearExtending(FlightPhaseNode):
    """
    Gear extending and retracting are section nodes, as they last for a
    finite period.
    
    For aircraft data such as the 737-5 frame used for testing, the transit
    is not recorded, so a dummy period of 5 seconds at gear down and
    gear up is included to allow for exceedance of gear transit limits.
    """
    def derive(self, gear_down=P('Gear Down'), frame=A('Frame'), airs=S('Airborne')):
        frame_name = frame.value if frame else None
        if frame_name in ['737-5']:
            edge_list=[]
            for air in airs:
                edge_list.append(find_edges(gear_down.array, air.slice))
            # We now have a list of lists and this trick flattens the result.
            for edge in sum(edge_list,[]):
                # We have no transition state, so allow 5 seconds for the
                # gear to extend.
                begin = edge-(0.5*gear_down.frequency)
                end = edge+(4.5*gear_down.frequency)
                self.create_phase(slice(begin, end))


class GearRetracting(FlightPhaseNode):
    def derive(self, gear_down=P('Gear Down'), frame=A('Frame'), airs=S('Airborne')):
        frame_name = frame.value if frame else None
        if frame_name in ['737-5']:
            edge_list=[]
            for air in airs:
                edge_list.append(find_edges(gear_down.array, air.slice, direction='falling_edges'))
            # We now have a list of lists and this trick flattens the result.
            for edge in sum(edge_list,[]):
                # We have no transition state, so allow 5 seconds for the
                # gear to retract.
                begin = edge-(0.5*gear_down.frequency)
                end = edge+(4.5*gear_down.frequency)
                self.create_phase(slice(begin, end))


def scan_ils(beam, ils_dots, height, scan_slice):
    '''
    beam = 'localizer' or 'glideslope'
    '''
    # Let's check to see if we have anything to work with...
    if np.ma.count(ils_dots[scan_slice])<5:
        return None
    # Find where we first see the ILS indication. We will start from 200ft to
    # avoid getting spurious glideslope readings (hence this code is the same
    # for glide and localizer).
    
    # Scan for going through 200ft, or in the case of a go-around, the lowest
    # point - hence 'closing' condition.
    idx_200 = index_at_value(height, 200, slice(scan_slice.stop, scan_slice.start, -1), endpoint='closing')

    # Now work back to 2.5 dots when the indication is first visible.
    dots_25 = index_at_value(np.ma.abs(ils_dots), 2.5, slice(idx_200, scan_slice.start, -1))
    if dots_25 == None:
        dots_25 = scan_slice.start

    # And now work forwards to the point of "Capture", defined as the first
    # time the ILS goes below 1 dot.
    if first_valid_sample(np.ma.abs(ils_dots[slice(dots_25, idx_200, +1)]))[1] < 1.0:
        # Aircraft came into the approach phase already on the centreline.
        ils_capture_idx = dots_25
    else:
        ils_capture_idx = index_at_value(np.ma.abs(ils_dots), 1.0, slice(dots_25, idx_200, +1))
    
    if beam == 'localizer':
        ils_end_idx = index_at_value(np.ma.abs(ils_dots), 2.5, slice(idx_200, scan_slice.stop))
        if ils_end_idx == None:
            ils_end_idx = scan_slice.stop
    elif beam == 'glideslope':
        ils_end_idx = idx_200
    else:
        raise ValueError,'Unrecognised beam type in scan_ils'

    return slice(ils_capture_idx, ils_end_idx)


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'
    """
    """
    def derive(self, ils_loc=P('ILS Localizer'), 
               alt_aal=P('Altitude AAL'), apps=S('Approach')):
        for app in apps:
            ils_app = scan_ils('localizer',ils_loc.array,alt_aal.array,app.slice)
            if ils_app != None:
                self.create_phase(ils_app)

  
'''
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    def derive(self, ils_loc = P('ILS Localizer'),
               ils_loc_ests = S('ILS Localizer Established')):
        # For most of the flight, the ILS will not be valid, so we scan only
        # the periods with valid data, ignoring short breaks:
        locs = np.ma.clump_unmasked(repair_mask(ils_loc.array))
        for loc_slice in locs:
            for ils_loc_est in ils_loc_ests:
                est_slice = ils_loc_est.slice
                if slices_overlap(loc_slice, est_slice):
                    before_established = slice(est_slice.start, loc_slice.start, -1)
                    begin = index_at_value(np.ma.abs(ils_loc.array),
                                                     3.0,
                                                     _slice=before_established)
                    end = est_slice.stop
                    self.create_phase(slice(begin, end))
                    '''

    
class ILSGlideslopeEstablished(FlightPhaseNode):
    name = "ILS Glideslope Established"
    """
    Within the Localizer Established phase, compute duration of approach with
    (repaired) Glideslope deviation continuously less than 1 dot,. Where > 10
    seconds, identify as Glideslope Established.
    """
    def derive(self, ils_gs = P('ILS Glideslope'),
               ils_loc_ests = S('ILS Localizer Established'),
               alt_aal=P('Altitude AAL')):
        # We don't accept glideslope approaches without localizer established
        # first, so this only works within that context. If you want to
        # follow a glidepath without a localizer, seek flight safety guidance
        # elsewhere.
        for ils_loc_est in ils_loc_ests:
            gs_est = scan_ils('glideslope', ils_gs.array, alt_aal.array, ils_loc_est.slice)
            self.create_phase(gs_est)
            

        """
        for ils_loc_est in ils_loc_ests:
            # Reduce the duration of the ILS localizer established period
            # down to minimum altitude. TODO: replace 100ft by variable ILS
            # category minima, possibly variable by operator.
            min_index = index_closest_value(alt_aal.array, 100, ils_loc_est.slice)
            
            # ^^^
            #TODO: limit this to 100ft min if the ILS Glideslope established threshold is reduced.            
            
            # Truncate the ILS establiched phase.
            ils_loc_2_min = slice(ils_loc_est.slice.start,
                                  min(ils_loc_est.slice.stop,min_index)) 
            gs = repair_mask(ils_gs.array[ils_loc_2_min]) # prepare gs data
            gsm = np.ma.masked_outside(gs,-1,1)  # mask data more than 1 dot
            ends = np.ma.flatnotmasked_edges(gsm)  # find the valid endpoints
            if ends is None:
                logging.debug("Did not establish localiser within +-1dot")
                continue
            elif ends[0] == 0 and ends[1] == -1:  # TODO: Pythonese this line !
                # All the data is within one dot, so the phase is already known
                self.create_phase(ils_loc_2_min)
            else:
                # Create the reduced duration phase
                reduced_phase = shift_slice(slice(ends[0],ends[1]),ils_loc_est.slice.start)
                # Cases where the aircraft shoots across the glidepath can
                # result in one or two samples within the range, in which
                # case the reduced phase will be None.
                if reduced_phase:
                    self.create_phase(reduced_phase)
            ##this_slice = ils_loc_est.slice
            ##on_slopes = np.ma.clump_unmasked(
                ##np.ma.masked_outside(repair_mask(ils_gs.array)[this_slice],-1,1))
            ##for on_slope in on_slopes:
                ##if slice_duration(on_slope, ils_gs.hz)>10:
                    ##self.create_phase(shift_slice(on_slope,this_slice.start))



class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                   app_land.slice.start))
                                                   """

class LevelFlight(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb For Flight Phases'),airs=S('Airborne')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        for air in airs:
            level_flight = np.ma.masked_outside(roc.array[air.slice], 
                                                -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                                RATE_OF_CLIMB_FOR_LEVEL_FLIGHT)
            level_slices = np.ma.clump_unmasked(level_flight)
            self.create_phases(shift_slices(level_slices, air.slice.start))


class OnGround(FlightPhaseNode):
    '''
    Includes start of takeoff run and part of landing run
    '''
    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        # Did the aircraft go fast enough to possibly become airborne?
        slow_where = np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD)
        self.create_phases(np.ma.clump_masked(slow_where))
    

class Landing(FlightPhaseNode):
    """
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        return 'Heading Continuous' in available and \
               'Altitude AAL For Flight Phases' in available and \
               'Fast' in available
    
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'), fast=S('Fast')):

        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data without having to go back and test against the airspeed array? TODO: Improve endpoint checks. DJ
            if speedy.slice.stop >= len(alt_aal.array) or \
               speedy.slice.stop == None:
                break
            
            landing_run = speedy.slice.stop
            datum = head.array[landing_run]
            
            first = landing_run - 300*alt_aal.frequency
            landing_begin = index_at_value(alt_aal.array,
                                          LANDING_THRESHOLD_HEIGHT,
                                          slice(first, landing_run))
 
            # The turn off the runway must lie within five minutes of the landing.
            last = landing_run + 300*head.frequency
            
            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            
            if landing_begin and landing_end:
                self.create_phases([slice(landing_begin, landing_end)])


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        return 'Heading Continuous' in available and \
               'Altitude AAL For Flight Phases' in available and \
               'Fast' in available
    
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               alt_rad=P('Altitude Radio')):

        # Note: This algorithm works across the entire data array, and
        # not just inside the speedy slice, so the final indexes are
        # absolute and not relative references.

        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.

            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if speedy.slice.start == None:
                break
            
            # The aircraft is part way down it's takeoff run at the start of 
            # the section.
            takeoff_run = speedy.slice.start
            
            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[takeoff_run]
            
            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = max(0, takeoff_run - 300*head.frequency)
            takeoff_begin = index_at_value(np.ma.abs(head.array-datum),
                                          HEADING_TURN_ONTO_RUNWAY,
                                          slice(takeoff_run, first, -1))

            # Where the data starts in line with the runway, default to the
            # start of the data
            if takeoff_begin is None:
                takeoff_begin = first
            
            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.
            
            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            if alt_rad:
                last = takeoff_run + 300*alt_rad.frequency
                takeoff_end = index_at_value(alt_rad.array,
                                             INITIAL_CLIMB_THRESHOLD,
                                             slice(takeoff_run, last))
            else:
                last = takeoff_run + 300*alt_aal.frequency
                takeoff_end = index_at_value(alt_aal.array,
                                             INITIAL_CLIMB_THRESHOLD,
                                             slice(takeoff_run, last))
 
            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            if takeoff_begin and takeoff_end:
                self.create_phases([slice(takeoff_begin, takeoff_end)])


class TOGA5MinRating(FlightPhaseNode):
    #TODO: Test
    """
    For engines, the period of high power operation is normally 5 minutes
    from the start of takeoff. Also applies in the case of a go-around.
    """
    def derive(self, toffs=S('Takeoff'), gas=S('Go Around')):
        for toff in toffs:
            self.create_phase(slice(toff.slice.start, toff.slice.start + 300))
        for ga in gas:
            self.create_phase(slice(ga.slice.start, ga.slice.start + 300))
            
    
class TaxiIn(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('On Ground'), lands=S('Landing')):
        land = lands[-1]
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                taxi_start = land.slice.stop
                taxi_stop = gnd.slice.stop
                self.create_phase(slice(taxi_start, taxi_stop), name="Taxi In")

        
class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('On Ground'), toffs=S('Takeoff')):
        toff = toffs[0]
        for gnd in gnds:
            if slices_overlap(gnd.slice, toff.slice):
                taxi_start = gnd.slice.start
                taxi_stop = toff.slice.start
                self.create_phase(slice(taxi_start, taxi_stop), name="Taxi Out")


class Taxiing(FlightPhaseNode):
    def derive(self, t_out=S('Taxi Out'), t_in=S('Taxi In')):
        self.create_phases(slices_or([s.slice for s in t_out],[s.slice for s in t_in]))
        
        
class TurningInAir(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES in the air
    """
    def derive(self, rate_of_turn=P('Rate Of Turn'), airborne=S('Airborne')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
            -RATE_OF_TURN_FOR_FLIGHT_PHASES,RATE_OF_TURN_FOR_FLIGHT_PHASES)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, air.slice) for air in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")

                
class TurningOnGround(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_TAXI_TURNS on the ground
    """
    def derive(self, rate_of_turn=P('Rate Of Turn'), ground=S('On Ground')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
            -RATE_OF_TURN_FOR_TAXI_TURNS,RATE_OF_TURN_FOR_TAXI_TURNS)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, gnd.slice) for gnd in ground]):
                self.create_phase(turn_slice, name="Turning On Ground")

