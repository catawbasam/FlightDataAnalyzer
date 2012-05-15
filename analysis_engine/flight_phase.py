import logging
import numpy as np

from analysis_engine.library import (find_edges,
                                     hysteresis, 
                                     index_at_value,
                                     index_closest_value,
                                     is_slice_within_slice,
                                     minimum_unmasked,
                                     peak_curvature, 
                                     repair_mask, 
                                     shift_slice, 
                                     shift_slices, 
                                     slice_duration,
                                     slices_overlap,
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
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), fast=S('Fast')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        for speedy in fast:
            midpoint = (speedy.slice.start + speedy.slice.stop) / 2
            # If the data starts in the climb, it must be already airborne.
            if roc.array[speedy.slice.start] > RATE_OF_CLIMB_FOR_LEVEL_FLIGHT :
                up = speedy.slice.start
            else:
                # Scan through the first half to find where the aircraft first
                # flies upwards
                up = index_at_value(roc.array, +RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                    slice(speedy.slice.start,midpoint))
            if not up:
                # The aircraft can have been airborne at the start of this
                # segment. If it goes down during this half of the data we
                # can assume it was airborne at the start of the segment.
                if index_at_value(roc.array, -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                  slice(speedy.slice.start,midpoint)):
                    up = speedy.slice.start
                else:
                    up = None
                    
            # Scan backwards through the latter half to find where the
            # aircraft last descends.
            lastpoint = int(speedy.slice.stop)
            if roc.array[lastpoint-1] < -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT:
                down = lastpoint
            elif lastpoint - midpoint < 2:
                down = lastpoint
            else:
                down = index_at_value(roc.array, -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                      slice(lastpoint,midpoint,-1))

            if not down:
                if index_at_value(roc.array, +RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                  slice(lastpoint,midpoint,-1)):
                    down = lastpoint
                else:
                    down = None

            if up and down:
                self.create_phase(slice(up,down))


'''
class Approach(FlightPhaseNode):
    """
    The 'Approach And Landing' phase descends but may also include a climb
    after a go-around. Approach is truncated to just the descent section.
    """
    # List the optimal parameter set here
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               apps=S('Approach And Landing')):
        for app in apps:
            begin = app.slice.start
            pit = np.ma.argmin(alt_AAL.array[app.slice]) + begin
            self.create_phase(slice(begin,pit,None))
'''

class ApproachAndGoAround(FlightPhaseNode):
    # List the optimal parameter set here
    def derive(self, alt_aal=P('Altitude AAL'),
               alt_rad = P('Altitude Radio'),
               climb=P('Climb For Flight Phases'),
               go_arounds=KTI('Go Around'),
               airs=S('Airborne')):
        # Prepare a home for the approach slices
        app_slice = []    
        for air in airs:
            if alt_rad:
                height = minimum_unmasked(alt_aal.array,alt_rad.array)
            else:
                height = alt_aal.array
            
            for ga in go_arounds:
                app_start = index_closest_value(height,INITIAL_APPROACH_THRESHOLD,slice(ga.index,air.slice.start, -1))
                app_stop = index_closest_value(height,500,slice(ga.index,air.slice.stop))
                app_slice.append(slice(app_start,app_stop))
        
        self.create_phases(app_slice)

class ApproachAndLanding(FlightPhaseNode):
    """
    This phase is used to identify an approach which may or may not include
    in a landing. It includes go-arounds, touch-and-go's and of course
    successful landings.
    
    Each Approach And Landing is associated with an airfield and a runway
    where possible. 
    
    The airfield is identified thus:

    if the aircraft lands:
        the airfield closest to the position recorded at maximum deceleration on
        the runway (i.e. LandingLatitude, LandingLongitude KPVs)
    else:
        the airfield closest to the aircraft position at the lowest point of 
        approach (i.e. ApproachMinimumLongitude, ApproachMinimumLatitude KPVs)

    The runway is identified thus:

    if the aircraft lands:
        identify using the runway bearing recorded at maximum deceleration
        (i.e. the LandingHeading KPV)
        
        if there are parallel runways:
            if the ILS is tuned and localizer data is valid:
                use the ApproachILSFrequency KPV to identify the runway

            elseif accurate position data is available:
                use the position (LandingLatitude, LandingLongitude)
                recorded at maximum deceleration to identify the runway

            else:
                use "*" to declare the runway not identified.
                
    else if the aircraft reaches the final approach phase:
        identify the runway bearing from the heading at lowest point of the 
        approach (ApproachMinimumHeading)

        if there are parallel runways:
            if the ILS is tuned and localizer data is valid:
                use ApproachILSFrequency to identify the runway
            else:
                use "*" to declare the runway not identified.
    
    """

    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):

        return 'Altitude AAL For Flight Phases' in available and \
               'Landing' in available
        
    # List the optimal parameter set here
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad = P('Altitude Radio'),
               lands=S('Landing')):
        # Prepare a home for the approach slices
        app_slice = []    
        if alt_rad:
            height = minimum_unmasked(alt_aal.array,
                                      alt_rad.array)
        else:
            height = alt_aal.array

        for land in lands:
            # Ideally we'd like to start at the initial approach threshold...
            app_start = index_at_value(height,
                                       INITIAL_APPROACH_THRESHOLD,
                                       slice(land.slice.start,0, -1))
            # ...but if this fails, take the end of the last climb.
            if app_start == None:
                app_start = index_at_value(height,
                                           INITIAL_APPROACH_THRESHOLD,
                                           slice(land.slice.start,0, -1), 
                                           endpoint='closing')
            app_slice.append(slice(app_start,land.slice.stop,None))
            
        self.create_phases(app_slice)


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt=P('Altitude For Climb Cruise Descent')):
        ccd = np.ma.masked_less(alt.array, ALTITUDE_FOR_CLB_CRU_DSC)
        self.create_phases(np.ma.clump_unmasked(ccd))


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
               tod=P('Top Of Descent'), 
               bod=P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod:
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
    def derive(self, alt=P('Altitude AAL For Flight Phases'),
               climb=P('Climb For Flight Phases'),
               lands=S('Landing'),
               airs=S('Airborne')):
        if not airs:
            return
        my_list=[]
        for air in airs:
            # Select periods below the initial approach threshold
            dlc = np.ma.masked_greater(alt.array[air.slice],
                                       INITIAL_APPROACH_THRESHOLD)
            dlc_list = np.ma.clump_unmasked(dlc)
            for this_dlc in dlc_list:
                this_alt = alt.array[air.slice][this_dlc]
                this_climb = climb.array[air.slice][this_dlc]
                # When is the next landing phase?
                land_start = lands.get_next(this_dlc.start)
                if land_start is None:
                    continue
                # OK, we want a real descent that does not end in a landing,
                # and where the climb exceeds 500ft. Note: Testing
                # "this_alt[0]" is acceptable as this must be valid as a
                # result of the mask and clump process above.
                if this_alt[0]-np.ma.min(this_alt) > 500 and \
                   this_dlc.stop < land_start.slice.start and \
                   np.ma.max(this_climb) > 500:
                    my_list.append(this_dlc)
        self.create_phases(shift_slices(my_list, air.slice.start))

        
class Fast(FlightPhaseNode):

    '''
    Data will have been sliced into single flights before entering the
    analysis engine, so we can be sure that there will be only one fast
    phase. This may have masked data within the phase, but by taking the
    notmasked edges we enclose all the data worth analysing.
    
    Therefore len(Fast) in [0,1]
    '''
    
    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        # Did the aircraft go fast enough to possibly become airborne?
        
        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        fast_samples = np.ma.notmasked_edges(test_array)
        
        if fast_samples is None:
            # Did not go fast enough, or was always fast.
            if np.ma.max(airspeed.array) > AIRSPEED_THRESHOLD:
                fast_samples = np.array([0, len(airspeed.array)])
            else:
                return
        elif fast_samples[0] == fast_samples[1]:
            # Airspeed array either starts or stops Fast.
            index = fast_samples[0]
            if airspeed.array[index] > airspeed.array[index+1]:
                # Airspeed slowing down, start at the beginning of the data.
                fast_samples[0] = np.ma.where(airspeed.array)[0][0]
            else:
                # Airspeed speeding up, start at the end of the data.
                fast_samples[1] = np.ma.where(airspeed.array)[0][-1]
        else:
            # Shift the samples to allow for the indexing at the beginning.
            fast_samples += 1
            
        self.create_phase(slice(*fast_samples))
 

class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               app_lands=S('Approach And Landing')):
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


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'
    """
    Duration of approach with (repaired) Localizer deviation
    continuously less than 1 dot, within either approach phase. Where this
    duration is over 20 seconds, identify as Localizer Established and create
    a phase accordingly. Reminder: 'Approach And Landing' phase is computed
    fromt the parameter 'Altitude AAL For Flight Phases' which in turn is a
    subset of the 'Fast' phase which requires the aircraft to be travelling
    at high speed. Therefore the lowest point is either the bottom of a
    go-around, touch-and-go or landing.
    """
    def scan_ils(self, ils_loc, start):

        # TODO: extract as settings
        LOCALIZER_ESTABLISHED_THRESHOLD = 1.0
        LOCALIZER_ESTABLISHED_MINIMUM_TIME = 30 # Seconds

        # Is the aircraft on the centreline during this phase?
        # TODO: Rethink the mask and thresholds.
        centreline = np.ma.masked_greater(np.ma.abs(repair_mask(ils_loc)),1.0)
        cls = np.ma.clump_unmasked(centreline)
        for cl in cls:
            if cl.stop-cl.start > 30:
                # Long enough to be established and not just crossing the ILS.
                self.create_phase(shift_slice(cl,start))
    

    def derive(self, aals=S('Approach And Landing'),
               aags=S('Approach And Go Around'),
               ils_loc=P('ILS Localizer')):
        for aal in aals:
            self.scan_ils(ils_loc.array[aal.slice],aal.slice.start)
        for aag in aags:
            self.scan_ils(ils_loc.array[aag.slice],aag.slice.start)

    ##'''
    ##Old code - TODO: Remove when new version working
            ###low_index=lowest.get_last(within_slice=aal.slice).index
            ###if np.ma.abs(ils_loc.array[low_index]) > ILS_MAX_SCALE:
                #### Not on ILS localizer at lowest point, so not established.
                ###break
            ###amplitude = np.ma.abs(ils_loc.array)
            ###in_range = np.ma.masked_outside(amplitude,-ILS_MAX_SCALE,ILS_MAX_SCALE)
            ###phases = np.ma.clump_unmasked(in_range)
            ###self.create_phase(phases[-1])
    ##'''


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
        for ils_loc_est in ils_loc_ests:
            # Reduce the duration of the ILS localizer established period
            # down to minimum altitude. TODO: replace 100ft by variable ILS
            # caterogry minima, possibly variable by operator.
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


"""
class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach And Landing')):
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
    def derive(self, airspeed=P('Airspeed')):
        # Did the aircraft go fast enough to possibly become airborne?
        slow_where = np.ma.masked_greater(repair_mask(airspeed.array),
                                       AIRSPEED_THRESHOLD)
        self.create_phases(np.ma.clump_unmasked(slow_where))
    

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
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               alt_rad=P('Altitude Radio For Flight Phases')
               ):
        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data without having to go back and test against the airspeed array? TODO: Improve endpoint checks. DJ
            if speedy.slice.stop >= len(alt_aal.array):
                break
            
            landing_run = speedy.slice.stop
            datum = head.array[landing_run]
            
            if alt_rad:
                first = landing_run - 300*alt_rad.frequency
                landing_begin = index_at_value(alt_rad.array,
                                            LANDING_THRESHOLD_HEIGHT,
                                            slice(first, landing_run))
            else:
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
            
            # If the data stops before this value is reached, substitute the
            # end of the data
            if landing_end == None:
                landing_end = len(head.array)-1
            
            """
            # Where possible use the point of peak curvature.
            try:
                landing_end = min(landing_end, 
                                  peak_curvature(head.array, slice(landing_run, last)))
            except ValueError:
                logging.debug("Lack of data for peak curvature of heading in landing")
                pass
            """

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
            if int(speedy.slice.start) == 0:
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
            
    
class Taxiing(FlightPhaseNode):
    #TODO: Test
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('On Ground'), toffs=S('Takeoff'), lands=S('Landing')):
        for gnd in gnds:
            taxi_start = gnd.slice.start
            taxi_stop = gnd.slice.stop
            for toff in toffs:
                if slices_overlap(gnd.slice, toff.slice):
                    taxi_stop = toff.slice.start
                    self.create_phase(slice(taxi_start, taxi_stop), name="Taxi Out")
            for land in lands:
                if slices_overlap(gnd.slice, land.slice):
                    taxi_start = land.slice.stop
                    self.create_phase(slice(taxi_start, taxi_stop), name="Taxi In")
            #self.create_phase(slice(taxi_start, taxi_stop), name="Taxi")
        
        
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

