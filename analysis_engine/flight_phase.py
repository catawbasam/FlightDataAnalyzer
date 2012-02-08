import logging
import numpy as np

from analysis_engine.library import (hysteresis, index_at_value,
                              index_closest_value,
                              is_slice_within_slice,
                              peak_curvature, repair_mask, 
                              shift_slice, shift_slices, slice_duration)
from analysis_engine.node import FlightPhaseNode, P, S, KTI
from analysis_engine.settings import (AIRSPEED_THRESHOLD,
                               ALTITUDE_FOR_CLB_CRU_DSC,
                               APPROACH_MIN_DESCENT,
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
                               )

    
class Airborne(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), fast=S('Fast')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        for speedy in fast:
            midpoint = (speedy.slice.start + speedy.slice.stop) / 2
            # If the data starts in the climb, it must be already airborne.
            if roc.array[speedy.slice.start] > RATE_OF_CLIMB_FOR_LEVEL_FLIGHT:
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
                    
            # Scan backwards through the latter half to find where the
            # aircraft last descends.
            lastpoint = int(speedy.slice.stop)
            if roc.array[lastpoint-1] < -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT:
                down = lastpoint
            else:
                down = index_at_value(roc.array, -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                      slice(lastpoint,midpoint,-1))

            if not down:
                if index_at_value(roc.array, +RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                  slice(lastpoint,midpoint,-1)):
                    down = lastpoint

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
               fast=P('Fast')):
        # Prepare a home for the approach slices
        app_slice = []    
        for speedy in fast:
            if alt_rad:
                height = np.ma.minimum(alt_aal.array,alt_rad.array)
            else:
                height = alt_aal.array
            
            for ga in go_arounds:
                app_start = index_closest_value(height,INITIAL_APPROACH_THRESHOLD,slice(ga.index,speedy.slice.start, -1))
                app_stop = index_closest_value(height,500,slice(ga.index,speedy.slice.stop))
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

    """
    
    ----------------------------------------------------------------------------
    Was not being called, although these two parameters were available.
    Commented out to see if things worked.
    ----------------------------------------------------------------------------
    
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):

        return 'Altitude AAL For Flight Phases' in available and \
               'Landing' in available
    """
    
    # List the optimal parameter set here
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad = P('Altitude Radio'),
               lands=S('Landing')):
        # Prepare a home for the approach slices
        app_slice = []    
        if alt_rad:
            height = np.ma.minimum(alt_aal.array,
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

"""
class ClimbFromBottomOfDescent(FlightPhaseNode):
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
"""
        
class Climbing(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), fast=S('Fast')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for speedy in fast:
            climbing = np.ma.masked_less(roc.array[speedy.slice],
                                         RATE_OF_CLIMB_FOR_CLIMB_PHASE)
            climbing_slices = np.ma.clump_unmasked(climbing)
            self.create_phases(shift_slices(climbing_slices, speedy.slice.start))

      
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
    def derive(self, roc=P('Rate Of Climb For Flight Phases'), airs=S('Fast')):
        # Rate of climb and descent limits of 500fpm gives good distinction
        # with level flight.
        for air in airs:
            descending = np.ma.masked_greater(roc.array[air.slice],
                                              RATE_OF_CLIMB_FOR_DESCENT_PHASE)
            desc_slices = np.ma.clump_unmasked(descending)
            self.create_phases(shift_slices(desc_slices, air.slice.start))


"""
class DescentToBottomOfDescent(FlightPhaseNode):
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
"""

class DescentLowClimb(FlightPhaseNode):
    def derive(self, alt=P('Altitude AAL For Flight Phases'),
               climb=P('Climb For Flight Phases'),
               lands=S('Landing'),
               fast=S('Fast')):
        my_list=[]
        for speedy in fast:
            # Select periods below the initial approach threshold
            dlc = np.ma.masked_greater(alt.array[speedy.slice],
                                       INITIAL_APPROACH_THRESHOLD)
            dlc_list = np.ma.clump_unmasked(dlc)
            for this_dlc in dlc_list:
                # When is the next landing phase?
                land_start = lands.get_next(this_dlc.start)
                if land_start is None:
                    continue
                # OK, we want a real dip that does not end in a landing, and
                # where the climb exceeds 500ft.
                if (np.ma.ptp(alt.array[0:69]) > 500 and
                    this_dlc.stop < land_start.slice.start and
                    np.ma.max(climb.array[speedy.slice][this_dlc]) > 500):
                    my_list.append(this_dlc)
        self.create_phases(my_list)

        
class Fast(FlightPhaseNode):
    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        # Did the aircraft go fast enough to possibly become airborne?
        fast_where = np.ma.masked_less(repair_mask(airspeed.array),
                                       AIRSPEED_THRESHOLD)
        fast_slices = np.ma.clump_unmasked(fast_where)
        self.create_phases(fast_slices)
 

class FinalApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               app_lands=S('Approach And Landing')):
        for app_land in app_lands:
            # From the lower of pressure or radio altitude reading 1000ft to
            # the first showing zero ft (bear in mind Altitude AAL is
            # computed from Altitude Radio below 50ft, so this is not
            # affected by pressure altitude variations at landing).
            app = np.ma.masked_outside(
                np.ma.minimum(alt_AAL.array,alt_rad.array),0,1000)
            self.create_phases(np.ma.clump_unmasked(app))


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
    def scan_ils(self, abs_ils_loc, start):

        # TODO: extract as settings
        LOCALIZER_ESTABLISHED_THRESHOLD = 1.0
        LOCALIZER_ESTABLISHED_MINIMUM_TIME = 30 # Seconds

        # Is the aircraft on the centreline during this phase?
        centreline = np.ma.masked_greater(abs_ils_loc,1.0)
        cls = np.ma.clump_unmasked(centreline)
        for cl in cls:
            if cl.stop-cl.start > 30:
                # Long enough to be established and not just crossing the ILS.
                self.create_phase(shift_slice(cl,start))
    

    def derive(self, aals=S('Approach And Landing'),
               aags=S('Approach And Go Around'),
               ils_loc=P('ILS Localizer')):
        for aal in aals:
            self.scan_ils(abs(ils_loc.array[aal.slice]),aal.slice.start)
        for aag in aags:
            self.scan_ils(abs(ils_loc.array[aag.slice]),aag.slice.start)

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

  
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    
    """
    @classmethod
    def can_operate(cls, available):
        return True
    """
    
    def derive(self, ils_loc = P('ILS Localizer'),
               fast = S('Fast')):
        return NotImplemented
     

    
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
            if ends[0] == 0 and ends[1] == -1:  # TODO: Pythonese this line !
                # All the data is within one dot, so the phase is already known
                self.create_phase(ils_loc_2_min)
            else:
                # Create the reduced duration phase
                self.create_phase(
                    shift_slice(slice(ends[0],ends[1]),ils_loc_est.slice.start))
            
            
            ##this_slice = ils_loc_est.slice
            ##on_slopes = np.ma.clump_unmasked(
                ##np.ma.masked_outside(repair_mask(ils_gs.array)[this_slice],-1,1))
            ##for on_slope in on_slopes:
                ##if slice_duration(on_slope, ils_gs.hz)>10:
                    ##self.create_phase(shift_slice(on_slope,this_slice.start))


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

"""
class OnGround(FlightPhaseNode):
    '''
    Includes before Liftoff and after Touchdown.
    '''
    def derive(self, airspeed=P('Airspeed')):
        # Did the aircraft go fast enough to possibly become airborne?
        fast_where = np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD)
        a,b = np.ma.flatnotmasked_edges(fast_where)
        self.create_phases([slice(a, b, None)])
    

(a) computes the inverse of the phase we want, 
(b) should be related to fast as below
(c) may be defunct

        fast_where = np.ma.masked_less(repair_mask(airspeed.array),
                                       AIRSPEED_THRESHOLD)
        fast_slices = np.ma.clump_unmasked(fast_where)
        self.create_phases(fast_slices)
    
"""

class Landing(FlightPhaseNode):
    """
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        if 'Heading Continuous' in available and \
           'Altitude AAL For Flight Phases' in available and \
           'Fast' in available:
            return True
        else:
            return False
    
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               alt_rad=P('Altitude Radio For Phases')
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
        if 'Heading Continuous' in available and \
           'Altitude AAL For Flight Phases' in available and\
           'Fast' in available:
            return True
        else:
            return False
    
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               alt_rad=P('Altitude Radio For Phases')):

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
                                          slice(first, takeoff_run))

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
            self.create_phases([slice(takeoff_begin, takeoff_end)])


class Turning(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES
    """
    def derive(self, rate_of_turn=P('Rate Of Turn'), airborne=S('Airborne')):
        turning = np.ma.masked_inside(
            hysteresis(repair_mask(rate_of_turn.array), HYSTERESIS_FPROT),
            RATE_OF_TURN_FOR_FLIGHT_PHASES * (-1.0),
            RATE_OF_TURN_FOR_FLIGHT_PHASES)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, a.slice) for a in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")
            else:
                self.create_phase(turn_slice, name="Turning On Ground")