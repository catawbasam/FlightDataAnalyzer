import numpy as np
# _ezclump clumps bool arrays into slices. Normally called by clump_masked
# and clump_unmasked but used here to clump discrete arrays.
from numpy.ma.extras import _ezclump

from analysis_engine.exceptions import DataFrameError

from analysis_engine.library import (
    bearing_and_distance,
    cycle_finder,
    find_edges,
    first_valid_sample,
    index_at_value,
    index_closest_value,
    is_index_within_slice,
    is_slice_within_slice,
    rate_of_change,
    repair_mask, 
    shift_slice, 
    shift_slices,
    slices_from_to,
    slices_overlap,
    slices_and,
    slices_or,
    slices_not,
    slices_remove_small_gaps
)

from analysis_engine.node import FlightPhaseNode, A, P, S, KTI, M

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME,
    AIRSPEED_THRESHOLD,
    BOUNCED_LANDING_THRESHOLD,
    GROUNDSPEED_FOR_MOBILE,
    HEADING_RATE_FOR_MOBILE,
    HEADING_TURN_OFF_RUNWAY,
    HEADING_TURN_ONTO_RUNWAY,
    HOLDING_MAX_GSPD,
    HOLDING_MIN_TIME,
    HYSTERESIS_FPALT_CCD,
    INITIAL_CLIMB_THRESHOLD,
    INITIAL_APPROACH_THRESHOLD,
    KTS_TO_MPS,
    LANDING_THRESHOLD_HEIGHT,
    VERTICAL_SPEED_FOR_CLIMB_PHASE,
    VERTICAL_SPEED_FOR_DESCENT_PHASE,
    VERTICAL_SPEED_FOR_LEVEL_FLIGHT,
    RATE_OF_TURN_FOR_FLIGHT_PHASES,
    RATE_OF_TURN_FOR_TAXI_TURNS
)


class Airborne(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        # Just find out when altitude above airfield is non-zero.
        for speedy in fast:
            # Stop here if the aircraft never went fast.
            if speedy.slice.start is None and speedy.slice.stop is None:
                break

            start_point = speedy.slice.start or 0
            stop_point = speedy.slice.stop or len(alt_aal.array)
            # First tidy up the data we're interested in
            work = repair_mask(alt_aal.array[start_point:stop_point])

            # Stop here if there is inadequate airborne data to process.
            if work is None:
                break

            airs = np.ma.clump_unmasked(np.ma.masked_less_equal(work, 0.0))
            # Make sure we propogate None ends to data which starts or ends in
            # midflight.
            for air in airs:
                begin = air.start
                if begin == 0 or speedy.slice.start is None:
                    begin = None
                end = air.stop
                if end == len(alt_aal.array) - (start_point or 0) or \
                   speedy.slice.stop is None:
                    end = None
                if begin is None or end is None:
                    self.create_phase(shift_slice(slice(begin, end),
                                                  start_point))
                else:
                    duration = end - begin
                    if (duration / alt_aal.hz) > AIRBORNE_THRESHOLD_TIME:
                        self.create_phase(shift_slice(slice(begin, end),
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
            back_up = descend.array - descend.array[ga.index - 1]
            ga_start = index_closest_value(back_up, 500,
                                           slice(ga.index,None, -1))
            ga_stop = index_closest_value(climb.array, 500,
                                          slice(ga.index, None))
            ga_slice.append(slice(ga_start, ga_stop))
        self.create_phases(ga_slice)


class Holding(FlightPhaseNode):
    """
    Holding is a process which involves multiple turns in a short period,
    normally in the same sense. We therefore compute the average rate of turn
    over a long period to reject short turns and pass the entire holding
    period.

    Note that this is the only function that should use "Heading Increasing"
    as we are only looking for turns, and not bothered about the sense or
    actual heading angle.
    """
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               hdg=P('Heading Increasing'), 
               lat=P('Latitude Smoothed'), lon=P('Longitude Smoothed')):
        _, height_bands = slices_from_to(alt_aal.array, 20000, 5000)
        # Three minutes should include two turn segments.
        turn_rate = rate_of_change(hdg, 3 * 60) 
        for height_band in height_bands:
            # We know turn rate will be positive because Heading Increasing only
            # increases.
            turn_bands = np.ma.clump_unmasked(
                np.ma.masked_less(turn_rate[height_band], 0.5))
            hold_bands=[]
            for turn_band in shift_slices(turn_bands, height_band.start):
                # Reject short periods and check that the average groundspeed was
                # low. The index is reduced by one sample to avoid overruns, and
                # this is fine because we are not looking for great precision in
                # this test.
                hold_sec = turn_band.stop - turn_band.start
                if (hold_sec > HOLDING_MIN_TIME*alt_aal.frequency):
                    start = turn_band.start
                    stop = turn_band.stop - 1
                    _, hold_dist = bearing_and_distance(
                        lat.array[start], lon.array[start],
                        lat.array[stop], lon.array[stop])
                    if hold_dist/KTS_TO_MPS/hold_sec < HOLDING_MAX_GSPD:
                        hold_bands.append(turn_band)

            self.create_phases(hold_bands)


class Approach(FlightPhaseNode):
    """
    This phase is used to identify an approach which may or may not include
    in a landing. It includes go-arounds, touch-and-go's and of course
    successful landings.

    We can then process all approaches to runways in the same way, which
    makes life easier later on.
    """
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               lands=S('Landing'), go_arounds=S('Go Around And Climbout')):
        # Prepare to extract the slices
        app_slices = []
        ga_slices = []

        for land in lands:
            app_start = index_closest_value(alt_aal.array,
                                            INITIAL_APPROACH_THRESHOLD,
                                            slice(land.slice.start, 0, -1))
            app_slices.append(slice(app_start,land.slice.stop,None))

        for ga in go_arounds:
            # Stretch the start point back to 3000ft (rather than 500ft)
            gapp_start = index_closest_value(alt_aal.array,
                                             INITIAL_APPROACH_THRESHOLD,
                                             slice(ga.slice.start, 0, -1))
            ga_slices.append(slice(gapp_start, ga.slice.stop,None))

        all_apps = slices_or(app_slices, ga_slices)

        if all_apps:
            self.create_phases(all_apps)
        else:
            self.warning('Flight with no valid approach or go-around phase. '
                         'Probably truncated data')


class BouncedLanding(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL'), airs=S('Airborne'),
               fast=S('Fast')):
        for speedy in fast:
            for air in airs:
                if slices_overlap(speedy.slice, air.slice):
                    start = air.slice.stop
                    stop = speedy.slice.stop
                    if start == stop:
                        stop += 1
                    scan = alt_aal.array[start:stop]
                    ht = max(scan)
                    if ht > BOUNCED_LANDING_THRESHOLD:
                        up = np.ma.clump_unmasked(np.ma.masked_less_equal(scan,
                                                                          0.0))
                        self.create_phase(shift_slice(slice(up[0].start,
                                                            up[-1].stop),
                                                      start))


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne')):
        for air in airs:
            pk_idxs, pk_vals = cycle_finder(alt_aal.array[air.slice],
                                            min_step=HYSTERESIS_FPALT_CCD)
            if pk_vals is not None:
                n = 0
                pk_idxs += air.slice.start or 0
                n_vals = len(pk_vals)
                while n < n_vals - 1:
                    pk_val = pk_vals[n]
                    pk_idx = pk_idxs[n]
                    next_pk_val = pk_vals[n + 1]
                    next_pk_idx = pk_idxs[n + 1]
                    if next_pk_val < pk_val:
                        self.create_phase(slice(None, next_pk_idx))
                        n += 1
                    else:
                        # We are going upwards from n->n+1, does it go down
                        # again?
                        if n + 2 < n_vals:
                            if pk_vals[n + 2] < next_pk_val:
                                # Hurrah! make that phase
                                self.create_phase(slice(pk_idx,
                                                        pk_idxs[n + 2]))
                                n += 2
                        else:
                            self.create_phase(slice(pk_idx, None))
                            n += 1


class Climb(FlightPhaseNode):
    def derive(self, 
               toc=KTI('Top Of Climb'),
               eot=KTI('Climb Start'), # AKA End Of Takeoff
               bod=KTI('Bottom Of Descent')):
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
                     closest_toc is None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            self.create_phase(slice(eot, closest_toc))        

        return 


class Climbing(FlightPhaseNode):
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(vert_spd.array[air.slice],
                                         VERTICAL_SPEED_FOR_CLIMB_PHASE)
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))


class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=S('Climb Cruise Descent'),
               tocs=KTI('Top Of Climb'),
               tods=KTI('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent at this
        # time.
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
    """ 
    Descending faster than 500fpm towards the ground
    """
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Vertical speed limits of 500fpm gives good distinction with level
        # flight.
        for air in airs:
            descending = np.ma.masked_greater(vert_spd.array[air.slice],
                                              VERTICAL_SPEED_FOR_DESCENT_PHASE)
            desc_slices = np.ma.clump_unmasked(descending)
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self, 
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent')):
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


class DescentToFlare(FlightPhaseNode):
    '''
    '''

    def derive(self,
            descents=S('Descent'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for descent in descents:
            end = index_at_value(alt_aal.array, 50.0, descent.slice)
            self.create_phase(slice(descent.slice.start, end))


class DescentLowClimb(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               descend=P('Descend For Flight Phases'),
               climb=P('Climb For Flight Phases'),
               fast=S('Fast')):
        my_list=[]

        # Select periods below the initial approach threshold, restricted to
        # periods fast enough to be airborne.
        for speedy in fast:
            dlc = np.ma.masked_greater(alt_aal.array[speedy.slice],
                                       INITIAL_APPROACH_THRESHOLD)
            dlc_list = shift_slices(np.ma.clump_unmasked(dlc),
                                    speedy.slice.start)

            for this_dlc in dlc_list:
                down_idx = np.ma.argmin(descend.array[this_dlc])
                down = descend.array[this_dlc][down_idx]
                up_idx = np.ma.argmax(climb.array[this_dlc])
                up = climb.array[this_dlc][up_idx]
                # OK, we want a descent of more than 500ft followed by a climb
                # of more than 500ft.
                if down < -500 and up > 500 and up_idx > down_idx:
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
        """
        Did the aircraft go fast enough to possibly become airborne?
        
        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        """
        fast_samples = np.ma.clump_unmasked(
            np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD))
        for fast_sample in fast_samples:
            start = fast_sample.start
            stop = fast_sample.stop
            if abs(airspeed.array[start] - AIRSPEED_THRESHOLD) > 20:
                start = None
            if abs(airspeed.array[stop - 1] - AIRSPEED_THRESHOLD) > 20:
                stop = None
            self.create_phase(slice(start, stop))


class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_phases(alt_aal.slices_from_to(500, 0))


class GearExtending(FlightPhaseNode):
    """
    Gear extending and retracting are section nodes, as they last for a
    finite period.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period of 5 seconds at gear down and gear up is included to
    allow for exceedance of gear transit limits.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available
    
    def derive(self, gear_down=M('Gear Down'), 
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning'),
               frame=A('Frame'), airs=S('Airborne')):
        frame_name = frame.value if frame else None

        # Aircraft with red warning captions to show travelling

        if frame_name in ['737-1', '737-3C']:
            gear_warn = np.ma.logical_or(gear_warn_l.array, gear_warn_r.array)
            gear_warn = np.ma.logical_or(gear_warn, gear_warn_n.array)
            slices = _ezclump(gear_warn)
            if gear_warn[0] == 0: 
                gear_moving = slices[1::2] 
            else: 
                gear_moving = slices[::2]             
            for air in airs:
                gear_moves = slices_and([air.slice], gear_moving)
                for gear_move in gear_moves:
                    if gear_down.array[gear_move.start - 1] == \
                            gear_down.array.state['Up']:
                        self.create_phase(gear_move)
        
        # Aircraft without red warning captions for travelling

        elif frame_name in ['737-5', '737-5_NON-EIS', '737-6', '737-i']:
            edge_list=[]
            for air in airs:
                edge_list.append(find_edges(gear_down.array.raw, air.slice))
            # We now have a list of lists and this trick flattens the result.
            for edge in sum(edge_list,[]):
                # We have no transition state, so allow 5 seconds for the
                # gear to extend.
                begin = edge
                end = edge+(5.0*gear_down.frequency)
                self.create_phase(slice(begin, end))
                
        else:
            raise DataFrameError(self.name, frame_name)



class GearRetracting(FlightPhaseNode):
    '''
    See Gear Extending for comments.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available

    def derive(self, gear_down=M('Gear Down'), 
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning'),
               frame=A('Frame'), airs=S('Airborne')):
        frame_name = frame.value if frame else None
        # Aircraft with red warning captions to show travelling

        if frame_name in ['737-1', '737-3C']:
            if not all((gear_warn_l, gear_warn_n, gear_warn_r, frame, airs)):
                raise DataFrameError(self.name, frame_name)
            gear_warn = np.ma.logical_or(gear_warn_l.array, gear_warn_r.array)
            gear_warn = np.ma.logical_or(gear_warn, gear_warn_n.array)
            slices = _ezclump(gear_warn)
            if gear_warn[0] == 0: 
                gear_moving = slices[1::2] 
            else: 
                gear_moving = slices[::2]             
            for air in airs:
                gear_moves = slices_and([air.slice], gear_moving)
                for gear_move in gear_moves:
                    if gear_down.array[gear_move.start - 1] == \
                       gear_down.array.state['Down']:
                        self.create_phase(gear_move)
        
        # Aircraft without red warning captions for travelling

        elif frame_name in ['737-5', '737-5_NON-EIS', '737-6', '737-i']:
            edge_list = []
            for air in airs:
                edge_list.append(find_edges(gear_down.array.raw, air.slice,
                                            direction='falling_edges'))
            # We now have a list of lists and this trick flattens the result.
            for edge in sum(edge_list,[]):
                # We have no transition state, so allow 5 seconds for the
                # gear to retract.
                begin = edge
                end = edge + (5.0 * gear_down.frequency)
                self.create_phase(slice(begin, end))
        else:
            raise DataFrameError(self.name, frame_name)


def scan_ils(beam, ils_dots, height, scan_slice):
    '''
    :param beam: 'localizer' or 'glideslope'
    :type beam: str
    '''
    if beam not in ['localizer', 'glideslope']:
        raise ValueError('Unrecognised beam type in scan_ils')

    # Let's check to see if we have anything to work with...
    if np.ma.count(ils_dots[scan_slice]) < 5:
        return None
    # Find where we first see the ILS indication. We will start from 200ft to
    # avoid getting spurious glideslope readings (hence this code is the same
    # for glide and localizer).

    # Scan for going through 200ft, or in the case of a go-around, the lowest
    # point - hence 'closing' condition.
    idx_200 = index_at_value(height, 200, slice(scan_slice.stop,
                                                scan_slice.start, -1),
                             endpoint='closing')

    # Now work back to 2.5 dots when the indication is first visible.
    dots_25 = index_at_value(np.ma.abs(ils_dots), 2.5,
                             slice(idx_200, scan_slice.start, -1))
    if dots_25 is None:
        dots_25 = scan_slice.start

    # And now work forwards to the point of "Capture", defined as the first
    # time the ILS goes below 1 dot.
    if int(dots_25) == int(idx_200):
        ils_capture_idx = dots_25
    elif first_valid_sample(
        np.ma.abs(ils_dots[dots_25:idx_200]))[1] < 1.0:
        # Aircraft came into the approach phase already on the centreline.
        ils_capture_idx = dots_25
    else:
        ils_capture_idx = index_at_value(np.ma.abs(ils_dots), 1.0,
                                         slice(dots_25, idx_200, +1))
        if ils_capture_idx is None:
            # Did we start with the ILS captured?
            if np.ma.abs(ils_dots[dots_25]) < 1.0:
                ils_capture_idx = dots_25

    if beam == 'localizer':
        ils_end_idx = index_at_value(np.ma.abs(ils_dots), 2.5,
                                     slice(idx_200, None))
        if ils_end_idx is None:
            # Can either never have captured, or data can end at less than 2.5
            # dots.
            countback_idx, last_loc = first_valid_sample(ils_dots[::-1])
            if abs(last_loc) < 2.5:
                ils_end_idx = len(ils_dots) - countback_idx -1
    elif beam == 'glideslope':
        ils_end_idx = idx_200
    else:
        raise ValueError("Unrecognised beam type '%s' in scan_ils" % beam)

    if ils_capture_idx and ils_end_idx:
        return slice(ils_capture_idx, ils_end_idx)
    else:
        return None


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'
    """
    """
    def derive(self, ils_loc=P('ILS Localizer'), 
               alt_aal=P('Altitude AAL For Flight Phases'), apps=S('Approach')):
        for app in apps:
            ils_app = scan_ils('localizer', ils_loc.array, alt_aal.array,
                               app.slice)
            if ils_app is not None:
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
            # Only look for glideslope established if the localizer was
            # established.
            if ils_loc_est.slice.start and ils_loc_est.slice.stop:
                gs_est = scan_ils('glideslope', ils_gs.array, alt_aal.array,
                                  ils_loc_est.slice)
                # If the glideslope signal is corrupt or there is no
                # glidepath (not fitted or out of service) there may be no
                # glideslope established phase, or the proportion of unmasked
                # values may be small.
                if gs_est:
                    good_data = np.ma.count(ils_gs.array[gs_est])
                    all_data = len(ils_gs.array[gs_est])
                    if good_data/all_data < 0.7:
                        self.warning('ILS glideslope signal poor quality in '
                                     'approach - considered not established.')
                        continue
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
                self.debug("Did not establish localiser within +-1dot")
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
    def derive(self, airs=S('Airborne'),
               vert_spd=P('Vertical Speed For Flight Phases')):
        # Vertical speed limit set to identify both level flight and end of
        # takeoff / start of landing.
        for air in airs:
            level_flight = np.ma.masked_outside(
                vert_spd.array[air.slice], -VERTICAL_SPEED_FOR_LEVEL_FLIGHT,
                VERTICAL_SPEED_FOR_LEVEL_FLIGHT)
            level_slices = np.ma.clump_unmasked(level_flight)
            self.create_phases(shift_slices(level_slices, air.slice.start))


class Grounded(FlightPhaseNode):
    '''
    Includes start of takeoff run and part of landing run.
    Was "On Ground" but this name conflicts with a recorded 737-6 parameter name.
    '''
    def derive(self, air=S('Airborne'), speed=P('Airspeed For Flight Phases')):
        data_end=len(speed.array)
        gnd_phases = slices_not(air.get_slices(), begin_at=0, end_at=data_end)
        if not gnd_phases:
            # Either all on ground or all in flight.
            median_speed = np.ma.median(speed.array)
            if median_speed > AIRSPEED_THRESHOLD:
                gnd_phases = [slice(None,None,None)]
            else:
                gnd_phases = [slice(0,data_end,None)]

        self.create_phases(gnd_phases)


class Mobile(FlightPhaseNode):
    """
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases. As Rate Of Turn is derived directly from heading, this
    phase is guaranteed to be operable for very basic aircraft.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Rate Of Turn' in available
    
    def derive(self, rot=P('Rate Of Turn'), gspd=P('Groundspeed')):
        move = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                         (np.ma.abs(rot.array),
                                          HEADING_RATE_FOR_MOBILE))
        
        if move is None:
            return # for the case where nothing happened
        
        if gspd:
            move_gspd = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                                  (np.ma.abs(gspd.array),
                                                   GROUNDSPEED_FOR_MOBILE))
            # moving is a numpy array so needs to be converted to a list of one
            # slice
            move[0] = min(move[0], move_gspd[0])
            move[1] = max(move[1], move_gspd[1])
            
        moves = [slice(move[0], move[1])]
        self.create_phases(moves)
        

class Landing(FlightPhaseNode):
    '''
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and thereby make sure the 50ft startpoint is exact.
    '''
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL'), fast=S('Fast')):

        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data
            # without having to go back and test against the airspeed array?
            # TODO: Improve endpoint checks. DJ
            if (speedy.slice.stop is None or \
                speedy.slice.stop >= len(alt_aal.array)):
                break

            landing_run = speedy.slice.stop
            datum = head.array[landing_run]

            first = landing_run - (300 * alt_aal.frequency)
            landing_begin = index_at_value(alt_aal.array,
                                           LANDING_THRESHOLD_HEIGHT,
                                           slice(first, landing_run))

            # The turn off the runway must lie within five minutes of the
            # landing.
            last = landing_run + (300 * head.frequency)

            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            if landing_end is None:
                # The data ran out before the aircraft left the runway so use
                # all we have.
                landing_end = len(head.array)-1

            self.create_phases([slice(landing_begin, landing_end)])


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase. 
    
    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and make sure the 35ft endpoint is exact.
    """
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL'),
               fast=S('Fast')):

        # Note: This algorithm works across the entire data array, and
        # not just inside the speedy slice, so the final indexes are
        # absolute and not relative references.

        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.

            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if speedy.slice.start is None:
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
            first = max(0, takeoff_run - (300 * head.frequency))
            takeoff_begin = index_at_value(np.ma.abs(head.array - datum),
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
            last = takeoff_run + (300 * alt_aal.frequency)
            takeoff_end = index_at_value(alt_aal.array, INITIAL_CLIMB_THRESHOLD,
                                         slice(takeoff_run, last))

            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            if takeoff_begin and takeoff_end:
                self.create_phases([slice(takeoff_begin, takeoff_end)])


class TakeoffRoll(FlightPhaseNode):
    '''
    Sub-phase primarily for use by the correlation tests
    '''
    def derive(self, lifts = S('Liftoff'), toffs = S('Takeoff')):
        for toff in toffs:
            for lift in lifts:
                if is_index_within_slice(lift.index, toff.slice):
                    self.create_phase(slice(toff.slice.start, lift.index))


class TakeoffRotation(FlightPhaseNode):
    '''
    '''
    def derive(self, lifts = S('Liftoff')):
        lift_index = lifts.get_first().index
        start = lift_index - 4
        end = lift_index + 4
        self.create_phase(slice(start, end))
        
    
################################################################################
# Takeoff/Go-Around Ratings


# TODO: Write some unit tests!
class Takeoff5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''
    def derive(self, toffs=S('Takeoff')):
        '''
        '''
        for toff in toffs:
            self.create_phase(slice(toff.slice.start, toff.slice.start + 300))


# TODO: Write some unit tests!
class GoAround5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''

    def derive(self, gas=S('Go Around And Climbout')):
        '''
        '''
        for ga in gas:
            self.create_phase(slice(ga.slice.start, ga.slice.start + 300))


################################################################################


class TaxiIn(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('Grounded'), lands=S('Landing')):
        land = lands.get_last()
        if not land:
            return
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                taxi_start = land.slice.stop
                taxi_stop = gnd.slice.stop
                self.create_phase(slice(taxi_start, taxi_stop),
                                  name="Taxi In")


class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('Grounded'), toffs=S('Takeoff')):
        if toffs:
            toff = toffs[0]
            for gnd in gnds:
                if slices_overlap(gnd.slice, toff.slice):
                    taxi_start = gnd.slice.start+1
                    taxi_stop = toff.slice.start-1
                    self.create_phase(slice(taxi_start, taxi_stop), name="Taxi Out")


class Taxiing(FlightPhaseNode):
    def derive(self, t_out=S('Taxi Out'), t_in=S('Taxi In')):
        taxi_slices = slices_or(t_out.get_slices(), t_in.get_slices())
        if taxi_slices:
            self.create_phases(taxi_slices)


class TurningInAir(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES in the air
    """
    def derive(self, rate_of_turn=P('Rate Of Turn'), airborne=S('Airborne')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -RATE_OF_TURN_FOR_FLIGHT_PHASES,
                                      RATE_OF_TURN_FOR_FLIGHT_PHASES)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, air.slice)
                    for air in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")


class TurningOnGround(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_TAXI_TURNS on the ground
    """
    def derive(self, rate_of_turn=P('Rate Of Turn'), ground=S('Grounded')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -RATE_OF_TURN_FOR_TAXI_TURNS,RATE_OF_TURN_FOR_TAXI_TURNS)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, gnd.slice)
                    for gnd in ground]):
                self.create_phase(turn_slice, name="Turning On Ground")


# NOTE: Python class name restriction: '2 Deg Pitch To 35 Ft'
class TwoDegPitchTo35Ft(FlightPhaseNode):
    """
    """
    
    name='2 Deg Pitch To 35 Ft'
    
    def derive(self, pitch=P('Pitch'), takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            pitch_2_deg_idx = index_at_value(pitch.array, 2.0, reversed_slice, 
                                             endpoint='closing')
            self.create_section(slice(pitch_2_deg_idx, takeoff.slice.stop),
                                begin=pitch_2_deg_idx,
                                end=takeoff.stop_edge)
