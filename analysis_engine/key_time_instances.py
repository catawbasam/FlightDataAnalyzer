import logging
import numpy as np

from analysis_engine.library import (hysteresis, index_at_value,
                                    is_index_within_slice,
                                    min_value,
                                    max_value, peak_curvature)
from analysis_engine.node import P, S, KTI, KeyTimeInstanceNode

from settings import (CLIMB_THRESHOLD,
                      RATE_OF_CLIMB_FOR_LIFTOFF,
                      RATE_OF_CLIMB_FOR_TOUCHDOWN,
                      SLOPE_FOR_TOC_TOD,
                      TAKEOFF_ACCELERATION_THRESHOLD
                      )


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
        section = slice(ccd_slice.start+peak_index, ccd_slice.stop, None)
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
    def derive(self, alt_std=P('Altitude STD'),
               dlc=S('Descent Low Climb')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for this_dlc in dlc:
            kti = np.ma.argmin(alt_std.array[this_dlc.slice])
            self.create_kti(kti + this_dlc.slice.start)
        
           
class ApproachAndLandingLowestPoint(KeyTimeInstanceNode):
    def derive(self, app_lands=S('Approach And Landing'),
               alt_std=P('Altitude STD'), touchdowns=KTI('Touchdown')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for app_land in app_lands:
            for touchdown in touchdowns:
                if is_index_within_slice(touchdown.index, app_land.slice):
                    kti = np.ma.argmin(alt_std.array[app_land.slice])
                    kti = min(kti, touchdown.index)
                    self.create_kti(kti + app_land.slice.start)
    

class ClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = index_at_value(alt_aal.array,
                                                 CLIMB_THRESHOLD, climb.slice)
            # The aircraft may be climbing, but starting from an altitude
            # above CLIMB_THRESHOLD. In this case no kti is created.
            if initial_climb_index:
                self.create_kti(initial_climb_index)


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
        return 'Descent Low Climb' in available and 'Altitude AAL' in available
        
    # List the optimal parameter set here
    
    def derive(self, dlcs=S('Descent Low Climb'),
               alt_aal=P('Altitude AAL'),
               alt_rad=P('Altitude Radio')):
        for dlc in dlcs:
            if alt_rad:
                pit = np.ma.argmin(alt_rad.array[dlc.slice])
            else:
                pit = np.ma.argmin(alt_aal.array[dlc.slice])
            self.create_kti(pit+dlc.slice.start)
        
        
    
    """
    OLD CODE:>
    
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters. If 'Altitude Radio For Flight
        # Phases' is available, that's a bonus and we will use it, but it is
        # not required.
        return 'Altitude AAL For Flight Phases' in available \
           and 'Approach And Landing' in available \
           and 'Climb For Flight Phases' in available
        
    # List the optimal parameter set here
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               approaches=S('Approach And Landing'),
               climb=P('Climb For Flight Phases')):
        for app in approaches:
            if np.ma.maximum(climb.array[app.slice]) > 500:
                # We must have been in an approach phase, then climbed at
                # least 500ft. Mark the lowest point.
                if alt_rad:
                    pit_index = np.ma.argmin(alt_rad.array[app.slice])
                else:
                    # In case this aircraft has no rad alt fitted
                    pit_index = np.ma.argmin(alt_AAL.array[app.slice])
                self.create_kti(app.slice.start+pit_index)
    """

class LandingPeakDeceleration(KeyTimeInstanceNode):
    """
    The landing has been found already, including and the flare and a little
    of the turn off the runway. Here we find the point of maximum
    deceleration, as this should lie between the touchdown when the aircraft
    may be drifting and the turnoff which could be at high speed, but should
    be at a gentler deceleration. This is subsequently used to identify the
    location and heading of the landing, as these will be more stable at peak
    deceleration than at the actual point of touchdown where the aircraft may
    still be have drift on.
    """
    def derive(self, landings=S('Landing'),  
               accel=P('Acceleration Longitudinal')):
        for land in landings:
            index, value = min_value(accel.array, _slice=land.slice)
            self.create_kti(index)


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
            # if this is the last point in the slice, it's come from
            # data that ends in the cruise, so we'll ignore this too.
            if n_tod==ccd_slice.stop - 1:
                break
            # Record the moment (with respect to this section of data)
            self.create_kti(n_tod)


class FlapStateChanges(KeyTimeInstanceNode):
    NAME_FORMAT = 'Flap %(flap)d'
    NAME_VALUES = {'flap': range(0, 46, 1)}
    
    def derive(self, flap=P('Flap')):
        # Mark all flap changes, irrespective of the aircraft type :o)
        previous = None
        for index, value in enumerate(flap.array):
            if value == previous:
                continue
            else:
                # Flap moved from previous setting, so record this change:
                self.create_kti(index, setting=value)


class TakeoffTurnOntoRunway(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to start when the aircraft turns
    # onto the runway, so at worst this KTI is just the start of that phase.
    # Where possible we compute the sharp point of the turn onto the runway.
    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff'),
               fast=S('Fast')):
        for toff in toffs:
            # Where possible use the point of peak curvature.
            try:
                # Ideally we'd like to work from the start of the Fast phase
                # backwards, but in case there is a problem with the phases,
                # use the midpoint. This avoids identifying the heading
                # change immediately after liftoff as a turn onto the runway.
                start_search=fast.get_next(toff.slice.start).slice.start
                if (start_search == None) or (start_search > toff.slice.stop):
                    start_search = (toff.slice.start+toff.slice.stop)/2
                takeoff_turn = peak_curvature(
                    head.array,slice(start_search,toff.slice.start,-1),
                    curve_sense='Bipolar')
            except ValueError:
                # If this didn't find a suitable point, revert to the start
                # of the takeoff phase.
                logging.debug \
                    ("Lack of data for peak curvature of heading in takeoff")
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
            self.create_kti(index)


class Liftoff(KeyTimeInstanceNode):
    def derive(self, roc=P('Rate Of Climb'), toffs=S('Takeoff')):
        for toff in toffs:
            # We scan the data backwards to find the last point where the
            # rate of climb passed the threshold, so transients during the
            # takeoff run will not affect the result.
            scan=slice(toff.slice.stop, toff.slice.start, -1)
            lift_index = index_at_value(roc.array, RATE_OF_CLIMB_FOR_LIFTOFF,
                                        scan)
            if lift_index:
                self.create_kti(lift_index)
            else:
                logging.warning("'%s' does not reach '%s' within '%s' section.",
                                roc.name, RATE_OF_CLIMB_FOR_LIFTOFF, toff.name)
            

class InitialClimbStart(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to run up to the start of the
    # initial climb, so this KTI is just at the end of that phase.
    def derive(self, toffs=S('Takeoff')):
        for toff in toffs:
            self.create_kti(toff.slice.stop)


"""
Landing KTIs are derived from the Landing Phase
"""

class LandingStart(KeyTimeInstanceNode):
    # The Landing flight phase is computed to start passing through 50ft
    # (nominally), so this KTI is just at the end of that phase.
    def derive(self, landings=S('Landing')):
        for landing in landings:
            self.create_kti(landing.slice.start)


class TouchAndGo(KeyTimeInstanceNode):
    #TODO: TESTS
    """
    In POLARIS we define a Touch and Go as a Go-Around that contacted the ground.
    
    TODO: Write a proper version when we have a better landing condition.
    """
    def derive(self, alt_AAL=P('Altitude AAL'), go_arounds=KTI('Go Around')):
        for ga in go_arounds:
            if alt_AAL.array[ga.index] == 0.0:
                # wheels on ground
                self.create_kti(ga.index)


class Touchdown(KeyTimeInstanceNode):
    def derive(self, roc=P('Rate Of Climb'), landings=S('Landing')):
        for landing in landings:
            # TODO: Find out why this is creating Touchdown's under 5 minutes
            # into the data.
            land_index = index_at_value(roc.array, RATE_OF_CLIMB_FOR_TOUCHDOWN,
                                        landing.slice)
            if land_index:
                self.create_kti(land_index)
            else:
                logging.warning("'%s' does not reach '%s' within '%s' section.",
                                roc.name, RATE_OF_CLIMB_FOR_TOUCHDOWN,
                                landing.name)


class LandingTurnOffRunway(KeyTimeInstanceNode):
    # See Takeoff Turn Onto Runway for description.
    def derive(self, head=P('Heading Continuous'),
               landings=S('Landing'),
               fast=P('Fast')):
        for landing in landings:
            try:
                start_search=fast.get_previous(landing.slice.stop).slice.stop
                if (start_search == None) or (start_search < landing.slice.start):
                    start_search = (landing.slice.start+landing.slice.stop)/2
                landing_turn = start_search + \
                    peak_curvature(head.array[
                        slice(start_search,landing.slice.stop)],
                                   curve_sense='Bipolar')
            except ValueError:
                logging.debug \
                    ("Lack of data for peak curvature of heading in takeoff")
                landing_turn = landing.slice.stop
            self.create_kti(landing_turn)
    
                

class LandingDecelerationEnd(KeyTimeInstanceNode):
    def derive(self, speed=P('Airspeed'), landings=S('Landing')):
        for landing in landings:
            end_decel = peak_curvature(speed.array[landing.slice])
            # Create the KTI if we have found one, otherwise point to the end
            # of the data, as sometimes recordings stop in mid-landing phase
            if end_decel:
                self.create_kti(end_decel+landing.slice.start)
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
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain heights when the aircraft is descending.
    '''
    NAME_FORMAT = '%(altitude)d Ft Descending'
    ALTITUDES = [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
                 1500, 2000, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000,
                 10000]
    NAME_VALUES = {'altitude': ALTITUDES}
    HYSTERESIS = 0 # Was 10 Q: Better as setting?
    
    def derive(self, descending=S('Descending'), alt_aal=P('Altitude AAL')):
        ##alt_array = hysteresis(alt_aal.array, self.HYSTERESIS)
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
                    self.create_kti(index, altitude=alt_threshold)


"""
-------------------------------------------------
Superceded by Descending conditions listed above.
-------------------------------------------------

class AltitudeInApproach(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain altitudes when the aircraft is in the approach phase.
    '''
    NAME_FORMAT = '%(altitude)d Ft In Approach'
    ALTITUDES = [1000, 1500, 2000, 3000]
    NAME_VALUES = {'altitude': ALTITUDES}
    HYSTERESIS = 10 # Q: Better as setting?
    
    def derive(self, approaches=S('Approach And Landing'),
               alt_aal=P('Altitude AAL')):
        alt_array = hysteresis(alt_aal.array, self.HYSTERESIS)
        for approach in approaches:
            for alt_threshold in self.ALTITUDES:
                # Will trigger a single KTI per height (if threshold is crossed)
                # per climbing phase.
                index = index_at_value(alt_array, alt_threshold, approach.slice)
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeInFinalApproach(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain heights when the aircraft is in the final approach.
    '''
    NAME_FORMAT = '%(altitude)d Ft In Final Approach'
    ALTITUDES = [100, 500]
    NAME_VALUES = {'altitude': ALTITUDES}
    HYSTERESIS = 10 # Q: Better as setting?
    
    def derive(self, approaches=S('Approach And Landing'),
               alt_aal=P('Altitude AAL')):
        # Attempt to smooth to avoid fluctuations triggering KTIs.
        # Q: Is this required?
        alt_array = hysteresis(alt_aal.array, self.HYSTERESIS)
        for approach in approaches:
            for alt_threshold in self.ALTITUDES:
                # Will trigger a single KTI per height (if threshold is crossed)
                # per climbing phase.
                index = index_at_value(alt_array, alt_threshold, approach.slice)
                if index:
                    self.create_kti(index, altitude=alt_threshold)
"""

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
