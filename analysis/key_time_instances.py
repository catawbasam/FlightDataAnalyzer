import logging
import numpy as np

from analysis.library import time_at_value_wrapped

from analysis.node import FlightPhaseNode, P, S, KTI

from analysis.node import KeyTimeInstance, KeyTimeInstanceNode

from settings import (CLIMB_THRESHOLD,
                      INITIAL_CLIMB_THRESHOLD,
                      LANDING_ACCELERATION_THRESHOLD,
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
    def derive(self, dlc=S('Descent Low Climb'),
               alt_std=P('Altitude STD')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for this_dlc in dlc:
            kti = np.ma.argmin(alt_std.array[this_dlc.slice])
            self.create_kti(kti + this_dlc.slice.start, 'Bottom Of Descent')
        
           
class ApproachAndLandingLowestPoint(KeyTimeInstanceNode):
    def derive(self, app_lands=S('Approach And Landing'),
               alt_std=P('Altitude STD')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for app_land in app_lands:
            kti = np.ma.argmin(alt_std.array[app_land.slice])
            self.create_kti(kti + app_land.slice.start, 'Approach And Landing Lowest Point')
    

class ClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = time_at_value_wrapped(alt_aal, climb,
                                                        CLIMB_THRESHOLD)
            self.create_kti(initial_climb_index, 'Climb Start')


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
        if 'Altitude AAL For Flight Phases' in available \
           and 'Approach And Landing' in available \
           and 'Climb For Flight Phases' in available:
            return True
        else:
            return False
        
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
                self.create_kti(app.slice.start+pit_index, 'Go Around')


class LandingPeakDeceleration(KeyTimeInstanceNode):
    """
    The landing has been found already, including and the flare and a little
    of the turn off the runway. Here we find the point of maximum
    deceleration, as this should lie between the touchdown when the aircraft
    may be drifting and the turnoff which could be at high speed, but should
    be at a gentler deceleration. This is used to identify the heading and
    location of the landing, as these will be more stable at peak
    deceleration than at the actual point of touchdown where the aircraft may
    still be have drift on.
    """
    def derive(self, landings=S('Landing'), head=P('Heading Continuous'), 
               accel=P('Acceleration Forwards For Flight Phases')):
        for land in landings:
            peak_decel_index = np.ma.argmin(accel.array[land.slice])
            peak_decel_index += land.slice.start
            self.create_kti(peak_decel_index, 'Landing Peak Deceleration')


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
            self.create_kti(n_toc, 'Top Of Climb')


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
            self.create_kti(n_tod, 'Top Of Descent')


class FlapStateChanges(KeyTimeInstanceNode):
    
    def derive(self, flap=P('Flap')):
        # Mark all flap changes, irrespective of the aircraft type :o)
        previous = None
        for index, value in enumerate(flap.array):
            if value == previous:
                continue
            else:
                # Flap moved from previous setting, so record this change:
                self.create_kti(index, 'Flap %d' % value)


# ===============================================================

"""
Takeoff KTIs are derived from the Takeoff Phase
"""

class TakeoffTurnOntoRunway(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to start when the aircraft turns
    # onto the runway, so this KTI is just at the start of that phase.
    def derive(self, toffs=S('Takeoff')):
        for toff in toffs:
            self.create_kti(toff.slice.start, 'Takeoff Turn Onto Runway')


class TakeoffStartAcceleration(KeyTimeInstanceNode):
    def derive(self, toffs=S('Takeoff'), fwd_acc=P('Acceleration Longitudinal')):
        for toff in toffs:
            start_accel = time_at_value_wrapped(fwd_acc, toff, 
                                                TAKEOFF_ACCELERATION_THRESHOLD)
            self.create_kti(toff.slice.start + start_accel, 'Takeoff Start Acceleration')

            
class Liftoff(KeyTimeInstanceNode):
    def derive(self, toffs=S('Takeoff'), roc=P('Rate Of Climb')):
        for toff in toffs:
            lift_time = time_at_value_wrapped(roc, toff, 
                                              RATE_OF_CLIMB_FOR_LIFTOFF)
            self.create_kti(toff.slice.start+lift_time, 'Liftoff')
            

class InitialClimbStart(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to run up to the start of the
    # initial climb, so this KTI is just at the end of that phase.
    def derive(self, toffs=S('Takeoff')):
        for toff in toffs:
            self.create_kti(toff.slice.stop, 'Initial Climb Start')


"""
Landing KTIs are derived from the Landing Phase
"""

class LandingStart(KeyTimeInstanceNode):
    # The Landing flight phase is computed to start passing through 50ft
    # (nominally), so this KTI is just at the end of that phase.
    def derive(self, landings=S('Landing')):
        for landing in landings:
            self.create_kti(landing.slice.start, 'Landing Start')


class Touchdown(KeyTimeInstanceNode):
    # TODO: Establish whether this works satisfactorily. If there are
    # problems with this algorithm we could compute the rate of descent
    # backwards from the runway for greater accuracy.
    def derive(self, landings=S('Landing'), roc=P('Rate Of Climb')):
        for landing in landings:
            land_time = time_at_value_wrapped(roc, landing, 
                                              RATE_OF_CLIMB_FOR_TOUCHDOWN)
            self.create_kti(landing.slice.start+land_time, 'Touchdown')


class LandingTurnOffRunway(KeyTimeInstanceNode):
    # The Landing phase is computed to end when the aircraft turns off the
    # runway, so this KTI is just at the start of that phase.
    def derive(self, landings=S('Landing')):
        for landing in landings:
            self.create_kti(landing.slice.stop, 'Landing Turn Off Runway')


class LandingStartDeceleration(KeyTimeInstanceNode):
    def derive(self, landings=S('Landing'), fwd_acc=P('Acceleration Longitudinal')):
        for landing in landings:
            start_accel = time_at_value_wrapped(fwd_acc, landing, 
                                                LANDING_ACCELERATION_THRESHOLD)
            self.create_kti(landing.slice.start+start_accel, 'Landing Start Deceleration')


#<<<< This style for all climbing events >>>>>

class _10FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented

    
class _20FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented

    
class _35FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented

    
class _50FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        for climb in climbing:
            index = time_at_value_wrapped(alt_aal,
                                          climb, 50)
            self.create_kti(index, '50 Ft In Initial Climb')


class _75FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        for climb in climbing:
            index = time_at_value_wrapped(alt_aal,
                                          climb, 50)
            self.create_kti(index, '50 Ft In Initial Climb')


class _100FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _150FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _200FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _300FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _400FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _500FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _750FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _1000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _1500FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _2000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _2500FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _3000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _3500FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _4000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _5000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _6000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _7000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _8000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _9000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _10000FtClimbing(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


'''
________Approach and Landing______________________________
'''
class _10000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _9000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _8000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _7000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _6000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _5000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _4000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _3500FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _3000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _2000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _1500FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _1000FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _750FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _500FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _400FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _300FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _200FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _150FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _100FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _75FtDescending(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL')):
        return NotImplemented


class _50FtToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _35FtDescending(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _20FtDescending(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _10FtDescending(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _5MinToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _4MinToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _3MinToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _2MinToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _90SecToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _1MinToTouchdown(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


