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
        
           
class ApproachAndLandingLowest(KeyTimeInstanceNode):
    def derive(self, app_lands=S('Approach And Landing'),
               alt_std=P('Altitude STD')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for app_land in app_lands:
            kti = np.ma.argmin(alt_std.array[app_land.slice])
            self.create_kti(kti + app_land.slice.start, 'Approach And Landing Lowest')
    

class ClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = time_at_value_wrapped(alt_aal, climb,
                                                        CLIMB_THRESHOLD)
            self.create_kti(initial_climb_index, 'Climb Start')


class GoAround(KeyTimeInstanceNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               fast=S('Fast'),
               climb=P('Climb For Flight Phases')):
        for sect in fast:
            flt = sect.slice
            '''
            app = np.ma.masked_where(np.ma.logical_or
                                     (np.ma.minimum(alt_AAL.array[flt],alt_rad.array[flt])>3000,
                                     climb.array[flt]>500), alt_AAL.array[flt])
            phases = np.ma.clump_unmasked(app[flt])
            for phase in phases:
                begin = phase.start
                end = phase.stop
                # Pit is the location of the pressure altitude minimum.
                pit = np.ma.argmin(app[flt][phase])
                # If this is at the start of the data, we are climbing 
                # through this height band. If it is at the end we may have
                # truncated data and we're only really interested in cases
                # where the data follows on from the go-around.
                if (0 != pit != end-begin-1):
                    self.create_kti(flt.start+begin+pit, 'Go Around')
                    '''
            app = np.ma.masked_where(np.ma.minimum(alt_AAL.array[flt],alt_rad.array[flt])>3000,
                                     alt_AAL.array[flt])
            phases = np.ma.clump_unmasked(app[flt])
            for phase in phases:
                begin = phase.start
                end = phase.stop
                # Pit is the location of the pressure altitude minimum.
                pit_index = np.ma.argmin(app[flt][phase])
                # If this is at the start of the data, we are climbing 
                # through this height band.
                if pit_index == 0:
                    continue
                # If it is at the end the data is probably truncated, or we 
                # may have landed.
                if pit_index == end-begin-1:
                    continue
                # Quick check that the pit was at the bottom of a descent.
                check_height = climb.array[flt][phase][pit_index]
                # OK. We were descending, and we have gone up after the pit was
                # reached. How far did we climb?
                peak = np.ma.maximum(climb.array[flt][phase][pit_index:])
                if peak>500:
                    self.create_kti(flt.start+begin+pit_index, 'Go Around')


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


