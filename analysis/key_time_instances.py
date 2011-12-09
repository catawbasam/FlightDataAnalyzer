import logging
import numpy as np

from analysis.library import time_at_value_wrapped

from analysis.node import FlightPhaseNode, P, S, KTI

from analysis.node import KeyTimeInstance, KeyTimeInstanceNode

from settings import (CLIMB_THRESHOLD,
                      INITIAL_CLIMB_THRESHOLD,
                      SLOPE_FOR_TOC_TOD
                      )

'''
kpt['FlapDeployed'] = []
kpt['FlapRetracted'] = []
for flap_operated_period in np.ma.flatnotmasked_contiguous(np.ma.masked_equal(fp.flap.data[block],0.0)):
    kpt['FlapDeployed'].append(first+flap_operated_period.start)
    kpt['FlapRetracted'].append(first+flap_operated_period.stop)
'''

class BottomOfDescent(KeyTimeInstanceNode):
    def derive(self, dlc=S('Descent Low Climb'),
               alt_std=P('Altitude STD')):
        # In the case of descents without landing, this finds the minimum
        # point of the dip.
        for this_dlc in dlc:
            kti = np.ma.argmin(alt_std.array[this_dlc.slice])
            self.create_kti(kti + this_dlc.slice.start, 'Bottom Of Descent')
        
           

class ClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_aal=P('Altitude AAL'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = time_at_value_wrapped(alt_aal, climb,
                                                        CLIMB_THRESHOLD)
            self.create_kti(initial_climb_index, 'Climb Start')


class GoAround(KeyTimeInstanceNode):
    def derive(self, alt_AAL=P('Altitude AAL For Phases'),
               alt_rad=P('Altitude Radio For Phases'),
               fast=S('Fast'),
               climb=P('Climb For Flight Phases')):
        for sect in fast:
            flt = sect.slice
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


class Liftoff(KeyTimeInstanceNode):
    def derive(self, air=S('Airborne')):
        # Basic version to operate with minimal valid data
        for each_section in air:
            self.create_kti(each_section.slice.start, 'Liftoff')
            

class Touchdown(KeyTimeInstanceNode):
    def derive(self, air=S('Airborne')):
        # Basic version to operate with minimal valid data
        for each_section in air:
            self.create_kti(each_section.slice.stop, 'Touchdown')


class InitialClimbStart(KeyTimeInstanceNode):
    def derive(self, alt_radio=P('Altitude Radio'), climbing=S('Climbing')):
        for climb in climbing:
            initial_climb_index = time_at_value_wrapped(alt_radio, climb, 
                                                        INITIAL_CLIMB_THRESHOLD)
            self.create_kti(initial_climb_index, 'Initial Climb Start')

class LandingGroundEffectStart(KeyTimeInstanceNode):
    def derive(self, alt_rad=P('Altitude Radio')):
        return NotImplemented

    
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


class _1000FtInApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _1500FtInApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _3000FtInApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


# Q: Is final approach distinction correct?
class _100FtInFinalApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _150FtInFinalApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented
                

class _500FtInFinalApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _1000FtInFinalApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _2000FtInFinalApproach(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _50FtInInitialClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _100FtInInitialClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _400FtInInitialClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _500FtInInitialClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _1000FtInClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _1500FtInClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _2000FtInClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _3500FtInClimb(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented


class _1MinToLanding(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _90SecToLanding(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _2MinToLanding(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _25FtInLanding(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _50FtInLanding(KeyTimeInstanceNode):
    def derive(self, touchdown=KTI('Touchdown')): # Q: Args?
        return NotImplemented


class _35FtInTakeoff(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD')): # Q: Args?
        return NotImplemented

    
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
