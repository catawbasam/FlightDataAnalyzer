import logging
import numpy as np

from analysis.node import FlightPhaseNode, P

from analysis.node import KeyTimeInstance, KeyTimeInstanceNode

from settings import SLOPE_FOR_TOC_TOD

'''
kpt['FlapDeployed'] = []
kpt['FlapRetracted'] = []
for flap_operated_period in np.ma.flatnotmasked_contiguous(np.ma.masked_equal(fp.flap.data[block],0.0)):
    kpt['FlapDeployed'].append(first+flap_operated_period.start)
    kpt['FlapRetracted'].append(first+flap_operated_period.stop)
'''
          

class LiftOff(KeyTimeInstanceNode):
    def derive(self, wow=P('Weight On Wheels')):
        #fp.inertial_rate_of_climb.seek(block, kpt['TakeoffEnd'], kpt['TakeoffStartEstimate'], LIFTOFF_RATE_OF_CLIMB)
        return NotImplemented
                 
                    
                    
class TouchDown(KeyTimeInstanceNode):
    def derive(self, wow=P('Weight On Wheels')):
        fp.inertial_rate_of_climb.seek(block, kpt['LandingEndEstimate'], kpt['LandingStart'], TOUCHDOWN_RATE_OF_DESCENT)
        return NotImplemented



class LandingGroundEffectStart(KeyTimeInstanceNode):
    def derive(self, alt_rad=P('Altitude Radio')):
        return NotImplemented

    
class TopOfClimbTopOfDescent(KeyTimeInstanceNode):
    def derive(self, alt_std=P('Altitude STD'), 
               ccd=P('Climb Cruise Descent')):
        # This checks for the top of climb and descent in each 
        # Climb/Cruise/Descent period of the flight.
        for ccd_slice in ccd:
            # First establish a simple monotonic timebase
            length = len(alt_std.array[ccd_slice])
            timebase = np.arange(len(alt_std.array[ccd_slice]))
            # Then scale this to the required altitude data slope
            slope = timebase * SLOPE_FOR_TOC_TOD
            # For airborne data only, subtract the slope from the climb...
            y = alt_std.array[ccd_slice] - slope
            # and the peak is the top of climb.
            n_toc = np.ma.argmax(y)
            # if this is the first point in the slice, it's come from
            # data that is already in the cruise, so we'll ignore this
            if n_toc>0:
                # Record the moment (with respect to this section of data)
                self.create_kti(ccd_slice.start + n_toc, 'Top Of Climb')

            # Then do the same for the descent
            y = alt_std.array[ccd_slice] + slope
            n_tod = np.ma.argmax(y)
            if n_tod<length-1:
                self.create_kti(ccd_slice.start + n_tod,'Top Of Descent')

        
        
class FlapStateChanges(KeyTimeInstanceNode):
    
    def derive(self, flap=P('flap')):
        # Mark all flap changes, irrespective of the aircraft type :o)
        previous = None
        for index, value in enumerate(flap.array):
            if value == previous:
                continue
            else:
                # Flap moved from previous setting, so record this change:
                self.create_kti(index, 'Flap %d' % value)
    
    
    
    