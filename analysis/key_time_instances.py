import logging
import numpy as np

from analysis.node import KeyTimeInstance, KeyTimeInstanceNode

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
    name = "Top of Climb and Top of Descent"
    dependencies = ['phase_airborne', 'altitude_std', 'altitude_std_smoothed'] #
    returns = ['top_of_climb', 'top_of_descent']
    
    def derive(self, airborne=P('Airborne'), alt_std=P('Altitude STD')): #altitude_std_smoothed): # TODO: Change to new parameter names.
        """
        Threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
        This was often OK, but one test data sample had a 4000ft climb 20 mins
        after level off. This led to increasing the threshold to 600 fpm in 3
        minutes which has been found to give good qualitative segregation
        between climb, cruise and descent phases.
        """
        # Updated 8/10/11 to allow for multiple cruise phases
        cruise_slices = np.ma.clump_unmasked(np.ma.masked_less(altitude_std_smoothed,10000))
        logging.info('This block has %d cruise phase.' % len(cruise_list))
        for cruise_slice in cruise_slices:
            # First establish a simple monotonic timebase
            timebase = np.arange(len(airspeed[cruise_slice]))
            # Then subtract (or for the descent, add) this slope to the altitude data
            slope = timebase * (600/float(180))
            # For airborne data only, compute a climb graph on a slope
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), alt_std.array[cruise_slice] - slope)
            # and the peak is the top of climb.
            n_toc = np.ma.argmax(y)
            
            # Record the moment (with respect to this cruise)
            kti_list.append(KeyTimeInstance(cruise_slice.start + n_toc, 'TopOfClimb'))
            
            # Let's find the top of descent.
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), alt_std.array[cruise_slice] + slope)
            n_tod = np.ma.argmax(y)
            self.create_kti(cruise_slice.start + n_tod, 'TopOfDescent')
        
        
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
    
    
    
    