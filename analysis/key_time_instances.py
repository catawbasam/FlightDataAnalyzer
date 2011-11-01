import logging
import numpy as np

from analysis.derived import KeyTimeInstance, KeyTimeInstanceNode

'''
kpt['FlapDeployed'] = []
kpt['FlapRetracted'] = []
for flap_operated_period in np.ma.flatnotmasked_contiguous(np.ma.masked_equal(fp.flap.data[block],0.0)):
    kpt['FlapDeployed'].append(first+flap_operated_period.start)
    kpt['FlapRetracted'].append(first+flap_operated_period.stop)
'''
          


# KTI
class TopOfClimbTopOfDescent(Derived):
    name = "Top of Climb and Top of Descent"
    dependencies = [PHASE_AIRBORNE, ALTITUDE_STD, ALTITUDE_STD_SMOOTHED]
    returns = [TOP_OF_CLIMB, TOP_OF_DESCENT]
    
    def derive(airborne_phase, altitude_std, altitude_std_smoothed):
        """
        Threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
        This was often OK, but one test data sample had a 4000ft climb 20 mins
        after level off. This led to increasing the threshold to 600 fpm in 3
        minutes which has been found to give good qualitative segregation
        between climb, cruise and descent phases.
        """
        # Updated 8/10/11 to allow for multiple cruise phases
        cruise_slices = np.ma.flatnotmasked_contiguous(np.ma.masked_less(altitude_std_smoothed,10000))
        logging.info('This block has %d cruise phase.' % len(cruise_list))
        kti_list = []
        for cruise_slice in cruise_slices:
        
            # First establish a simple monotonic timebase
            timebase = np.arange(len(airspeed[cruise_slice]))
            # Then subtract (or for the descent, add) this slope to the altitude data
            slope = timebase * (600/float(180))
            # For airborne data only, compute a climb graph on a slope
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), altitude_std[cruise_slice] - slope)
            # and the peak is the top of climb.
            n_toc = np.ma.argmax(y)
            
            # Record the moment (with respect to this cruise)
            kti_list.append(KeyTimeInstance(cruise_slice.start + n_toc, 'TopOfClimb'))
            
            # Let's find the top of descent.
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), altitude_std[cruise_slice] + slope)
            n_tod = np.ma.argmax(y)
    
            kti_list.append(KeyTimeInstance(cruise_slice.start + n_tod, 'TopOfDescent'))
                    
        return kti_list
        
        
class FlapStateChanges(Derived):
    dependencies = [FLAP]
    
    def derive(self, ph, params):
        flap = params[FLAP].data        
        # Mark all flap changes, irrespective of the aircraft type :o)
        kti_list = []
        previous = None
        for index, value in enumerate(flap):
            if value == previous:
                continue
            else:
                # Flap moved from previous setting, so record this change:
                kti = KeyTimeInstance(index, 'Flap %d' % value)
                kti_list.append(kti)
        return kti_list