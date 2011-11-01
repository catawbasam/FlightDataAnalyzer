import numpy as np
import datetime
from collections import namedtuple

from analysis.derived import  KeyPointValue, KeyPointValueNode
from analysis.settings import DESCENT_MIN_DURATION

# Find when the flap was last taken up and first put down

#TODO: Change to KeyPointValueNode and generate kpv using self.create_kpv()
###-------------------------------------------------------------------------------
### Key Point Values
### ================
    
##class MaxMachCruise(KeyPointValueNode):
    ##dependencies = [MACH, ALTITUDE_STD]
    
    ##def derive(self, params):
        ##return max(params[MACH][PHASE_CRUISE])


class RateOfDescentHigh(DerivedParameterNode):
    dependencies = [RATE_OF_CLIMB]
    # Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
    DESCENT_MIN_DURATION = 10
    
    def derive(self, ph, params):
        rate_of_climb = params[RATE_OF_CLIMB]
        #TODO: Merge with below RateOfDescentMax accepting a flightphase arg
        kpv_list = []
        for descent_period in np.ma.flatnotmasked_contiguous(ph['Descending']):
            duration = descent_period.stop - descent_period.start
            if duration > self.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb[descent_period])
                howfast = rate_of_climb[descent_period][when]
                kpv = KeyPointValue(descent_period.start+when, howfast, 'RateOfDescent')
                kpv_list.append(kpv)
        return kpv_list
                
                
class RateOfDescentMax(DerivedParameterNode):
    dependencies = [RATE_OF_CLIMB]
    # Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
    DESCENT_MIN_DURATION = 10
    
    def derive(self, ph, params):
        rate_of_climb = params[RATE_OF_CLIMB]
        kpv_list = []
        for descent_period in np.ma.flatnotmasked_contiguous(ph['Descent']):
            duration = descent_period.stop - descent_period.start
            if duration > self.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb[descent_period])
                howfast = rate_of_climb[descent_period][when]
                kpv = KeyPointValue(descent_period.start+when, howfast)
                kpv_list.append(kpv)
        return kpv_list
             
                
class AirspeedMax(DerivedParameterNode):
    dependencies = [AIRSPEED]
    
    def derive(self, ph, params):
        airspeed = params[AIRSPEED].data
        # Use Numpy to locate the maximum airspeed, then go back and get the value.
        n = np.ma.argmax(airspeed)
        airspeed_max = airspeed[n]
        # Create a key point value for this.
        return KeyPointValue(n, airspeed_max)
    
    
class LevelFlightMaxAirspeed(DerivedParameterNode):
    dependencies = [AIRSPEED]
    LEVEL_FLIGHT_DURATION = 60
    
    def derive(self, ph, params):
        airspeed = params[AIRSPEED].data
        kpv_list = []
        for level_flight_period in np.ma.flatnotmasked_contiguous(ph['LevelFlight']):
            #FIXME: Does this assume 1Hz input when comparing duration against period stop/start?
            duration = level_flight_period.stop - level_flight_period.start
            if duration > self.LEVEL_FLIGHT_DURATION:
                when = np.ma.argmax(fp.airspeed.data[block][level_flight_period])
                howfast = fp.airspeed.data[block][level_flight_period][when]
                kpv = KeyPointValue(level_flight_period.start+when, howfast)
                kpv_list.append(kpv)
            else:
                logging.debug('Short duration %d of level flight ignored', duration)
        return kpv_list
    
#TODO:
#toc = altitude_std[kpt['TopOfClimb']] # Indexing n_toc into the reduced array [block]
#kpv['Altitude_TopOfClimb'] = [(kpt['TopOfClimb'], toc, altitude_std)]
#kpv['LandingTurnOffRunway'] = [(block.start+kpt['LandingEndEstimate'],(head_mag[kpt['LandingEndEstimate']] - head_landing), head_mag.param_name)]
#kpv['Head_Landing'] = [(block.start+kpt['LandingEndEstimate'], head_landing%360, head_mag.param_name)]  # Convert to normal compass heading for display
#tod = altitude_std[kpt['TopOfDescent']] # Indexing n_toc into the reduced array [block]
#kpv['Altitude_TopOfDescent'] = [(kpt['TopOfDescent'], tod, altitude_std)]
#kpv['Head_Takeoff'] = [(block.start+kpt['TakeoffStartEstimate'], head_takeoff%360, head_mag.param_name)] # Convert to normal compass heading for display
#kpv['TakeoffTurnOntoRunway'] = [(block.start+turn_onto_runway,head_takeoff - head_mag[turn_onto_runway],head_mag.param_name)]


