import numpy as np
import datetime
from collections import namedtuple

from analysis.node import  KeyPointValue, KeyPointValueNode
from analysis import settings

"""
TODO:
=====

* Generate kpv using self.create_kpv()

* Move Max to start rather than end?
"""

##########################################
# KPV about the flight

class TakeoffAirport(KeyPointValueNode):
    dependencies = ['Lift Off']
    def derive(self, params):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class ApproachAirport(KeyPointValueNode):
    dependencies = ['Descent']
    def derive(self, params):
        return NotImplemented
    
class LandingAirport(KeyPointValueNode):
    dependencies = ['Touch Down']
    def derive(self, params):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class TakeoffAltitude(KeyPointValueNode):
    dependencies = ['Lift Off', TakeoffAirport]
    def derive(self, params):
        return NotImplemented
    
class LandingAltitude(KeyPointValueNode):
    dependencies = ['Touch Down', LandingAirport]
    def derive(self, params):
        return NotImplemented


                
                
##########################################
# KPV from A6RKA_KPVvalues.xls


class IndicatedAirspeedAtLiftOff(KeyPointValueNode):
    dependencies = ['Lift Off', 'Indicated Airspeed']
    def derive(self, params):
        return NotImplemented
    
class PitchAtLiftOff(KeyPointValueNode):
    dependencies = ['Lift Off', 'Pitch']
    def derive(self, params):
        return NotImplemented
   
   
class FlapAtLiftOff(KeyPointValueNode):
    dependencies = ['Lift Off', 'Flap']
    def derive(self, params):
        return NotImplemented

class IndicatedAirspeedAt35Feet(KeyPointValueNode):
    """ Based on Radio Altitude
    """
    dependencies = ['Indicated Airspeed', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class NormalgLiftOffTo35FeetMax(KeyPointValueNode):
    dependencies = ['Normal g', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class NormalgMaxDeviation(KeyPointValueNode):
    """ For discussion - why have Max and Min Normal g when it's just the max 
    distance from 0.98 that's interesting?
    """
    dependencies = ['Normal g', 'Airborne']
    def derive(self, params):
        STANDARD_GRAVITY = 9.80665
        normg_in_air = params['Normal g'].data[params['Airborne']]
        gdiff = np.ma.absolute(normg_in_air - STANDARD_GRAVITY)
        max_index = gdiff.argmax()
        return self.create_kpv(max_index, gdiff[max_index])    
    
class Pitch35To400FeetMax(KeyPointValueNode):
    dependencies = ['Pitch', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class Pitch1000To100FeetMax(KeyPointValueNode):
    dependencies = ['Pitch', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
    
class Pitch5FeetToTouchDownMax(KeyPointValueNode):
    dependencies = ['Pitch', 'Radio Altitude', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
    
class PitchCycles(KeyPointValueNode):
    """ Count
    """
    dependencies = ['Pitch']
    def derive(self, params):
        return NotImplemented
    
class Pitch35To400FeetMin(KeyPointValueNode):
    dependencies = ['Pitch', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class Pitch1000To100FeetMin(KeyPointValueNode):
    dependencies = ['Pitch', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
    
class Pitch20FeetToTouchDownMin(KeyPointValueNode):
    """ Q: This is 20 feet, the max uses 5 feet
    """
    dependencies = ['Pitch', 'Radio Altitude', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
class PitchRateLiftOffTo35FeetMax(KeyPointValueNode):
    dependencies = ['Pitch Rate', 'Lift Off', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class PitchRate35To1500FeetMax(KeyPointValueNode):
    dependencies = ['Pitch Rate', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
    
    
class RollBelow20FeetMax(KeyPointValueNode): # absolute max?
    dependencies = ['Roll', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
   
class RollBetween100And500FeetMax(KeyPointValueNode): # absolute max?
    dependencies = ['Roll', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
    
class RollBetween500And1500FeetMax(KeyPointValueNode):  # absolue max?
    dependencies = ['Roll', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented   
    
class RollAbove1500FeetMax(KeyPointValueNode):
    dependencies = ['Roll', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
    
class RollCycles1000FeetToTouchDown(KeyPointValueNode):
    dependencies = ['Roll', 'Altitude AAL', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
class MaxAltitudeWithFlaps(KeyPointValueNode):
    """ It's max Altitude not Max Flaps
    """
    dependencies = ['Flap', 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class AltitudeStdMax(KeyPointValueNode):
    dependencies = ['Atitude Std']
    def derive(self, params):
        alt = params['Altitude Std']
        max_index = alt.argmax()
        return self.create_kpv(max_index, alt[max_index])
        
class IndicatedAirspeedMax(KeyPointValueNode):
    dependencies = ['Indicated Airspeed']
    
    def derive(self, params):
        airspeed = params['Indicated Airspeed']
        # Use Numpy to locate the maximum airspeed, then go back and get the value.
        index = np.ma.argmax(airspeed)
        airspeed_max = airspeed[index]
        return self.create_kpv(index, airspeed_max)
    
class MACHMax(KeyPointValueNode):
    name = 'MACH Max'
    dependencies = ['MACH']
    def derive(self, params):
        return NotImplemented
    

class IndicatedAirspeedAtTouchDown(KeyPointValueNode):
    dependencies = ['IndicatedAirspeed', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
class GroundSpeedOnGroundMax(KeyPointValueNode):
    dependencies = ['Ground Speed', 'On Ground']
    def derive(self, params):
        return NotImplemented

class FlapAtTouchDown(KeyPointValueNode):
    dependencies = ['Flap', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
class GrossWeightAtTouchDown(KeyPointValueNode):
    dependencies = ['Gross Weight', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
class EGTMax(KeyPointValueNode): # which engine? or all engines? # or all and each!?
    dependencies = ['Engine (1) EGT', 'Engine (2) EGT', 'Engine (3) EGT', 'Engine (4) EGT']
    ##returns = "EGT Max"  # add which engine?
    NAME_FORMAT = 'EGT Max %{engine}'
    RETURN_OPTIONS = {'engine': dependencies + ['Engine (*) EGT']}

    @classmethod
    def can_operate(cls, available):
        if set(cls.dependencies).intersection(available):
            return True  # if ANY are available
        else:
            return False  # we have no EGT recorded on any engines
        
    def derive(self, params):
        ##egt1 = params['Engine (1) EGT']
        ##egt2 = params['Engine (2) EGT'] 
        ##egt3 = params['Engine (3) EGT'] 
        ##egt4 = params['Engine (4) EGT']
        
        kmax = vmax = imax = None
        for k, v in params.iteritems():
            _imax = v.argmax()
            _vmax = v[_imax]
            if _vmax > vmax:
                imax = _imax # index of max
                vmax = _vmax # max value
                kmax = k # param name of max eng
        return self.create_kpv(imax, vmax, engine=kmax) # include engine using kmax?
    
    
class MagneticHeadingAtLiftOff(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    dependencies = ['Magnetic Heading', 'Lift Off']
    def derive(self, params):
        return NotImplemented
    
class MagneticHeadingAtTouchDown(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    dependencies = ['Magnetic Heading', 'Touch Down']
    def derive(self, params):
        return NotImplemented
    
# TODO: Trouble with naming these
#class LatgOnGround??
#class Pitch rate (from 2 degrees of pitch?) to 35 feet minimum

    
'''Shortcut: copy n paste below!
(KeyPointValueNode):
    dependencies = []
    def derive(self, params):
        return NotImplemented
'''

##########################################
# KPV from DJ Code

class AccelerationNormalMax(KeyPointValueNode):
    dependencies = ['Normal Acceleration']
    def derive(self, params):
        # Use Numpy to locate the maximum g, then go back and get the value.
        n_acceleration_normal_max = np.ma.argmax(fp.acceleration_normal.data[block])
        acceleration_normal_max = fp.acceleration_normal.data[block][n_acceleration_normal_max]
        # Create a key point value for this.
        kpv['AccelerationNormalMax']=[(block.start+n_acceleration_normal_max,acceleration_normal_max,fp.airspeed.param_name)]
    
    
class RateOfDescentHigh(KeyPointValueNode):
    dependencies = ['Rate Of Climb', 'Descending']
    
    def derive(self, params):
        rate_of_climb = params['Rate Of Climb']
        #TODO: Merge with below RateOfDescentMax accepting a flightphase arg
        ##kpv_list = []
        for descent_period in np.ma.flatnotmasked_contiguous(params['Descending']):
            duration = descent_period.stop - descent_period.start
            if duration > settings.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb[descent_period])
                howfast = rate_of_climb[descent_period][when]
                kpv = self.create_kpv(descent_period.start+when, howfast)
                ##kpv_list.append(kpv)
        ##return kpv_list
                
                
class RateOfDescentMax(KeyPointValueNode):
    dependencies = ['Rate Of Climb', 'Descent']
    # Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
    DESCENT_MIN_DURATION = 10
    
    def derive(self, ph, params):
        rate_of_climb = params['Rate Of Climb']
        kpv_list = []
        for descent_period in np.ma.flatnotmasked_contiguous(params['Descent']):
            duration = descent_period.stop - descent_period.start
            if duration > self.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb[descent_period])
                howfast = rate_of_climb[descent_period][when]
                kpv = self.create_kpv(descent_period.start+when, howfast)
                kpv_list.append(kpv)
        return kpv_list
             
                

    
    
class MaxIndicatedAirspeedLevelFlight(KeyPointValueNode):
    dependencies = ['Indicated Airspeed', 'Level Flight']
    def derive(self, params):
        airspeed = params['Indicated Airspeed']
        kpv_list = []
        for level_slice in np.ma.flatnotmasked_contiguous(params['Level Flight']):
            duration = level_slice.stop - level_slice.start
            if duration > settings.LEVEL_FLIGHT_MIN_DURATION:
                # stable for long enough
                when = np.ma.argmax(airspeed[level_slice])
                howfast = airspeed[level_slice][when]
                kpv = self.create_kpv(level_slice.start+when, howfast)
                kpv_list.apend(kpv)
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


