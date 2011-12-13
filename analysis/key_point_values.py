import numpy as np
import datetime
from collections import namedtuple

from analysis.node import  KeyPointValue, KeyPointValueNode, KTI, P, S
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
    def derive(self, lift_off=KTI('Lift Off')):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class ApproachAirport(KeyPointValueNode):
    def derive(self, descent=S('Descent')):
        return NotImplemented
    
class LandingAirport(KeyPointValueNode):
    def derive(self, touch_down=KTI('Touch Down')):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class TakeoffAltitude(KeyPointValueNode):
    def derive(self, lift_off=KTI('Lift Off'), takeoff_airport=TakeoffAirport):
        return NotImplemented
    
class LandingAltitude(KeyPointValueNode):
    def derive(self, touch_down=KTI('Touch Down'),
               landing_airport=LandingAirport):
        return NotImplemented


                
                
##########################################
# KPV from A6RKA_KPVvalues.xls


class IndicatedAirspeedAtLiftOff(KeyPointValueNode):
    def derive(self, lift_off=KTI('Lift Off'),
               indicated_airspeed=P('Indicated Airspeed')):
        return NotImplemented
    
class PitchAtLiftOff(KeyPointValueNode):
    def derive(self, lift_off=KTI('Lift Off'), pitch=P('Pitch')):
        return NotImplemented
   
   
class FlapAtLiftOff(KeyPointValueNode):
    def derive(self, lift_off=KTI('Lift Off'), flap=P('Flap')):
        return NotImplemented

class IndicatedAirspeedAt35Feet(KeyPointValueNode):
    """ Based on Altitude Radio
    """
    def derive(self, indicated_airspeed=P('Indicated Airspeed'),
               alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class NormalgLiftOffTo35FeetMax(KeyPointValueNode):
    def derive(self, norm_g=P('Normal g'), alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class NormalgMaxDeviation(KeyPointValueNode):
    """ For discussion - why have Max and Min Normal g when it's just the max 
    distance from 0.98 that's interesting?
    """
    def derive(self, norm_g=P('Normal g'), airborne=S('Airborne')):
        STANDARD_GRAVITY = 9.80665
        for airborne_slice in airborne:
            normg_in_air = norm_g.array.data[airborne_slice]
            gdiff = np.ma.absolute(normg_in_air - STANDARD_GRAVITY)
            max_index = gdiff.argmax()
            self.create_kpv(max_index, gdiff[max_index])    
    
class Pitch35To400FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class Pitch1000To100FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        return NotImplemented
    
class Pitch5FeetToTouchDownMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_rad=P('Altitude Radio'),
               touch_down=KTI('Touch Down')):
        return NotImplemented
    
    
class PitchCycles(KeyPointValueNode):
    """ Count
    """
    def derive(self, pitch=P('Pitch')):
        return NotImplemented
    
class Pitch35To400FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class Pitch1000To100FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL')):
        return NotImplemented
    
class Pitch20FeetToTouchDownMin(KeyPointValueNode):
    """ Q: This is 20 feet, the max uses 5 feet
    """
    def derive(self, pitch=P('Pitch'), alt_rad=P('Altitude Radio'),
               touch_down=KTI('Touch Down')):
        return NotImplemented
    
class PitchRateLiftOffTo35FeetMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), lift_off=KTI('Lift Off'),
               alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class PitchRate35To1500FeetMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), alt_aal=P('Altitude AAL')):
        return NotImplemented
    
    
class RollBelow20FeetMax(KeyPointValueNode): # absolute max?
    def derive(self, roll=P('Roll'), alt_rad=P('Altitude Radio')):
        return NotImplemented
   
class RollBetween100And500FeetMax(KeyPointValueNode): # absolute max?
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        return NotImplemented
    
class RollBetween500And1500FeetMax(KeyPointValueNode):  # absolue max?
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        return NotImplemented   
    
class RollAbove1500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL')):
        return NotImplemented
    
class RollCycles1000FeetToTouchDown(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL'),
               touch_down=KTI('Touch Down')):
        return NotImplemented
    
class AltitudeWithFlapsMax(KeyPointValueNode):
    """ It's max Altitude not Max Flaps
    """
    def derive(self, flap=P('Flap'), alt_std=P('Altitude Std')):
        return NotImplemented
    
class AltitudeStdMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude Std')):
        max_index = alt_std.array.argmax()
        self.create_kpv(max_index, alt_std[max_index])
        
class IndicatedAirspeedMax(KeyPointValueNode):
    def derive(self, airspeed=P('Indicated Airspeed')):
        # Use Numpy to locate the maximum airspeed, then get the value.
        index = airspeed.array.argmax()
        airspeed_max = airspeed.array[index]
        self.create_kpv(index, airspeed_max)
    
class MACHMax(KeyPointValueNode):
    name = 'MACH Max'
    def derive(self, mach=P('MACH')):
        return NotImplemented
    

class IndicatedAirspeedAtTouchDown(KeyPointValueNode):
    def derive(self, airspeed=P('Indicated Airspeed'),
               touch_down=KTI('Touch Down')):
        return NotImplemented
    
class GroundSpeedOnGroundMax(KeyPointValueNode):
    def derive(self, ground_speed=P('Ground Speed'), on_ground=P('On Ground')):
        return NotImplemented

class FlapAtTouchDown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touch_down=KTI('Touch Down')):
        return NotImplemented
    
class GrossWeightAtTouchDown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight'), 
               touch_down=KTI('Touch Down')):
        return NotImplemented
    
class EngEGTMax(KeyPointValueNode):
    name = 'Eng EGT Max'
    ##NAME_FORMAT = 'Eng EGT Max %(engine)s'
    ##NAME_FORMAT = 'Eng EGT Max'
    ##RETURN_OPTIONS = {'engine': ['Eng (%d) EGT' % n for n in range(1,5)]}

    @classmethod
    def can_operate(cls, available):
        if set(cls.get_dependency_names()).intersection(available):
            return True  # if ANY are available
        else:
            return False  # we have no EGT recorded on any engines
        
    def derive(self, egt1=P('Eng (1) EGT'), egt2=P('Eng (2) EGT'),
               egt3=P('Eng (3) EGT'), egt4=P('Eng (4) EGT')):
        kmax = vmax = imax = None
        for p in (egt1, egt2, egt3, egt4):
            _imax = p.array.argmax()
            _vmax = p.array[_imax]
            if _vmax > vmax:
                imax = _imax # index of max
                vmax = _vmax # max value
                kmax = p.name # param name of max eng
        self.create_kpv(imax, vmax)
    
    
class MagneticHeadingAtLiftOff(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), lift_off=KTI('Lift Off')):
        return NotImplemented
    
class MagneticHeadingAtTouchDown(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), touch_down=KTI('Touch Down')):
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
    def derive(self, normal_acceleration=P('Normal Acceleration'),
               airspeed=P('Airspeed')):
        # Use Numpy to locate the maximum g, then go back and get the value.
        n_acceleration_normal_max = np.ma.argmax(normal_acceleration.array.data[block])
        acceleration_normal_max = normal_acceleration.array.data[block][n_acceleration_normal_max]
        # Create a key point value for this. TODO: Change to self.create_kpv()?
        self.create_kpv(block.start+n_acceleration_normal_max,
                        acceleration_normal_max)
    
    
class RateOfDescentHigh(KeyPointValueNode):
    
    def derive(self, rate_of_climb=P('Rate Of Climb'),
               descending=S('Descending')):
        #TODO: Merge with below RateOfDescentMax accepting a flightphase arg
        for descent_slice in descending:
            duration = descent_slice.stop - descent_slice.start
            if duration > settings.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb.array[descent_slice])
                howfast = rate_of_climb.array[descent_slice][when]
                self.create_kpv(descent_slice.start+when, howfast)
                
                
class RateOfDescentMax(KeyPointValueNode):
    # Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
    DESCENT_MIN_DURATION = 10
    
    def derive(self, rate_of_climb=P('Rate Of Climb'), descent=S('Descent')):
        for descent_slice in descent:
            duration = descent_slice.stop - descent_slice.start
            if duration > self.DESCENT_MIN_DURATION:
                when = np.ma.argmax(rate_of_climb.array[descent_slice])
                howfast = rate_of_climb.array[descent_slice][when]
                self.create_kpv(descent_slice.start+when, howfast)
             
                

    
    
class MaxIndicatedAirspeedLevelFlight(KeyPointValueNode):
    def derive(self, airspeed=P('Indicated Airspeed'),
               level_flight=S('Level Flight')):
        for level_slice in level_flight:
            duration = level_slice.stop - level_slice.start
            if duration > settings.LEVEL_FLIGHT_MIN_DURATION:
                # stable for long enough
                when = np.ma.argmax(airspeed.array[level_slice])
                howfast = airspeed.array[level_slice][when]
                self.create_kpv(level_slice.start+when, howfast)
            else:
                logging.debug('Short duration %d of level flight ignored',
                              duration)
            
    
    
    
    
class AirspeedMinusVref500ftTo0ftMax(KeyPointValueNode):
    NAME_FORMAT = 'Airspeed Minus Vref 500ft to 0ft Max' #TODO: auto replace with name?!
    
    def derive(self, airspeed_minus_vref=P('AirspeedMinusVref'), 
               _500ft_to_0ft=S('500ft To 0ft')):  #Q: Label this as the list of kpv sections?

        for sect in _500ft_to_0ft:
            ##max_spd = airspeed_minus_vref.array[sect].max()
            ##when = np.ma.where(airspeed_minus_vref.array[sect] == max_spd)[0][0] + sect.start
            
            when = np.ma.argmax(airspeed_minus_vref.array[sect]) + sect.start
            max_spd = airspeed_minus_vref.array[when]
            self.create_kpv(when, max_spd)
    
    
    
    
    
    
    
    
    
    
    
#TODO:
#toc = altitude_std[kpt['TopOfClimb']] # Indexing n_toc into the reduced array [block]
#kpv['Altitude_TopOfClimb'] = [(kpt['TopOfClimb'], toc, altitude_std)]
#kpv['LandingTurnOffRunway'] = [(block.start+kpt['LandingEndEstimate'],(head_mag[kpt['LandingEndEstimate']] - head_landing), head_mag.param_name)]
#kpv['Head_Landing'] = [(block.start+kpt['LandingEndEstimate'], head_landing%360, head_mag.param_name)]  # Convert to normal compass heading for display
#tod = altitude_std[kpt['TopOfDescent']] # Indexing n_toc into the reduced array [block]
#kpv['Altitude_TopOfDescent'] = [(kpt['TopOfDescent'], tod, altitude_std)]
#kpv['Head_Takeoff'] = [(block.start+kpt['TakeoffStartEstimate'], head_takeoff%360, head_mag.param_name)] # Convert to normal compass heading for display
#kpv['TakeoffTurnOntoRunway'] = [(block.start+turn_onto_runway,head_takeoff - head_mag[turn_onto_runway],head_mag.param_name)]






