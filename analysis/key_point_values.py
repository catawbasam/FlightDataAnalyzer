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
    def derive(self, liftoff=KTI('Liftoff')):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class ApproachAirport(KeyPointValueNode):
    def derive(self, descent=S('Descent')):
        return NotImplemented
    
class LandingAirport(KeyPointValueNode):
    def derive(self, touchdown=KTI('Touchdown')):
        ##KeyPointValue(n, 'ICAO', 'Takeoff Airport')
        ##KeyPointValue(n, '09L', 'Takeoff Runway')
        return NotImplemented
    
class TakeoffAltitude(KeyPointValueNode):
    def derive(self, liftoff=KTI('Liftoff'), takeoff_airport=TakeoffAirport):
        return NotImplemented
    
class LandingAltitude(KeyPointValueNode):
    def derive(self, touchdown=KTI('Touchdown'),
               landing_airport=LandingAirport):
        return NotImplemented


                
                
##########################################
# KPV from A6RKA_KPVvalues.xls


class IndicatedAirspeedAtLiftOff(KeyPointValueNode):
    def derive(self, liftoff=KTI('Liftoff'),
               indicated_airspeed=P('Indicated Airspeed')):
        return NotImplemented
    
class PitchAtLiftOff(KeyPointValueNode):
    def derive(self, liftoff=KTI('Liftoff'), pitch=P('Pitch')):
        return NotImplemented
   
   
class FlapAtLiftOff(KeyPointValueNode):
    def derive(self, liftoff=KTI('Liftoff'), flap=P('Flap')):
        return NotImplemented

class IndicatedAirspeedAt35Feet(KeyPointValueNode):
    """ Based on Altitude Radio
    """
    def derive(self, airspeed=P('Indicated Airspeed'),
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
               touchdown=KTI('Touchdown')):
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
               touchdown=KTI('Touchdown')):
        return NotImplemented
    
class PitchRateLiftOffTo35FeetMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), liftoff=KTI('Liftoff'),
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
               touchdown=KTI('Touchdown')):
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
               touchdown=KTI('Touchdown')):
        return NotImplemented
    
class GroundSpeedOnGroundMax(KeyPointValueNode):
    def derive(self, ground_speed=P('Ground Speed'), on_ground=P('On Ground')):
        return NotImplemented

class FlapAtTouchDown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdown=KTI('Touchdown')):
        return NotImplemented
    
class GrossWeightAtTouchDown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight'), 
               touchdown=KTI('Touchdown')):
        return NotImplemented
    
class EGTMax(KeyPointValueNode): # which engine? or all engines? # or all and each!?
    ##returns = "EGT Max"  # add which engine?
    NAME_FORMAT = 'EGT Max %(engine)s'
    # FIXME: In the following line, dependencies is not defined when the file
    # is parsed.
    #RETURN_OPTIONS = {'engine': dependencies + ['Engine (*) EGT']}

    @classmethod
    def can_operate(cls, available):
        if set(cls.dependencies).intersection(available):
            return True  # if ANY are available
        else:
            return False  # we have no EGT recorded on any engines
        
    def derive(self, egt1=P('Engine (1) EGT'), egt2=P('Engine (2) EGT'),
               egt3=P('Engine (3) EGT'), egt4=P('Engine (4) EGT')):
        kmax = vmax = imax = None
        for p in (egt1, egt2, egt3, egt4):
            _imax = p.array.argmax()
            _vmax = p.array[_imax]
            if _vmax > vmax:
                imax = _imax # index of max
                vmax = _vmax # max value
                kmax = p.name # param name of max eng
        self.create_kpv(imax, vmax, engine=kmax) # include engine using kmax?
    
    
class MagneticHeadingAtLiftOff(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), liftoff=KTI('Liftoff')):
        return NotImplemented
    
class MagneticHeadingAtTouchDown(KeyPointValue):
    """ Shouldn't this be difference between aircraft heading and runway heading???
    """
    def derive(self, heading=P('Magnetic Heading'), touchdown=KTI('Touchdown')):
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


class NormalGAirborneMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('Acceleration Normal'),
               airborne=S('Airborne')):
        return NotImplemented


class NormalGDuringTakeOffMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('Acceleration Normal'),
               liftoff=KTI('Liftoff')):
        return NotImplemented


class AltitudeMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), airborne=S('Airborne')):
        return NotImplemented


class AltitudeWithFlapsMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), flap=P('Flap'),
               airborne=S('Airborne')):
        return NotImplemented


class FlapAtTouchdown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdown=KTI('Touchdown')):
        return NotImplemented


# FIXME: Bounced Landing name duplicated between KPV and Section!
class BouncedLanding(KeyPointValueNode):
    def derive(self, bounced_landing=S('Bounced Landing Section')):
        return NotImplemented


class ROD500ToTouchdownMax(KeyPointValueNode):
    name = 'ROD 500 To Touchdown Max'
    def derive(self, roc=P('Rate Of Climb'),
               _500_ft_in_final_approach=KTI('500 Ft In Final Approach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented
    

class ROD1000To500FeetMax(KeyPointValueNode):
    name = 'ROD 1000 To 500 Feet Max'
    def derive(self, roc=P('Rate Of Climb'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _500_ft_in_final_approach=KTI('500 Ft In Final Approach')):
        return NotImplemented


class ROD2000To1000FeetMax(KeyPointValueNode):
    name = 'ROD 2000 To 1000 Feet Max'
    def derive(self, roc=P('Rate Of Climb'),
               _2000_ft_in_approach=KTI('2000 Ft In Approach'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach')):
        return NotImplemented


class DontSinkWarning(KeyPointValueNode):
    def derive(self, gpws_dont_sink=P('GPWS Dont Sink')):
        return NotImplemented


class HeightAtConfigChangeLiftOffTo3500FeetMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),
               liftoff=KTI('Liftoff'),
               _3500_ft_in_climb=KTI('3500 Ft In Climb')):
        return NotImplemented


class EGTTakeoffMax(KeyPointValueNode):
    name = 'EGT Takeoff Max'
    def derive(self, eng_n_egt=P('Engine (N) EGT'), liftoff=KTI('Liftoff')):
        return NotImplemented


class AirspeedWithFlapXMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed')):
        return NotImplemented


class GlideslopeWarning(KeyPointValueNode):
    def derive(self, gpws_glideslope=P('GPWS Glideslope')):
        return NotImplemented


class GlideslopeDeviation1000To150FeetMax(KeyPointValueNode):
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _150_ft_in_final_approach=KTI('150 Ft In Final Approach')):
        return NotImplemented


class GlideslopeDeviation1500To1000FeetMax(KeyPointValueNode):
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               _1500_ft_in_approach=KTI('1500 Ft In Approach'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach')):
        return NotImplemented


class HeightAtGoAroundMin(KeyPointValueNode):
    def derive(self, go_around=P('Go Around'),
               alt_radio=P('Altitude Radio')): # FIXME: Go Around parameter??
        return NotImplemented


class SinkRateWarning(KeyPointValueNode):
    def derive(self, gpws_sink_rate=P('GPWS Sink Rate')):
        return NotImplemented


class Normalg20FeetToGroundMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('Acceleration Normal')):
        return NotImplemented


class HeadingDeviation100KtsToLiftOffMax(KeyPointValueNode):
    def derive(self, head_mag=P('Heading Magnetic'), airspeed=P('Airspeed'),
               liftoff=KTI('Liftoff')):
        return NotImplemented


class Height1MinuteToTouchdown(KeyPointValueNode):
    def derive(self, altitude_aal=P('Altitude AAL'),
               _1_min_to_landing=KTI('1 Min To Landing')):
        return NotImplemented


class Height2MinutesToTouchdown(KeyPointValueNode):
    def derive(self, altitude_aal=P('Altitude AAL'),
               _2_min_to_landing=KTI('2 Min To Landing')):
        return NotImplemented


class HeightLost1000To2000FeetMax(KeyPointValueNode):
    def derive(self, altitude_std=P('Altitude STD'),
               _1000_ft_in_climb=KTI('1000 Ft In Climb'),
               _2000_ft_in_climb=KTI('2000 Ft In Climb')):
        return NotImplemented


class HeightLost50To1000Max(KeyPointValueNode):
    def derive(self, altitude_std=P('Altitude STD'),
               _50_ft_in_initial_climb=KTI('_50 Ft In Initial Climb'),
               _1000_ft_in_climb=KTI('1000 Ft In Climb')):
        return NotImplemented


class Height1MinuteToTouchdown(KeyPointValueNode):
    def derive(self, altitude_aal=P('Altitude AAL'),
               _1_min_to_landing=KTI('1 Min To Landing')):
        return NotImplemented


class Height2MinutesToTouchdown(KeyPointValueNode):
    def derive(self, altitude_aal=P('Altitude AAL'),
               _2_min_to_landing=KTI('2 Min To Landing')):
        return NotImplemented


class LateralGOnGroundDurationAboveLimit(KeyPointValueNode):
    def derive(self, acc_lat=P('Acceleration Lateral'),
               taxi_out=S('Taxi Out'), taxi_in=S('Taxi In')):
        return NotImplemented


class N13000FeetToTouchdownMax(KeyPointValueNode):
    name = 'N1 3000 Feet To Touchdown Max'
    def derive(self, eng_n1_avg=P('Engine N1 Average'),
               _3000_ft_in_approach=KTI('3000 Ft In Approach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class N1TakeoffMax(KeyPointValueNode):
    name = 'N1 Takeoff Max'
    def derive(self, eng_n1_max=P('Engine N1 Maximum'), liftoff=KTI('Liftoff')):
        return NotImplemented


class FlapAtGearSelectedDown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_sel_down=P('Gear Selected Down')):
        return NotImplemented


class HeightAtConfigChange1500FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'),
               _1500_ft_in_approach=KTI('1500 Ft In Approach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class HeightAtConfigChange1500FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'),
               _1500_ft_in_approach=KTI('1500 Ft In Approach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class SuspectedLevelBust(KeyPointValueNode):
    def derive(self, level_bust=S('Level Bust')):
        return NotImplemented


class LocaliserDeviation1000To150FeetMax(KeyPointValueNode):
    def derive(self, ils_loc=P('ILS Localizer'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _150_ft_in_final_approach=KTI('150 Ft In Final Approach')):
        return NotImplemented


class LocaliserDeviation1500To1000FeetMax(KeyPointValueNode):
    def derive(self, ils_loc=P('ILS Localizer'),
               _1500_ft_in_approach=KTI('1500 Ft In Approach'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach')):
        return NotImplemented


class RadAlt20FeetToTouchdownDurationOverLimit(KeyPointValueNode):
    def derive(self, alt_rad=P('Altitude Radio'),
               _25_ft_in_landing=KTI('_25 Ft In Landing'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class LowPowerLessthan500Ft10Sec(KeyPointValueNode):
    def derive(self, eng_n1_min=P('Engine N1 Minimum'),
               _500_ft_in_final_approach=KTI('500 Ft In Final Approach')):
        return NotImplemented


class GroundspeedRTOMax(KeyPointValueNode):
    name = 'Groundspeed RTO Max'
    def derive(self, groundspeed=P('Groundspeed'),
               rejected_takeoff=S('Rejected Takeoff')):
        return NotImplemented


class N1Max(KeyPointValueNode):
    name = 'NI Max'
    def derive(self, eng_n_n1=P('Engine (N) N1')):
        return NotImplemented


class N2CyclesMax(KeyPointValueNode):
    name = 'N2 Cycles Max'
    def derive(self, eng_n_n2=P('Engine (N) N2')):
        return NotImplemented


class N2Max(KeyPointValueNode):
    name = 'N2 Max'
    def derive(self, eng_n_n2=P('Engine (N) N2')):
        return NotImplemented


class OIPMax(KeyPointValueNode):
    name = 'OIP Max'
    def derive(self, eng_n_oil_press=P('Engine (N) Oil Press')):
        return NotImplemented


class OIPMin(KeyPointValueNode):
    name = 'OIP Min'
    def derive(self, eng_n_oil_press_low=P('Engine (N) Oil Press Low')):
        return NotImplemented


class OITMax(KeyPointValueNode):
    name = 'OIT Max'
    def derive(self, eng_n_oil_temp=P('Engine (1) Oil Temp')):
        return NotImplemented


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class PitchCyclesMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch')):
        return NotImplemented


class Pitch35To400FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _35_ft_in_takeoff=KTI('35 Ft In Takeoff'),
               _400_ft_in_initial_climb=KTI('400 Ft In Initial_Climb')):
        return NotImplemented


class Pitch1000To100FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _100_ft_in_final_approach=KTI('100 Ft In Final Approach')):
        return NotImplemented


class PitchAtLiftOff(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), liftoff=KTI('Liftoff')):
        return NotImplemented


class Pitch5FeetToGroundMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), landing=S('Landing')):
        return NotImplemented


class Pitch35To400FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _35_ft_in_takeoff=KTI('35 Ft In Takeoff'),
               _400_ft_in_initial_climb=KTI('400 Ft In Initial_Climb')):
        return NotImplemented


class Pitch1000To100FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _100_ft_in_final_approach=KTI('100 Ft In Final Approach')):
        return NotImplemented


class Pitch20FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _25_ft_in_landing=KTI('25 Ft In Landing'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class PitchRateDuringTakeOffMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), liftoff=KTI('Liftoff')):
        return NotImplemented


class PitchRate35To1500FeetMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'),
               _35_ft_in_takeoff=KTI('35 Ft In Takeoff'),
               _1500_ft_in_climb=KTI('1500 Ft In Climb')):
        return NotImplemented


class PitchRateFrom2DegreesOfPitchTo35FeetMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), liftoff=KTI('Liftoff'),
               _35_ft_in_takeoff=KTI('35 Ft In Takeoff')):
        return NotImplemented


class PowerOnWithSpeedbrakesDeployedDurationGreaterThanLimit(KeyPointValueNode):
    def derive(self, engine_n1_average=P('Engine N1 Average'),
               speedbrake=P('Speedbrake')):
        return NotImplemented


class PullUpWarning(KeyPointValueNode):
    def derive(self, gpws_pull_up=P('GPWS Pull Up')):
        return NotImplemented


class RollCycles1000FtToTouchdownMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class RollBetween100And500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _100_ft_in_initial_climb=KTI('100 Ft In Initial_Climb'),
               _500_ft_in_initial_climb=KTI('500 Ft In Initial_Climb')):
        return NotImplemented


class RollBetween500And1500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _500_ft_in_initial_climb=KTI('500 Ft In Initial_Climb'),
               _1500_ft_in_climb=KTI('1500 Ft In Climb')):
        return NotImplemented


class RollBelow20FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _25_ft_in_landing=KTI('25 Ft In Landing'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class RollAbove1500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_rad=P('Altitude Radio')):
        return NotImplemented


class RudderReversalAbove50Feet(KeyPointValueNode):
    def derive(self, rudder_reversal=S('Rudder Reversal')):
        return NotImplemented


class AirspeedWithGearSelectedDownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_down=P('Gear Selected Down')):
        return NotImplemented


class AirspeedAtGearSelectedDown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_down=P('Gear Selected Down')):
        return NotImplemented


class AirspeedAtGearSelectedUp(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_sel_up=P('Gear Selected Up')):
        return NotImplemented


class GroundspeedOnGroundMax(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'), on_ground=S('On Ground')):
        return NotImplemented


class MACHMMOMax(KeyPointValueNode):
    name = 'MACH MMO Max'
    def derive(self, mach=P('MACH')):
        return NotImplemented


class AirspeedMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedVref500FeetToTouchdownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class Airspeed1000To500FeetMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap1Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap2Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap5Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap15Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap25Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap40Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap30Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedWithFlap10Max(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedVrefAtTouchdown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedBelow3000FeetMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class GroundspeedOnGroundWithRateOfChangeOfHeadingGreaterThanLimitMax(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed')):
        return NotImplemented


class AirspeedV2AtLiftOff(KeyPointValueNode):
    name = 'Airspeed V2 At Lift Off'
    def derive(self, airspeed=P('Airspeed'), liftoff=KTI('Liftoff')):
        return NotImplemented


class AirspeedBetween90SToTouchdownAndTouchdownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               _90_sec_to_landing=KTI('90 Sec To Landing'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class AirspeedV235To400FeetMin(KeyPointValueNode):
    name = 'Airspeed V2 35 To 400 Feet Min'
    def derive(self, airspeed=P('Airspeed'),
               _35_ft_in_takeoff=KTI('35 Ft In Takeoff'),
               _400_ft_in_initial_climb=KTI('400 Ft In Initial_Climb')):
        return NotImplemented


class AirspeedV2400To1500FeetMin(KeyPointValueNode):
    name = 'Airspeed V2 400 To 1500 Feet Min'
    def derive(self, airspeed=P('Airspeed'),
               _400_ft_in_initial_climb=KTI('400 Ft In Initial_Climb'),
               _1500_ft_in_climb=KTI('1500 Ft In Climb')):
        return NotImplemented


class AirspeedV2AtLiftOff(KeyPointValueNode):
    name = 'Airspeed V2 At Lift Off'
    def derive(self, airspeed=P('Airspeed')):
        return NotImplemented


class AirspeedVrefLast2MinutesToTouchdownMin(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               _2_min_to_landing=KTI('2 Min To Landing'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class SpeedbrakesDeployed1000To30Feet(KeyPointValueNode):
    def derive(self, speedbrake=P('Speedbrake'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _25_ft_in_landing=KTI('25 Ft In Landing')):
        return NotImplemented


class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), speedbrake=P('Speedbrake')):
        return NotImplemented


class StickShakerActivated(KeyPointValueNode):
    def derive(self, stick_shaker=P('Stick Shaker')):
        return NotImplemented


class TerrainWarning(KeyPointValueNode):
    def derive(self, gpws_terrain=P('GPWS Terrain')):
        return NotImplemented


class TerrainPullUpWarning(KeyPointValueNode):
    def derive(self, gpws_terrain_pull_up=P('GPWS Terrain Pull Up')):
        return NotImplemented


class ThrottleCycles1000FeetToTouchdownMax(KeyPointValueNode):
    def derive(self, thrust_lever_n=P('Thrust Lever (N)')):
        return NotImplemented


class TooLowFlapWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_flap=P('GPWS Too Low Flap')):
        return NotImplemented


class TooLowGearWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_gear=P('GPWS Too Low Gear')):
        return NotImplemented


class TooLowTerrainWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_terrain=P('GPWS Too Low Terrain')):
        return NotImplemented


class NormalGAirborneMin(KeyPointValueNode):
    def derive(self, acc_norm=P('Acceleration Normal'), airborne=S('Airborne')):
        return NotImplemented


class VibrationN1GreaterThanLimitDurationGreaterThanLimit(KeyPointValueNode):
    name = 'Vibration N1 Greater Than Limit Duration Greater Than Limit'
    def derive(self, eng_n_vib_n1=P('Engine (N) Vib N1'),
               airborne=S('Airborne')):
        return NotImplemented


class VibrationN2GreaterThanLimitDurationGreaterThanLimit(KeyPointValueNode):
    name = 'Vibration N2 Greater Than Limit Duration Greater Than Limit'
    def derive(self, eng_n_vib_n2=P('Engine (N) Vib N2'),
               airborne=S('Airborne')):
        return NotImplemented


class WindshearWarningBelow1500Feet(KeyPointValueNode):
    def derive(self, gpws_windshear=P('GPWS Windshear'),
               alt_aal=P('Altitude AAL')):
        return NotImplemented


class TaxiSpeedStraight(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'), rot=P('Rate Of Turn')):
        return NotImplemented


class TaxiSpeedTurning(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'), rot=P('Rate Of Turn')):
        return NotImplemented


class LowPowerInFinalApproach10Sec(KeyPointValueNode):
    def derive(self, eng_n_avg=P('Engine (N) Average'),
               _1000_ft_in_approach=KTI('1000 Ft In Approach'),
               _50_ft_in_landing=KTI('50 Ft In Landing')):
        return NotImplemented


