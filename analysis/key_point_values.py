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
    
class EGTMax(KeyPointValueNode): # which engine? or all engines? # or all and each!?
    ##returns = "EGT Max"  # add which engine?
    NAME_FORMAT = 'EGT Max %(engine)s'
    RETURN_OPTIONS = {'engine': dependencies + ['Engine (*) EGT']}

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



class NormalGAirborneMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('AccelerationNormal'),
               airborne=S('Airborne')):
        return NotImplemented


class NormalGDuringTakeOffMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('AccelerationNormal'),
               takeoff=S('Takeoff')):
        return NotImplemented


class AltitudeMax(KeyPointValueNode):
    def derive(self, altitude_std=P('AltitudeSTD'), airborne=S('Airborne')):
        return NotImplemented


class AltitudeWithFlapsMax(KeyPointValueNode):
    def derive(self, altitude_std=P('AltitudeSTD'), flap=P('Flap'),
               airborne=S('Airborne')):
        return NotImplemented


class FlapAtTouchdown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdown=KTI('Touchdown')):
        return NotImplemented


class BouncedLanding(KeyPointValueNode):
    def derive(self, bounced_landing=S('BouncedLanding')):
        return NotImplemented


class ROD500ToTouchdownMax(KeyPointValueNode):
    name = 'ROD500ToTouchdownMax'
    def derive(self, rate_of_climb=P('RateOfClimb'),
               _500_ft_in_final_approach=KTI('_500FtInFinalApproach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class ROD1000To500FeetMax(KeyPointValueNode):
    name = 'ROD1000To500FeetMax'
    def derive(self, rate_of_climb=P('RateOfClimb'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _500_ft_in_final_approach=KTI('_500FtInFinalApproach')):
        return NotImplemented


class ROD2000To1000FeetMax(KeyPointValueNode):
    name = 'ROD2000To1000FeetMax'
    def derive(self, roc=P('RateOfClimb'),
               _2000_ft_in_approach=KTI('_2000FtInApproach'),
               _1000_ft_in_approach=KTI('_1000FtInApproach')):
        return NotImplemented


class DontSinkWarning(KeyPointValueNode):
    def derive(self, gpws_dont_sink=P('GPWSDontSink')):
        return NotImplemented


class HeightAtConfigChangeLiftOffTo3500FeetMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'), altitude_aal=P('AltitudeAAL'),
               liftoff=KTI('Liftoff'), _3500_ft_in_climb=KTI('_3500FtInClimb')):
        return NotImplemented


class EGTTakeoffMax(KeyPointValueNode):
    name = 'EGTTakeoffMax'
    def derive(self, engine_(n)_egt=P('Engine(N)EGT'), takeoff=S('Takeoff')):
        return NotImplemented


class EGTMax(KeyPointValueNode):
    name = 'EGTMax'
    def derive(self, engine_(n)_egt=P('Engine(N)EGT')):
        return NotImplemented


class AirspeedWithFlapXMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed')):
        return NotImplemented


class GlideslopeWarning(KeyPointValueNode):
    def derive(self, gpws_glideslope=P('GPWSGlideslope')):
        return NotImplemented


class GlideslopeDeviation1000To150FeetMax(KeyPointValueNode):
    def derive(self, ils_glideslope=P('ILSGlideslope'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _150_ft_in_final_approach=KTI('_150FtInFinalApproach')):
        return NotImplemented


class GlideslopeDeviation1500To1000FeetMax(KeyPointValueNode):
    def derive(self, ils_glideslope=P('ILSGlideslope'),
               _1500_ft_in_approach=KTI('_1500FtInApproach'),
               _1000_ft_in_approach=KTI('_1000FtInApproach')):
        return NotImplemented


class HeightAtGoAroundMin(KeyPointValueNode):
    def derive(self, go_around=P('GoAround'), alt_rad=P('AltitudeRadio')):
        return NotImplemented


class SinkRateWarning(KeyPointValueNode):
    def derive(self, gpws_sink_rate=P('GPWSSinkRate')):
        return NotImplemented


class Normalg20FeetToGroundMax(KeyPointValueNode):
    def derive(self, acceleration_normal=P('AccelerationNormal')):
        return NotImplemented


class HeadingDeviation100KtsToLiftOffMax(KeyPointValueNode):
    def derive(self, head_mag=P('HeadingMagnetic'),
               airspeed=P('Airspeed'), liftoff=KTI('Liftoff')):
        return NotImplemented


class Height1MinuteToTouchdown(KeyPointValueNode):
    def derive(self, alt_aal=P('AltitudeAAL'),
               _1_min_to_landing=KTI('_1MinToLanding')):
        return NotImplemented


class Height2MinutesToTouchdown(KeyPointValueNode):
    def derive(self, alt_aal=P('AltitudeAAL'),
               _2_min_to_landing=KTI('_2MinToLanding')):
        return NotImplemented


class HeightLost1000To2000FeetMax(KeyPointValueNode):
    def derive(self, altitude_std=P('AltitudeSTD'),
               _1000_ft_in_climb=KTI('_1000FtInClimb'),
               _2000_ft_in_climb=KTI('_2000FtInClimb')):
        return NotImplemented


class HeightLost50To1000Max(KeyPointValueNode):
    def derive(self, alt_std=P('AltitudeSTD'),
               _50_ft_in_initial_climb=KTI('_50FtInInitial_Climb'),
               _1000_ft_in_climb=KTI('_1000FtInClimb')):
        return NotImplemented


class Height1MinuteToTouchdown(KeyPointValueNode):
    def derive(self, alt_aal=P('AltitudeAAL'),
               _1_min_to_landing=KTI('_1MinToLanding')):
        return NotImplemented


class Height2MinutesToTouchdown(KeyPointValueNode):
    def derive(self, alt_aal=P('AltitudeAAL'),
               _2_min_to_landing=KTI('_2MinToLanding')):
        return NotImplemented


class LateralGOnGroundDurationAboveLimit(KeyPointValueNode):
    def derive(self, acceleration_lateral=P('AccelerationLateral'),
               taxi_out=S('TaxiOut'), taxi_in=S('TaxiIn')):
        return NotImplemented


class N13000FeetToTouchdownMax(KeyPointValueNode):
    name = 'N13000FeetToTouchdownMax'
    def derive(self, eng_n1_avg=P('EngineN1Average'),
               _3000_ft_in_approach=KTI('_3000FtInApproach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class N1TakeoffMax(KeyPointValueNode):
    name = 'N1TakeoffMax'
    def derive(self, eng_n1_max=P('EngineN1Maximum'),
               takeoff=S('Takeoff')):
        return NotImplemented


class FlapAtGearSelectedDown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_sel_down=P('GearSelectedDown')):
        return NotImplemented


class HeightAtConfigChange1500FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'),
               _1500_ft_in_approach=KTI('_1500FtInApproach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class HeightAtConfigChange1500FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, flap=P('Flap'),
               _1500_ft_in_approach=KTI('_1500FtInApproach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class SuspectedLevelBust(KeyPointValueNode):
    def derive(self, level_bust=S('LevelBust')):
        return NotImplemented


class LocaliserDeviation1000To150FeetMax(KeyPointValueNode):
    def derive(self, ils_localizer=P('ILSLocalizer'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _150_ft_in_final_approach=KTI('_150FtInFinalApproach')):
        return NotImplemented


class LocaliserDeviation1500To1000FeetMax(KeyPointValueNode):
    def derive(self, ils_localizer=P('ILSLocalizer'),
               _1500_ft_in_approach=KTI('_1500FtInApproach'),
               _1000_ft_in_approach=KTI('_1000FtInApproach')):
        return NotImplemented


class RadAlt20FeetToTouchdownDurationOverLimit(KeyPointValueNode):
    def derive(self, alt_radio=P('AltitudeRadio'),
               _25_ft_in_landing=KTI('_25FtInLanding'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class LowPowerLessthan500Ft10Sec(KeyPointValueNode):
    def derive(self, eng_n1_min=P('EngineN1Minimum'),
               _500_ft_in_final_approach=KTI('_500FtInFinalApproach')):
        return NotImplemented


class GroundspeedRTOMax(KeyPointValueNode):
    name = 'GroundspeedRTOMax'
    def derive(self, groundspeed=P('Groundspeed'),
               rejected_takeoff=S('RejectedTakeoff')):
        return NotImplemented


class NIMax(KeyPointValueNode):
    name = 'NIMax'
    def derive(self, eng_n1=P('EngineN1')):
        return NotImplemented


class N2CyclesMax(KeyPointValueNode):
    name = 'N2CyclesMax'
    def derive(self, eng_n2=P('EngineN2')):
        return NotImplemented


class N2Max(KeyPointValueNode):
    name = 'N2Max'
    def derive(self, eng_n2=P('Engine(N)N2')):
        return NotImplemented


class OIPMax(KeyPointValueNode):
    name = 'OIPMax'
    def derive(self, eng_oil_press=P('EngineOilPress')):
        return NotImplemented


class OIPMin(KeyPointValueNode):
    name = 'OIPMin'
    def derive(self, eng_oil_press_low=P('EngineOilPressLow')):
        return NotImplemented


class OITMax(KeyPointValueNode):
    name = 'OITMax'
    def derive(self, eng_1_oil_temp=P('Engine1OilTemp')):
        return NotImplemented


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self, gross_weight=P('GrossWeight'), touchdown=KTI('Touchdown')):
        return NotImplemented


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self):
        return NotImplemented


class PitchCyclesMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch')):
        return NotImplemented


class Pitch35To400FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _35_ft_in_takeoff=KTI('_35FtInTakeoff'),
               _400_ft_in_initial_climb=KTI('_400FtInInitial_Climb')):
        return NotImplemented


class Pitch1000To100FeetMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _100_ft_in_final_approach=KTI('_100FtInFinalApproach')):
        return NotImplemented


class PitchAtLiftOff(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), liftoff=KTI('Liftoff')):
        return NotImplemented


class Pitch5FeetToGroundMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), landing=S('Landing')):
        return NotImplemented


class Pitch35To400FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _35_ft_in_takeoff=KTI('_35FtInTakeoff'),
               _400_ft_in_initial_climb=KTI('_400FtInInitial_Climb')):
        return NotImplemented


class Pitch1000To100FeetMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _100_ft_in_final_approach=KTI('_100FtInFinalApproach')):
        return NotImplemented


class Pitch20FeetToTouchdownMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'),
               _25_ft_in_landing=KTI('_25FtInLanding'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class PitchRateDuringTakeOffMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('PitchRate'), takeoff=S('Takeoff')):
        return NotImplemented


class PitchRate35To1500FeetMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('PitchRate'),
               _35_ft_in_takeoff=KTI('_35FtInTakeoff'),
               _1500_ft_in_climb=KTI('_1500FtInClimb')):
        return NotImplemented


class PitchRateFrom2DegreesOfPitchTo35FeetMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('PitchRate'),
               liftoff=KTI('Liftoff'),
               _35_ft_in_takeoff=KTI('_35FtInTakeoff')):
        return NotImplemented


class PowerOnWithSpeedbrakesDeployedDurationGreaterThanLimit(KeyPointValueNode):
    def derive(self, eng_n1_avg=P('EngineN1Average'),
               speedbrake=P('Speedbrake')):
        return NotImplemented


class PullUpWarning(KeyPointValueNode):
    def derive(self, gpws_pull_up=P('GPWSPullUp')):
        return NotImplemented


class GroundspeedRTOMax(KeyPointValueNode):
    name = 'GroundspeedRTOMax'
    def derive(self, ):
        return NotImplemented


class RollCycles1000FtToTouchdownMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class RollBetween100And500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _100_ft_in_initial_climb=KTI('_100FtInInitial_Climb'),
               _500_ft_in_initial_climb=KTI('_500FtInInitial_Climb')):
        return NotImplemented


class RollBetween500And1500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               _500_ft_in_initial_climb=KTI('_500FtInInitial_Climb'),
               _1500_ft_in_climb=KTI('_1500FtInClimb')):
        return NotImplemented


class RollBelow20FeetMax(KeyPointValueNode):
    def derive(self,roll=P('Roll'),
               _25_ft_in_landing=KTI('_25FtInLanding'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class RollAbove1500FeetMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'),
               altitude_radio=P('AltitudeRadio')):
        return NotImplemented


class RudderReversalAbove50Feet(KeyPointValueNode):
    def derive(self,
               rudder_reversal=S('RudderReversal')):
        return NotImplemented


class AirspeedWithGearSelectedDownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_down=P('GearSelectedDown')):
        return NotImplemented


class AirspeedAtGearSelectedDown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_down=P('GearSelectedDown')):
        return NotImplemented


class AirspeedAtGearSelectedUp(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               gear_sel_up=P('GearSelectedUp')):
        return NotImplemented


class GroundspeedOnGroundMax(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'), on_ground=S('OnGround')):
        return NotImplemented


class MachMMOMax(KeyPointValueNode):
    name = 'MachMMOMax'
    def derive(self, mach=P('Mach')):
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
    name = 'AirspeedV2AtLiftOff'
    def derive(self, airspeed=P('Airspeed'),
               liftoff=KTI('Liftoff')):
        return NotImplemented


class AirspeedBetween90SToTouchdownAndTouchdownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               _90_secs_to_landing=KTI('_90SecsToLanding'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class AirspeedV235To400FeetMin(KeyPointValueNode):
    name = 'AirspeedV235To400FeetMin'
    def derive(self, airspeed=P('Airspeed'),
               _35_ft_in_takeoff=KTI('_35FtInTakeoff'),
               _400_ft_in_initial_climb=KTI('_400FtInInitial_Climb')):
        return NotImplemented


class AirspeedV2400To1500FeetMin(KeyPointValueNode):
    name = 'AirspeedV2400To1500FeetMin'
    def derive(self, airspeed=P('Airspeed'),
               _400_ft_in_initial_climb=KTI('_400FtInInitial_Climb'),
               _1500_ft_in_climb=KTI('_1500FtInClimb')):
        return NotImplemented


class AirspeedV2AtLiftOff(KeyPointValueNode):
    name = 'AirspeedV2AtLiftOff'
    def derive(self, ):
        return NotImplemented


class AirspeedVrefLast2MinutesToTouchdownMin(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'),
               _2_min_to_landing=KTI('_2MinToLanding'),
               touchdown=KTI('Touchdown')):
        return NotImplemented


class SpeedbrakesDeployed1000To30Feet(KeyPointValueNode):
    def derive(self, speedbrake=P('Speedbrake'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _25_ft_in_landing=KTI('_25FtInLanding')):
        return NotImplemented


class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), speedbrake=P('Speedbrake')):
        return NotImplemented


class StickShakerActivated(KeyPointValueNode):
    def derive(self, stick_shaker=P('StickShaker')):
        return NotImplemented


class TerrainWarning(KeyPointValueNode):
    def derive(self, gpws_terrain=P('GPWSTerrain')):
        return NotImplemented


class Terrain,PullUpWarning(KeyPointValueNode):
    def derive(self, gpws_terrain_pull_up=P('GPWSTerrainPullUp')):
        return NotImplemented


class ThrottleCycles1000FeetToTouchdownMax(KeyPointValueNode):
    def derive(self, thrust_lever_n=P('ThrustLeverN')):
        return NotImplemented


class TooLowFlapWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_flap=P('GPWSTooLowFlap')):
        return NotImplemented


class TooLowGearWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_gear=P('GPWSTooLowGear')):
        return NotImplemented


class TooLowTerrainWarning(KeyPointValueNode):
    def derive(self, gpws_too_low_terrain=P('GPWSTooLowTerrain')):
        return NotImplemented


class NormalGAirborneMin(KeyPointValueNode):
    def derive(self, acceleration_normal=P('AccelerationNormal'),
               airborne=S('Airborne')):
        return NotImplemented


class VibrationN1GreaterThanLimitDurationGreaterThanLimit(KeyPointValueNode):
    name = 'VibrationN1GreaterThanLimitDurationGreaterThanLimit'
    def derive(self, eng_n_vib_n1=P('EngineNVibN1'), airborne=S('Airborne')):
        return NotImplemented


class VibrationN2GreaterThanLimitDurationGreaterThanLimit(KeyPointValueNode):
    name = 'VibrationN2GreaterThanLimitDurationGreaterThanLimit'
    def derive(self, eng_n_vib_n2=P('EngineNVibN2'), airborne=S('Airborne')):
        return NotImplemented


class WindshearWarningBelow1500Feet(KeyPointValueNode):
    def derive(self, gpws_windshear=P('GPWSWindshear'),
               alt_aal=P('AltitudeAAL')):
        return NotImplemented


class TaxiSpeedStraight(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'),
               rate_of_turn=P('RateOfTurn')):
        return NotImplemented


class TaxiSpeedTurning(KeyPointValueNode):
    def derive(self, groundspeed=P('Groundspeed'),
               rate_of_turn=P('RateOfTurn')):
        return NotImplemented


class LowPowerInFinalApproach10Sec(KeyPointValueNode):
    def derive(self, eng_n_avg=P('Eng(N)Average'),
               _1000_ft_in_approach=KTI('_1000FtInApproach'),
               _50_ft_in_landing=KTI('_50FtInLanding')):
        return NotImplemented
