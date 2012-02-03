import logging
import numpy as np

from analysis.node import A, DerivedParameterNode, KPV, KTI, P, S, Parameter
from analysis.model_information import (get_aileron_map, get_config_map,
                                        get_flap_map, get_slat_map)
from analysis.library import (align, 
                              first_order_lag,
                              first_order_washout,
                              hysteresis, 
                              interleave,
                              rate_of_change, 
                              repair_mask,
                              round_to_nearest,
                              step_values,
                              straighten_headings,
                              vstack_params)

from settings import (AZ_WASHOUT_TC,
                      HYSTERESIS_FPALT,
                      HYSTERESIS_FPALT_CCD,
                      HYSTERESIS_FP_RAD_ALT,
                      HYSTERESIS_FPIAS, 
                      HYSTERESIS_FPROC,
                      GRAVITY,
                      KTS_TO_FPS,
                      RATE_OF_CLIMB_LAG_TC
                      )

#-------------------------------------------------------------------------------
# Derived Parameters


# Q: What do we do about accessing KTIs - params['a kti class name'] is a list of kti's
#   - could have a helper - filter_for('kti_name', take_max=True) # and possibly take_first, take_min, take_last??

# Q: Accessing information like ORIGIN / DESTINATION

# Q: What about V2 Vref etc?


class AccelerationVertical(DerivedParameterNode):
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_lat=P('Acceleration Lateral'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        """
        Resolution of three accelerations to compute the vertical
        acceleration (perpendicular to the earth surface). Upwards = +ve
        """
        # Simple Numpy algorithm working on masked arrays
        pitch_rad = np.radians(pitch.array)
        roll_rad = np.radians(roll.array)
        resolved_in_pitch = acc_long.array * np.ma.sin(pitch_rad) \
                            + acc_norm.array * np.ma.cos(pitch_rad)
        self.array = resolved_in_pitch * np.ma.cos(roll_rad) \
                     - acc_lat.array * np.ma.sin(roll_rad)
        

class AccelerationForwards(DerivedParameterNode):
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch')):
        """
        Resolution of three body axis accelerations to compute the forward
        acceleration, that is, in the direction of the aircraft centreline
        when projected onto the earth's surface. Forwards = +ve
        """
        # Simple Numpy algorithm working on masked arrays
        pitch_rad = np.radians(pitch.array)
        self.array = acc_long.array * np.cos(pitch_rad)\
                     - acc_norm.array * np.sin(pitch_rad)


class AccelerationSideways(DerivedParameterNode):
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_lat=P('Acceleration Lateral'),
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        """
        Resolution of three body axis accelerations to compute the lateral
        acceleration, that is, in the direction perpendicular to the aircraft centreline
        when projected onto the earth's surface. Right = +ve.
        """
        pitch_rad = np.radians(pitch.array)
        roll_rad = np.radians(roll.array)
        # Simple Numpy algorithm working on masked arrays
        resolved_in_pitch = acc_long.array * np.sin(pitch_rad) \
                            + acc_norm.array * np.cos(pitch_rad)
        self.array = resolved_in_pitch * np.sin(roll_rad) \
                     + acc_lat.array * np.cos(roll_rad)

"""
-------------------------------------------------------------------------------
Superceded by Truck and Trailer analysis of airspeed during takeoff and landing
-------------------------------------------------------------------------------
class AccelerationForwardsForFlightPhases(DerivedParameterNode):
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        if 'Airspeed' in available:
            return True
        elif 'Acceleration Longitudinal' in available:
            return True
        else:
            return False
        
    # List the optimal parameter set here
    def derive(self, acc_long=P('Acceleration Longitudinal'),
               airspeed=P('Airspeed')):
        '''
        Acceleration or deceleration on the runway is used to identify the
        runway heading. For the Hercules aircraft there is no longitudinal
        accelerometer, so rate of change of airspeed is used instead.
        '''
        if not acc_long: #  TODO: remove this inversion. Herc testing only.
            self.array = repair_mask(acc_long.array)
        else:
            '''
            This calculation is included for the few aircraft that do not
            have a longitudinal accelerometer installed, so we can identify
            acceleration or deceleration on the runway.
            '''
            # TODO: Remove float from line below
            aspd = P('Aspd',array=repair_mask(np.ma.array(airspeed.array.data, dtype='float')),frequency=airspeed.frequency)
            # Tacky smoothing to see how it works. TODO fix !
            roc_aspd = rate_of_change(aspd,1.5) * KTS_TO_FPS/GRAVITY
            self.array =  roc_aspd 
-------------------------------------------------------------------------------
Superceded by Truck and Trailer analysis of airspeed during takeoff and landing
-------------------------------------------------------------------------------
"""
            

class AirspeedForFlightPhases(DerivedParameterNode):
    def derive(self, airspeed=P('Airspeed')):
        self.array = hysteresis(airspeed.array, HYSTERESIS_FPIAS)


class AirspeedMinusV2(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, airspeed=P('Airspeed'), v2=P('V2')):
        self.array = airspeed.array - v2

class AirspeedMinusVref(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, airspeed=P('Airspeed'), vref=P('Vref')):
        self.array = airspeed.array - vref


class AirspeedTrue(DerivedParameterNode):
    #dependencies = ['SAT', 'VMO', 'MMO', 'Indicated Airspeed', 'Altitude QNH']
    # TODO: Move required dependencies from old format above to derive kwargs.
    def derive(self, ias = P('Airspeed'),
               alt_std = P('Altitude STD'),
               sat = P('SAT')):
        return NotImplemented
    

class AltitudeAAL(DerivedParameterNode):
    """
    TODO: Altitude Parameter to account for transition altitudes for airports
    between "altitude above mean sea level" and "pressure altitude relative
    to FL100". Ideally use the BARO selection switch when recorded, else the
    Airport elevation where provided, else guess based on location (USA =
    18,000ft, Europe = 3,000ft)
    """
    # Dummy for testing DJ TODO: Replace with one that takes radio altitude and local minima into account.
    name = 'Altitude AAL'
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):# alt_std=P('Altitude STD'), alt_rad=P('Altitude Radio')):
        # TODO: This is a hack! Implement separate derive method for Altitude
        # AAL.
        logging.warning("Altitude AAL is not properly implemented!!")
        self.array = alt_aal.array

    
class AltitudeAALForFlightPhases(DerivedParameterNode):
    name = 'Altitude AAL For Flight Phases'
    # This crude parameter is used for flight phase determination,
    # and only uses airspeed and pressure altitude for robustness.
    def derive(self, alt_std=P('Altitude STD'), fast=S('Fast')):
        
        # Initialise the array to zero, so that the altitude above the airfield
        # will be 0ft when the aircraft cannot be airborne.
        self.array = np.ma.zeros(len(alt_std.array))
        
        altitude = repair_mask(alt_std.array) # Remove small sections of corrupt data
        ##print 'fast, len(alt_std), alt_std.offset', fast, len(alt_std.array), alt_std.offset
        for speedy in fast:
            begin = speedy.slice.start
            end = speedy.slice.stop
            peak = np.ma.argmax(altitude[speedy.slice])
            # We override any negative altitude variations that occur at
            # takeoff or landing rotations. This parameter is only used for
            # flight phase determination so it is important that it behaves
            # in a predictable manner.
            ##print end
            self.array[begin:begin+peak] = np.ma.maximum(0.0,altitude[begin:begin+peak]-altitude[begin])
            self.array[begin+peak:end] = np.ma.maximum(0.0,altitude[begin+peak:end]-altitude[end-1])
    
    
class AltitudeForClimbCruiseDescent(DerivedParameterNode):
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis(alt_std.array, HYSTERESIS_FPALT_CCD)
    
    
class AltitudeForFlightPhases(DerivedParameterNode):
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis(repair_mask(alt_std.array), HYSTERESIS_FPALT)
    
    
class AltitudeRadio(DerivedParameterNode):
    """
    This function allows for the distance between the radio altimeter antenna
    and the main wheels of the undercarriage.

    The parameter raa_to_gear is measured in feet and is positive if the
    antenna is forward of the mainwheels.
    """
    def derive(self, alt_rad=P('Altitude Radio Sensor'), pitch=P('Pitch'),
               main_gear_to_alt_rad=A('Main Gear To Altitude Radio')):
        # Align the pitch attitude samples to the Radio Altimeter samples,
        # ready for combining them.
        pitch_rad = np.radians(pitch.array)
        # Now apply the offset if one has been provided
        self.array = alt_rad.array - np.sin(pitch_rad) * main_gear_to_alt_rad.value


class AltitudeRadioForFlightPhases(DerivedParameterNode):
    def derive(self, alt_rad=P('Altitude Radio')):
        self.array = hysteresis(repair_mask(alt_rad.array), HYSTERESIS_FP_RAD_ALT)


class AltitudeQNH(DerivedParameterNode):
    name = 'Altitude QNH'
    def derive(self, param=P('Altitude AAL')):
        return NotImplemented


class AltitudeSTD(DerivedParameterNode):
    name = 'Altitude STD'
    @classmethod
    def can_operate(cls, available):
        high_and_low = 'Altitude STD High' in available and \
            'Altitude STD Low' in available
        rough_and_ivv = 'Altitude STD Rough' in available and \
            'Inertial Vertical Speed' in available
        return high_and_low or rough_and_ivv
    
    def _high_and_low(self, alt_std_high, alt_std_low, top=18000, bottom=17000):
        # Create empty array to write to.
        alt_std = np.ma.empty(len(alt_std_high.array))
        alt_std.mask = np.ma.mask_or(alt_std_high.array.mask,
                                     alt_std_low.array.mask)
        difference = top - bottom
        # Create average of high and low. Where average is above crossover,
        # source value from alt_std_high. Where average is below crossover,
        # source value from alt_std_low.
        average = (alt_std_high.array + alt_std_low.array) / 2
        source_from_high = average > top
        alt_std[source_from_high] = alt_std_high.array[source_from_high]
        source_from_low = average < bottom
        alt_std[source_from_low] = alt_std_low.array[source_from_low]
        source_from_high_or_low = np.ma.logical_or(source_from_high,
                                                   source_from_low)
        crossover = np.ma.logical_not(source_from_high_or_low)
        crossover_indices = np.ma.where(crossover)[0]
        high_values = alt_std_high.array[crossover]
        low_values = alt_std_low.array[crossover]
        for index, high_value, low_value in zip(crossover_indices,
                                                high_values,
                                                low_values):
            average_value = average[index]
            high_multiplier = (average_value - bottom) / float(difference)
            low_multiplier = abs(1 - high_multiplier)
            crossover_value = (high_value * high_multiplier) + \
                (low_value * low_multiplier)
            alt_std[index] = crossover_value
        return alt_std
    
    def _rough_and_ivv(self, alt_std_rough, ivv):
        alt_std_with_lag = first_order_lag(alt_std_rough.array, 10,
                                           alt_std_rough.hz)
        mask = np.ma.mask_or(alt_std_with_lag.mask, ivv.array.mask)
        return np.ma.masked_array(alt_std_with_lag + (ivv.array / 60.0),
                                  mask=mask)
    
    def derive(self, alt_std_high=P('Altitude STD High'),
               alt_std_low=P('Altitude STD Low'),
               alt_std_rough=P('Altitude STD Rough'),
               ivv=P('Inertial Vertical Speed')): # Q: Is IVV name correct?
        if alt_std_high and alt_std_low:
            self.array = self._high_and_low(alt_std_high, alt_std_low)
            ##crossover = np.ma.logical_and(average > 17000, average < 18000)
            ##crossover_indices = np.ma.where(crossover)
            ##for crossover_index in crossover_indices:
                
            ##top = 18000
            ##bottom = 17000
            ##av = (alt_std_high + alt_std_low) / 2
            ##ratio = (top - av) / (top - bottom)
            ##if ratio > 1.0:
                ##ratio = 1.0
            ##elif ratio < 0.0:
                ##ratio = 0.0
            ##alt = alt_std_low * ratio + alt_std_high * (1.0 - ratio)
            ##alt_std  = alt_std * 0.8 + alt * 0.2
            
            #146-300 945003 (01) 
            #-------------------
            ##Set the thresholds for changeover from low to high scales.
            #top = 18000
            #bottom = 17000
            #
            #av = (ALT_STD_HIGH + ALT_STD_LOW) /2
            #ratio = (top - av) / (top - bottom)
            #
            #IF (ratio > 1.0) THEN ratio = 1.0 ENDIF
            #IF (ratio < 0.0) THEN ratio = 0.0 ENDIF
            #
            #alt = ALT_STD_LOW * ratio + ALT_STD_HIGH * (1.0 - ratio)
            #
            ## Smoothing to reduce unsightly noise in the signal. DJ
            #ALT_STDC = ALT_STDC * 0.8 + alt * 0.2
        elif alt_std_rough and ivv:
            self.array = self._rough_and_ivv(alt_std_rough, ivv)
            #ALT_STDC = (last_alt_std * 0.9) + (ALT_STD * 0.1) + (IVVR / 60.0)
            


class AltitudeTail(DerivedParameterNode):
    """
    This function allows for the distance between the radio altimeter antenna
    and the point of the airframe closest to tailscrape.
   
    The parameter gear_to_tail is measured in feet and is the distance from 
    the main gear to the point on the tail most likely to scrape the runway.
    """
    #TODO: Review availability of Attribute "Dist Gear To Tail"
    def derive(self, alt_rad = P('Altitude Radio'), 
               pitch = P('Pitch'),
               dist_gear_to_tail=A('Dist Gear To Tail')):
        # Align the pitch attitude samples to the Radio Altimeter samples,
        # ready for combining them.
        pitch_rad= np.radians(pitch.array)
        # Now apply the offset
        self.array = alt_rad.array - np.sin(pitch_rad) * dist_gear_to_tail.value
        

class ClimbForFlightPhases(DerivedParameterNode):
    #TODO: Optimise with numpy operations
    def derive(self, alt_std=P('Altitude STD'), airs=S('Fast')):
        self.array = np.ma.zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            ax = air.slice
            # Initialise the tracking altitude value
            curr_alt = alt_std.array[ax][0]
            self.array[ax][0] = 0.0
            for count in xrange(1, int(ax.stop - ax.start)):
                if alt_std.array[ax][count] < alt_std.array[ax][count-1]:
                    # Going down, keep track of current altitude
                    curr_alt = alt_std.array[ax][count]
                    self.array[ax][count] = 0.0
                else:
                    self.array[ax][count] = alt_std.array[ax][count] - curr_alt
    
    
class DistanceTravelled(DerivedParameterNode):
    "Distance travelled in Nautical Miles. Calculated using Groundspeed"
    #Q: could be validated using the track flown or distance between origin 
    # and destination
    def derive(self, gspd=P('Groundspeed')):
        return NotImplemented


class DistanceToLanding(DerivedParameterNode):
    # Q: Is this distance to final landing, or distance to each approach
    # destination (i.e. resets once reaches point of go-around)
    def derive(self, dist=P('Distance Travelled'), tdwns=KTI('Touchdown')):
               ##ils_gs=P('Glideslope Deviation'),
               ##ldg=P('LandingAirport')):
        return NotImplemented
    


class Eng_EGTAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EGT Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) EGT'),
               eng2=P('Eng (2) EGT'),
               eng3=P('Eng (3) EGT'),
               eng4=P('Eng (4) EGT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        
class Eng_EGTMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EGT Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) EGT'),
               eng2=P('Eng (2) EGT'),
               eng3=P('Eng (3) EGT'),
               eng4=P('Eng (4) EGT')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.max(axis=0)


class Eng_EGTMin(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EGT Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) EGT'),
               eng2=P('Eng (2) EGT'),
               eng3=P('Eng (3) EGT'),
               eng4=P('Eng (4) EGT')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.min(axis=0)


class Eng_N1Avg(DerivedParameterNode):
    name = "Eng (*) N1 Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_N1Max(DerivedParameterNode):
    name = "Eng (*) N1 Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        
        
class Eng_N1Min(DerivedParameterNode):
    name = "Eng (*) N1 Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_N2Avg(DerivedParameterNode):
    name = "Eng (*) N2 Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_N2Max(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) N2 Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_N2Min(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) N2 Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_OilTempAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Temp Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        

class Eng_OilTempMin(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Temp Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.min(axis=0)


class Eng_OilTempMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Temp Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.max(axis=0)


class Eng_OilPressAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Press Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        
        
class Eng_OilPressMax(DerivedParameterNode):
    #TODO: TEST
    #Q: Press or Pressure?
    name = "Eng (*) Oil Press Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.max(axis=0)


class Eng_OilPressMin(DerivedParameterNode):
    name = 'Eng (*) Oil Press Min'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.min(axis=0)


class Eng_VibN1Max(DerivedParameterNode):
    name = 'Eng (*) Vib N1 Max'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Vib N1'),
               eng2=P('Eng (2) Vib N1'),
               eng3=P('Eng (3) Vib N1'),
               eng4=P('Eng (4) Vib N1')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.max(axis=0)
        
        
class Eng_VibN2Max(DerivedParameterNode):
    name = 'Eng (*) Vib N2 Max'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Vib N2'),
               eng2=P('Eng (2) Vib N2'),
               eng3=P('Eng (3) Vib N2'),
               eng4=P('Eng (4) Vib N2')):
        eng = vstack_params(eng1, eng2, eng3, eng4)
        self.array = eng.max(axis=0)


class FuelQty(DerivedParameterNode):
    '''
    May be replaced by an LFL parameter of the same name if available.
    
    Sum of fuel in left, right and middle tanks where available.
    '''
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               fuel_qty1=P('Fuel Qty (1)'),
               fuel_qty2=P('Fuel Qty (2)'),
               fuel_qty3=P('Fuel Qty (3)')):
        # Repair array masks to ensure that the summed values are not too small
        # because they do not include masked values.
        for param in filter(bool, [fuel_qty1, fuel_qty2, fuel_qty3]):
            param.array = repair_mask(param.array)
        stacked_params = vstack_params(fuel_qty1, fuel_qty2, fuel_qty3)
        self.array = np.ma.sum(stacked_params, axis=0)


class FlapLever(DerivedParameterNode):
    """
    Steps raw Flap angle from lever into detents.
    """
    def derive(self, flap=P('Flap Lever Position'), series=A('Series'), family=A('Family')):
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except ValueError:
            # no flaps mapping, round to nearest 5 degrees
            logging.warning("No flap settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(flap.array, 5.0)
        else:
            self.array = step_values(flap.array, flap_steps)
        
            
class Flap(DerivedParameterNode):
    """
    Steps raw Flap angle into detents.
    """
    def derive(self, flap=P('Flap Surface'), series=A('Series'), family=A('Family')):
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except ValueError:
            # no flaps mapping, round to nearest 5 degrees
            logging.warning("No flap settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(flap.array, 5.0)
        else:
            self.array = step_values(flap.array, flap_steps)
        
            
class Slat(DerivedParameterNode):
    """
    Steps raw Slat angle into detents.
    """
    def derive(self, slat=P('Slat Surface'), series=A('Series'), family=A('Family')):
        try:
            slat_steps = get_slat_map(series.value, family.value) 
        except ValueError:
            # no slats mapping, round to nearest 5 degrees
            logging.warning("No slat settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(slat.array, 5.0)
        else:
            self.array = step_values(slat.array, slat_steps)
            
            
            
class Config(DerivedParameterNode):
    """
    Multi-state with the following mapping:
    {
        0 : '0',
        1 : '1',
        2 : '1 + F',
        3 : '2(a)',  #Q: should display be (a) or 2* or 1* ?!
        4 : '2',
        5 : '3(b)',
        6 : '3',
        7 : 'FULL',
    }
    (a) corresponds to CONF 1*
    (b) corresponds to CONF 2*
    
    Note: Does not use the Flap Lever position. This parameter reflects the
    actual config state of the aircraft rather than the intended state
    represented by the selected lever position.
    
    Note: Values that do not map directly to a required state are masked with
    the data being random (memory alocated)
    """
    @classmethod
    def can_operate(cls, available):
        if 'Flap' in available and 'Slat' in available \
           and 'Series' in available and 'Family' in available:
            return True
        
    def derive(self, flap=P('Flap'), slat=P('Slat'), aileron=P('Aileron'), 
               series=A('Series'), family=A('Family')):
        #TODO: manu=A('Manufacturer') - we could ensure this is only done for Airbus?
        
        mapping = get_config_map(series.value, family.value)        
        qty_param = len(mapping.itervalues().next())
        if qty_param == 3 and not aileron:
            # potential problem here!
            logging.warning("Aileron not available, so will calculate Config using only slat and flap")
            qty_param = 2
        elif qty_param == 2 and aileron:
            # only two items in values tuple
            logging.debug("Aileron available but not required for Config calculation")
            pass
        
        #TODO: Scale each parameter individually to ensure uniqueness
        # sum the required parameters
        summed = vstack_params(*(flap, slat, aileron)[:qty_param]).sum(axis=0)
        
        # create a placeholder array fully masked
        self.array = np.ma.empty_like(flap.array)
        self.array.mask=True
        for state, values in mapping.iteritems():
            s = sum(values[:qty_param])
            # unmask bits we know about
            self.array[summed == s] = state
            
    
class GearSelectedDown(DerivedParameterNode):
    # And here is where the nightmare starts.
    # Sometimes recorded
    # Sometimes interpreted from other signals
    # There's no pattern to how this is worked out.
    # For aircraft with a Gear Selected Down parameter let's try this...
    def derive(self, param=P('Gear Selected Down FDR')):
        return NotImplemented


class GearSelectedUp(DerivedParameterNode):
    def derive(self, param=P('Gear Selected Up FDR')):
        pass


class HeadingContinuous(DerivedParameterNode):
    """
    For all internal computing purposes we use this parameter which does not
    jump as it passes through North. To recover the compass display, modulus
    ("%360" in Python) returns the value to display to the user.
    """
    def derive(self, head_mag=P('Heading Magnetic')):
        self.array = repair_mask(straighten_headings(head_mag.array))


class HeadingMagnetic(DerivedParameterNode):
    '''
    This class currently exists only to give the 146-301 Magnetic Heading.
    '''
    def derive(self, head_mag=P('RECORDED MAGNETIC HEADING')):
        self.array = head_mag.array


class HeadingTrue(DerivedParameterNode):
    #TODO: TESTS
    # Requires the computation of a magnetic deviation parameter linearly 
    # changing from the deviation at the origin to the destination.
    def derive(self, head = P('Heading Continuous'),
               dev = P('Magnetic Deviation')):
        self.array = head + dev.array
    

class ILSLocalizerGap(DerivedParameterNode):
    def derive(self, ils_loc = P('Localizer Deviation'),
               alt_aal = P('Altitude AAL')):
        return NotImplemented

    
class ILSGlideslopeGap(DerivedParameterNode):
    def derive(self, ils_gs = P('Glideslope Deviation'),
               alt_aal = P('Altitude AAL')):
        return NotImplemented
 
    
class MACH(DerivedParameterNode):
    def derive(self, ias = P('Airspeed'), tat = P('TAT'),
               alt = P('Altitude Std')):
        return NotImplemented
        

class RateOfClimb(DerivedParameterNode):
    """
    This routine derives the rate of climb from the vertical acceleration, the
    Pressure altitude and the Radio altitude.
    
    We use pressure altitude rate above 100ft and radio altitude rate below
    50ft, with a progressive changeover across that range. Below 100ft the
    pressure altitude information is affected by the flow field around the
    aircraft, while above 50ft there is an increasing risk of changes in
    ground profile affecting the radio altimeter signal.
    
    Complementary first order filters are used to combine the acceleration
    data and the height data. A high pass filter on the altitude data and a
    low pass filter on the acceleration data combine to form a consolidated
    signal.
    
    By merging the altitude rate signals, we avoid problems of altimeter
    datums affecting the transition as these will have been washed out by the
    filter stage first.
    
    Long term errors in the accelerometers are removed by washing out the
    acceleration term with a longer time constant filter before use. The
    consequence of this is that long period movements with continued
    acceleration will be underscaled slightly. As an example the test case
    with a 1ft/sec^2 acceleration results in an increasing rate of climb of
    55 fpm/sec, not 60 as would be theoretically predicted.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters. If 'Altitude Radio For Flight
        # Phases' is available, that's a bonus and we will use it, but it is
        # not required.
        return 'Altitude STD' in available
    
    def derive(self, 
               az = P('Acceleration Vertical'),
               alt_std = P('Altitude STD'),
               alt_rad = P('Altitude Radio')):
        if az and alt_rad:
            # Use the complementary smoothing approach

            roc_alt_std = first_order_washout(alt_std.array,
                                              RATE_OF_CLIMB_LAG_TC, az.hz)
            roc_alt_rad = first_order_washout(alt_rad.array,
                                              RATE_OF_CLIMB_LAG_TC, az.hz)
                    
            # Use pressure altitude rate above 100ft and radio altitude rate
            # below 50ft with progressive changeover across that range.
            # up to 50 ft radio 0 < std_rad_ratio < 1 over 100ft radio
            std_rad_ratio = np.maximum(np.minimum(
                (alt_rad.array.data-50.0)/50.0,
                1),0)
            roc_altitude = roc_alt_std*std_rad_ratio +\
                roc_alt_rad*(1.0-std_rad_ratio)
                
            roc_altitude /= RATE_OF_CLIMB_LAG_TC # Remove washout gain  
            
            # Lag this rate of climb
            az_washout = first_order_washout (az.array, AZ_WASHOUT_TC, az.hz, initial_value = az.array[0])
            inertial_roc = first_order_lag (az_washout, RATE_OF_CLIMB_LAG_TC, az.hz, gain=GRAVITY*RATE_OF_CLIMB_LAG_TC)
            self.array = (roc_altitude + inertial_roc) * 60.0
        else:
            # The period for averaging altitude only data has been chosen
            # from careful inspection of Hercules data, where the pressure
            # altitude signal resolution is of the order of 9 ft/bit.
            # Extension to wider timebases, or averaging with more samples,
            # smooths the data more but equally more samples are affected by
            # corrupt source data. So, change the "3" only after careful
            # consideration.
            self.array = rate_of_change(alt_std,3)*60


class RateOfClimbForFlightPhases(DerivedParameterNode):
    def derive(self, alt_std = P('Altitude STD')):
        self.array = hysteresis(rate_of_change(repair_mask(alt_std),3)*60,
                                HYSTERESIS_FPROC)


class Relief(DerivedParameterNode):
    # also known as Terrain
    
    # Quickly written without tests as I'm really editing out the old dependencies statements :-(
    def derive(self, alt_aal = P('Altitude AAL'),
               alt_rad = P('Radio Altitude')):
        self.array = alt_aal - alt_rad


class Speedbrake(DerivedParameterNode):
    def derive(self, param=P('Speedbrake FDR')):
        # There will be a recorded parameter, but varying types of correction will 
        # need to be applied according to the aircraft type and data frame.
        self.array = param


'''

Better done together

class SmoothedLatitude(DerivedParameterNode): # TODO: Old dependency format.
    dependencies = ['Latitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class SmoothedLongitude(DerivedParameterNode): # TODO: Old dependency format.
    dependencies = ['Longitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
'''


class RateOfTurn(DerivedParameterNode):
    def derive(self, head=P('Heading Continuous')):
        #TODO: Review whether half_width 1 is applicable at multi Hz
        self.array = rate_of_change(head, 1)


class Pitch(DerivedParameterNode):
    def derive(self, p1=P('Pitch (1)'), p2=P('Pitch (2)')):
        self.hz = p1.hz * 2
        self.offset = min(p1.offset, p2.offset)
        self.array = interleave (p1, p2)


class ThrottleLever(DerivedParameterNode):
    def derive(self, tla1=P('Throttle Lever Angle (1)'), 
               tla2=P('Throttle Lever Angle (2)')):
        ##self.hz = tla1.hz * 2
        ##self.offset = min(tla1.offset, tla2.offset)
        ##self.array = interleave (tla1, tla2)
        return NotImplemented


