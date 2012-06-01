import logging
import numpy as np
from tempfile import TemporaryFile
from operator import attrgetter
from math import floor, radians

from analysis_engine.settings import API_HANDLER
from analysis_engine.api_handler import get_api_handler
    
from analysis_engine.model_information import (get_aileron_map, 
                                               get_config_map,
                                               get_flap_map,
                                               get_slat_map)
from analysis_engine.node import A, DerivedParameterNode, KPV, KTI, P, S, Parameter

from analysis_engine.library import (align,
                                     bearings_and_distances,
                                     blend_alternate_sensors,
                                     blend_two_parameters,
                                     clip,
                                     coreg,
                                     first_valid_sample,
                                     first_order_lag,
                                     first_order_washout,
                                     ground_track,
                                     hysteresis,
                                     index_at_value,
                                     index_of_datetime,
                                     integrate,
                                     interleave,
                                     ils_localizer_align,
                                     interpolate_and_extend,
                                     is_index_within_slice,
                                     is_slice_within_slice,
                                     last_valid_sample,
                                     latitudes_and_longitudes,
                                     merge_sources,
                                     merge_two_parameters,
                                     np_ma_ones_like,
                                     np_ma_masked_zeros_like,
                                     np_ma_zeros_like,
                                     rate_of_change, 
                                     repair_mask,
                                     rms_noise,
                                     round_to_nearest,
                                     runway_distances,
                                     runway_heading,
                                     runway_length,
                                     slices_overlap,
                                     smooth_track,
                                     step_values,
                                     straighten_headings,
                                     track_linking,
                                     value_at_index,
                                     vstack_params,
                                     
                                     alt2press,
                                     alt2press_ratio,
                                     alt2sat,
                                     cas2dp,
                                     cas_alt2mach,
                                     dp_over_p2mach,
                                     dp2tas,
                                     machtat2sat,
                                     _alt2press_ratio_gradient,
                                     _alt2press_ratio_isothermal)

from settings import (AIRSPEED_THRESHOLD,
                      ALTITUDE_AAL_LAG_TC,
                      AZ_WASHOUT_TC,
                      AT_WASHOUT_TC,
                      FEET_PER_NM,
                      GROUNDSPEED_LAG_TC,
                      HYSTERESIS_FPALT,
                      HYSTERESIS_FPALT_CCD,
                      HYSTERESIS_FPIAS,
                      HYSTERESIS_FP_RAD_ALT,
                      HYSTERESIS_FPROC,
                      GRAVITY_IMPERIAL,
                      GRAVITY_METRIC,
                      KTS_TO_FPS,
                      KTS_TO_MPS,
                      METRES_TO_FEET,
                      RATE_OF_CLIMB_LAG_TC,
                      TRANSITION_ALT_RAD_TO_STD,
                      )

# There is no numpy masked array function for radians, so we just multiply thus:
deg2rad = radians(1.0)

class AccelerationVertical(DerivedParameterNode):
    """
    Resolution of three accelerations to compute the vertical
    acceleration (perpendicular to the earth surface). Result is in g,
    retaining the 1.0 datum and positive upwards.
    """
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_lat=P('Acceleration Lateral'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        pitch_rad = pitch.array*deg2rad
        roll_rad = roll.array*deg2rad
        resolved_in_roll = acc_norm.array*np.ma.cos(roll_rad)\
            - acc_lat.array * np.ma.sin(roll_rad)
        self.array = resolved_in_roll * np.ma.cos(pitch_rad) \
                     + acc_long.array * np.ma.sin(pitch_rad)
        

class AccelerationForwards(DerivedParameterNode):
    """
    Resolution of three body axis accelerations to compute the forward
    acceleration, that is, in the direction of the aircraft centreline
    when projected onto the earth's surface. Forwards = +ve
    """
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch')):
        pitch_rad = pitch.array*deg2rad
        self.array = acc_long.array * np.cos(pitch_rad)\
                     - acc_norm.array * np.sin(pitch_rad)


class AccelerationAcrossTrack(DerivedParameterNode):
    """
    The forward and sideways ground-referenced accelerations are resolved
    into along track and across track coordinates in preparation for
    groundspeed computations.
    """
    def derive(self, acc_fwd=P('Acceleration Forwards'),
               acc_side=P('Acceleration Sideways'),
               drift=P('Drift')):
        drift_rad = drift.array*deg2rad
        self.array = acc_side.array * np.cos(drift_rad)\
            - acc_fwd.array * np.sin(drift_rad)


class AccelerationAlongTrack(DerivedParameterNode):
    """
    The forward and sideways ground-referenced accelerations are resolved
    into along track and across track coordinates in preparation for
    groundspeed computations.
    """
    def derive(self, acc_fwd=P('Acceleration Forwards'), 
               acc_side=P('Acceleration Sideways'), 
               drift=P('Drift')):
        drift_rad = drift.array*deg2rad
        self.array = acc_fwd.array * np.cos(drift_rad)\
                     + acc_side.array * np.sin(drift_rad)


class AccelerationSideways(DerivedParameterNode):
    """
    Resolution of three body axis accelerations to compute the lateral
    acceleration, that is, in the direction perpendicular to the aircraft centreline
    when projected onto the earth's surface. Right = +ve.
    """
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_lat=P('Acceleration Lateral'),
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        pitch_rad = pitch.array*deg2rad
        roll_rad = roll.array*deg2rad
        # Simple Numpy algorithm working on masked arrays
        resolved_in_pitch = acc_long.array * np.sin(pitch_rad) \
                            + acc_norm.array * np.cos(pitch_rad)
        self.array = resolved_in_pitch * np.sin(roll_rad) \
                     + acc_lat.array * np.cos(roll_rad)


class AirspeedForFlightPhases(DerivedParameterNode):
    def derive(self, airspeed=P('Airspeed')):
        self.array = hysteresis(repair_mask(airspeed.array), HYSTERESIS_FPIAS)


class AirspeedMinusV2(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, airspeed=P('Airspeed'), v2=P('V2')):
        self.array = airspeed.array - v2.array


class AirspeedMinusV2For3Sec(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, spd_v2=P('Airspeed Minus V2')):
        self.array = clip(spd_v2.array, 3.0, spd_v2.frequency)
        

class AirspeedMinusV2For5Sec(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, airspeed=P('Airspeed Minus V2')):
        self.array = clip(spd_v2.array, 5.0, spd_v2.frequency)
        

class AirspeedMinusVref(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, airspeed=P('Airspeed'), vref=A('FDR Vref')):
        self.array = airspeed.array - vref.value


class AirspeedMinusVrefFor3Sec(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, spd_vref=P('Airspeed Minus Vref')):
        self.array = clip(spd_vref.array, 3.0, spd_vref.frequency)

        
class AirspeedMinusVrefFor5Sec(DerivedParameterNode):
    #TODO: TESTS
    def derive(self, spd_vref=P('Airspeed Minus Vref')):
        self.array = clip(spd_vref.array, 5.0, spd_vref.frequency)

        
class AirspeedTrue(DerivedParameterNode):
    """
    True airspeed is computed from the recorded airspeed and pressure
    altitude. We assume that the recorded airspeed is indicated or computed,
    and that the pressure altitude is on standard (1013mB = 29.92 inHg).
    
    There are a few aircraft still operating which do not record the air
    temperature, so only these two parameters are required for the algorithm
    to run.
    
    Where air temperature is available, we accept Toal Air Temperature (TAT)
    and include this accordingly.
    
    Thanks are due to Kevin Horton of Ottawa for permission to derive the
    code here from his AeroCalc library.
    
    True airspeed is also extended to the ends of the takeoff and landing
    run, in particular so that we can estimate the minimum airspeed at which
    thrust reversers are used.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Airspeed' in available and 'Altitude STD' in available
    
    def derive(self, cas_p = P('Airspeed'),
               alt_std_p = P('Altitude STD'),
               tat_p = P('TAT'), 
               toffs=S('Takeoff'), lands=S('Landing'), 
               gspd=P('Groundspeed'), acc_fwd=P('Acceleration Forwards')):
        
        cas = cas_p.array
        alt_std = alt_std_p.array
        if tat_p:
            tat = tat_p.array
            dp = cas2dp(cas)
            mach = cas_alt2mach(cas, alt_std)
            sat = machtat2sat(mach, tat)
            tas = dp2tas(dp, alt_std, sat)
            combined_mask= np.logical_or(
                np.logical_or(cas_p.array.mask,alt_std_p.array.mask),
                tas.mask)
        else:
            dp = cas2dp(cas)
            sat = alt2sat(alt_std)
            tas = dp2tas(dp, alt_std, sat)
            combined_mask= np.logical_or(cas_p.array.mask,alt_std_p.array.mask)
            
        tas_from_airspeed = np.ma.masked_less(np.ma.array(data=tas, mask=combined_mask),50)
        tas_valids = np.ma.clump_unmasked(tas_from_airspeed)
        
        # Now see if we can extend this during the takeoff phase, using
        # either recorded groundspeed or failing that integrating
        # acceleration:
        for toff in toffs:
            for tas_valid in tas_valids:
                tix = tas_valid.start
                if is_index_within_slice(tix, toff.slice):
                    tas_0 = tas_from_airspeed[tix]
                    wind = tas_0 - gspd.array[tix]
                    scope = slice(toff.slice.start, tix)
                    if gspd:
                        tas_from_airspeed[scope] = gspd.array[scope] + wind
                    else:
                        tas_from_airspeed[scope] = \
                            integrate(acc_fwd.array[scope], acc_fwd.frequency, 
                                      initial_value=tas_0,
                                      scale=GRAVITY_IMPERIAL/KTS_TO_FPS, 
                                      direction='backwards')
                    
        # Then see if we can do the same for the landing phase:
        for land in lands:
            for tas_valid in tas_valids:
                tix = tas_valid.stop - 1
                if is_index_within_slice(tix, land.slice):
                    tas_0 = tas_from_airspeed[tix]
                    wind = tas_0 - gspd.array[tix]
                    scope = slice(tix + 1, land.slice.stop)
                    if gspd:
                        tas_from_airspeed[scope] = gspd.array[scope] + wind
                    else:
                        tas_from_airspeed[scope] = \
                            integrate(acc_fwd.array[scope], acc_fwd.frequency,
                                      initial_value=tas_0, 
                                      scale=GRAVITY_IMPERIAL/KTS_TO_FPS)
                    
        self.array = tas_from_airspeed
        

class AltitudeAAL(DerivedParameterNode):
    """
    Altitude Parameter to account for transition altitudes for airports
    between "altitude above mean sea level" and "pressure altitude relative
    to FL100". Ideally use the BARO selection switch when recorded, else the
    Airport elevation where provided, else guess based on location (USA =
    18,000ft, Europe = 3,000ft)

    This is the main altitude measure used during flight analysis.
    
    Where radio altimeter data is available, this is used for altitudes up to
    100ft and thereafter the pressure altitude signal is used. The two are
    "joined" together at the sample above 100ft in the climb or descent as
    appropriate.
    
    If no radio altitude signal is available, the simple measure based on
    pressure altitude only is used, which provides workable solutions except
    that the point of takeoff and landing may be inaccurate.
    
    This parameter includes a rejection of bounced landings of less than 35ft
    height. To monitor these accurately, look at the raw altitude radio
    parameter.
    """    
    name = "Altitude AAL"
    units = 'ft'

    @classmethod
    def can_operate(cls, available):
        return all([d in available for d in ('Altitude STD', 'Fast')])
        
    def derive(self, alt_std = P('Altitude STD'),
               alt_rad = P('Altitude Radio'),
               speedies = S('Fast')):
       
        # alt_aal will be zero on the airfield
        alt_aal = np_ma_zeros_like(alt_std.array)
  
        if alt_rad:
            for speedy in speedies:
                quick = speedy.slice
                # Make use of the fact that radio altimeters give negative readings on the ground.
                from_0_to_100ft = np.ma.masked_outside(alt_rad.array[quick], 0.0, 100.0)
                ralt_sections = np.ma.clump_unmasked(from_0_to_100ft )
                
                for ralt_section in ralt_sections:
                    if np.ma.max(alt_rad.array[quick][ralt_section]) > 35.0:
                        # We have a reading between 35ft and 100ft so this looks good.
                        alt_aal[quick][ralt_section] = alt_rad.array[quick][ralt_section]
                    else:
                        # Probably a bounced landing, which we will ignore.
                        pass                    
  
                # Checking for masked greater tnan 100ft makes this match the
                # previous mask operation in the case of a masked rad alt
                # sample on the 100ft threshold.
                below_100ft = np.ma.masked_greater(alt_rad.array[quick], 100.0)
                baro_sections = np.ma.clump_masked(below_100ft)
                
                for baro_section in baro_sections:
                    # TODO: Consider if we should default to zero or baro alt in the case the tests below fail in the case of corrupt data?
                    begin_index = (quick.start or 0) + baro_section.start
                    peak_index = np.ma.argmax(alt_std.array[quick][baro_section]) + begin_index
                    end_index = (quick.start or 0) + baro_section.stop
                    
                    up_diff = None
                    dn_diff = None
                    for ralt_section in ralt_sections:
                        if ralt_section.stop == baro_section.start:
                            up_diff =  alt_std.array[begin_index] - alt_rad.array[begin_index]
                        elif ralt_section.start == baro_section.stop:
                            dn_diff = alt_std.array[end_index] - alt_rad.array[end_index]
                            
                    if up_diff == None and dn_diff == None:
                        # Neither edge was found. Did we get close the ground at some point?
                        low_index = np.ma.argmin(alt_rad.array[begin_index:end_index])
                        if low_index:
                            # We did, so use that as a datum
                            mid_diff = alt_std.array[low_index] - alt_rad.array[low_index]
                            alt_aal[begin_index:end_index] = alt_std.array[begin_index:end_index] - mid_diff
                        else:
                            # We didn't so there is nothing to be done but copy the altitude across.
                            alt_aal[begin_index:end_index] = alt_std.array[begin_index:end_index]
                    elif up_diff == None:
                        # No rising edge was found, so stretch the descending edge to the end of the data.
                        alt_aal[begin_index:end_index] = alt_std.array[begin_index:end_index] - dn_diff
                    elif dn_diff == None:
                        #No falling edge was found...
                        alt_aal[begin_index:end_index] = alt_std.array[begin_index:end_index] - up_diff
                    else:
                        alt_aal[begin_index:peak_index] = alt_std.array[begin_index:peak_index] - up_diff
                        alt_aal[peak_index:end_index] = alt_std.array[peak_index:end_index] - dn_diff

                self.array = alt_aal

        else:
            for speedy in speedies:
                quick = speedy.slice
                begin_index = quick.start or 0
                peak_index = np.ma.argmax(alt_std.array[quick]) + begin_index
                end_index = quick.stop or len(alt_std.array)
                
                up_datum = alt_std.array[begin_index]
                alt_aal[begin_index:peak_index] = alt_std.array[begin_index:peak_index] - up_datum
                
                dn_datum = alt_std.array[end_index]
                alt_aal[peak_index:end_index] = alt_std.array[peak_index:end_index] - dn_datum
                
                # This backstop trap for negative values is necessary as aircraft without rad alts will indicate negative altitudes as they land. 
                self.array = np.ma.maximum(alt_aal, 0.0)
        

    
class AltitudeAALForFlightPhases(DerivedParameterNode):
    name = 'Altitude AAL For Flight Phases'
    units = 'ft'
    
    # This parameter repairs short periods of masked data, making it suitable
    # for detecting altitude bands on the climb and descent. The parameter
    # should not be used to compute KPV values themselves, to avoid using
    # interpolated values in an event.
    
    def derive(self, alt_aal=P('Altitude AAL')):
        
        self.array = repair_mask(alt_aal.array)
   

'''
class AltitudeAALSmoothed(DerivedParameterNode):
    """
    A smoothed version of this altitude signal is available that includes
    inertial smoothing to provide a higher sample rate signal. This may be
    used for accurate determination of variations during turbulence or bumpy
    landings.
    
    """    
    name = "Altitude AAL"
    units = 'ft'

    @classmethod
    def can_operate(cls, available):
        #TODO: Improve accuracy of this method. For example, the code does
        #not cater for the availability of Altitude Radio but Rate Of Climb
        #not being available.
        smoothing_params = all([d in available for d in ('Liftoff',
                                                         'Touchdown',
                                                         'Takeoff',
                                                         'Landing',
                                                         'Rate Of Climb',
                                                         'Altitude STD',
                                                         'Altitude Radio',
                                                         'Airspeed')])
        fallback = 'Altitude AAL For Flight Phases' in available
        return smoothing_params or fallback
    
    def derive(self, liftoffs=KTI('Liftoff'),
               touchdowns=KTI('Touchdown'),
               takeoffs=S('Takeoff'),
               landings=S('Landing'),
               roc = P('Rate Of Climb'),
               alt_std = P('Altitude STD'),
               alt_rad = P('Altitude Radio'),
               airspeed = P('Airspeed'),
               alt_aal_4fp = P('Altitude AAL For Flight Phases'),):
        if liftoffs and touchdowns and landings and roc and alt_std \
           and alt_rad and airspeed:
            # Initialise the array to zero, so that the altitude above the airfield
            # will be 0ft when the aircraft cannot be airborne.
            alt_aal = np_ma_zeros_like(alt_std.array) 
            # Actually creates a masked copy filled with zeros.
            
            ordered_ktis = sorted(liftoffs + touchdowns,
                                  key=attrgetter('index'))
            
            for next_index, first_kti in enumerate(ordered_ktis, start=1):
                # Iterating over pairs of 'Liftoff' and 'Touchdown' KTIs ordered
                # by index. Expecting Touchdowns followed by Liftoffs.
                try:
                    second_kti = ordered_ktis[next_index]
                except IndexError:
                    break
                in_air_slice = slice(first_kti.index, second_kti.index)
                
                             
                # Use pressure altitude to ensure data is filled between
                # Liftoff and Touchdown KTIs.
                alt_aal[in_air_slice] = alt_std.array[in_air_slice]
                peak_index = np.ma.argmax(alt_std.array[in_air_slice]) + \
                                        in_air_slice.start
                if first_kti.name == 'Liftoff':
                    threshold_index = index_at_value(alt_rad.array,
                                                     TRANSITION_ALT_RAD_TO_STD,
                                                     _slice=in_air_slice)
                    join_index = int(threshold_index)
                    difference = alt_rad.array[join_index] - \
                        alt_std.array[join_index]
                    alt_aal[join_index:peak_index] += difference
                    pre_threshold = slice(in_air_slice.start, join_index)
                    alt_aal[pre_threshold] = alt_rad.array[pre_threshold]
                
                if second_kti.name == 'Touchdown':
                    reverse_slice = slice(in_air_slice.stop,
                                          in_air_slice.start, -1)
                    threshold_index = index_at_value(alt_rad.array,
                                                     TRANSITION_ALT_RAD_TO_STD,
                                                     _slice=reverse_slice)
                    join_index = int(threshold_index)+1
                    difference = alt_rad.array[join_index] - \
                        alt_std.array[join_index]
                    alt_aal[peak_index:join_index] += difference
                    post_threshold = slice(join_index, in_air_slice.stop)
                    alt_aal[post_threshold] = alt_rad.array[post_threshold]
        
            # Use the complementary smoothing approach
            roc_lag = first_order_lag(roc.array,
                                      ALTITUDE_AAL_LAG_TC, roc.hz,
                                      gain=ALTITUDE_AAL_LAG_TC/60.0)            
            alt_aal_lag = first_order_lag(alt_aal, ALTITUDE_AAL_LAG_TC, roc.hz)
            alt_aal = alt_aal_lag + roc_lag
            # Force result to zero at low speed and on the ground.
            alt_aal[airspeed.array < AIRSPEED_THRESHOLD] = 0
            #alt_aal[alt_rad.array < 0] = 0
            self.array = np.ma.maximum(0.0,alt_aal)
            
        else:
            self.array = np.ma.copy(alt_aal_4fp.array) 
            '''
    
class AltitudeForClimbCruiseDescent(DerivedParameterNode):
    """
    This parameter includes a huge hysteresis and is solely to identify climb
    and descent phases. DO NOT use this for event detection.
    """
    units = 'ft'
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis(repair_mask(alt_std.array), HYSTERESIS_FPALT_CCD)
    
    
class AltitudeForFlightPhases(DerivedParameterNode):
    """
    This parameter includes a hysteresis to avoid repeated phase changes in
    the cruise. DO NOT use this for event detection.
    """
    units = 'ft'
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis(repair_mask(alt_std.array), HYSTERESIS_FPALT)
    

class AltitudeRadio(DerivedParameterNode):
    """
    There is a wide variety of radio altimeter installations including linear
    and non-linear transducers with various transfer functions, and two or
    three sensors may be installed each with different timing and
    inaccuracies to be compensated.
    
    The input data is stored 'temporarily' in parameters named Altitude Radio
    (A) to (D), and the frame details are augmented by a frame qualifier
    which identifies which formula to apply.
    
    :param frame: The frame attribute, e.g. '737-i'
    :type frame: An attribute
    :param frame_qual: The frame qualifier, e.g. 'Altitude_Radio_D226A101_1_16D'
    :type frame_qual: An attribute
    
    :returns Altitude Radio with values typically taken as the mean between
    two valid sensors.
    :type parameter object.
    """
    @classmethod
    def can_operate(cls, available):
        if 'Altitude Radio (A)' in available and \
           'Altitude Radio (B)' in available:
            return True
    
    align_to_first_dependency = False
    
    def derive(self, frame = A('Frame'),
               frame_qual = A('Frame Qualifier'),
               source_A = P('Altitude Radio (A)'),
               source_B = P('Altitude Radio (B)'),
               source_C = P('Altitude Radio (C)'),
               source_D = P('Altitude Radio (D)')):
        
        frame_name = frame.value if frame else None
        frame_qualifier = frame_qual.value if frame_qual else None
        
        if frame_name in ['737-3C']:
            # Alternate samples (A) for this frame have latency of over 1
            # second, so do not contribute to the height measurements
            # available. For this reason we only blend the two good sensors.
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_B, source_C)
            
        elif frame_name in ['737-4', '737-4_Analogue']:
            self.array, self.frequency, self.offset = \
                merge_two_parameters(source_A, source_B)
        
        elif frame_name in ['737-5']:
            alt_rad_efis = frame_qualifier and 'Altitude_Radio_EFIS' in frame_qualifier
            alt_rad_d226a101_1_16d = \
                frame_qualifier and 'Altitude_Radio_D226A101_1_16D' in frame_qualifier
            if alt_rad_efis:
                self.array, self.frequency, self.offset = \
                    blend_two_parameters(source_C, source_D)
                self.array = np.ma.masked_greater(self.array, 2600)
            elif frame_qualifier and 'Altitude_Radio_None' in frame_qualifier:
                pass # Some old 737 aircraft have no rad alt recorded.
            else:
                raise ValueError,'737-5 frame needs qualifier Altitude_Radio_Analogue/EFIS/None'
        
        else:
            logging.warning("No specified Altitude Radio (*) merging for frame "
                            "'%s' so using source (A)", frame_name)
            self.array = source_A.array


class AltitudeRadioForFlightPhases(DerivedParameterNode):
    def derive(self, alt_rad=P('Altitude Radio')):
        self.array = hysteresis(repair_mask(alt_rad.array), HYSTERESIS_FP_RAD_ALT)


class AltitudeQNH(DerivedParameterNode):
    """
    TODO: Complete this code when airport elevations are included in the database.
    
    This altitude Parameter is for events based upon height above sea level,
    not standard altitude or airfield elevation. For example, in the US the
    speed high below 10,000ft is based on height above sea level. Ideally use
    the BARO selection switch when recorded, else based upon the transition
    height for the departing airport in the climb and the arrival airport in
    the descent. If no such data is available, transition at 18,000 ft (USA
    standard). because there is no European standard transition height.
    """
    name = 'Altitude QNH'
    units = 'ft'
    
    
    def derive(self, alt_aal=P('Altitude AAL'), 
               land = A('FDR Landing Airport'),
               toff = A('FDR Takeoff Airport')):
        # Break the "journey" at the midpoint.
        peak = np.ma.argmax(alt_aal.array)
        alt_qnh = np.ma.copy(alt_aal.array)

        """
        # Add the elevation of the takeoff airport (above sea level) to the
        # climb portion. If this fails, make sure it's inhibited.
        try:
            alt_qnh[:peak]+=toff.value['elevation']
        except:
            alt_qnh[:peak]=np.ma.masked
        
        # Same for the downward leg of the journey.
        try:
            alt_qnh[peak:]+=land.value['elevation']
        except:
            alt_qnh[peak:]=np.ma.masked
        """
        
        self.array = alt_qnh


class AltitudeSTD(DerivedParameterNode):
    """
    This section allows for manipulation of the altitude recordings from
    different types of aircraft. Problems often arise due to combination of
    the fine and coarse parts of the data and many different types of
    correction have been developed to cater for these cases.
    """
    name = 'Altitude STD'
    units = 'ft'
    @classmethod
    def can_operate(cls, available):
        high_and_low = 'Altitude STD Coarse' in available and \
            'Altitude STD Fine' in available
        coarse_and_ivv = 'Altitude STD Coarse' in available and \
            'Rate Of Climb' in available
        return high_and_low or coarse_and_ivv
    
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
    
    def _coarse_and_ivv(self, alt_std_coarse, ivv):
        alt_std_with_lag = first_order_lag(alt_std_coarse.array, 10,
                                           alt_std_coarse.hz)
        mask = np.ma.mask_or(alt_std_with_lag.mask, ivv.array.mask)
        return np.ma.masked_array(alt_std_with_lag + (ivv.array / 60.0),
                                  mask=mask)
    
    def derive(self, alt_std_coarse=P('Altitude STD Coarse'),
               alt_std_fine=P('Altitude STD Fine'),
               ivv=P('Rate Of Climb')):
        if alt_std_high and alt_std_low:
            self.array = self._high_and_low(alt_std_coarse, alt_std_fine)
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
        elif alt_std_coarse and ivv:
            self.array = self._coarse_and_ivv(alt_std_coarse, ivv)
            #ALT_STDC = (last_alt_std * 0.9) + (ALT_STD * 0.1) + (IVVR / 60.0)


class Autopilot(DerivedParameterNode):
    """
    Placeholder for combining multi-channel AP modes into a single consistent status.
    """
    def derive(self, ap_eng=P('Autopilot Engaged'),
               frame=A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            self.array = ap_eng.array
       

class Autothrottle(DerivedParameterNode):
    """
    Placeholder for combining multi-channel AP modes into a single consistent status.
    """
    def derive(self, at_eng=P('Autothrottle Engaged'),
               frame=A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            self.array = at_eng.array
 
        
class AltitudeTail(DerivedParameterNode):
    """
    This function allows for the distance between the radio altimeter antenna
    and the point of the airframe closest to tailscrape.
   
    The parameter gear_to_tail is measured in metres and is the distance from 
    the main gear to the point on the tail most likely to scrape the runway.
    """
    units = 'ft'
    #TODO: Review availability of Attribute "Dist Gear To Tail"
    def derive(self, alt_rad=P('Altitude Radio'), pitch=P('Pitch'),
               ground_to_tail=A('Ground To Lowest Point Of Tail'),
               dist_gear_to_tail=A('Main Gear To Lowest Point Of Tail')):
        # Align the pitch attitude samples to the Radio Altimeter samples,
        # ready for combining them.
        pitch_rad = pitch.array*deg2rad
        # Now apply the offset
        gear2tail = dist_gear_to_tail.value * METRES_TO_FEET
        ground2tail = ground_to_tail.value * METRES_TO_FEET
        self.array = (alt_rad.array - np.sin(pitch_rad) * gear2tail)


class ClimbForFlightPhases(DerivedParameterNode):
    """
    This computes climb segments, and resets to zero as soon as the aircraft
    descends. Very useful for measuring climb after an aborted approach etc.
    """
    def derive(self, alt_std=P('Altitude STD'), airs=S('Fast')):
        self.array = np.ma.zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            ups = np.ma.clump_unmasked(np.ma.masked_less(deltas,0.0))
            for up in ups:
                self.array[air.slice][up] = np.ma.cumsum(deltas[up])    

            
class DescendForFlightPhases(DerivedParameterNode):
    """
    This computes descent segments, and resets to zero as soon as the aircraft
    climbs Used for measuring descents, e.g. following a suspected level bust.
    """
    def derive(self, alt_std=P('Altitude STD'), airs=S('Fast')):
        self.array = np.ma.zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            downs = np.ma.clump_unmasked(np.ma.masked_greater(deltas,0.0))
            for down in downs:
                self.array[air.slice][down] = np.ma.cumsum(deltas[down])
    
    
class ControlColumn(DerivedParameterNode):
    '''
    The position of the control column blended from the position of the captain
    and first officer's control columns.
    '''
    align_to_first_dependency = False
    def derive(self,
               posn_capt=P('Control Column (Capt)'),
               posn_fo=P('Control Column (FO)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(posn_capt, posn_fo)


class ControlColumnForceCapt(DerivedParameterNode):
    '''
    The force applied by the captain to the control column.  This is dependent
    on who has master control of the aircraft and this derived parameter
    selects the appropriate slices of data from the foreign and local forces.
    '''
    name = 'Control Column Force (Capt)'
    def derive(self,
               force_local=P('Control Column Force (Local)'),
               force_foreign=P('Control Column Force (Foreign)'),
               fcc_master=P('FCC Local Limited Master')):
        self.array = np.ma.where(fcc_master.array != 1,
                                 force_local.array,
                                 force_foreign.array)


class ControlColumnForceFO(DerivedParameterNode):
    '''
    The force applied by the first officer to the control column.  This is
    dependent on who has master control of the aircraft and this derived
    parameter selects the appropriate slices of data from the foreign and local
    forces.
    '''
    name = 'Control Column Force (FO)'
    def derive(self,
               force_local=P('Control Column Force (Local)'),
               force_foreign=P('Control Column Force (Foreign)'),
               fcc_master=P('FCC Local Limited Master')):
        self.array = np.ma.where(fcc_master.array == 1,
                                 force_local.array,
                                 force_foreign.array)


class ControlColumnForce(DerivedParameterNode):
    '''
    The combined force from the captain and the first officer.
    '''
    def derive(self,
               force_capt=P('Control Column Force (Capt)'),
               force_fo=P('Control Column Force (FO)')):
        self.array = force_capt.array + force_fo.array
        # TODO: Check this summation is correct in amplitude and phase.
        # Compare with Boeing charts for the 737NG.



class ControlWheel(DerivedParameterNode):
    '''
    The position of the control wheel blended from the position of the captain
    and first officer's control wheels.
    '''
    align_to_first_dependency = False
    def derive(self,
               posn_capt=P('Control Wheel (Capt)'),
               posn_fo=P('Control Wheel (FO)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(posn_capt, posn_fo)


class DistanceToLanding(DerivedParameterNode):
    """
    Ground distance to cover before touchdown.
    
    Note: This parameter gets closer to zero approaching the final touchdown,
    but then increases as the aircraft decelerates on the runway.
    """
    units = 'nm'
    # Q: Is this distance to final landing, or distance to each approach
    # destination (i.e. resets once reaches point of go-around)
    def derive(self, dist=P('Distance Travelled'), tdwns=KTI('Touchdown')):
        if tdwns:
            dist_flown_at_tdwn = dist.array[tdwns.get_last().index]
            self.array = np.ma.abs(dist_flown_at_tdwn - dist.array)
        else:
            self.array = np.zeros_like(dist.array)
            self.array.mask=True        


class DistanceTravelled(DerivedParameterNode):
    """
    Distance travelled in Nautical Miles. Calculated using integral of Groundspeed"
    """
    units = 'nm'
    def derive(self, gspd=P('Groundspeed')):
        self.array = integrate(gspd.array, gspd.frequency, scale=1.0/3600.0)
        

class Eng_EGTAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EGT Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
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
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) EGT'),
               eng2=P('Eng (2) EGT'),
               eng3=P('Eng (3) EGT'),
               eng4=P('Eng (4) EGT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_EGTMin(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EGT Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) EGT'),
               eng2=P('Eng (2) EGT'),
               eng3=P('Eng (3) EGT'),
               eng4=P('Eng (4) EGT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_EPRAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EPR Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_EPRMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) EPR Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_FuelFlow(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Fuel Flow"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
            
    def derive(self, 
               eng1=P('Eng (1) Fuel Flow'),
               eng2=P('Eng (2) Fuel Flow'),
               eng3=P('Eng (3) Fuel Flow'),
               eng4=P('Eng (4) Fuel Flow')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.sum(engines, axis=0)
      

class Eng_ITTAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) ITT Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) ITT'),
               eng2=P('Eng (2) ITT'),
               eng3=P('Eng (3) ITT'),
               eng4=P('Eng (4) ITT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_ITTMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) ITT Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) ITT'),
               eng2=P('Eng (2) ITT'),
               eng3=P('Eng (3) ITT'),
               eng4=P('Eng (4) ITT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_ITTMin(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) ITT Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) ITT'),
               eng2=P('Eng (2) ITT'),
               eng3=P('Eng (3) ITT'),
               eng4=P('Eng (4) ITT')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_N1Avg(DerivedParameterNode):
    name = "Eng (*) N1 Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
    
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
        return any([d in available for d in cls.get_dependency_names()])
    
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
        return any([d in available for d in cls.get_dependency_names()])
    
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
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_N2Max(DerivedParameterNode):
    name = "Eng (*) N2 Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
            
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_N2Min(DerivedParameterNode):
    name = "Eng (*) N2 Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
    
    def derive(self, 
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_N3Avg(DerivedParameterNode):
    name = "Eng (*) N3 Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_N3Max(DerivedParameterNode):
    name = "Eng (*) N3 Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_N3Min(DerivedParameterNode):
    name = "Eng (*) N3 Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
    
    def derive(self, 
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_OilTempAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Temp Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
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
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_OilTempMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Temp Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_OilPressAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Oil Press Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
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
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_OilPressMin(DerivedParameterNode):
    #TODO: TEST
    name = 'Eng (*) Oil Press Min'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)

               
class Eng_TorqueAvg(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Torque Avg"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)


class Eng_TorqueMin(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Torque Min"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)


class Eng_TorqueMax(DerivedParameterNode):
    #TODO: TEST
    name = "Eng (*) Torque Max"
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_VibN1Max(DerivedParameterNode):
    #TODO: TEST
    name = 'Eng (*) Vib N1 Max'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Vib N1'),
               eng2=P('Eng (2) Vib N1'),
               eng3=P('Eng (3) Vib N1'),
               eng4=P('Eng (4) Vib N1')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        
        
class Eng_VibN2Max(DerivedParameterNode):
    #TODO: TEST
    name = 'Eng (*) Vib N2 Max'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
        
    def derive(self, 
               eng1=P('Eng (1) Vib N2'),
               eng2=P('Eng (2) Vib N2'),
               eng3=P('Eng (3) Vib N2'),
               eng4=P('Eng (4) Vib N2')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class Eng_VibN3Max(DerivedParameterNode):
    #TODO: TEST
    name = 'Eng (*) Vib N3 Max'
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        if any([d in available for d in cls.get_dependency_names()]):
            return True
        
    def derive(self, 
               eng1=P('Eng (1) Vib N3'),
               eng2=P('Eng (2) Vib N3'),
               eng3=P('Eng (3) Vib N3'),
               eng4=P('Eng (4) Vib N3')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)


class FuelQty(DerivedParameterNode):
    '''
    May be supplanted by an LFL parameter of the same name if available.
    
    Sum of fuel in left, right and middle tanks where available.
    '''
    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any([d in available for d in cls.get_dependency_names()])
    
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


class GearDown(DerivedParameterNode):
    """
    A simple binary parameter, 0 = gear not down, 1 = gear down.
    Highly aircraft dependent, so likely to be extended.
    """
    align_to_first_dependency = False
    def derive(self, gl=P('Gear (L) Down'),
               gn=P('Gear (N) Down'),
               gr=P('Gear (R) Down'),
               frame=A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            # 737-5 has nose gear sampled alternately with mains. No obvious
            # way to accommodate mismatch of the main gear positions, so
            # assume that the right wheel does the same as the left !
            self.array, self.frequency, self.offset = merge_two_parameters(gl, gn)

class GearSelectedDown(DerivedParameterNode):
    """
    Derivation of gear selection for aircraft without this separately
    recorded. Where Gear Selected Down is recorded, this derived parameter
    will be skipped automatically.
    """
    def derive(self, gear=P('Gear Down'), frame=A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            self.array = gear.array

        
class GearSelectedUp(DerivedParameterNode):
    def derive(self, gear=P('Gear Down'), frame=A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            self.array = 1 - gear.array

        
class GrossWeightSmoothed(DerivedParameterNode):
    '''
    Gross weight is usually sampled at a low rate and can be very poor in the
    climb, often indicating an increase in weight at takeoff and this effect
    may not end until the aircraft levels in the cruise. Also some aircraft
    weight data saturates at high AUW values, and while the POLARIS Analysis
    Engine can mask this data a subsitute is needed for takeoff weight (hence
    V2) calculations. This can only be provided by extrapolation backwards
    from data available later in the flight.
    
    This routine makes the best of both worlds by using fuel flow to compute
    short term changes in weight and mapping this onto the level attitude
    data. We avoid using the recorded fuel weight in this calculation,
    however it is used in the Zero Fuel Weight calculation.
    '''
    align_to_first_dependency = False
    
    def derive(self, ff = P('Eng (*) Fuel Flow'),
               gw = P('Gross Weight'),
               climbs = S('Climbing'),
               descends = S('Descending')
               ):
        flow = repair_mask(ff.array)
        fuel_to_burn = np.ma.array(integrate (flow/3600.0, ff.frequency,  direction='reverse'))

        to_burn_valid = []
        to_burn_all = []
        gw_valid = []
        gw_all = []
        for gw_index in gw.array.nonzero()[0]:
            # Keep all the values
            gw_all.append(gw.array.data[gw_index])
            ff_time = ((gw_index/gw.frequency)+gw.offset-ff.offset)*ff.frequency
            to_burn_all.append(value_at_index(fuel_to_burn, ff_time))
            
            # Skip values which are within Climbing or Descending phases.
            if any([is_index_within_slice(gw_index, c.slice) for c in climbs]) or \
               any([is_index_within_slice(gw_index, d.slice) for d in descends]):
                continue
            gw_valid.append(gw.array.data[gw_index])
            ff_time = ((gw_index/gw.frequency)+gw.offset-ff.offset)*ff.frequency
            to_burn_valid.append(value_at_index(fuel_to_burn, ff_time))
        
        use_valid = len(gw_valid) > 5
        use_all = len(gw_all) > 2
        offset = None
        
        if use_valid or use_all:
            if use_valid:
                corr, slope, offset = coreg(np.ma.array(gw_valid), indep_var=np.ma.array(to_burn_valid))
            elif use_all:
                corr, slope, offset = coreg(np.ma.array(gw_all), indep_var=np.ma.array(to_burn_all))
            if corr < 0.5:
                offset = gw_all[0] - to_burn_all[0]
        elif len(gw_all) == 1:
            offset = gw_all[0] - to_burn_all[0]
            
        if offset == None:
            logging.warning("Cannot smooth Gross Weight. Using the original data")
            self.frequency = ff.frequency
            self.offset = ff.offset
            self.array = align(gw, ff)
        else:
            self.array = fuel_to_burn + offset


class FlapLever(DerivedParameterNode):
    """
    Steps raw Flap angle from lever into detents.
    """
    def derive(self, flap=P('Flap Lever'), series=A('Series'), family=A('Family')):
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except KeyError:
            # no flaps mapping, round to nearest 5 degrees
            logging.warning("No flap settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(flap.array, 5.0)
        else:
            self.array = step_values(flap.array, flap_steps)
        
            
'''
:TODO: Resolve the processing of different base sets of data
'''

class FlapSurface(DerivedParameterNode):
    """
    Gather the recorded flap parameters and convert into a single analogue.
    """
    align_to_first_dependency = False

    @classmethod
    def can_operate(cls, available):
        return ('Altitude AAL' in available) or \
               ('Flap (1)' in available and \
                'Flap (2)' in available)

    def derive(self, flap_A=P('Flap (1)'), flap_B=P('Flap (2)'),
               frame=A('Frame'),
               apps=S('Approach'),               
               alt_aal=P('Altitude AAL')):
        frame_name = frame.value if frame else None

        if frame_name in ['737-5']:
            self.array, self.frequency, self.offset = merge_two_parameters(flap_A, flap_B)

        if frame_name in ['L382-Hercules']:
            # Flap is not recorded, so invent one of the correct length.
            flap_herc = np.ma.array(np.zeros_like(alt_aal.array))
            if apps:
                for app in apps:
                    # The flap setting is not recorded, so we have to assume that
                    # the flap is probably set to 50% above 1000ft, and 100% from
                    # 500ft down.
                    scope = app.slice
                    flap_herc[scope] = np.ma.where(alt_aal.array[scope]>1000.0,100.0,50.0)
            self.array = np.ma.array(flap_herc)
            self.frequency, self.offset = alt_aal.frequency, alt_aal.offset
            
        if frame_name in ['146-301']:
            # Flap Surface is computed within the LFL
            pass
                   
                            
class Flap(DerivedParameterNode):
    """
    Steps raw Flap angle from surface into detents.
    """
    def derive(self, flap=P('Flap Surface'),
               series=A('Series'), family=A('Family')):
        
        """
        Steps raw Flap angle into detents.
        """
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except KeyError:
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
        except KeyError:
            # no slats mapping, round to nearest 5 degrees
            logging.warning("No slat settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(slat.array, 5.0)
        else:
            self.array = step_values(slat.array, slat_steps)
            
            
class SlopeToLanding(DerivedParameterNode):
    """
    This parameter was developed as part of the Artificical Intelligence
    analysis of approach profiles, 'Identifying Abnormalities in Aircraft
    Flight Data and Ranking their Impact on the Flight' by Dr Edward Smart,
    Institute of Industrial Research, University of Portsmouth.
    http://eprints.port.ac.uk/4141/
    """
    def derive(self, alt_aal=P('Altitude AAL'), dist=P('Distance To Landing')):
        self.array = alt_aal.array / (dist.array * FEET_PER_NM)
    
    
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
        return 'Flap' in available and \
               'Slat' in available and \
               'Series' in available and \
               'Family' in available
        
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


####class GearSelectedDown(DerivedParameterNode):
####    # And here is where the nightmare starts.
####    # Sometimes recorded
####    # Sometimes interpreted from other signals
####    # There's no pattern to how this is worked out.
####    # For aircraft with a Gear Selected Down parameter let's try this...
####    def derive(self, param=P('Gear Selected Down')):
####        return NotImplemented


class GroundspeedAlongTrack(DerivedParameterNode):
    """
    Inertial smoothing provides computation of groundspeed data when the
    recorded groundspeed is unreliable. For example, during sliding motion on
    a runway during deceleration. This is not good enough for long period
    computation, but is an improvement over aircraft where the groundspeed
    data stops at 40kn or thereabouts.
    """
    def derive(self, gndspd=P('Groundspeed'),
               at=P('Acceleration Along Track'),
               alt_aal=P('Altitude AAL'),
               glide = P('ILS Glideslope')):
        at_washout = first_order_washout(at.array, AT_WASHOUT_TC, gndspd.hz, 
                                         gain=GROUNDSPEED_LAG_TC*GRAVITY_METRIC)
        self.array = first_order_lag(gndspd.array*KTS_TO_MPS + at_washout,
                                     GROUNDSPEED_LAG_TC,gndspd.hz)
    
        
        """
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        import csv
        spam = csv.writer(open('beans.csv', 'wb'))
        spam.writerow(['at', 'gndspd', 'at_washout', 'self', 'alt_aal','glide'])
        for showme in range(0, len(at.array)):
            spam.writerow([at.array.data[showme], 
                           gndspd.array.data[showme]*KTS_TO_FPS,
                           at_washout[showme], 
                           self.array.data[showme],
                           alt_aal.array[showme],glide.array[showme]])
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        """

class HeadingContinuous(DerivedParameterNode):
    """
    For all internal computing purposes we use this parameter which does not
    jump as it passes through North. To recover the compass display, modulus
    ("%360" in Python) returns the value to display to the user.
    """
    units = 'deg'
    def derive(self, head_mag=P('Heading')):
        self.array = repair_mask(straighten_headings(head_mag.array))


class HeadingTrue(DerivedParameterNode):
    """
    Compensates for magnetic variation, which will have been computed previously.
    """
    units = 'deg'
    def derive(self, head=P('Heading Continuous'),
               var = P('Magnetic Variation')):
        self.array = head.array + var.array
        
        """
        # We copy the masked array to transfer the mask array. All the data
        # values will be overwritten, but the mask will not be affected by
        # conversion from magnetic to true headings.
        true_array = np.ma.copy(head.array)        
        start_dt = start_datetime.value
        first_liftoff = liftoffs.get_first()
        if not takeoff_airport.value or \
           not takeoff_airport.value['magnetic_variation'] or not \
           approaches.value or not first_liftoff:
            self.array.mask = True
            return
        orig_index = first_liftoff.index
        orig_mag_var = takeoff_airport.value['magnetic_variation']
        variations = []
        
        for approach in approaches.value:
            dest_index = index_of_datetime(start_dt, approach['datetime'],
                                           self.frequency)
            dest_mag_var = approach['airport'].get('magnetic_variation')
            if not dest_mag_var:
                logging.warning("Cannot calculate '%s' with a missing magnetic "
                                "variation for airport with ID '%s'.",
                                self.name, approach['airport']['id'])
                self.array.mask = True
                return
            variations.append({'slice': slice(orig_index, dest_index),
                               'orig_mag_var': orig_mag_var,
                               'dest_mag_var': dest_mag_var})
            orig_index = dest_index
            orig_mag_var = dest_mag_var
        
        start_index = 0
        for variation in variations:
            orig_mag_var = variation['orig_mag_var']
            dest_mag_var = variation['dest_mag_var']
            variation_slice = variation['slice']
            
            orig_slice = slice(start_index, variation_slice.start)
            true_array[orig_slice] = head.array[orig_slice] + orig_mag_var
            mag_var_diff = dest_mag_var - orig_mag_var
            variation_duration = variation_slice.stop - variation_slice.start
            step = mag_var_diff / variation_duration
            true_array[variation_slice] = head.array[variation_slice] + \
                np.arange(orig_mag_var, dest_mag_var, step)
            start_index = variation_slice.stop
        # Apply landing airport magnetic varation to remainder of array.
        end_slice = slice(start_index, None)
        true_array[end_slice] = true_array[end_slice] + dest_mag_var
        self.array = true_array
        """


class ILSFrequency(DerivedParameterNode):
    """
    This code is based upon the normal operation of an Instrument Landing
    System whereby the left and right receivers are tuned to the same runway
    ILS frequency. This allows independent monitoring of the approach by the
    two crew.
    
    If there is a problem with the system, users can inspect the (L) and (R)
    signals separately, although the normal use will show valid ILS data when
    both are tuned to the same frequency.
    
    """
    name = "ILS Frequency"
    align_to_first_dependency = False
    def derive(self, f1=P('ILS (L) Frequency'),f2=P('ILS (R) Frequency')):
        # Mask invalid frequencies
        f1_trim = np.ma.masked_outside(f1.array,108.10,111.95)
        f2_trim = np.ma.masked_outside(f2.array,108.10,111.95)
        # and mask where the two receivers are not matched
        self.array = np.ma.array(data = f1_trim.data,
                                 mask = np.ma.masked_not_equal(f1_trim-f2_trim,0.0).mask)
        

class ILSLocalizer(DerivedParameterNode):
    name = "ILS Localizer"
    align_to_first_dependency = False
    def derive(self, loc_1=P('ILS (L) Localizer'),loc_2=P('ILS (R) Localizer'), 
               freq=P("ILS Frequency")):
        self.array, self.frequency, self.offset = blend_two_parameters(loc_1, loc_2)
        # Would like to do this, except the frequemcies don't match
        # self.array.mask = np.ma.logical_or(self.array.mask, freq.array.mask)
               
       
class ILSGlideslope(DerivedParameterNode):
    name = "ILS Glideslope"
    align_to_first_dependency = False
    def derive(self, gs_1=P('ILS (L) Glideslope'),gs_2=P('ILS (R) Glideslope'), 
               freq=P("ILS Frequency")):
        self.array, self.frequency, self.offset = blend_two_parameters(gs_1, gs_2)
        # Would like to do this, except the frequemcies don't match
        # self.array.mask = np.ma.logical_or(self.array.mask, freq.array.mask)
       

class ILSRange(DerivedParameterNode):
    name = "ILS Range"
    """
    Range is computed from the track where available, otherwise estimated
    from available groundspeed or airspeed parameters.
    
    It is (currently) in feet from the localizer antenna.
    """
    
    ##@classmethod
    ##def can_operate(cls, available):
        ##return True
    
    def derive(self, lat=P('Latitude Prepared'),
               lon = P('Longitude Prepared'),
               glide = P('ILS Glideslope'),
               gspd = P('Groundspeed'),
               tas = P('Airspeed True'),
               alt_aal = P('Altitude AAL'),
               loc_established = S('ILS Localizer Established'),
               gs_established = S('ILS Glideslope Established'),
               precise = A('Precise Positioning'),
               app_info = A('FDR Approaches'),
               #final_apps = S('Final Approach'),
               start_datetime = A('Start Datetime')
               ):
        ils_range = np_ma_masked_zeros_like(gspd.array)
        
        for this_loc in loc_established:
            # Scan through the recorded approaches to find which matches this
            # localizer established phase.
            for approach in app_info.value:
                # line up an approach slice
                start = index_of_datetime(start_datetime.value,
                                          approach['slice_start_datetime'],
                                          self.frequency)
                stop = index_of_datetime(start_datetime.value,
                                         approach['slice_stop_datetime'],
                                         self.frequency)
                approach_slice = slice(start, stop)
                if slices_overlap(this_loc.slice, approach_slice):
                    # we've found a matching approach where the localiser was established
                    break
            else:
                logging.warning("No approach found within slice '%s'.",this_loc)
                continue

            runway = approach['runway']
            if not runway:
                logging.warning("Approach runway information not available. "
                                "No support for Airports without Runways! "
                                "Details: %s", approach)
                continue
            
            if precise.value:
                # Convert (prepared) latitude & longitude for the whole phase
                # into range from the threshold. (threshold = {})
                if 'localizer' in runway:
                    threshold = runway['localizer']
                elif 'end' in runway:
                    threshold = runway['end']
                else:
                    pass
                    # TODO: Set threshold is where the touchdown happened.
                    
                brg, ils_range[this_loc.slice] = \
                    bearings_and_distances(repair_mask(lat.array[this_loc.slice]),
                                           repair_mask(lon.array[this_loc.slice]),
                                           threshold)
                ils_range[this_loc.slice] *= METRES_TO_FEET
                continue # move onto next loc_established
                
            #-----------------------------
            #else: non-precise positioning
            
            # Use recorded groundspeed where available, otherwise estimate
            # range using true airspeed. This is because there are aircraft
            # which record ILS but not groundspeed data.
            if gspd:
                speed = np.ma.where(gspd.array.mask[this_loc.slice], \
                                    tas.array.data[this_loc.slice], \
                                    gspd.array.data[this_loc.slice]) 
            else:
                speed = tas.array.data[this_loc.slice]
                
            # Estimate range by integrating back from zero at the end of the
            # phase to high range values at the start of the phase.
            spd_repaired = repair_mask(speed)
            ils_range[this_loc.slice] = integrate(
                spd_repaired, gspd.frequency, scale=KTS_TO_FPS, direction='reverse')
            
            try:
                start_2_loc, gs_2_loc, end_2_loc, pgs_lat, pgs_lon = \
                    runway_distances(runway)  
            except KeyError:
                logging.warning("Runway did not have required information in "
                                "'%s', '%s'.", self.name, runway)
                continue
            if 'glideslope' in runway:
                # The runway has an ILS glideslope antenna
                
                for this_gs in gs_established:                    
                    if is_slice_within_slice(this_gs.slice, this_loc.slice):
                        # we'll take the first one!
                        break
                else:
                    # we didn't find a period where the glideslope was
                    # established at the same time as the localiser
                    logging.warning("No glideslope established at same time as localiser")
                    continue
                    
                # Compute best fit glidepath. The term (1-.13 x glideslope
                # deviation) caters for the aircraft deviating from the
                # planned flightpath. 1 dot low is about 0.76 deg, or 13% of
                # a 3 degree glidepath. Not precise, but adequate accuracy
                # for the small error we are correcting for here.
                corr, slope, offset = coreg(
                    alt_aal.array[this_gs.slice]* (1-0.13*glide.array[this_gs.slice]),
                    ils_range[this_gs.slice])

                # Shift the values in this approach so that the range = 0 at
                # 0ft on the projected ILS slope, then reference back to the
                # localizer antenna.                  
                datum_2_loc = gs_2_loc * METRES_TO_FEET + offset/slope
                
            else:
                # Case of an ILS approach using localizer only.
                for this_app in final_apps:
                    if is_slice_within_slice(this_app.slice, this_loc.slice):
                        # we'll take the first one!
                        break
                else:
                    # we didn't find a period where the approach was within the localiser
                    logging.warning("Approaches were not fully established with localiser")
                    continue
                    
                corr, slope, offset = coreg(
                    alt_aal.array[this_app.slice], ils_range[this_app.slice])
                
                # Touchdown point nominally 1000ft from start of runway
                datum_2_loc = (start_2_loc*METRES_TO_FEET-1000) - offset/slope
                        
                
            # Adjust all range values to relate to the localizer antenna by
            # adding the landing datum to localizer distance.
            ils_range[this_loc.slice] += datum_2_loc

        self.array = ils_range
   
    
class LatitudeSmoothed(DerivedParameterNode):
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return 'Latitude Prepared' in available
     
    units = 'deg'
    
    # Note order of longitude and latitude - data aligned to latitude here.
    def derive(self, lat = P('Latitude Prepared'),
               lon = P('Longitude Prepared'),
               loc_est = S('ILS Localizer Established'),
               ils_range = P('ILS Range'),
               ils_loc = P('ILS Localizer'),
               gspd = P('Groundspeed'),
               hdg=P('Heading True'),
               tas = P('Airspeed True'),
               precise =A('Precise Positioning'),
               toff = S('Takeoff'),
               app_info = A('FDR Approaches'),
               toff_rwy = A('FDR Takeoff Runway'),
               start_datetime = A('Start Datetime'),
               ):
        
        if ils_range:
            if len(app_info.value) != len(loc_est):
                # Q: Is this still True?
                logging.warning("Cannot Smooth latitude if the number of '%s'"
                                "Sections is not equal to the number of approaches.",
                                loc_est.name)
                self.array = lat.array
                return
            
            self.array, _ = adjust_track(lon,lat,loc_est,ils_range,ils_loc,
                                            gspd,hdg,tas,precise,toff,
                                            app_info,toff_rwy,start_datetime)

            if self.array == None:
                self.array = lat.array

        else:
            # For aircraft without ILS recorded
            # TODO: Revise this to be embedded into adjust_track
            self.array = lat.array
        
class LongitudeSmoothed(DerivedParameterNode):
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters.
        return 'Longitude Prepared' in available
    
    units = 'deg'
    
    # Note order of longitude and latitude - data aligned to longitude here.
    def derive(self, lon = P('Longitude Prepared'),
               lat = P('Latitude Prepared'),
               loc_est = S('ILS Localizer Established'),
               ils_range = P('ILS Range'),
               ils_loc = P('ILS Localizer'),
               gspd = P('Groundspeed'),
               hdg = P('Heading True'),
               tas = P('Airspeed True'),
               precise =A('Precise Positioning'),
               toff = S('Takeoff'),
               app_info = A('FDR Approaches'),
               toff_rwy = A('FDR Takeoff Runway'),
               start_datetime = A('Start Datetime'),
               ):
        if ils_range:
            if len(app_info.value) != len(loc_est) :
                # Q: Is this still True?
                logging.warning("Cannot Smooth longitude if the number of '%s'"
                                "Sections is not equal to the number of approaches.",
                                loc_est.name)
                self.array = lon.array
                return        
    
            _, self.array = adjust_track(lon,lat,loc_est,ils_range,ils_loc,
                                         gspd,hdg,tas,precise,toff,
                                         app_info,toff_rwy,start_datetime)
            
            if self.array == None:
                self.array = lon.array
            
        else:
            # For aircraft without ILS recorded
            self.array = lon.array
            
        
        
def adjust_track(lon,lat,loc_est,ils_range,ils_loc,gspd,hdg,tas,
                 precise,toff,app_info,toff_rwy,start_datetime):

    # Set up a working space.
    lat_adj = np.ma.array(data=lat.array.data, mask=True, copy=True)
    lon_adj = np.ma.array(data=lon.array.data, mask=True, copy=True)

    #-----------------------------------------------------------------------
    # Use synthesized track for takeoffs where necessary
    #-----------------------------------------------------------------------
    
    first_toff = toff.get_first()
    
    # We compute the ground track using best available data.
    if gspd:
        speed = gspd.array
        freq = gspd.frequency
    else:
        speed = tas.array
        freq = tas.frequency

    if precise.value:
        if first_toff:
            # Compute a smooth taxi out track.
            [lat_adj[:first_toff.slice.start], lon_adj[:first_toff.slice.start]] = \
                ground_track(lat.array[first_toff.slice.start],
                             lon.array[first_toff.slice.start],
                             speed[:first_toff.slice.start],
                             hdg.array[:first_toff.slice.start],
                             freq,
                             'takeoff')
        else:
            logging.warning("Cannot smooth taxi out without a takeoff.")

        # Either way, we allow the recorded track to be used for the takeoff unchanged.
        pass

    else:
        #For not precise aircraft, synthesize the takeoff roll as well as the taxi out.
        if first_toff and toff_rwy.value:
            # Compute takeoff track from start of runway using integrated
            # groundspeed, down runway centreline to end of takeoff (35ft
            # altitude). An initial value of 100m puts the aircraft at a
            # reasonable position with respect to the runway start.
            rwy_dist = np.ma.array(                        
                data = integrate(speed[first_toff.slice], freq, initial_value=100, 
                                 scale=KTS_TO_MPS),
                mask = speed.mask[first_toff.slice])
    
            # The start location has been read from the database.
            start_locn = toff_rwy.value['start']
    
            # Similarly the runway bearing is derived from the runway endpoints
            # (this gives better visualisation images than relying upon the
            # nominal runway heading). This is converted to a numpy masked array
            # of the length required to cover the takeoff phase.
            rwy_hdg = runway_heading(toff_rwy.value)
            rwy_brg = np_ma_ones_like(speed[first_toff.slice])*rwy_hdg
            
            # And finally the track down the runway centreline is
            # converted to latitude and longitude.
            lat_adj[first_toff.slice], lon_adj[first_toff.slice] = \
                latitudes_and_longitudes(rwy_brg, 
                                         rwy_dist, 
                                         start_locn)                    
    
            [lat_adj[:first_toff.slice.start],
            lon_adj[:first_toff.slice.start]] = \
                ground_track(lat_adj.data[first_toff.slice.start],
                             lon_adj.data[first_toff.slice.start],
                             speed[:first_toff.slice.start],
                             hdg.array[:first_toff.slice.start],
                             freq, 'takeoff')
        else:
            logging.warning("Cannot smooth takeoff without runway details.")

    #-----------------------------------------------------------------------
    # Use ILS track for approach and landings in all localizer approches
    #-----------------------------------------------------------------------
    
    for this_loc in loc_est:    
        # Join with ILS bearings (inherently from the localizer) and
        # revert the ILS track from range and bearing to lat & long
        # coordinates.
        # Scan through the recorded approaches to find which matches this
        # localizer established phase.
        for approach in app_info.value:
            # line up an approach slice
            start = index_of_datetime(start_datetime.value,
                                      approach['slice_start_datetime'],
                                      lon.frequency)
            stop = index_of_datetime(start_datetime.value,
                                     approach['slice_stop_datetime'],
                                     lon.frequency)
            approach_slice = slice(start, stop)
            if slices_overlap(this_loc.slice, approach_slice):
                # we've found a matching approach where the localiser was established
                break
        else:
            logging.warning("No approach found within slice '%s'.",this_loc)
            continue
        
        runway = approach['runway']
        if not runway:
            logging.error("Approach runway information not available.")
            raise NotImplementedError(
                "No support for Airports without Runways! Details: %s" % approach)    
        
        if 'localizer' in runway:
            reference = runway['localizer']
            
            if 'beam_width' in reference:
                # Compute the localizer scale factor (degrees per dot)
                # Half the beam width is 2 dots full scale
                scale = (reference['beam_width']/2.0) / 2.0
            else:
                # Normal scaling of a localizer gives 700ft width at the threshold
                scale = np.degrees(np.arctan2(700/2, runway_length(runway)*METRES_TO_FEET)) / 2.0
                
            # Adjust the ils data to be degrees from the reference point.
            bearings = ils_loc.array[this_loc.slice] * scale + \
                runway_heading(runway)+180
            
            # Adjust distance units
            distances = ils_range.array[this_loc.slice] / METRES_TO_FEET
            
            # Tweek the localizer position to be on the start:end centreline
            localizer_on_cl = ils_localizer_align(runway)
            
            # At last, the conversion of ILS localizer data to latitude and longitude
            lat_adj[this_loc.slice], lon_adj[this_loc.slice] = \
                latitudes_and_longitudes(bearings, distances, localizer_on_cl)
            
            # Alignment of the ILS Range causes corrupt first samples.
            # TODO: Fix this by propogating the ILS Range mask correctly.
            lat_adj[this_loc.slice.start] = np.ma.masked
            lon_adj[this_loc.slice.start] = np.ma.masked
            
            # Finally, tack the ground track onto the end of the landing run
            join_idx = this_loc.slice.stop - 1
            [lat_adj[join_idx:], lon_adj[join_idx:]] = \
                ground_track(lat_adj.data[join_idx], lon_adj.data[join_idx],
                             speed[join_idx:], hdg.array[join_idx:], 
                             freq, 'landing')
            
    # --- Merge Tracks and return ---
    lat_adj = track_linking(lat.array, lat_adj)
    lon_adj = track_linking(lon.array, lon_adj)

    return lat_adj, lon_adj


class Mach(DerivedParameterNode):
    def derive(self, cas = P('Airspeed'), alt = P('Altitude STD')):
        dp = cas2dp(cas.array)
        p = alt2press(alt.array)
        self.array = dp_over_p2mach(dp/p)
       
class MagneticVariation(DerivedParameterNode):
    """
    This computes local magnetic deviation values on the runways and
    interpolates between one airport and the next. The values at each airport
    are kept constant.
    
    The main idea here is that we can easily identify the ends of the runway
    and the heading on the runway, but it is far harder to find data on the
    magnetic variation at an airport. Especially in difficult locations like
    Africa or post-war zones.
    """
    def derive(self, head=P('Heading Continuous'),
               head_land = KTI('Heading At Landing'),
               head_toff = KTI('Heading At Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'), 
               land_rwy = A('FDR Landing Runway')):
        
        def first_turn(x):
            # Heading continuous can be huge after a few turns in the hold,
            # so we need this to straighten things out! I bet there's a more
            # Pythonic way to do this, but at least it's simple.
            return x - floor(x/180.0 + 0.5)*180.0        
        
        if toff_rwy.value == None and land_rwy.value == None:
            # Return an array of zeros as we know no better
            # TODO: Perhaps fill in from the database values or a computed value?
            self.array = np_ma_zeros_like(head.array)

        else:
            # Make a masked copy of the heading array, then insert deviations at
            # just the points we know. "interpolate_and_extend" is designed to
            # replace the masked values with linearly interpolated values between
            # two known points, and extrapolate to the ends of the array.
            dev = np.ma.masked_all_like(head.array)
            if head_toff:
                takeoff_heading = head_toff.get_first()
                try:
                    dev[takeoff_heading.index] = first_turn(\
                        runway_heading(toff_rwy.value) - takeoff_heading.value)
                except:
                    dev[takeoff_heading.index] = 0.0
                    
            if land_rwy.value:
                landing_heading = head_land.get_last()
                try:
                    dev[landing_heading.index] = first_turn( \
                        runway_heading(land_rwy.value) - landing_heading.value)
                except:
                    dev[landing_heading.index] = 0.0
            
            self.array = interpolate_and_extend(dev)

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
        # List the minimum required parameters.
        return 'Altitude STD' in available
    
    def derive(self, 
               az = P('Acceleration Vertical'),
               alt_std = P('Altitude STD'),
               alt_rad = P('Altitude Radio'),
               speed=P('Airspeed')):

        def inertial_rate_of_climb(alt_std_repair, frequency, alt_rad_repair, az_repair):
            # Uses the complementary smoothing approach
            
            # This is the accelerometer washout term, with considerable gain.
            # The initialisation "initial_value=az.array[clump][0]" is very
            # important, as without this the function produces huge spikes at
            # each start of a data period.
            az_washout = first_order_washout (az_repair, 
                                              AZ_WASHOUT_TC, frequency, 
                                              gain=GRAVITY_IMPERIAL,
                                              initial_value=az_repair[0])
            inertial_roc = first_order_lag (az_washout, 
                                            RATE_OF_CLIMB_LAG_TC, 
                                            frequency, 
                                            gain=RATE_OF_CLIMB_LAG_TC)
    
            # Both sources of altitude data are differentiated before
            # merging, as we mix height rate values to minimise the effect of
            # changeover of sources.
            roc_alt_std = first_order_washout(alt_std_repair,
                                              RATE_OF_CLIMB_LAG_TC, frequency,
                                              gain=1/RATE_OF_CLIMB_LAG_TC)
            roc_alt_rad = first_order_washout(alt_rad_repair,
                                              RATE_OF_CLIMB_LAG_TC, frequency,
                                              gain=1/RATE_OF_CLIMB_LAG_TC)
                    
            # Use pressure altitude rate above 100ft and radio altitude rate
            # below 50ft with progressive changeover across that range.
            # up to 50 ft radio 0 < std_rad_ratio < 1 over 100ft radio
            std_rad_ratio = np.maximum(np.minimum((alt_rad_repair-50.0)/50.0,
                                                  1),0)
            roc_altitude = roc_alt_std*std_rad_ratio +\
                roc_alt_rad*(1.0-std_rad_ratio)
            
            return (roc_altitude + inertial_roc) * 60.0

        if az and alt_rad:
            # Make space for the answers
            self.array = np.ma.masked_all_like(alt_std.array)
            
            # Fix minor dropouts
            az_repair = repair_mask(az.array)
            alt_rad_repair = repair_mask(alt_rad.array, frequency=alt_rad.frequency, repair_duration=None)
            alt_std_repair = repair_mask(alt_std.array, frequency=alt_std.frequency)
            
            # np.ma.getmaskarray ensures we have complete mask arrays even if
            # none of the samples are masked (normally returns a single
            # "False" value. We ignore the rad alt mask because we are only
            # going to use the radio altimeter values below 100ft, and short
            # transients will have been repaired. By repairing with the
            # repair_duration=None option, we ignore the masked saturated
            # values at high altitude.
            
            az_masked = np.ma.array(data = az_repair.data, 
                                    mask = np.ma.logical_or(
                                        np.ma.getmaskarray(az_repair),
                                        np.ma.getmaskarray(alt_std_repair)))
            
            # We are going to compute the answers only for ranges where all
            # the required parameters are available.
            clumps = np.ma.clump_unmasked(az_masked)
            for clump in clumps:
                self.array[clump] = inertial_rate_of_climb(
                    alt_std_repair[clump], az.frequency,
                    alt_rad_repair[clump], az_repair[clump])
            
        else:
            # The period for averaging altitude only data has been chosen
            # from careful inspection of Hercules data, where the pressure
            # altitude signal resolution is of the order of 9 ft/bit.
            # Extension to wider timebases, or averaging with more samples,
            # smooths the data more but equally more samples are affected by
            # corrupt source data. So, change the "6" only after careful
            # consideration.
            self.array = rate_of_change(alt_std,6)*60
         
         
class RateOfClimbForFlightPhases(DerivedParameterNode):
    """
    A simple and robust rate of climb parameter suitable for identifying
    flight phases. DO NOT use this for event detection.
    """
    def derive(self, alt_std = P('Altitude STD')):
        # This uses a scaled hysteresis parameter. See settings for more detail.
        threshold = HYSTERESIS_FPROC * max(1, rms_noise(alt_std.array))  
        # The max(1, prevents =0 case when testing with artificial data.
        self.array = hysteresis(rate_of_change(alt_std,6)*60,threshold)


class Relief(DerivedParameterNode):
    """
    Also known as Terrain, this is zero at the airfields. There is a small
    cliff in mid-flight where the Altitude AAL changes from one reference to
    another.
    """
    def derive(self, alt_aal = P('Altitude AAL'),
               alt_rad = P('Altitude Radio')):
        self.array = alt_aal.array - alt_rad.array


class CoordinatesStraighten(object):
    '''
    Superclass for LatitudePrepared and LongitudePrepared.
    '''
    units = 'deg'
    def _smooth_coordinates(self, coord1, coord2):
        """
        Acceleration along track only used to determine the sample rate and
        alignment of the resulting smoothed track parameter.
        
        :param coord1: Either 'Latitude' or 'Longitude' parameter.
        :type coord1: DerivedParameterNode
        :param coord2: Either 'Latitude' or 'Longitude' parameter.
        :type coord2: DerivedParameterNode
        :returns: coord1 smoothed.
        :rtype: np.ma.masked_array
        """        
        coord1_s = repair_mask(coord1.array)
        coord2_s = repair_mask(coord2.array)
        
        # Join the masks, so that we only consider positional data when both are valid:
        coord1_s.mask = np.ma.logical_or(np.ma.getmaskarray(coord1.array),
                                         np.ma.getmaskarray(coord2.array))
        coord2_s.mask = coord1_s.mask
        # Preload the output with masked values to keep dimension correct 
        array = np_ma_masked_zeros_like(coord1_s)  
        
        # Now we just smooth the valid sections.
        tracks = np.ma.clump_unmasked(coord1_s)
        for track in tracks:
            # Reject any data with invariant positions, i.e. sitting on stand.
            if np.ma.ptp(coord1_s[track])>0.0 and np.ma.ptp(coord2_s[track])>0.0:
                coord1_s_track, coord2_s_track, cost = \
                    smooth_track(coord1_s[track], coord2_s[track])
                array[track] = coord1_s_track
        return array
        

class LongitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    This removes the jumps in longitude arising from the poor resolution of
    the recorded signal.
    """
    def derive(self,
               lon=P('Longitude'),
               lat=P('Latitude')):

        self.array = self._smooth_coordinates(lon, lat)

    
class LatitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    This removes the jumps in latitude arising from the poor resolution of
    the recorded signal.
    """
    def derive(self, 
               lat=P('Latitude'), 
               lon=P('Longitude')):
        self.array = self._smooth_coordinates(lat, lon)


class RateOfTurn(DerivedParameterNode):
    """
    Simple rate of change of heading. 
    """
    def derive(self, head=P('Heading Continuous')):
        self.array = rate_of_change(head, 2)


class Pitch(DerivedParameterNode):
    """
    Combination of pitch signals from two sources where required.
    """
    units = 'deg'
    align_to_first_dependency = False
    def derive(self, p1=P('Pitch (1)'), p2=P('Pitch (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(p1, p2)


class PitchRate(DerivedParameterNode):
    """
    Computes rate of change of pitch attitude over a two second period.
    
    Comment: A two second period is used to remove excessive short period
    transients which the pilot could not realistically be asked to control.
    It also means that low sample rate data (one airline reported having
    pitch sampled at 1Hz) will still give comparable results. The drawback is
    that very brief transients, for example due to rough handling or
    turbulence, will not be detected.
    """
    def derive(self, pitch=P('Pitch')):
        self.array = rate_of_change(pitch, 2.0)


class Roll(DerivedParameterNode):
    """
    Combination of roll signals from two sources where required.
    """
    units = 'deg'
    align_to_first_dependency = False
    def derive(self, r1=P('Roll (1)'), r2=P('Roll (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(r1, r2)


class RollRate(DerivedParameterNode):
    # TODO: Tests.
    def derive(self, roll=P('Roll')):
        self.array = rate_of_change(roll, 2.0)


class ThrottleLevers(DerivedParameterNode):
    """
    A synthetic throttle lever angle, based on the average of the two. Allows
    for simple identification of changes in power etc.
    """
    def derive(self,
               tla1=P('Eng (1) Throttle Lever'), 
               tla2=P('Eng (2) Throttle Lever')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(tla1, tla2)

class ThrustReversers(DerivedParameterNode):
    """
    A single parameter with values 0=all stowed, 1=all deployed, 0.5=in transit.
    This saves subsequent algorithms having to check the various flags for each
    engine.
    """
    def derive(self, e1_left_dep=P('Eng (1) Thrust Reverser (L) Deployed'),
               e1_left_out=P('Eng (1) Thrust Reverser (L) Not Stowed'),
               e1_right_dep=P('Eng (1) Thrust Reverser (R) Deployed'),
               e1_right_out=P('Eng (1) Thrust Reverser (R) Not Stowed'),
               e2_left_dep=P('Eng (2) Thrust Reverser (L) Deployed'),
               e2_left_out=P('Eng (2) Thrust Reverser (L) Not Stowed'),
               e2_right_dep=P('Eng (2) Thrust Reverser (R) Deployed'),
               e2_right_out=P('Eng (2) Thrust Reverser (R) Not Stowed'),
               frame = A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            all_tr = e1_left_dep.array + e1_left_out.array + \
                e1_right_dep.array + e1_right_out.array + \
                e2_left_dep.array + e2_left_out.array + \
                e2_right_dep.array + e2_right_out.array
            self.array = step_values(all_tr/8.0, [0,0.5,1])
            

class Headwind(DerivedParameterNode):
    """
    This is the headwind, negative values are tailwind.
    """
    def derive(self, windspeed=P('Wind Speed'), wind_dir=P('Wind Direction'), 
               head=P('Heading True')):
        rad_scale = radians(1.0)
        self.array = windspeed.array * np.ma.cos((wind_dir.array-head.array)*rad_scale)
                

class Tailwind(DerivedParameterNode):
    """
    This is the tailwind component.
    """
    def derive(self, hwd=P('Headwind')):
        self.array = -hwd.array
        
        
class WindAcrossLandingRunway(DerivedParameterNode):
    """
    This is the windspeed across the final landing runway, positive wind from left to right.
    """
    def derive(self, windspeed=P('Wind Speed'), wind_dir=P('Wind Direction'), 
               land_rwy = A('FDR Landing Runway')):
        land_heading = runway_heading(land_rwy.value)
        if land_heading:
            self.array = windspeed.array * np.ma.sin((land_heading - wind_dir.array)*deg2rad)
        else:
            self.array = np_ma_masked_zeros_like(wind_dir.array)
                

class Aileron(DerivedParameterNode):
    '''
    Blends alternate aileron samples. Note that this technique requires both
    aileron samples to be scaled similarly and have positive sign for
    positive rolling moment. That is, port aileron down and starboard aileron
    up have positive sign.
    '''
    # TODO: TEST
    name = 'Aileron'

    @classmethod
    def can_operate(cls, available):
       a = set(['Aileron (L)', 'Aileron (R)'])
       b = set(['Aileron (L) Inboard', 'Aileron (R) Inboard', 'Aileron (L) Outboard', 'Aileron (R) Outboard'])
       x = set(available)
       return not (a - x) or not (b - x)

    def derive(self,
               al=P('Aileron (L)'),
               ar=P('Aileron (R)'),
               ali=P('Aileron (L) Inboard'),
               ari=P('Aileron (R) Inboard'),
               alo=P('Aileron (L) Outboard'),
               aro=P('Aileron (R) Outboard')):
        
        if al and ar:
            self.array, self.frequency, self.offset = blend_two_parameters(al, ar)
        else:
            return NotImplemented


class AileronTrim(DerivedParameterNode): # RollTrim
    '''
    '''
    # TODO: TEST
    name = 'Aileron Trim' # Roll Trim

    def derive(self,
               atl=P('Aileron Trim (L)'),
               atr=P('Aileron Trim (R)')):
        self.array, self.frequency, self.offset = blend_two_parameters(atl, atr)


class Elevator(DerivedParameterNode):
    '''
    Blends alternate elevator samples
    '''
    def derive(self,
               el=P('Elevator (L)'),
               er=P('Elevator (R)')):
        self.array, self.frequency, self.offset = blend_two_parameters(el, er)


class ElevatorTrim(DerivedParameterNode): # PitchTrim
    '''
    '''
    # TODO: TEST
    name = 'Elevator Trim' # Pitch Trim

    def derive(self,
               etl=P('Elevator Trim (L)'),
               etr=P('Elevator Trim (R)')):
        return NotImplemented


class Spoiler(DerivedParameterNode):
    '''
    '''
    def derive(self, spoiler_2=P('Spoiler (2)'),
               spoiler_7=P('Spoiler (7)'),
               frame = A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            '''
            We indicate the angle of either raised spoiler, ignoring sense of
            direction as it augments the roll.
            '''
            self.offset = (spoiler_2.offset + spoiler_7.offset) / 2.0
            self.array = np.ma.maximum(spoiler_2.array,spoiler_7.array)
            # Force small angles to indicate zero.
            self.array = np.ma.where(self.array < 2.0, 0.0, self.array)

class Speedbrake(DerivedParameterNode):
    '''
    '''
    align_to_first_dependency = False
    
    def derive(self, spoiler_2=P('Spoiler (2)'),
               spoiler_7=P('Spoiler (7)'),
               frame = A('Frame')):
        frame_name = frame.value if frame else None
        
        if frame_name in ['737-5']:
            '''
            For the 737-5 frame, we do not have speedbrake handle position recorded,
            so the use of the speedbrake is identified by both spoilers being
            extended at the same time.
            '''
            self.offset = (spoiler_2.offset + spoiler_7.offset) / 2.0
            self.array = np.ma.minimum(spoiler_2.array,spoiler_7.array)
            # Force small angles to indicate zero.
            self.array = np.ma.where(self.array < 2.0, 0.0, self.array)


class StickShaker(DerivedParameterNode):
    def derive(self, shake=P('STICK SHAKER-LEFT')):
        self.array, self.frequency, self.offset = \
            shake.array, shake.frequency, shake.offset
