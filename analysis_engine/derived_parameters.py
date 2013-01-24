import numpy as np
from math import ceil, floor, radians

from analysis_engine.exceptions import DataFrameError

from analysis_engine.model_information import (get_conf_map,
                                               get_flap_map,
                                               get_slat_map)
from analysis_engine.node import (
    A, DerivedParameterNode, MultistateDerivedParameterNode, KPV, KTI, M, P, S)
from analysis_engine.library import (air_track,
                                     align,
                                     all_of,
                                     alt2press,
                                     alt2sat,
                                     bearing_and_distance,
                                     bearings_and_distances,
                                     blend_two_parameters,
                                     cas2dp,
                                     cas_alt2mach,
                                     clip,
                                     coreg,
                                     cycle_finder,
                                     datetime_of_index,
                                     dp2tas,
                                     dp_over_p2mach,
                                     filter_vor_ils_frequencies,
                                     first_valid_sample,
                                     first_order_lag,
                                     first_order_washout,
                                     ground_track,
                                     ground_track_precise,
                                     hysteresis,
                                     index_at_value,
                                     integrate,
                                     ils_localizer_align,
                                     interpolate,
                                     is_day,
                                     is_index_within_slice,
                                     last_valid_sample,
                                     latitudes_and_longitudes,
                                     localizer_scale,
                                     machtat2sat,
                                     mask_inside_slices,
                                     mask_outside_slices,
                                     merge_two_parameters,
                                     moving_average,
                                     np_ma_ones_like,
                                     np_ma_masked_zeros_like,
                                     np_ma_zeros_like,
                                     offset_select,
                                     peak_curvature,
                                     rate_of_change, 
                                     repair_mask,
                                     rms_noise,
                                     round_to_nearest,
                                     runway_distances,
                                     runway_heading,
                                     runway_length,
                                     runway_snap_dict,
                                     shift_slice,
                                     slices_between,
                                     slice_multiply,
                                     slices_not,
                                     slices_or,
                                     smooth_track,
                                     step_values,
                                     straighten_headings,
                                     track_linking,
                                     vstack_params)
from analysis_engine.velocity_speed import get_vspeed_map

from settings import (AZ_WASHOUT_TC,
                      FEET_PER_NM,
                      HYSTERESIS_FPIAS,
                      HYSTERESIS_FPROC,
                      GRAVITY_IMPERIAL,
                      KTS_TO_FPS,
                      KTS_TO_MPS,
                      METRES_TO_FEET,
                      METRES_TO_NM,
                      VERTICAL_SPEED_LAG_TC)

# There is no numpy masked array function for radians, so we just multiply thus:
deg2rad = radians(1.0)

class AccelerationLateralOffsetRemoved(DerivedParameterNode):
    """
    This process attempts to remove datum errors in the lateral accelerometer.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Acceleration Lateral' in available
    
    units = 'g'
    
    def derive(self, acc=P('Acceleration Lateral'), 
               offset=KPV('Acceleration Lateral Offset')):
        if offset:
            self.array = acc.array - offset[0].value
        else:
            self.array = acc.array
    
    
class AccelerationNormalOffsetRemoved(DerivedParameterNode):
    """
    This process attempts to remove datum errors in the normal accelerometer.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Acceleration Normal' in available
    
    units = 'g'
    
    def derive(self, acc=P('Acceleration Normal'), 
               offset=KPV('Acceleration Normal Offset')):
        if offset:
            self.array = acc.array - offset[0].value + 1.0 # 1.0 to reset datum.
        else:
            self.array = acc.array
    
    
class AccelerationVertical(DerivedParameterNode):
    """
    Resolution of three accelerations to compute the vertical
    acceleration (perpendicular to the earth surface). Result is in g,
    retaining the 1.0 datum and positive upwards.
    """

    units = 'g'

    def derive(self, acc_norm=P('Acceleration Normal Offset Removed'), 
               acc_lat=P('Acceleration Lateral'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        # FIXME: FloatingPointError: underflow encountered in multiply
        pitch_rad = pitch.array*deg2rad
        roll_rad = roll.array*deg2rad
        resolved_in_roll = acc_norm.array * np.ma.cos(roll_rad)\
            - acc_lat.array * np.ma.sin(roll_rad)
        self.array = resolved_in_roll * np.ma.cos(pitch_rad) \
                     + acc_long.array * np.ma.sin(pitch_rad)
        

class AccelerationForwards(DerivedParameterNode):
    """
    Resolution of three body axis accelerations to compute the forward
    acceleration, that is, in the direction of the aircraft centreline
    when projected onto the earth's surface. 
    
    Forwards = +ve, Constant sensor errors not washed out.
    """

    units = 'g'

    def derive(self, acc_norm=P('Acceleration Normal Offset Removed'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch')):
        pitch_rad = pitch.array * deg2rad
        self.array = acc_long.array * np.ma.cos(pitch_rad)\
                     - acc_norm.array * np.ma.sin(pitch_rad)


class AccelerationAcrossTrack(DerivedParameterNode):
    """
    The forward and sideways ground-referenced accelerations are resolved
    into along track and across track coordinates in preparation for
    groundspeed computations.
    """

    units = 'g'

    def derive(self, acc_fwd=P('Acceleration Forwards'),
               acc_side=P('Acceleration Sideways'),
               drift=P('Drift')):
        drift_rad = drift.array*deg2rad
        self.array = acc_side.array * np.ma.cos(drift_rad)\
            - acc_fwd.array * np.ma.sin(drift_rad)


class AccelerationAlongTrack(DerivedParameterNode):
    """
    The forward and sideways ground-referenced accelerations are resolved
    into along track and across track coordinates in preparation for
    groundspeed computations.
    """

    units = 'g'

    def derive(self, acc_fwd=P('Acceleration Forwards'), 
               acc_side=P('Acceleration Sideways'), 
               drift=P('Drift')):
        drift_rad = drift.array*deg2rad
        self.array = acc_fwd.array * np.ma.cos(drift_rad)\
                     + acc_side.array * np.ma.sin(drift_rad)


class AccelerationSideways(DerivedParameterNode):
    """
    Resolution of three body axis accelerations to compute the lateral
    acceleration, that is, in the direction perpendicular to the aircraft
    centreline when projected onto the earth's surface. Right = +ve.
    """

    units = 'g'

    def derive(self, acc_norm=P('Acceleration Normal Offset Removed'), 
               acc_lat=P('Acceleration Lateral'),
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        pitch_rad = pitch.array * deg2rad
        roll_rad = roll.array * deg2rad
        # Simple Numpy algorithm working on masked arrays
        resolved_in_pitch = (acc_long.array * np.ma.sin(pitch_rad)
                             + acc_norm.array * np.ma.cos(pitch_rad))
        self.array = (resolved_in_pitch * np.ma.sin(roll_rad)
                      + acc_lat.array * np.ma.cos(roll_rad))


class AirspeedForFlightPhases(DerivedParameterNode):

    units = 'kts'

    def derive(self, airspeed=P('Airspeed')):
        self.array = hysteresis(
            repair_mask(airspeed.array, repair_duration=None), HYSTERESIS_FPIAS)


################################################################################
# Airspeed Minus V2 (Airspeed relative to V2 or a fixed value.)


# TODO: Write some unit tests!
# TODO: Ensure that this derived parameter supports fixed values.
class AirspeedMinusV2(DerivedParameterNode):
    '''
    Airspeed on takeoff relative to:

    - V2    -- Airbus, Boeing, or any other aircraft that has V2.
    - Fixed -- Prop aircraft, or as required.
    
    A fixed value will most likely be zero making this relative airspeed
    derived parameter the same as the original absolute airspeed parameter.
    '''

    units = 'kts'

    def derive(self, airspeed=P('Airspeed'), v2=P('V2')):
        '''
        Where V2 is recorded, a low permitted rate of change of 1.0 kt/sec
        (specified in the Parameter Operating Limit section of the POLARIS
        database) forces all false data to be masked, leaving only the
        required valid data. By repairing the mask with duration = None, the
        valid data is extended. For example, 737-3C data only records V2 on
        the runway and it needs to be extended to permit V-V2 KPVs to be
        recorded during the climbout.
        '''
        # If the data starts in mid-flight, there may be no valid V2 values.
        if np.ma.count(v2.array):
            repaired_v2 = repair_mask(v2.array, 
                                      copy=True, 
                                      repair_duration=None, 
                                      extrapolate=True)
            self.array = airspeed.array - repaired_v2
        else:
            self.array = np_ma_zeros_like(airspeed.array)

            #param.invalid = 1
            #param.array.mask = True
            #hdf.set_param(param, save_data=False, save_mask=True)
            #logger.info("Marked param '%s' as invalid as ptp %.2f "\
                        #"did not exceed minimum change %.2f",
                        #ptp, min_change)


# TODO: Write some unit tests!
class AirspeedMinusV2For3Sec(DerivedParameterNode):
    '''
    Airspeed on takeoff relative to V2 over a 3 second window.

    See the derived parameter 'Airspeed Minus V2'.
    '''

    def derive(self, spd_v2=P('Airspeed Minus V2')):
        '''
        '''
        self.array = clip(spd_v2.array, 3.0, spd_v2.frequency)
        

# TODO: Write some unit tests!
class AirspeedMinusV2For5Sec(DerivedParameterNode):
    '''
    Airspeed on takeoff relative to V2 over a 5 second window.

    See the derived parameter 'Airspeed Minus V2'.
    '''

    units = 'kts'

    def derive(self, spd_v2=P('Airspeed Minus V2')):
        '''
        '''
        self.array = clip(spd_v2.array, 5.0, spd_v2.frequency)
        

################################################################################
# Airspeed Relative (Airspeed relative to Vapp, Vref or a fixed value.)

class AirspeedReference(DerivedParameterNode):
    '''
    Airspeed on approach reference value:

    - Vapp  -- Airbus
    - Vref  -- Boeing
    - Fixed -- Prop aircraft, or as required.

    A fixed value will most likely be zero making this relative airspeed
    derived parameter the same as the original absolute airspeed parameter.
    '''

    units = 'kts'

    @classmethod
    def can_operate(cls, available):
        works_with_any = ['Vapp', 'Vref', 'AFR Vapp', 'AFR Vref']
        existing_values = any([_node in available for _node in works_with_any])
        
        x = set(available)
        base_for_lookup = ['Airspeed', 'Gross Weight Smoothed', 'Series',
                           'Family', 'Approach And Landing']
        airbus = set(base_for_lookup + ['Configuration']).issubset(x)
        boeing = set(base_for_lookup + ['Flap']).issubset(x)
        return existing_values or airbus or boeing

    def derive(self,
               spd=P('Airspeed'),
               gw=P('Gross Weight Smoothed'),
               flap=P('Flap Lever Detent'),
               conf=P('Configuration'),
               vapp=P('Vapp'),
               vref=P('Vref'),
               afr_vapp=A('AFR Vapp'),
               afr_vref=A('AFR Vref'),
               apps=S('Approach And Landing'),
               series=A('Series'),
               family=A('Family')):
        # docstring no longer accurate?
        ##'''
        ##Currently a work in progress. We should use a recorded parameter if
        ##it's available, failing that a computed forumla reflecting the
        ##aircraft condition and failing that a single value from the achieved
        ##flight file. Achieved flight records without a recorded value will be
        ##repeated thoughout the flight, calculated values will be calculated
        ##for each approach.
        ##Rises KeyError if no entires for Family/Series in vspeed lookup map.
        ##'''
        if vapp:
            # Vapp is recorded so use this
            self.array = vapp.array
        elif vref:
            # Vref is recorded so use this
            self.array = vref.array
        elif spd and (afr_vapp or afr_vref):
            # Vapp/Vref is supplied from FDR so use this
            afr_vspeed = afr_vapp or afr_vref
            self.array = np.ma.zeros(len(spd.array), np.double)
            self.array.mask = True
            for approach in apps:
                self.array[approach.slice] = afr_vspeed.value
        else:
            # elif apps and spd and gw and (flap or conf):
            # No values recorded or supplied so lookup in vspeed tables
            setting_param = flap or conf
            
            # Was:
            #self.array = np.ma.zeros(len(spd.array), np.double)
            #self.array.mask=True
            # Better:
            self.array = np_ma_masked_zeros_like(spd.array)

            vspeed_class = get_vspeed_map(series.value, family.value)

            if vspeed_class:
                vspeed_table = vspeed_class() # instansiate VelocitySpeed object
                # allow up to 2 superframe values to be repaired 
                # (64*2=128 + a bit)

                try:
                    repaired_gw = repair_mask(gw.array, repair_duration=130,
                                          copy=True, extrapolate=True)
                except:
                    repaired_gw = np_ma_masked_zeros_like(gw.array)
                    
                for approach in apps:
                    index = np.ma.argmax(setting_param.array[approach.slice])
                    weight = repaired_gw[approach.slice][index]
                    setting = setting_param.array[approach.slice][index]
                    vspeed = vspeed_table.airspeed_reference(weight, setting)
                    self.array[approach.slice] = vspeed
            else:
                # aircraft does not use vspeeds
                pass


# TODO: Write some unit tests!
# TODO: Ensure that this derived parameter supports Vapp and fixed values.
class AirspeedRelative(DerivedParameterNode):
    '''
    See AirspeedReference for details.
    '''

    units = 'kts'

    def derive(self, airspeed=P('Airspeed'), vref=P('Airspeed Reference')):
        '''
        '''
        self.array = airspeed.array - vref.array


# TODO: Write some unit tests!
class AirspeedRelativeFor3Sec(DerivedParameterNode):
    '''
    Airspeed on approach relative to Vapp/Vref over a 3 second window.

    See the derived parameter 'Airspeed Relative'.
    '''

    units = 'kts'

    def derive(self, spd_vref=P('Airspeed Relative')):
        '''
        '''
        try:
            self.array = clip(spd_vref.array, 3.0, spd_vref.frequency)
        except ValueError:
            self.warning("'%s' is completely masked within '%s'. Output array "
                         "will also be masked.", spd_vref.name, self.name)
            self.array = np_ma_zeros_like(spd_vref.array, mask=True)

        
# TODO: Write some unit tests!
class AirspeedRelativeFor5Sec(DerivedParameterNode):
    '''
    Airspeed on approach relative to Vapp/Vref over a 5 second window.

    See the derived parameter 'Airspeed Relative'.
    '''

    units = 'kts'

    def derive(self, spd_vref=P('Airspeed Relative')):
        '''
        '''
        try:
            self.array = clip(spd_vref.array, 5.0, spd_vref.frequency)
        except ValueError:
            self.warning("'%s' is completely masked within '%s'. Output array "
                         "will also be masked.", spd_vref.name, self.name)
            self.array = np_ma_zeros_like(spd_vref.array, mask=True)


################################################################################

        
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

    units = 'kts'

    @classmethod
    def can_operate(cls, available):
        return 'Airspeed' in available and 'Altitude STD' in available
    
    def derive(self, cas_p = P('Airspeed'), alt_std_p = P('Altitude STD'),
               tat_p = P('TAT'), toffs=S('Takeoff'), lands=S('Landing'), 
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
                np.logical_or(cas_p.array.mask, alt_std_p.array.mask),
                tas.mask)
        else:
            dp = cas2dp(cas)
            sat = alt2sat(alt_std)
            tas = dp2tas(dp, alt_std, sat)
            combined_mask= np.logical_or(cas_p.array.mask,alt_std_p.array.mask)
            
        tas_from_airspeed = np.ma.masked_less(
            np.ma.array(data=tas, mask=combined_mask), 50)
        tas_valids = np.ma.clump_unmasked(tas_from_airspeed)
        
        if gspd:
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
                                integrate(acc_fwd.array[scope],
                                          acc_fwd.frequency, 
                                          initial_value=tas_0,
                                          scale=GRAVITY_IMPERIAL / KTS_TO_FPS, 
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
                                integrate(acc_fwd.array[scope],
                                          acc_fwd.frequency,
                                          initial_value=tas_0, 
                                          scale=GRAVITY_IMPERIAL / KTS_TO_FPS)
                    
        self.array = tas_from_airspeed
        

class AltitudeAAL(DerivedParameterNode):
    """
    This is the main altitude measure used during flight analysis.
    
    Where radio altimeter data is available, this is used for altitudes up to
    100ft and thereafter the pressure altitude signal is used. The two are
    "joined" together at the sample above 100ft in the climb or descent as
    appropriate.
    
    If no radio altitude signal is available, the simple measure based on
    pressure altitude only is used, which provides workable solutions except
    that the point of takeoff and landing may be inaccurate.
    
    This parameter includes a rejection of bounced landings of less than 35ft
    height.
    """    
    name = "Altitude AAL"
    units = 'ft'

    @classmethod
    def can_operate(cls, available):
        return 'Altitude STD Smoothed' in available and 'Fast' in available
    
    def compute_aal(self, mode, alt_std, low_hb, high_gnd, alt_rad=None):
        
        alt_result = np_ma_zeros_like(alt_std)

        def shift_alt_std():
            '''
            Return Altitude STD Smoothed shifted relative to 0 for cases where we do not
            have a reliable Altitude Radio.
            '''
            try:
                # Test case => NAX_8_LN-NOE_20120109063858_02_L3UQAR___dev__sdb.002.hdf5
                # Look over the first 500ft of climb (or less if the data doesnt' get that high).
                to = index_at_value(alt_std, min(alt_std[0]+500, np.ma.max(alt_std)))
                # Seek the point where the altitude first curves upwards.
                idx = int(peak_curvature(alt_std, slice(None,to)))
                # The liftoff most probably arose in the preceding 20 seconds.
                rotate = slice(max(idx-20,0),idx)
                # Draw a straight line across this period with a ruler.
                p,m,c = coreg(alt_std[rotate])
                ruler = np.ma.arange(rotate.stop-rotate.start)*m+c
                # Measure how far the altitude is below the ruler.
                delta = alt_std[rotate] - ruler
                # The liftoff occurs where the gap is biggest because this is
                # where the wing lift has caused the local pressure to
                # increase, hence the altitude appears to decrease.
                pit = alt_std[np.ma.argmin(delta)]
            except:
                # If something odd about the data causes a problem with this
                # technique, use a simpler solution. This can give
                # significantly erroneous results in the case of sloping
                # runways, but it's the most robust technique.
                pit = np.ma.min(alt_std)
            alt_result = alt_std - pit
            return np.ma.maximum(alt_result, 0.0)

        if alt_rad is None:
            # This backstop trap for negative values is necessary as aircraft
            # without rad alts will indicate negative altitudes as they land.            
            return shift_alt_std()
        
        if mode != 'land':
            return alt_std - high_gnd
        
        alt_rad_aal = np.ma.maximum(alt_rad, 0.0)
        ralt_sections = np.ma.clump_unmasked(
            np.ma.masked_outside(alt_rad_aal, 0.0, 100.0))
        
        if not ralt_sections:
            # Altitude Radio did not drop below 100.
            return shift_alt_std()
    
        baro_sections = slices_not(ralt_sections, begin_at=0,
                                   end_at=len(alt_std))

        for ralt_section in ralt_sections:
            alt_result[ralt_section] = alt_rad_aal[ralt_section]
            
            for baro_section in baro_sections:
                begin_index = baro_section.start
                
                if ralt_section.stop == baro_section.start:
                    alt_diff = (alt_std[begin_index:begin_index + 60] -
                                alt_rad[begin_index:begin_index + 60])
                    slip, up_diff = first_valid_sample(alt_diff)
                    if slip is None:
                        up_diff = 0.0
                    else:
                        # alt_std is invalid at the point of handover
                        # so stretch the radio signal until we can
                        # handover.
                        fix_slice = slice(begin_index,
                                          begin_index + slip) 
                        alt_result[fix_slice] = alt_rad[fix_slice]
                        begin_index += slip
                        
                    alt_result[begin_index:] = \
                        alt_std[begin_index:] - up_diff
        return alt_result
        
    def derive(self, alt_std = P('Altitude STD Smoothed'),
               alt_rad = P('Altitude Radio'),
               speedies = S('Fast')):
        # Altitude Radio was taken as the prime reference to ensure the
        # minimum ground clearance passing peaks is accurately reflected.
        # However, when the Altitude Radio signal is sampled at a lower rate
        # than the Altitude STD Smoothed, this results in a lower sample rate
        # for a primary analysis parameter, and this is why Altitude STD
        # Smoothed is now the primary reference.
        
        # alt_aal will be zero on the airfield, so initialise to zero.
        alt_aal = np_ma_zeros_like(alt_std.array)
  
        for speedy in speedies:
            quick = speedy.slice
            if speedy.slice == slice(None, None, None):
                self.array = alt_aal
                break

            # We set the minimum height for detecting flights to the smaller
            # of 5,000 ft or 90% of the height range covered. This ensures
            # that low altitude "hops" are still treated as complete flights
            # while more complex flights are processed as climbs and descents
            # of 5000 ft or more.
            dh = min(np.ma.ptp(alt_std.array[quick])*0.9, 5000.0)
            alt_idxs, alt_vals = cycle_finder(alt_std.array[quick],
                                              min_step=dh)

            # Reference to start of arrays for simplicity hereafter.
            alt_idxs += quick.start or 0
            
            n = 0
            dips = []
            # List of dicts, with each sublist containing:
            
            # 'type' of item 'land' or 'over_gnd' or 'high'
            
            # 'slice' for this part of the data
            # if 'type' is 'land' the land section comes at the beginning of the
            # slice (i.e. takeoff slices are normal, landing slices are
            # reversed)
            # 'over_gnd' or 'air' are normal slices.

            # 'alt_std' as:
            # 'land' = the pressure altitude on the ground
            # 'over_gnd' = the pressure altitude when flying closest to the
            #              ground
            # 'air' = the lowest pressure altitude in this slice

            # 'highest_ground' in this area
            # 'land' = the pressure altitude on the ground
            # 'over_gnd' = the pressure altitude minus the radio altitude when
            #              flying closest to the ground
            # 'air' = None (the aircraft was too high for the radio altimeter to
            #         register valid data
            
            n_vals = len(alt_vals)
            while n < n_vals - 1:
                alt = alt_vals[n]
                alt_idx = alt_idxs[n]
                next_alt = alt_vals[n + 1]
                next_alt_idx = alt_idxs[n + 1]
                
                if next_alt > alt:
                    # Rising section.
                    dips.append({
                        'type': 'land',
                        'slice': slice(alt_idx, next_alt_idx),
                        'alt_std': alt,
                        'highest_ground': alt,
                    })
                    n += 1
                    continue
                
                if n + 2 >= n_vals:
                    # Falling section. Slice it backwards to use the same code
                    # as for takeoffs.
                    dips.append({
                        'type': 'land',
                        'slice': slice(next_alt_idx - 1, alt_idx - 1, -1),
                        'alt_std': next_alt,
                        'highest_ground': next_alt,
                    })
                    n += 1                    
                    continue
                
                if alt_vals[n + 2] > next_alt:
                    # A down and up section.
                    down_up = slice(alt_idx, alt_idxs[n + 2])
                    # Is radio altimeter data both supplied and valid in this
                    # range?
                    if alt_rad and np.ma.count(alt_rad.array[down_up]) > 0:
                        # Let's find the lowest rad alt reading 
                        # (this may not be exactly the highest ground, but 
                        # it was probably the point of highest concern!)
                        arg_hg_max = \
                            np.ma.argmin(alt_rad.array[down_up]) + \
                            alt_idxs[n]
                        hg_max = alt_std.array[arg_hg_max] - \
                            alt_rad.array[arg_hg_max]
                        if np.ma.count(hg_max):
                            # The rad alt measured height above a peak...
                            dips.append({
                                'type': 'over_gnd',
                                'slice': down_up,
                                'alt_std': alt_std.array[arg_hg_max],
                                'highest_ground': hg_max,
                            })
                    else:
                        # We have no rad alt data we can use.
                        # TODO: alt_std code needs careful checking.
                        if dips:
                            prev_dip = dips[-1]
                        if dips and prev_dip['type'] == 'high':
                            # Join this dip onto the previous one
                            prev_dip['slice'] = \
                                slice(prev_dip['slice'].start,
                                      alt_idxs[n + 2])
                            prev_dip['alt_std'] = \
                                min(prev_dip['alt_std'],
                                    next_alt)
                        else:
                            dips.append({
                                'type': 'high',
                                'slice': down_up,
                                'alt_std': next_alt,
                                'highest_ground': None,
                            })
                    n += 2
                else:
                    raise ValueError('Problem in Altitude AAL where data '
                                     'should dip, but instead has a peak.')

            for n, dip in enumerate(dips):
                if dip['type'] == 'high':
                    if n == 0:
                        if len(dips) == 1:
                            # Arbitrary offset in indeterminate case.
                            dip['alt_std'] = dip['highest_ground'] + 1000 
                        else:
                            next_dip = dips[n + 1]
                            dip['highest_ground'] = \
                                dip['alt_std'] - next_dip['alt_std'] + \
                                next_dip['highest_ground']
                    elif n == len(dips) - 1:
                        prev_dip = dips[n - 1]
                        dip['highest_ground'] = \
                            dip['alt_std'] - prev_dip['alt_std'] + \
                            prev_dip['highest_ground']
                    else:
                        # Here is the most commonly used, and somewhat
                        # arbitrary code. For a dip where no radio
                        # measurement of the ground is available, what height
                        # can you use as the datum? The lowest ground
                        # elevation in the preceding and following sections
                        # is practical, a little optimistic perhaps, but
                        # useable until we find a case otherwise.
                        next_dip = dips[n + 1]
                        prev_dip = dips[n - 1]
                        dip['highest_ground'] = min(prev_dip['highest_ground'],
                                                    next_dip['highest_ground'])

            for dip in dips:
                if alt_rad:
                    alt_aal[dip['slice']] = \
                        self.compute_aal(dip['type'],
                                         alt_std.array[dip['slice']],
                                         dip['alt_std'],
                                         dip['highest_ground'],
                                         alt_rad=alt_rad.array[dip['slice']])
                else:
                    alt_aal[dip['slice']] = \
                        self.compute_aal(dip['type'],
                                         alt_std.array[dip['slice']],
                                         dip['alt_std'], dip['highest_ground'])
        self.array = alt_aal


class AltitudeAALForFlightPhases(DerivedParameterNode):
    name = 'Altitude AAL For Flight Phases'
    units = 'ft'
    
    # This parameter repairs short periods of masked data, making it suitable
    # for detecting altitude bands on the climb and descent. The parameter
    # should not be used to compute KPV values themselves, to avoid using
    # interpolated values in an event.
    
    def derive(self, alt_aal=P('Altitude AAL')):
        self.array = repair_mask(alt_aal.array, repair_duration=None)
   

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

    units = 'ft'
    align = False

    @classmethod
    def can_operate(cls, available):
        return ('Altitude Radio (A)' in available and
                'Altitude Radio (B)' in available)
    
    def derive(self, frame = A('Frame'),
               frame_qual = A('Frame Qualifier'),
               source_A = P('Altitude Radio (A)'),
               source_B = P('Altitude Radio (B)'),
               source_C = P('Altitude Radio (C)'),
               source_L = P('Altitude Radio EFIS (L)'),
               source_R = P('Altitude Radio EFIS (R)')):
        
        frame_name = frame.value if frame else ''
        frame_qualifier = frame_qual.value if frame_qual else None

        # 737-1 & 737-i has Altitude Radio recorded.
        
        if frame_name in ['737-3A', '737-3B', '737-3C', '757-DHL']:
            # 737-3* comment:
            # Alternate samples (A) for this frame have latency of over 1
            # second, so do not contribute to the height measurements
            # available. For this reason we only blend the two good sensors.
            # - discovered with 737-3C and 3A/3B have same LFL for this param.
            
            # 757-DHL comment:
            # Altitude Radio (B) comes from the Right altimeter, and is
            # sampled in word 26 of the frame. Altitude Radio (C) comes from
            # the Centre altimeter, is sample in word 104. Altitude Radio (A)
            # comes from the EFIS system, and includes excessive latency so
            # is not used.
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_B, source_C)
            
        elif frame_name in ['737-4', '737-4_Analogue']:
            if frame_qualifier and 'Altitude_Radio_EFIS' in frame_qualifier:
                self.array, self.frequency, self.offset = \
                    blend_two_parameters(source_L, source_R)
            else:
                self.array, self.frequency, self.offset = \
                    blend_two_parameters(source_A, source_B)
        
        elif frame_name in ('737-5', '737-5_NON-EIS'):
            ##if frame_qualifier and 'Altitude_Radio_EFIS' in frame_qualifier or\
               ##frame_qualifier and 'Altitude_Radio_ARINC_552' in frame_qualifier:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_A, source_B)
            ##elif frame_qualifier and 'Altitude_Radio_None' in frame_qualifier:
                ##pass # Some old 737 aircraft have no rad alt recorded.
            ##else:
                ##raise ValueError,'737-5 frame Altitude Radio qualifier not recognised.'

        elif frame_name in ['CRJ-700-900', 'E135-145']:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_A, source_B)
        else:
            raise DataFrameError(self.name, frame_name)


class AltitudeSTDSmoothed(DerivedParameterNode):
    """
    :param frame: The frame attribute, e.g. '737-i'
    :type frame: An attribute
    
    :returns Altitude STD Smoothed as a local average where the original source is unacceptable, but unchanged otherwise.
    :type parameter object.
    """

    name = "Altitude STD Smoothed"
    units = 'ft'
    
    @classmethod
    def can_operate(cls, available):
        if 'Altitude STD' in available:
            return True
    
    def derive(self, alt = P('Altitude STD'), frame = A('Frame')):
        
        frame_name = frame.value if frame else ''
        
        if frame_name in ['737-i', '737-6']:
            # The altitude signal is measured in steps of 32 ft so needs
            # smoothing. A 5-point Gaussian distribution was selected as a
            # balance between smoothing effectiveness and excessive
            # manipulation of the data.
            gauss = [0.054488683, 0.244201343, 0.402619948, 0.244201343, 0.054488683]
            self.array = moving_average(alt.array, window=5, weightings=gauss)
        elif frame_name in ['E135-145']:
            # Here two sources are sampled alternately, so this form of
            # weighting merges the two to create a smoothed average.
            self.array = moving_average(alt.array, window=3, 
                                        weightings=[0.25,0.5,0.25], pad=True)
        else:
            self.array = alt.array


# TODO: Account for 'Touch & Go' - need to adjust QNH for additional airfields!
class AltitudeQNH(DerivedParameterNode):
    '''
    This altitude is above mean sea level. From the takeoff airfield to the
    highest altitude above airfield, the altitude QNH is referenced to the
    takeoff airfield elevation, and from that point onwards it is referenced
    to the landing airfield elevation.

    We can determine the elevation in the following ways:

    1. Take the average elevation between the start and end of the runway.
    2. Take the general elevation of the airfield.

    If we can only determine the takeoff elevation, the landing elevation
    will using the same value as the error will be the difference in pressure
    altitude between the takeoff and landing airports on the day which is
    likely to be less than forcing it to 0. Therefore landing elevation is
    used if the takeoff elevation cannot be determined.

    If we are unable to determine either the takeoff or landing elevations,
    we use the Altitude AAL parameter.    
    '''

    name = 'Altitude QNH'
    units = 'ft'

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL' in available and 'Altitude Peak' in available

    def derive(self, alt_aal=P('Altitude AAL'), alt_peak=KTI('Altitude Peak'),
            l_apt=A('FDR Landing Airport'), l_rwy=A('FDR Landing Runway'),
            t_apt=A('FDR Takeoff Airport'), t_rwy=A('FDR Takeoff Runway')):
        '''
        We attempt to adjust Altitude AAL by adding elevation at takeoff and
        landing. We need to know the takeoff and landing runway to get the most
        precise elevation, falling back to the airport elevation if they are
        not available.
        '''
        alt_qnh = np.ma.copy(alt_aal.array)  # copy only required for test case

        # Attempt to determine elevation at takeoff:
        t_elev = None
        if t_rwy:
            t_elev = self._calc_rwy_elev(t_rwy.value)
        if t_elev is None and t_apt:
            t_elev = self._calc_apt_elev(t_apt.value)

        # Attempt to determine elevation at landing:
        l_elev = None
        if l_rwy:
            l_elev = self._calc_rwy_elev(l_rwy.value)
        if l_elev is None and l_apt:
            l_elev = self._calc_apt_elev(l_apt.value)

        if t_elev is None and l_elev is None:
            self.warning("No Takeoff or Landing elevation, using Altitude AAL")
            self.array = alt_qnh
            return  # BAIL OUT!
        elif t_elev is None:
            self.warning("No Takeoff elevation, using %dft at Landing", l_elev)
            smooth = False
            t_elev = l_elev
        elif l_elev is None:
            self.warning("No Landing elevation, using %dft at Takeoff", t_elev)
            smooth = False
            l_elev = t_elev
        else:
            # both have valid values
            smooth = True
        
        # Break the "journey" at the "midpoint" - actually max altitude aal -
        # and be sure to account for rise/fall in the data and stick the peak
        # in the correct half:
        peak = alt_peak.get_first()  # NOTE: Fix for multiple approaches...
        fall = alt_aal.array[peak.index - 1] > alt_aal.array[peak.index + 1]
        peak = peak.index 
        if fall:
            peak += int(fall)

        # Add the elevation at takeoff to the climb portion of the array:
        alt_qnh[:peak] += t_elev

        # Add the elevation at landing to the descent portion of the array:
        alt_qnh[peak:] += l_elev

        # Attempt to smooth out any ugly transitions due to differences in
        # pressure so that we don't get horrible bumps in visualisation:
        if smooth:
            # step jump transforms into linear slope
            delta = np.ma.ptp(alt_qnh[peak - 1:peak + 1])
            width = ceil(delta * alt_aal.frequency / 3)
            window = slice(peak - width, peak + width + 1)
            alt_qnh[window] = np.ma.masked
            repair_mask(
                array=alt_qnh,
                repair_duration=window.stop - window.start,
            )

        self.array = alt_qnh

    @staticmethod
    def _calc_apt_elev(apt):
        '''
        '''
        return apt.get('elevation')

    @staticmethod
    def _calc_rwy_elev(rwy):
        '''
        '''
        elev_s = rwy.get('start', {}).get('elevation')
        elev_e = rwy.get('end', {}).get('elevation')
        if elev_s is None:
            return elev_e
        if elev_e is None:
            return elev_s
        # FIXME: Determine based on liftoff/touchdown coordinates?
        return (elev_e + elev_s) / 2


'''
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
            'Vertical Speed' in available
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
               ivv=P('Vertical Speed')):
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
            '''


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
        pitch_rad = pitch.array * deg2rad
        # Now apply the offset
        gear2tail = dist_gear_to_tail.value * METRES_TO_FEET
        ground2tail = ground_to_tail.value * METRES_TO_FEET
        # Prepare to add back in the negative rad alt reading as the aircraft
        # settles on its oleos
        min_rad = np.ma.min(alt_rad.array)
        self.array = (alt_rad.array + ground2tail - 
                      np.ma.sin(pitch_rad) * gear2tail - min_rad)

class AutopilotEngaged(MultistateDerivedParameterNode):
    '''
    This function assumes that either autopilot is capable of controlling the aircraft.
    
    This function will be extended to be multi-state parameter with states
    Off/Single/Dual as a first step towards monitoring autoland function.
    '''
    
    align=False
    
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):
        return any(d in available for d in cls.get_dependency_names())

    def derive(self, ap1=M('Autopilot (1) Engaged'), 
               ap2=M('Autopilot (2) Engaged'),
               ap3=M('Autopilot (3) Engaged')):


        if ap3 == None:
            # Only got a duplex autopilot.
            self.array = np.ma.max(ap1.array.raw, ap2.array.raw)
            self.offset = offset_select('mean', [ap1, ap2])
        else:
            self.array = np.ma.max(ap1.array.raw, ap2.array.raw, ap3.array.raw)
            self.offset = offset_select('mean', [ap1, ap2, ap3])
     
        self.frequency = ap1.frequency


'''
class Autothrottle(DerivedParameterNode):
    name = 'AT Engaged'
    """
    Placeholder for combining multi-channel AP modes into a single consistent status.

    Not required for 737-5 frame as AT Engaged is recorded directly.
    """
'''
 
        


class ClimbForFlightPhases(DerivedParameterNode):
    """
    This computes climb segments, and resets to zero as soon as the aircraft
    descends. Very useful for measuring climb after an aborted approach etc.
    """

    units = 'ft'

    def derive(self, alt_std=P('Altitude STD Smoothed'), airs=S('Fast')):
        self.array = np.ma.zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            ups = np.ma.clump_unmasked(np.ma.masked_less(deltas,0.0))
            for up in ups:
                self.array[air.slice][up] = np.ma.cumsum(deltas[up])    


class Daylight(MultistateDerivedParameterNode):
    '''
    Makes use of 64 second superframe boundaries.
    
    '''
    align = True
    align_frequency = 1/64.0
    align_offset = 0.0
    
    values_mapping = {
        0 : 'Night',
        1 : 'Day'
        }

    def derive(self, 
               latitude=P('Latitude Smoothed'),
               longitude=P('Longitude Smoothed'),
               start_datetime=A('Start Datetime'),
               duration=A('HDF Duration')):
        # Set default to 'Day'
        array_len = duration.value * self.frequency
        self.array = np.ma.ones(array_len)
        for step in xrange(0, int(array_len)):
            curr_dt = datetime_of_index(start_datetime.value, step, 1)
            lat = latitude.array[step]
            lon = longitude.array[step]
            if lat and lon:
                if not is_day(curr_dt, lat, lon):
                    # Replace values with Night
                    self.array[step] = 0
                else:
                    continue  # leave array as 1
            else:
                # either is masked or recording 0.0 which is invalid too
                self.array[step] = np.ma.masked
                
            
class DescendForFlightPhases(DerivedParameterNode):
    """
    This computes descent segments, and resets to zero as soon as the aircraft
    climbs Used for measuring descents, e.g. following a suspected level bust.
    """

    units = 'ft'

    def derive(self, alt_std=P('Altitude STD Smoothed'), airs=S('Fast')):
        self.array = np.ma.zeros(len(alt_std.array))
        repair_mask(alt_std.array) # Remove small sections of corrupt data
        for air in airs:
            deltas = np.ma.ediff1d(alt_std.array[air.slice], to_begin=0.0)
            downs = np.ma.clump_unmasked(np.ma.masked_greater(deltas,0.0))
            for down in downs:
                self.array[air.slice][down] = np.ma.cumsum(deltas[down])
    
    
class AOA(DerivedParameterNode):

    name = 'AOA'
    units = 'deg'

    def derive(self, aoa_l=P('AOA (L)'), aoa_r=P('AOA (R)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(aoa_l, aoa_r)
        

class ControlColumn(DerivedParameterNode):
    '''
    The position of the control column blended from the position of the captain
    and first officer's control columns.
    '''
    align = False
    units = 'deg'

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
    units = 'deg'

    def derive(self,
               force_local=P('Control Column Force (Local)'),
               force_foreign=P('Control Column Force (Foreign)'),
               fcc_master=M('FCC Local Limited Master')):
        
        # FCC (R) == 1 and this form is used as the numpy array comparison
        # tilts at using the MappedArray object.
        
        self.array = np.ma.where(fcc_master.array.data != 1,
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
    units = 'lbf'

    def derive(self,
               force_local=P('Control Column Force (Local)'),
               force_foreign=P('Control Column Force (Foreign)'),
               fcc_master=M('FCC Local Limited Master')):

        # FCC (R) == 1 and this form is used as the numpy array comparison
        # tilts at using the MappedArray object.

        self.array = np.ma.where(fcc_master.array.data == 1,
                                 force_local.array,
                                 force_foreign.array)


class ControlColumnForce(DerivedParameterNode):
    '''
    The combined force from the captain and the first officer.
    '''

    units = 'lbf'

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

    align = False
    units = 'deg'
    
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
            self.array.mask = True


class DistanceTravelled(DerivedParameterNode):
    '''
    Distance travelled in Nautical Miles. Calculated using integral of
    Groundspeed.
    '''

    units = 'nm'

    def derive(self, gspd=P('Groundspeed')):
        self.array = integrate(gspd.array, gspd.frequency, scale=1.0 / 3600.0)


class Drift(DerivedParameterNode):

    units = 'deg'

    def derive(self, drift_1=P('Drift (1)'), drift_2=P('Drift (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(drift_1, drift_2)
        


################################################################################
# Pack Valves

class BrakePressure(DerivedParameterNode):
    """
    Gather the recorded brake parameters and convert into a single analogue.
    
    This node allows for expansion for different types, and possibly
    operation in primary and standby modes.
    """
    align_to_first_dependency = False

    @classmethod
    def can_operate(cls, available):
        return ('Brake (L) Press' in available and \
                'Brake (R) Press' in available)

    def derive(self, brake_L=P('Brake (L) Press'), brake_R=P('Brake (R) Press'),
               frame=A('Frame')):
        frame_name = frame.value if frame else ''

        if frame_name.startswith('737-'):
            self.array, self.frequency, self.offset = blend_two_parameters(brake_L, brake_R)
            
        else:
            raise DataFrameError(self.name, frame_name)

################################################################################
# Pack Valves


class PackValvesOpen(MultistateDerivedParameterNode):
    '''
    Integer representation of the combined pack configuration.
    '''

    name = 'Pack Valves Open'

    align = False

    values_mapping = {
        0: 'All closed',
        1: 'One engine low flow',
        2: 'Flow level 2',
        3: 'Flow level 3',
        4: 'Both engines high flow',
    }

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with both 'ECS Pack (1) On' and 'ECS Pack (2) On' ECS Pack High Flows are optional
        return all_of(['ECS Pack (1) On', 'ECS Pack (2) On' ], available)

    def derive(self,
            p1=P('ECS Pack (1) On'), p1h=P('ECS Pack (1) High Flow'),
            p2=P('ECS Pack (2) On'), p2h=P('ECS Pack (2) High Flow')):
        '''
        '''
        # TODO: account properly for states/frame speciffic fixes
        # Sum the open engines, allowing 1 for low flow and 1+1 for high flow
        # each side.
        flow = p1.array.raw + p2.array.raw
        if p1h and p2h:
            flow = p1.array.raw * (1 + p1h.array.raw) \
                 + p2.array.raw * (1 + p2h.array.raw)
        self.array = flow
        self.offset = offset_select('mean', [p1, p1h, p2, p2h])


################################################################################
# Engine EPR


# TODO: Write some unit tests!
class Eng_EPRAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Avg'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_EPRMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Max'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_EPRMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) EPR Min'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) EPR'),
               eng2=P('Eng (2) EPR'),
               eng3=P('Eng (3) EPR'),
               eng4=P('Eng (4) EPR')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine Fuel Flow


# TODO: Write some unit tests!
class Eng_FuelFlow(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Fuel Flow'
    units = 'lbs/h'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Fuel Flow'),
               eng2=P('Eng (2) Fuel Flow'),
               eng3=P('Eng (3) Fuel Flow'),
               eng4=P('Eng (4) Fuel Flow')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.sum(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine Gas Temperature


# TODO: Write some unit tests!
class Eng_GasTempAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Avg'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_GasTempMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Max'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_GasTempMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Gas Temp Min'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Gas Temp'),
               eng2=P('Eng (2) Gas Temp'),
               eng3=P('Eng (3) Gas Temp'),
               eng4=P('Eng (4) Gas Temp')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine N1


class Eng_N1Avg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N1 Avg'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_N1Max(DerivedParameterNode):
    '''
    This returns the highest N1 in any sample period for up to four engines.
    The offset is returned as the mean of the operating samples, to represent
    the overall period irrespective of the timing of the samples.
    '''

    name = 'Eng (*) N1 Max'
    units = '%'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        

class Eng_N1Min(DerivedParameterNode):
    '''
    This returns the lowest N1 in any sample period for up to four engines.
    The offset is returned as the mean of the operating samples, to represent
    the overall period irrespective of the timing of the samples.
    '''

    name = 'Eng (*) N1 Min'
    units = '%'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine N2


class Eng_N2Avg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N2 Avg'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_N2Max(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N2 Max'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_N2Min(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N2 Min'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N2'),
               eng2=P('Eng (2) N2'),
               eng3=P('Eng (3) N2'),
               eng4=P('Eng (4) N2')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine N3


class Eng_N3Avg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N3 Avg'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_N3Max(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N3 Max'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


class Eng_N3Min(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) N3 Min'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) N3'),
               eng2=P('Eng (2) N3'),
               eng3=P('Eng (3) N3'),
               eng4=P('Eng (4) N3')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine Oil Pressure


# TODO: Write some unit tests!
class Eng_OilPressAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Avg'
    units = 'psi'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_OilPressMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Max'
    units = 'psi'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_OilPressMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Press Min'
    units = 'psi'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Press'),
               eng2=P('Eng (2) Oil Press'),
               eng3=P('Eng (3) Oil Press'),
               eng4=P('Eng (4) Oil Press')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])

################################################################################
# Engine Oil Quantity


# TODO: Write some unit tests!
class Eng_OilQtyAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Avg'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_OilQtyMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Max'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_OilQtyMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Qty Min'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Qty'),
               eng2=P('Eng (2) Oil Qty'),
               eng3=P('Eng (3) Oil Qty'),
               eng4=P('Eng (4) Oil Qty')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])

################################################################################
# Engine Oil Temperature


# TODO: Write some unit tests!
class Eng_OilTempAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Avg'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        avg_array = np.ma.average(engines, axis=0)

        if np.ma.count(avg_array) != 0:
            self.array = avg_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(avg_array)


# TODO: Write some unit tests!
class Eng_OilTempMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Max'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        engines = vstack_params(eng1, eng2, eng3, eng4)
        max_array = np.ma.max(engines, axis=0)
        
        if np.ma.count(max_array) != 0:
            self.array = max_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(max_array)


# TODO: Write some unit tests!
class Eng_OilTempMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Oil Temp Min'
    units = 'C'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Oil Temp'),
               eng2=P('Eng (2) Oil Temp'),
               eng3=P('Eng (3) Oil Temp'),
               eng4=P('Eng (4) Oil Temp')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        min_array = np.ma.min(engines, axis=0)
        
        if np.ma.count(min_array) != 0:
            self.array = min_array
            self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])
        else:
            # Some aircraft have no oil temperature sensors installed, so
            # quit now if there is no valid result.
            self.array = np_ma_masked_zeros_like(min_array)



################################################################################
# Engine Torque


# TODO: Write some unit tests!
class Eng_TorqueAvg(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Avg'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.average(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_TorqueMax(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Max'
    units = '%'
  
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


# TODO: Write some unit tests!
class Eng_TorqueMin(DerivedParameterNode):
    '''
    '''

    name = 'Eng (*) Torque Min'
    units = '%'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Torque'),
               eng2=P('Eng (2) Torque'),
               eng3=P('Eng (3) Torque'),
               eng4=P('Eng (4) Torque')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.min(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])


################################################################################
# Engine Vibration (N1)


# TODO: Write some unit tests!
class Eng_VibN1Max(DerivedParameterNode):
    '''
    This derived parameter condenses all the available first shaft order
    vibration measurements into a single consolidated value.
    '''

    name = 'Eng (*) Vib N1 Max'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Vib N1'),
               eng2=P('Eng (2) Vib N1'),
               eng3=P('Eng (3) Vib N1'),
               eng4=P('Eng (4) Vib N1'),
               fan1=P('Eng (1) Vib N1 Fan'),
               fan2=P('Eng (2) Vib N1 Fan'),
               lpt1=P('Eng (1) Vib N1 Turbine'),
               lpt2=P('Eng (2) Vib N1 Turbine')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4, fan1, fan2, lpt1, lpt2)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4, fan1, fan2, lpt1, lpt2])


################################################################################
# Engine Vibration (N2)


# TODO: Write some unit tests!
class Eng_VibN2Max(DerivedParameterNode):
    '''
    This derived parameter condenses all the available second shaft order
    vibration measurements into a single consolidated value.
    '''

    name = 'Eng (*) Vib N2 Max'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Vib N2'),
               eng2=P('Eng (2) Vib N2'),
               eng3=P('Eng (3) Vib N2'),
               eng4=P('Eng (4) Vib N2'),
               hpc1=P('Eng (1) Vib N2 Compressor'),
               hpc2=P('Eng (2) Vib N2 Compressor'),
               hpt1=P('Eng (1) Vib N2 Turbine'),
               hpt2=P('Eng (2) Vib N2 Turbine')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4, hpc1, hpc2, hpt1, hpt2)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4, hpc1, hpc2, hpt1, hpt2])


################################################################################
# Engine Vibration (N3)


# TODO: Write some unit tests!
class Eng_VibN3Max(DerivedParameterNode):
    '''
    This derived parameter condenses all the available third shaft order
    vibration measurements into a single consolidated value.
    '''

    name = 'Eng (*) Vib N3 Max'

    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with any combination of parameters available:
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               eng1=P('Eng (1) Vib N3'),
               eng2=P('Eng (2) Vib N3'),
               eng3=P('Eng (3) Vib N3'),
               eng4=P('Eng (4) Vib N3')):
        '''
        '''
        engines = vstack_params(eng1, eng2, eng3, eng4)
        self.array = np.ma.max(engines, axis=0)
        self.offset = offset_select('mean', [eng1, eng2, eng3, eng4])

################################################################################


class FuelQty(DerivedParameterNode):
    '''
    May be supplanted by an LFL parameter of the same name if available.
    
    Sum of fuel in left, right and middle tanks where available.
    '''

    align = False

    @classmethod
    def can_operate(cls, available):
        # works with any combination of params available
        return any(d in available for d in cls.get_dependency_names())
    
    def derive(self, 
               fuel_qty1=P('Fuel Qty (1)'),
               fuel_qty2=P('Fuel Qty (2)'),
               fuel_qty3=P('Fuel Qty (3)'),
               fuel_qty_aux=P('Fuel Qty (Aux)')):
        params = []
        for param in (fuel_qty1, fuel_qty2, fuel_qty3, fuel_qty_aux):
            if not param:
                continue
            # Repair array masks to ensure that the summed values are not too small
            # because they do not include masked values.
            try:
                param.array = repair_mask(param.array)
            except ValueError as err:
                # Q: Should we be creating a summed Fuel Qty parameter when
                # omitting a masked parameter? The resulting array will contain
                # values lower than expected. The same problem will occur if
                # a parameter has been marked invalid, though we will not
                # be aware of the problem within a derive method.
                self.warning('Skipping %s while calculating %s: %s. Summed '
                             'fuel quantity may be lower than expected.',
                             param, self, err)
            else:
                params.append(param)
        
        stacked_params = vstack_params(*params)
        self.array = np.ma.sum(stacked_params, axis=0)
        self.offset = offset_select('mean', params)


################################################################################
# Landing Gear


class GearDown(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is up or down.
    '''

    align = False
    values_mapping = {
        0: 'Up',
        1: 'Down',
    }
    
    @classmethod
    def can_operate(cls, available):
        '''
        '''
        return any(d in available for d in cls.get_dependency_names())
    
    def derive(self,
               gl=M('Gear (L) Down'),
               gn=M('Gear (N) Down'),
               gr=M('Gear (R) Down')):
        # Join all available gear parameters and use whichever are available.
        v = vstack_params(gl, gn, gr)
        wheels_down = v.sum(axis=0) >= (v.shape[0] / 2.0)
        self.array = np.ma.where(wheels_down, self.state['Down'], self.state['Up'])


class GearOnGround(MultistateDerivedParameterNode):
    '''
    Combination of left and right main gear signals.
    '''

    align = False
    values_mapping = {
        0: 'Air',
        1: 'Ground',
    }
    
    @classmethod
    def can_operate(cls, available):
        '''
        '''
        return any(d in available for d in cls.get_dependency_names())


    def derive(self,
            gl=M('Gear (L) On Ground'),
            gr=M('Gear (R) On Ground')):
        '''
        Note that this is not needed on the following frames which record this
        parameter directly: 737-4, 737-i
        '''
        ##frame_name = frame.value if frame else ''
        ##if frame_name.startswith('737-'):
        
        if gl and gr:
            if gl.offset == gr.offset:
                # A common case is for the left and right gear to be mapped
                # onto different bits of the same word. In this case we
                # accept that either wheel on the ground equates to gear on
                # ground.
                self.array = np.ma.logical_or(gl.array, gr.array)
                self.frequency = gl.frequency
                self.offset = gl.offset
            else:
                # If the paramters are not co-located, then
                # merge_two_parameters creates the best combination possible.
                self.array, self.frequency, self.offset = merge_two_parameters(gl, gr)
        elif gl:
            self.array, self.frequency, self.offset = gl.array, gl.frequency, gl.offset
        else: # gr
            self.array, self.frequency, self.offset = gr.array, gr.frequency, gr.offset
        


class GearDownSelected(MultistateDerivedParameterNode):
    '''
    Derivation of gear selection for aircraft without this separately recorded.
    Where 'Gear Down Selected' is recorded, this derived parameter will be
    skipped automatically.
    
    Red warnings are included as the selection may first be indicated by one
    of the red warning lights coming on, rather than the gear status
    changing.
    '''

    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available

    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    def derive(self, gear_down=M('Gear Down'),
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning')):

        dn = gear_down.array.raw
        
        if gear_warn_l and gear_warn_n and gear_warn_r:
            # Join all available gear parameters and use whichever are available.
            v = vstack_params(dn, 
                              gear_warn_l.array.raw, 
                              gear_warn_n.array.raw, 
                              gear_warn_r.array.raw)
            wheels_dn = v.sum(axis=0) > 0
            self.array = np.ma.where(wheels_dn, self.state['Down'], self.state['Up'])
        else:
            self.array = dn
        self.frequency = gear_down.frequency
        self.offset = gear_down.offset
        


class GearUpSelected(MultistateDerivedParameterNode):
    '''
    Derivation of gear selection for aircraft without this separately recorded.
    Where 'Gear Up Selected' is recorded, this derived parameter will be
    skipped automatically.
    
    Red warnings are included as the selection may first be indicated by one
    of the red warning lights coming on, rather than the gear status
    changing.
    '''
    
    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available

    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    def derive(self, gear_down=M('Gear Down'),
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning')):

        up = 1 - gear_down.array.raw
        
        if gear_warn_l and gear_warn_n and gear_warn_r:
            # Join all available gear parameters and use whichever are available.
            v = vstack_params(up, 
                              gear_warn_l.array.raw, 
                              gear_warn_n.array.raw, 
                              gear_warn_r.array.raw)
            wheels_up = v.sum(axis=0) > 0
            self.array = np.ma.where(wheels_up, self.state['Up'], self.state['Down'])
        else:
            self.array = up
        self.frequency = gear_down.frequency
        self.offset = gear_down.offset
        


################################################################################


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
    units = 'kgs'
    
    def derive(self, ff=P('Eng (*) Fuel Flow'),
               gw=P('Gross Weight'),
               climbs=S('Climbing'),
               descends=S('Descending'),
               fast=S('Fast')):
        
        gw_masked = gw.array.copy()
        gw_masked = mask_inside_slices(gw.array, climbs.get_slices())
        gw_masked = mask_inside_slices(gw.array, descends.get_slices())
        gw_masked = mask_outside_slices(gw.array, fast.get_slices())
        
        gw_nonzero = gw.array.nonzero()[0]
        
        try:
            gw_valid_index = gw_masked.nonzero()[0][-1]
        except IndexError:
            self.warning(
                "'%s' had no valid samples within '%s' section, but outside "
                "of '%s' and '%s'. Reverting to '%s'.", self.name, fast.name,
                climbs.name, descends.name, gw.name)
            self.array = gw.array
            return
        
        flow = repair_mask(ff.array)
        fuel_to_burn = np.ma.array(integrate(flow / 3600.0, ff.frequency,
                                             direction='reverse'))        
        
        offset = gw_masked[gw_valid_index] - fuel_to_burn[gw_valid_index]
        
        self.array = fuel_to_burn + offset
        
        # Test that the resulting array is sensible compared with Gross Weight.
        test_index = len(gw_nonzero) / 2
        test_difference = \
            abs(gw.array[test_index] - self.array[test_index]) > 1000
        if test_difference > 1000: # Q: Is 1000 too large?
            raise ValueError(
                "'%s' difference from '%s' at half-way point is greater than "
                "'%s': '%s'." % self.name, gw.name, 1000, test_difference)


class Groundspeed(DerivedParameterNode):
    """
    This caters for cases where some preprocessing is required.
    :param frame: The frame attribute, e.g. '737-i'
    :type frame: An attribute
    :returns groundspeed as the mean between two valid sensors.
    :type parameter object.
    """
    units = 'kts'
    align = False
    
    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude STD','Groundspeed (1)','Groundspeed (2)'),
                      available)
        
    def derive(self,
               alt = P('Altitude STD'),
               source_A = P('Groundspeed (1)'),
               source_B = P('Groundspeed (2)'),
               frame = A('Frame')):
        
        frame_name = frame.value if frame else ''
        
        if frame_name in ['757-DHL']:
            # The coding in this frame is unique as it only uses two bits for
            # the first digit of the BCD-encoded groundspeed, limiting the
            # recorded value range to 399 kts. At altitude the aircraft can
            # exceed this so a fiddle is required to sort this out.
            altitude = align(alt, source_A) # Caters for different sample rates.
            adjust_A = np.logical_and(source_A.array<200, altitude>8000).data*400
            source_A.array += adjust_A
            adjust_B = np.logical_and(source_B.array<200, altitude>8000).data*400
            source_B.array += adjust_B
            self.array, self.frequency, self.offset = \
                blend_two_parameters(source_A, source_B)
            
        else:
            raise DataFrameError(self.name, frame_name)


class FlapLeverDetent(DerivedParameterNode):
    """
    Steps raw Flap angle from lever into detents.
    """
    units = 'deg'

    def derive(self, flap=P('Flap Lever'), series=A('Series'), family=A('Family')):
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except KeyError:
            # no flaps mapping, round to nearest 5 degrees
            self.warning("No flap settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(flap.array, 5.0)
        else:
            self.array = step_values(flap.array, flap_steps)


class FlapSurface(DerivedParameterNode):
    """
    Gather the recorded flap parameters and convert into a single analogue.
    """
    align = False
    units = 'deg'

    @classmethod
    def can_operate(cls, available):
        return ('Altitude AAL' in available) and \
               ('Flap (L)' in available or \
                'Flap (R)' in available)

    def derive(self, flap_A=P('Flap (L)'), flap_B=P('Flap (R)'),
               frame=A('Frame'),
               apps=S('Approach'),               
               alt_aal=P('Altitude AAL')):
        frame_name = frame.value if frame else ''

        if frame_name.startswith('737-') or frame_name in ['757-DHL']:
            self.array, self.frequency, self.offset = blend_two_parameters(flap_A,
                                                                           flap_B)

        elif frame_name in ['L382-Hercules']:
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

        else:
            raise DataFrameError(self.name, frame_name)
                   
                            
class Flap(DerivedParameterNode):
    """
    Steps raw Flap angle from surface into detents.
    """

    units = 'deg'

    @classmethod
    def can_operate(cls, available):
        return all_of(('Flap Surface', 'Series', 'Family'), available)
    
    def derive(self, flap=P('Flap Surface'),
               series=A('Series'), family=A('Family')):
        try:
            flap_steps = get_flap_map(series.value, family.value) 
        except KeyError:
            # no flaps mapping, round to nearest 5 degrees
            self.warning("No flap settings - rounding to nearest 5")
            # round to nearest 5 degrees
            self.array = round_to_nearest(flap.array, 5.0)
        else:
            self.array = step_values(flap.array, flap_steps)
        
  
'''
class SlatSurface(DerivedParameterNode):
    """
    """
    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),

    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),
    
    s1f = M('Slat (1) Fully Extended'),
    s1t = M('Slat (1) In Transit'),
    s1m = M('Slat (1) Mid Extended'),
'''

class Slat(DerivedParameterNode):
    """
    Steps raw Slat angle into detents.
    """
    def derive(self, slat=P('Slat Surface'), series=A('Series'), family=A('Family')):
        try:
            slat_steps = get_slat_map(series.value, family.value) 
        except KeyError:
            # no slats mapping, round to nearest 5 degrees
            self.warning("No slat settings - rounding to nearest 5")
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
    
    
class Configuration(DerivedParameterNode):
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
    actual configuration state of the aircraft rather than the intended state
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
        
        mapping = get_conf_map(series.value, family.value)        
        qty_param = len(mapping.itervalues().next())
        if qty_param == 3 and not aileron:
            # potential problem here!
            self.warning("Aileron not available, so will calculate Configuration using only slat and flap")
            qty_param = 2
        elif qty_param == 2 and aileron:
            # only two items in values tuple
            self.debug("Aileron available but not required for Configuration calculation")
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


'''

TODO: Revise computation of sliding motion

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
'''

class HeadingContinuous(DerivedParameterNode):
    """
    For all internal computing purposes we use this parameter which does not
    jump as it passes through North. To recover the compass display, modulus
    (val % 360 in Python) returns the value to display to the user.
    """
    units = 'deg'
    def derive(self, head_mag=P('Heading')):
        self.array = repair_mask(straighten_headings(head_mag.array))


# TODO: Absorb this derived parameter into the 'Holding' flight phase.
class HeadingIncreasing(DerivedParameterNode):
    """
    This parameter is computed to allow holding patterns to be identified. As
    the aircraft can enter a hold turning in one direction, then do a
    teardrop and continue with turns in the opposite direction, we are
    interested in the total angular changes, not the sign of these changes.    
    """
    units = 'deg'
    def derive(self, head=P('Heading Continuous')):
        rot = np.ma.ediff1d(head.array, to_begin = 0.0)
        self.array = integrate(np.ma.abs(rot), head.frequency)


class HeadingTrueContinuous(DerivedParameterNode):
    units = 'deg'
    def derive(self, hdg=P('Heading True')):
        self.array = straighten_headings(hdg.array)


class HeadingTrue(DerivedParameterNode):
    """
    Compensates for magnetic variation, which will have been computed previously.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Heading Continuous' in available
    
    units = 'deg'
    def derive(self, head=P('Heading Continuous'),
               var = P('Magnetic Variation')):
        if var:
            self.array = (head.array + var.array)%360.0
        else:
            # Default to magnetic if we know no better.
            self.array = head.array


class ILSFrequency(DerivedParameterNode):
    """
    This code is based upon the normal operation of an Instrument Landing
    System whereby the left and right receivers are tuned to the same runway
    ILS frequency. This allows independent monitoring of the approach by the
    two crew.
    
    If there is a problem with the system, users can inspect the (1) and (2)
    signals separately, although the normal use will show valid ILS data when
    both are tuned to the same frequency.
    """
    
    name = "ILS Frequency"
    units='MHz'
    align = False
    
    @classmethod
    def can_operate(cls, available):
        return ('ILS (1) Frequency' in available and 
                'ILS (2) Frequency' in available) or \
               ('ILS-VOR (1) Frequency' in available and
                'ILS-VOR (2) Frequency' in available)    

    def derive(self, f1=P('ILS (1) Frequency'),f2=P('ILS (2) Frequency'),
               f1v=P('ILS-VOR (1) Frequency'), f2v=P('ILS-VOR (2) Frequency')):
               ##frame = A('Frame')):

        ##frame_name = frame.value if frame else ''
        
        # On some frames only one ILS frequency recording works
        if False:
            pass
        ##if frame_name in ['737-6'] and \
           ##(np.ma.count(f2.array) == 0 or np.ma.ptp(f2.array) == 0.0):
            ##self.array = f1.array
            
        # In all cases other than those identified above we look for both
        # receivers being tuned together to form a valid signal
        else:
            if f1 and f2:
                first = f1.array
                second = f2.array
            else:
                first = f1v.array
                second = f2v.array
                
            # Mask invalid frequencies
            f1_trim = filter_vor_ils_frequencies(first, 'ILS')
            f2_trim = filter_vor_ils_frequencies(second, 'ILS')

            # and mask where the two receivers are not matched
            mask = np.ma.masked_not_equal(f1_trim - f2_trim, 0.0).mask
            self.array = np.ma.array(data=f1_trim.data, mask=mask)


class ILSLocalizer(DerivedParameterNode):
    
    name = "ILS Localizer"
    units = 'dots'
    align = False

    def derive(self, loc_1=P('ILS (1) Localizer'),loc_2=P('ILS (2) Localizer')):
        self.array, self.frequency, self.offset = blend_two_parameters(loc_1, loc_2)
               
       
class ILSGlideslope(DerivedParameterNode):

    name = "ILS Glideslope"
    units = 'dots'
    align = False

    def derive(self, gs_1=P('ILS (1) Glideslope'),gs_2=P('ILS (2) Glideslope')):
        self.array, self.frequency, self.offset = blend_two_parameters(gs_1, gs_2)
        # Would like to do this, except the frequencies don't match
        # self.array.mask = np.ma.logical_or(self.array.mask, freq.array.mask)
       

class AimingPointRange(DerivedParameterNode):
    """
    Aiming Point Range is derived from the Approach Range. The units are
    converted to nautical miles ready for plotting and the datum is offset to
    either the ILS Glideslope Antenna position where an ILS is installed or
    the nominal threshold position where there is no ILS installation.
    """

    unit = 'nm'

    def derive(self, app_rng=P('Approach Range'),
               approaches = A('FDR Approaches'),
               ):
        self.array = np_ma_masked_zeros_like(app_rng.array)
        
        for approach in approaches.value:
            runway = approach.get('runway')
            if not runway:
                # no runway to establish distance to glideslope antenna
                continue
            try:
                extend = runway_distances(runway)[1] # gs_2_loc
            except (KeyError, TypeError):
                extend = runway_length(runway) - 1000 / METRES_TO_FEET
                
            s = slice(approach['slice_start'], approach['slice_stop'])
            self.array[s] = (app_rng.array[s] - extend) / METRES_TO_NM 


class CoordinatesSmoothed(object):
    '''
    Superclass for SmoothedLatitude and SmoothedLongitude classes as they share
    the adjust_track methods.
    
    _adjust_track_pp is used for aircraft with precise positioning, usually
    GPS based and qualitatively determined by a recorded track that puts the
    aircraft on the correct runway. In these cases we only apply fine
    adjustment of the approach and landing path using ILS localizer data to
    position the aircraft with respect to the runway centreline.
    
    _adjust_track_ip is for aircraft with imprecise positioning. In these
    cases we use all the data available to correct for errors in the recorded
    position at takeoff, approach and landing.
    '''
    def taxi_out_track_pp(self, lat, lon, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi out track.
        '''

        lat_out, lon_out, wt = ground_track_precise(lat, lon, speed, hdg,
                                                    freq, 'takeoff')
        return lat_out, lon_out

    def taxi_in_track_pp(self, lat, lon, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi in track.
        '''
        lat_in, lon_in, wt = ground_track_precise(lat, lon, speed, hdg, freq, 
                                              'landing')
        return lat_in, lon_in
        
    def taxi_out_track(self, toff_slice, lat_adj, lon_adj, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi out track.
        TODO: Include lat & lon corrections for precise positioning tracks.
        '''
        lat_out, lon_out = \
            ground_track(lat_adj[toff_slice.start],
                         lon_adj[toff_slice.start],
                         speed[:toff_slice.start],
                         hdg.array[:toff_slice.start],
                         freq, 
                         'takeoff')
        return lat_out, lon_out
    
    def taxi_in_track(self, lat_adj, lon_adj, speed, hdg, freq):
        '''
        Compute a groundspeed and heading based taxi in track.
        '''        
        lat_in, lon_in = ground_track(lat_adj[0],
                                      lon_adj[0],
                                      speed,
                                      hdg,
                                      freq,
                                      'landing')
        return lat_in, lon_in
    
    def _adjust_track(self, lon, lat, ils_loc, app_range, hdg, gspd, tas, 
            toff, toff_rwy, approaches, mobile, precise):
        
        # Set up a working space.
        lat_adj = np_ma_masked_zeros_like(hdg.array)
        lon_adj = np_ma_masked_zeros_like(hdg.array)
        
        mobiles = [s.slice for s in mobile]
        begin = mobiles[0].start
        end = mobiles[-1].stop
        
        ils_join_offset = None
        
        #------------------------------------
        # Use synthesized track for takeoffs
        #------------------------------------

        # We compute the ground track using best available data.
        if gspd:
            speed = gspd.array
            freq = gspd.frequency
        else:
            speed = tas.array
            freq = tas.frequency

        try:
            toff_slice = toff[0].slice
        except:
            toff_slice = None

        if toff_slice and precise:
            try:
                lat_out, lon_out = self.taxi_out_track_pp(lat.array[begin:toff_slice.start], 
                                                          lon.array[begin:toff_slice.start], 
                                                          speed[begin:toff_slice.start],
                                                          hdg.array[begin:toff_slice.start],
                                                          freq)
            except ValueError:
                self.exception("'%s'. Using non smoothed coordinates for Taxi Out",
                             self.__class__.__name__)
                lat_out = lat.array[begin:toff_slice.start]
                lon_out = lon.array[begin:toff_slice.start]
            lat_adj[begin:toff_slice.start] = lat_out
            lon_adj[begin:toff_slice.start] = lon_out

        elif toff_slice and toff_rwy and toff_rwy.value:
                
            start_locn_recorded = runway_snap_dict(
                toff_rwy.value,lat.array[toff_slice.start], 
                lon.array[toff_slice.start])
            start_locn_default = toff_rwy.value['start']
            _,distance = bearing_and_distance(start_locn_recorded['latitude'], 
                                              start_locn_recorded['longitude'],
                                              start_locn_default['latitude'],  
                                              start_locn_default['longitude'])
        
            if distance < 50:
                # We may have a reasonable start location, so let's use that
                start_locn = start_locn_recorded
                initial_displacement = 0.0
            else:
                # The recorded start point is way off, default to 50m down the track.
                start_locn = start_locn_default
                initial_displacement = 50.0
    
            # Compute takeoff track from start of runway using integrated
            # groundspeed, down runway centreline to end of takeoff (35ft
            # altitude). An initial value of 100m puts the aircraft at a
            # reasonable position with respect to the runway start.
            rwy_dist = np.ma.array(                        
                data = integrate(speed[toff_slice], freq, 
                                 initial_value=initial_displacement, 
                                 scale=KTS_TO_MPS),
                mask = np.ma.getmaskarray(speed[toff_slice]))
    
            # Similarly the runway bearing is derived from the runway endpoints
            # (this gives better visualisation images than relying upon the
            # nominal runway heading). This is converted to a numpy masked array
            # of the length required to cover the takeoff phase.
            rwy_hdg = runway_heading(toff_rwy.value)
            rwy_brg = np_ma_ones_like(speed[toff_slice])*rwy_hdg
            
            # The track down the runway centreline is then converted to
            # latitude and longitude.
            lat_adj[toff_slice], lon_adj[toff_slice] = \
                latitudes_and_longitudes(rwy_brg, 
                                         rwy_dist, 
                                         start_locn)                    
    
            lat_out, lon_out = self.taxi_out_track(toff_slice, lat_adj, lon_adj, speed, hdg, freq)
            lat_adj[:toff_slice.start] = lat_out
            lon_adj[:toff_slice.start] = lon_out
            
        else:
            print 'Cannot smooth taxi out' 

        #-----------------------------------------------------------------------
        # Use ILS track for approach and landings in all localizer approches
        #-----------------------------------------------------------------------
        
        for approach in approaches.value:
            
            this_app_slice = slice_multiply(slice(approach['slice_start'], 
                                                  approach['slice_stop']),
                                            freq)

            if 'runway' not in approach or approach['runway'] == None:
                continue
            runway = approach['runway']
             
            if 'ILS localizer established' in approach:
                
                this_loc_slice = slice_multiply(approach['ILS localizer established'],freq)
                    
                # Adjust the ils data to be degrees from the reference point.
                scale = localizer_scale(runway)
                bearings = ils_loc.array[this_loc_slice] * scale + \
                    runway_heading(runway)+180
                
                if precise:
                    
                    # Tweek the localizer position to be on the start:end centreline
                    localizer_on_cl = ils_localizer_align(runway)
                    
                    # Find distances from the localizer
                    _, distances = bearings_and_distances(lat.array[this_loc_slice], 
                                                          lon.array[this_loc_slice], 
                                                          localizer_on_cl)
                    
                    
                    # At last, the conversion of ILS localizer data to latitude and longitude
                    lat_adj[this_loc_slice], lon_adj[this_loc_slice] = \
                        latitudes_and_longitudes(bearings, distances, localizer_on_cl)
                    
                else: # Imprecise navigation but with an ILS tuned.
                    
                    # Adjust distance units
                    distances = app_range.array[this_loc_slice]
                    
                    if np.ma.count(distances)/float(len(distances)) < 0.8:
                        continue # Insufficient range data to make this worth computing.
                    
                    # Tweek the localizer position to be on the start:end centreline
                    localizer_on_cl = ils_localizer_align(runway)
                    
                    # At last, the conversion of ILS localizer data to latitude and longitude
                    lat_adj[this_loc_slice], lon_adj[this_loc_slice] = \
                        latitudes_and_longitudes(bearings, distances,
                                                 localizer_on_cl)
                    
                # Alignment of the ILS Localizer Range causes corrupt first
                # samples.
                lat_adj[this_loc_slice.start] = np.ma.masked
                lon_adj[this_loc_slice.start] = np.ma.masked
                
                ils_join_offset = None
                if approach['type'] == 'LANDING':
                    # Remember where we lost the ILS, in preparation for the taxi in.
                    ils_join, _ = last_valid_sample(lat_adj[this_loc_slice])
                    if ils_join:
                        ils_join_offset = this_loc_slice.start + ils_join
              
            else:
                # No localizer in this approach
                
                if precise:
                    # Without an ILS we can do no better than copy the prepared arrray data forwards.
                    lat_adj[this_app_slice] = lat.array[this_app_slice]
                    lon_adj[this_app_slice] = lon.array[this_app_slice]
                else:
                    # Adjust distance units
                    distances = app_range.array[this_app_slice]
        
                    bearings = np_ma_ones_like(distances) * (runway_heading(runway)+180)%360.0
                    
                    # Reference point for visual approaches is the runway end.
                    ref_point = runway['end']
                    
                    # We convert the range data
                    lat_adj[this_app_slice], lon_adj[this_app_slice] = \
                        latitudes_and_longitudes(bearings, distances,
                                                 ref_point)
                        
            # The computation of a ground track is not ILS dependent and does
            # not depend upon knowing the runway details.
            if approach['type'] == 'LANDING':

                if 'Landing Turn Off Runway' in approach:
                    turn_offset = int(floor(approach['Landing Turn Off Runway'])) * freq
                else:
                    turn_offset = None
                
                # This function returns the lowest non-None offset.
                join_idx = min(filter(bool, [ils_join_offset, turn_offset]))
                    
                if join_idx and (len(lat_adj) > join_idx): # We have some room to extend over.
                        
                    if precise:
                        # Set up the point of handover
                        lat.array[join_idx] = lat_adj[join_idx]
                        lon.array[join_idx] = lon_adj[join_idx]
                        try:
                            lat_in, lon_in = self.taxi_in_track_pp(lat.array[join_idx:end],
                                                                   lon.array[join_idx:end],
                                                                   speed[join_idx:end],
                                                                   hdg.array[join_idx:end],
                                                                   freq)
                        except ValueError:
                            self.exception("'%s'. Using non smoothed coordinates for Taxi In",
                                           self.__class__.__name__)
                            lat_in = lat.array[join_idx:end]
                            lon_in = lon.array[join_idx:end]
                    else:
                        if join_idx and (len(lat_adj) > join_idx):
                            scan_back = slice(join_idx, this_app_slice.start, -1)
                            lat_join = first_valid_sample(lat_adj[scan_back])
                            lon_join = first_valid_sample(lon_adj[scan_back])
                            join_idx -= max(lat_join.index, lon_join.index) # step back to make sure the join location is not masked.
                            lat_in, lon_in = self.taxi_in_track(lat_adj[join_idx:end], 
                                                                lon_adj[join_idx:end], 
                                                                speed[join_idx:end], 
                                                                hdg.array[join_idx:end], 
                                                                freq)
                    
                    lat_adj[join_idx:end] = lat_in
                    lon_adj[join_idx:end] = lon_in
   
        return lat_adj, lon_adj


class LatitudeSmoothed(DerivedParameterNode, CoordinatesSmoothed):
    """
    From a prepared Latitude parameter, which may have been created by
    straightening out a recorded latitude data set, or from an estimate using
    heading and true airspeed, we now match the data to the available runway
    data. (Airspeed is included as an alternative to groundspeed so that the
    algorithm has wider applicability).
    
    Where possible we use ILS data to make the landing data as accurate as
    possible, and we create ground track data with groundspeed and heading if
    available.
    
    Once these sections have been created, the parts are 'stitched' together
    to make a complete latitude trace.
    
    The first parameter in the derive method is heading_continuous, which is
    always available and which should always have a sample rate of 1Hz. This
    ensures that the resulting computations yield a smoothed track with 1Hz
    spacing, even if the recorded latitude and longitude have only 0.25Hz
    sample rate.
    """
        
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        return 'Latitude Prepared' in available and \
               'Longitude Prepared' in available and \
               'Approach Range' in available and \
               'Heading Continuous' in available and \
               'Airspeed True' in available and \
               'FDR Approaches' in available
    
    units = 'deg'
    
    def derive(self, lat = P('Latitude Prepared'),
               lon = P('Longitude Prepared'),
               head_mag=P('Heading Continuous'),
               ils_loc = P('ILS Localizer'),
               app_range = P('Approach Range'),
               hdg = P('Heading True Continuous'),
               gspd = P('Groundspeed'),
               tas = P('Airspeed True'),
               precise =A('Precise Positioning'),
               toff = S('Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'),
               approaches = A('FDR Approaches'),
               mobile=S('Mobile'),
               ):
        precision = bool(getattr(precise, 'value', False))
        
        if hdg == None:
            hdg = head_mag

        lat_adj, lon_adj = self._adjust_track(lon, lat, ils_loc, app_range, hdg,
                                            gspd, tas, toff, toff_rwy,
                                            approaches, mobile, precision)
        self.array = track_linking(lat.array, lat_adj)
        

class LongitudeSmoothed(DerivedParameterNode, CoordinatesSmoothed):
    """
    See Latitude Smoothed for notes.
    """
    
    units = 'deg'
    ##align_frequency = 1.0
    ##align_offset = 0.0
    
    @classmethod
    def can_operate(cls, available):
        return 'Latitude Prepared' in available and \
               'Longitude Prepared' in available and \
               'Approach Range' in available and \
               'Heading Continuous' in available and \
               'Airspeed True' in available and \
               'FDR Approaches' in available
    
    def derive(self, lat = P('Latitude Prepared'),
               lon = P('Longitude Prepared'),
               head_mag=P('Heading Continuous'),
               ils_loc = P('ILS Localizer'),
               app_range = P('Approach Range'),
               hdg = P('Heading True Continuous'),
               gspd = P('Groundspeed'),
               tas = P('Airspeed True'),
               precise =A('Precise Positioning'),
               toff = S('Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'),
               approaches = A('FDR Approaches'),
               mobile=S('Mobile'),
               ):
        precision = bool(getattr(precise, 'value', False))
        
        if hdg == None:
            hdg = head_mag
        lat_adj, lon_adj = self._adjust_track(lon, lat, ils_loc, app_range, hdg,
                                            gspd, tas, toff, toff_rwy,
                                            approaches, mobile, precision)
        self.array = track_linking(lon.array, lon_adj)

class Mach(DerivedParameterNode):
    '''
    Mach derived from air data parameters for aircraft where no suitable Mach
    data is recorded.
    '''
    
    units = 'Mach'
    
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
    Africa or post-war zones. Also, by using the aircraft compass values to
    work out the variation, we inherently accommodate compass drift for that
    day.
    """
    
    units = 'deg'
    
    def derive(self, head=P('Heading Continuous'),
               head_land = KPV('Heading At Landing'),
               head_toff = KPV('Heading At Takeoff'),
               toff_rwy = A('FDR Takeoff Runway'), 
               land_rwy = A('FDR Landing Runway')):
        
        def first_turn(x):
            # Heading continuous can be huge after a few turns in the hold,
            # so we need this to straighten things out! I bet there's a more
            # Pythonic way to do this, but at least it's simple.
            return x - floor(x/180.0 + 0.5)*180.0        
        
        # Make a masked copy of the heading array, then insert deviations at
        # just the points we know. "interpolate" is designed to
        # replace the masked values with linearly interpolated values between
        # two known points, and extrapolate to the ends of the array. It also
        # substitutes a zero array in case neither is available.
        dev = np.ma.masked_all_like(head.array)
        if head_toff:
            takeoff_heading = head_toff.get_first()
            try:
                dev[takeoff_heading.index] = first_turn(\
                    runway_heading(toff_rwy.value) - takeoff_heading.value)
            except:
                dev[takeoff_heading.index] = 0.0
                
        if land_rwy:
            landing_heading = head_land.get_last()
            try:
                dev[landing_heading.index] = first_turn( \
                    runway_heading(land_rwy.value) - landing_heading.value)
            except:
                dev[landing_heading.index] = 0.0
        
        self.array = interpolate(dev)

class VerticalSpeedInertial(DerivedParameterNode):
    '''
    See 'Vertical Speed' for pressure altitude based derived parameter.
    
    This routine derives the vertical speed from the vertical acceleration, the
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
    with a 1ft/sec^2 acceleration results in an increasing vertical speed of
    55 fpm/sec, not 60 as would be theoretically predicted.
    '''

    units = 'fpm'

    def derive(self, 
               az = P('Acceleration Vertical'),
               alt_std = P('Altitude STD Smoothed'),
               alt_rad = P('Altitude Radio'),
               speed=P('Airspeed')):

        def inertial_vertical_speed(alt_std_repair, frequency, alt_rad_repair, az_repair):
            # Uses the complementary smoothing approach
            
            # This is the accelerometer washout term, with considerable gain.
            # The initialisation "initial_value=az_repair[0]" is very
            # important, as without this the function produces huge spikes at
            # each start of a data period.
            az_washout = first_order_washout (az_repair, 
                                              AZ_WASHOUT_TC, frequency, 
                                              gain=GRAVITY_IMPERIAL,
                                              initial_value=az_repair[0])
            inertial_roc = first_order_lag (az_washout, 
                                            VERTICAL_SPEED_LAG_TC, 
                                            frequency, 
                                            gain=VERTICAL_SPEED_LAG_TC)
    
            # Both sources of altitude data are differentiated before
            # merging, as we mix height rate values to minimise the effect of
            # changeover of sources.
            roc_alt_std = first_order_washout(alt_std_repair,
                                              VERTICAL_SPEED_LAG_TC, frequency,
                                              gain=1/VERTICAL_SPEED_LAG_TC)
            roc_alt_rad = first_order_washout(alt_rad_repair,
                                              VERTICAL_SPEED_LAG_TC, frequency,
                                              gain=1/VERTICAL_SPEED_LAG_TC)
                    
            # Use pressure altitude rate above 100ft and radio altitude rate
            # below 50ft with progressive changeover across that range.
            # up to 50 ft radio 0 < std_rad_ratio < 1 over 100ft radio
            std_rad_ratio = np.maximum(np.minimum((alt_rad_repair-50.0)/50.0,
                                                  1),0)
            roc_altitude = roc_alt_std*std_rad_ratio +\
                roc_alt_rad*(1.0-std_rad_ratio)
            
            return (roc_altitude + inertial_roc) * 60.0


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
            self.array[clump] = inertial_vertical_speed(
                alt_std_repair[clump], az.frequency,
                alt_rad_repair[clump], az_repair[clump])

   
class VerticalSpeed(DerivedParameterNode):
    '''
    The period for averaging altitude data is a trade-off between transient
    response and noise rejection. 
    
    Some older aircraft have poor resolution, and the 4 second timebase leaves a
    noisy signal. We have inspected Hercules data, where the resolution is of the
    order of 9 ft/bit, and data from the BAe 146 where the resolution is 15ft. In
    these cases the wider timebase with greater smoothing is necessary, albeit at
    the expense of transient response.
    
    For most aircraft however, a period of 4 seconds is used. This has been
    found to give good results, and is also the value used to compute the
    recorded Vertical Speed parameter on Airbus A320 series aircraft
    (although in that case the data is delayed, and the aircraft cannot know
    the future altitudes!).    
    '''

    units = 'fpm'
    
    @classmethod
    def can_operate(cls, available):
        if 'Altitude STD Smoothed' in available:
            return True

    def derive(self, alt_std=P('Altitude STD Smoothed'), frame=A('Frame')):
        frame_name = frame.value if frame else ''
        
        if frame_name in ['Hercules', '146']:
            timebase = 8.0
        else:
            timebase = 4.0
        self.array = rate_of_change(alt_std, timebase) * 60.0
        

class VerticalSpeedForFlightPhases(DerivedParameterNode):
    """
    A simple and robust vertical speed parameter suitable for identifying
    flight phases. DO NOT use this for event detection.
    """

    units = 'fpm'

    def derive(self, alt_std = P('Altitude STD Smoothed')):
        # This uses a scaled hysteresis parameter. See settings for more detail.
        threshold = HYSTERESIS_FPROC * max(1, rms_noise(alt_std.array))  
        # The max(1, prevents =0 case when testing with artificial data.
        self.array = hysteresis(rate_of_change(alt_std, 6) * 60, threshold)


class Relief(DerivedParameterNode):
    """
    Also known as Terrain, this is zero at the airfields. There is a small
    cliff in mid-flight where the Altitude AAL changes from one reference to
    another, however this normally arises where Altitude Radio is out of its
    operational range, so will be masked from view.
    """
    
    units = 'ft'
    
    def derive(self, alt_aal = P('Altitude AAL'),
               alt_rad = P('Altitude Radio')):
        self.array = alt_aal.array - alt_rad.array


class CoordinatesStraighten(object):
    '''
    Superclass for LatitudePrepared and LongitudePrepared.
    '''
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
        coord1_s = coord1.array
        coord2_s = coord2.array
        
        # Join the masks, so that we only consider positional data when both are valid:
        coord1_s.mask = np.ma.logical_or(np.ma.getmaskarray(coord1.array),
                                         np.ma.getmaskarray(coord2.array))
        coord2_s.mask = np.ma.getmaskarray(coord1_s)
        # Preload the output with masked values to keep dimension correct 
        array = np_ma_masked_zeros_like(coord1_s)  
        
        # Now we just smooth the valid sections.
        tracks = np.ma.clump_unmasked(coord1_s)
        for track in tracks:
            # Reject any data with invariant positions, i.e. sitting on stand.
            if np.ma.ptp(coord1_s[track])>0.0 and np.ma.ptp(coord2_s[track])>0.0:
                coord1_s_track, coord2_s_track, cost = \
                    smooth_track(coord1_s[track], coord2_s[track], coord1.frequency)
                array[track] = coord1_s_track
        return array
        
        
class LongitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    See Latitude Smoothed for notes.
    """

    units = 'deg'

    @classmethod
    def can_operate(cls, available):
        return ('Longitude' in available and 'Latitude' in available) or\
               ('Airspeed True' in available and \
                'Heading True' in available and \
                'Latitude At Takeoff' in available and \
                'Longitude At Takeoff' in available and \
                'Latitude At Landing' in available and \
                'Longitude At Landing' in available) 
    
    # Note hdg is alignment master to force 1Hz operation when latitude &
    # longitude are only recorded at 0.25Hz.
    def derive(self, 
               lat=P('Latitude'), lon=P('Longitude'),
               hdg=P('Heading True'),tas=P('Airspeed True'), 
               lat_lift=KPV('Latitude At Takeoff'),
               lon_lift=KPV('Longitude At Takeoff'),
               lat_land=KPV('Latitude At Landing'),
               lon_land=KPV('Longitude At Landing')):

        if lat and lon:
            """
            This removes the jumps in longitude arising from the poor resolution of
            the recorded signal.
            """
            self.array = self._smooth_coordinates(lon, lat)
        else:
            _, lon_array = air_track(lat_lift.get_first().value, lon_lift.get_first().value, 
                                     lat_land.get_last().value, lon_land.get_last().value, 
                                     tas.array, hdg.array, tas.frequency)
            self.array = lon_array
    
class LatitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    See Latitude Smoothed for notes.
    """

    units = 'deg'

    @classmethod
    def can_operate(cls, available):
        return ('Latitude' in available and 'Longitude' in available) or\
               ('Airspeed True' in available and \
                'Heading True' in available and \
                'Latitude At Takeoff' in available and \
                'Longitude At Takeoff' in available and \
                'Latitude At Landing' in available and \
                'Longitude At Landing' in available) 
    
    # Note hdg is alignment master to force 1Hz operation when latitude &
    # longitude are only recorded at 0.25Hz.
    def derive(self, 
               lat=P('Latitude'),lon=P('Longitude'),
               hdg=P('Heading True'),tas=P('Airspeed True'), 
               lat_lift=KPV('Latitude At Takeoff'),
               lon_lift=KPV('Longitude At Takeoff'),
               lat_land=KPV('Latitude At Landing'),
               lon_land=KPV('Longitude At Landing')):

        if lat and lon:
            self.array = self._smooth_coordinates(lat, lon)
        else:
            lat_array, _ = air_track(lat_lift.get_first().value, lon_lift.get_first().value, 
                                     lat_land.get_last().value, lon_land.get_last().value, 
                                     tas.array, hdg.array, tas.frequency)
            self.array = lat_array


class RateOfTurn(DerivedParameterNode):
    """
    Simple rate of change of heading. 
    """

    units = 'deg/sec'
    
    def derive(self, head=P('Heading Continuous')):
        # add a little hysteresis to the rate of change to smooth out minor 
        # changes
        roc = rate_of_change(head, 4)
        self.array = hysteresis(roc, 0.1)
        # trouble is that we're loosing the nice 0 values, so force include!
        self.array[(self.array <= 0.05) & (self.array >= -0.05)] = 0


class Pitch(DerivedParameterNode):
    """
    Combination of pitch signals from two sources where required.
    """
    units = 'deg'
    align = False
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
    
    units = 'deg/sec'
    
    def derive(self, pitch=P('Pitch')):
        self.array = rate_of_change(pitch, 2.0)


class Roll(DerivedParameterNode):
    """
    Combination of roll signals from two sources where required.
    """
    units = 'deg'
    align = False
    def derive(self, r1=P('Roll (1)'), r2=P('Roll (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(r1, r2)


class RollRate(DerivedParameterNode):
    # TODO: Tests.
    
    units = 'deg/sec'
    
    def derive(self, roll=P('Roll')):
        self.array = rate_of_change(roll, 2.0)


class ThrottleLevers(DerivedParameterNode):
    """
    A synthetic throttle lever angle, based on the average of the two. Allows
    for simple identification of changes in power etc.
    """
    
    units = 'deg'
    
    def derive(self,
               tla1=P('Eng (1) Throttle Lever'), 
               tla2=P('Eng (2) Throttle Lever')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(tla1, tla2)


class ThrustAsymmetry(DerivedParameterNode):
    '''
    Thrust asymmetry based on N1.
    
    For EPR rated aircraft, this measure should still be applicable as we are
    not applying a manufaturer's limit to the value, rather this is being
    used to identify imbalance of thrust and as the thrust comes from engine
    speed, N1 is still applicable. 
    
    If we have EPR rated engines, but no N1 recorded, a possible solution
    would be to treat EPR=2.0 as 100% and EPR=1.0 as 0% so the Thrust
    Asymmetry would be simply (EPRmax-EPRmin)*100.

    For propellor aircraft the product of prop speed and torgue should be
    used to provide a similar single asymmetry value.
    '''
    
    def derive(self, max_n1=P('Eng (*) N1 Max'), min_n1=P('Eng (*) N1 Min')):
        self.array = max_n1.array - min_n1.array


class ThrustReversers(MultistateDerivedParameterNode):
    '''
    A single parameter with multistate mapping as below.
    '''
    @classmethod
    def can_operate(cls, available):
        return all_of((
            'Eng (1) Thrust Reverser (L) Deployed',
            'Eng (1) Thrust Reverser (L) Unlocked',
            'Eng (1) Thrust Reverser (R) Deployed',
            'Eng (1) Thrust Reverser (R) Unlocked',
            'Eng (2) Thrust Reverser (L) Deployed',
            'Eng (2) Thrust Reverser (L) Unlocked',
            'Eng (2) Thrust Reverser (R) Deployed',
            'Eng (2) Thrust Reverser (R) Unlocked',
        ), available) or all_of((
            'Eng (1) Thrust Reverser Unlocked',
            'Eng (1) Thrust Reverser Deployed',
            'Eng (2) Thrust Reverser Unlocked',
            'Eng (2) Thrust Reverser Deployed',
        ), available)
        
    # We are interested in all stowed, all deployed or any other combination.
    # The mapping "In Transit" is used for anythingn other than the fully
    # established conditions, so for example one locked and the other not is
    # still treated as in transit.
    values_mapping = {
        0: 'Stowed',
        1: 'In Transit',
        2: 'Deployed',
    }

    def derive(self,
            e1_lft_dep=P('Eng (1) Thrust Reverser (L) Deployed'),
            e1_lft_out=P('Eng (1) Thrust Reverser (L) Unlocked'),
            e1_rgt_dep=P('Eng (1) Thrust Reverser (R) Deployed'),
            e1_rgt_out=P('Eng (1) Thrust Reverser (R) Unlocked'),
            e2_lft_dep=P('Eng (2) Thrust Reverser (L) Deployed'),
            e2_lft_out=P('Eng (2) Thrust Reverser (L) Unlocked'),
            e2_rgt_dep=P('Eng (2) Thrust Reverser (R) Deployed'),
            e2_rgt_out=P('Eng (2) Thrust Reverser (R) Unlocked'),
            
            e1_unlk=P('Eng (1) Thrust Reverser Unlocked'),
            e1_depd=P('Eng (1) Thrust Reverser Deployed'),
            e2_unlk=P('Eng (2) Thrust Reverser Unlocked'),
            e2_depd=P('Eng (2) Thrust Reverser Deployed'),
            
            frame=A('Frame')):
        frame_name = frame.value if frame else ''
        
        if frame_name in ['737-4', '737-1',
                          '737-5', '737-5_NON-EIS', 
                          '737-6', '737-6_NON-EIS', 
                          '737-i', '737-300_PTI']:
            all_tr = \
                e1_lft_dep.array.raw + e1_lft_out.array.raw + \
                e1_rgt_dep.array.raw + e1_rgt_out.array.raw + \
                e2_lft_dep.array.raw + e2_lft_out.array.raw + \
                e2_rgt_dep.array.raw + e2_rgt_out.array.raw
            
            result = np_ma_ones_like(e1_lft_dep.array.raw)
            result = np.ma.where(all_tr==0, 0, result)
            result = np.ma.where(all_tr==8, 2, result)
            self.array = result
            
        elif frame_name in ['737-3C']:
            all_tr = \
                e1_unlk.array.raw + e1_depd.array.raw + \
                e2_unlk.array.raw + e2_depd.array.raw
            
            result = np_ma_ones_like(e1_unlk.array.raw)
            result = np.ma.where(all_tr==0, 0, result)
            result = np.ma.where(all_tr==4, 2, result)
            self.array = result
                
        else:
            raise DataFrameError(self.name, frame_name)


class TurbulenceRMSG(DerivedParameterNode):
    """
    Simple RMS g measurement of turbulence over a 5-second period.
    """
    
    name = 'Turbulence RMS g'
    units = 'RMS g'
    
    def derive(self, acc=P('Acceleration Vertical')):
        width=int(acc.frequency*5+1)
        mean = moving_average(acc.array, window=width)
        acc_sq = (acc.array)**2.0
        n__sum_sq = moving_average(acc_sq, window=width)
        # Rescaling required as moving average is over width samples, whereas
        # we have only width - 1 gaps; fences and fence posts again !
        core = (n__sum_sq - mean**2.0)*width/(width-1.0)
        self.array = np.ma.sqrt(core)

#------------------------------------------------------------------
# WIND RELATED PARAMETERS
#------------------------------------------------------------------
class WindDirectionContinuous(DerivedParameterNode):
    """
    Like the aircraft heading, this does not jump as it passes through North.
    """
    units = 'deg'
    def derive(self, wind_head=P('Wind Direction')):
        self.array = straighten_headings(wind_head.array)


class Headwind(DerivedParameterNode):
    """
    This is the headwind, negative values are tailwind.
    """

    units = 'kts'
    
    @classmethod
    def can_operate(cls, available):
        if all_of(('Wind Speed',
                   'Wind Direction Continuous', 
                   'Heading True Continuous'), available):
            return True
    
    def derive(self, windspeed=P('Wind Speed'), 
               wind_dir=P('Wind Direction Continuous'), 
               head=P('Heading True Continuous'), 
               toffs=S('Takeoff'),               
               alt_aal=P('Altitude AAL'), 
               gspd=P('Groundspeed'), 
               aspd=P('Airspeed True')):
        
        rad_scale = radians(1.0)
        headwind = windspeed.array * np.ma.cos((wind_dir.array-head.array)*rad_scale)

        # If we have airspeed and groundspeed, overwrite the values for the
        # first hundred feet after takeoff. Note this is done in a
        # deliberately crude manner so that the different computations may be
        # identified easily by the analyst.
        if gspd and aspd and alt_aal and toffs:
            # We merge takeoff slices with altitude slices to extend the takeoff phase to 100ft.
            for climb in slices_or(alt_aal.slices_from_to(0, 100),
                                   [s.slice for s in toffs]):
                headwind[climb] = aspd.array[climb] - gspd.array[climb]
        
        self.array = headwind
                

class Tailwind(DerivedParameterNode):
    """
    This is the tailwind component.
    """

    units = 'kts'
    
    def derive(self, hwd=P('Headwind')):
        self.array = -hwd.array
        

class TAT(DerivedParameterNode):
    """
    Blends data from two air temperature sources.
    """
    name = "TAT"
    units = 'C'
    align = False
    
    def derive(self, 
               source_1 = P('TAT (1)'),
               source_2 = P('TAT (2)')):
        
        # Alternate samples (1)&(2) are blended.
        self.array, self.frequency, self.offset = \
            blend_two_parameters(source_1, source_2)


class V2(DerivedParameterNode):
    '''
    Based on weight and flap at time of Liftoff, first liftoff only.
    '''

    units = 'kts'
    
    @classmethod
    def can_operate(cls, available):
        available = set(available)
        afr = 'AFR V2' in available
        base_for_lookup = ['Airspeed', 'Gross Weight At Liftoff', 'Series',
                           'Family']
        airbus = set(base_for_lookup + ['Configuration']).issubset(available)
        boeing = set(base_for_lookup + ['Flap']).issubset(available)
        return afr or airbus or boeing

    def derive(self, 
               spd=P('Airspeed'),
               flap=P('Flap Lever Detent'),
               conf=P('Configuration'),
               afr_v2=A('AFR V2'),
               weight_liftoff=KPV('Gross Weight At Liftoff'),
               series=A('Series'),
               family=A('Family')):
        
        # Initialize the result space.
        self.array = np_ma_masked_zeros_like(spd.array)
        self.array.mask = True

        if afr_v2:
            # v2 supplied, use this
            self.array = afr_v2.value
        elif weight_liftoff:
            vspeed_class = get_vspeed_map(series.value, family.value)
            setting_param = flap or conf
            if vspeed_class:
                vspeed_table = vspeed_class()
                index = weight_liftoff[0].index
                weight = weight_liftoff[0].value
                setting = setting_param.array[index]
                vspeed = vspeed_table.v2(weight, setting)
                if vspeed is not None:
                    self.array[0:] = vspeed
            else:
                # Aircraft doesnt use V2
                self.array.mask = False
        else:
            # no lift off leave zero masked array
            pass


class WindAcrossLandingRunway(DerivedParameterNode):
    """
    This is the windspeed across the final landing runway, positive wind from left to right.
    """

    units = 'kts'
    
    def derive(self, windspeed=P('Wind Speed'), wind_dir=P('Wind Direction Continuous'), 
               land_rwy = A('FDR Landing Runway')):
        
        if land_rwy and land_rwy.value:
            land_heading = runway_heading(land_rwy.value)
        else:
            land_heading = None
            
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
    units = 'deg'
    
    @classmethod
    def can_operate(cls, available):
        a = set(['Aileron (L) Inboard', 'Aileron (R) Inboard', 'Aileron (L) Outboard', 'Aileron (R) Outboard'])
        x = set(available)
        return 'Aileron (L)' in available or \
               'Aileron (R)' in available or \
               not (a - x)

    def derive(self,
               al=P('Aileron (L)'),
               ar=P('Aileron (R)'),
               ali=P('Aileron (L) Inboard'),
               ari=P('Aileron (R) Inboard'),
               alo=P('Aileron (L) Outboard'),
               aro=P('Aileron (R) Outboard')):
        
        if al and ar:
            self.array, self.frequency, self.offset = blend_two_parameters(al, ar)
        elif al or ar:
            self.array = al.array if al else ar.array
            self.frequency = al.frequency if al else ar.frequency
            self.offset = al.offset if al else ar.offset
        else:
            return NotImplemented


class AileronTrim(DerivedParameterNode): # RollTrim
    '''
    '''
    # TODO: TEST
    name = 'Aileron Trim' # Roll Trim
    units = 'deg'

    def derive(self,
               atl=P('Aileron Trim (L)'),
               atr=P('Aileron Trim (R)')):
        self.array, self.frequency, self.offset = blend_two_parameters(atl, atr)


class Elevator(DerivedParameterNode):
    '''
    Blends alternate elevator samples. If either elevator signal is invalid,
    this reverts to just the working sensor.
    '''

    units = 'deg'
    
    def derive(self,
               el=P('Elevator (L)'),
               er=P('Elevator (R)')):
        
        if el and er:
            self.array, self.frequency, self.offset = blend_two_parameters(el, er)
        else:
            self.array = el.array if el else er.array
            self.frequency = el.frequency if el else er.frequency
            self.offset = el.offset if el else er.offset
        

################################################################################
# Speedbrake


# TODO: Write some unit tests!
class Speedbrake(DerivedParameterNode):
    '''
    Spoiler angle in degrees, zero flush with the wing and positive up.

    Spoiler positions are recorded in different ways on different aircraft,
    hence the frame specific sections in this class.
    '''

    units = 'deg'
    align = False

    @classmethod
    def can_operate(cls, available):
        '''
        Note: The frame name cannot be accessed within this method to determine
              which parameters are required.
        '''
        x = available
        return ('Frame' in x and 'Spoiler (2)' in x and 'Spoiler (7)' in x) \
            or ('Frame' in x and 'Spoiler (4)' in x and 'Spoiler (9)' in x)

    def spoiler_737(self, spoiler_a, spoiler_b):
        '''
        We indicate the angle of the lower of the two raised spoilers, as
        this represents the drag element. Differential deployment is used to
        augments roll control, so the higher of the two spoilers is relating
        to roll control. Small values are ignored as these arise from control
        trim settings.
        '''
        offset = (spoiler_a.offset + spoiler_b.offset) / 2.0
        array = np.ma.minimum(spoiler_a.array, spoiler_b.array)
        # Force small angles to indicate zero:
        array = np.ma.where(array < 3.0, 0.0, array)
        return array, offset

    def derive(self,
            spoiler_2=P('Spoiler (2)'), spoiler_7=P('Spoiler (7)'),
            spoiler_4=P('Spoiler (4)'), spoiler_9=P('Spoiler (9)'),
            frame=A('Frame')):
        '''
        '''
        frame_name = frame.value if frame else ''

        if frame_name in ['737-3A', '737-3B', '737-3C']:
            self.array, self.offset = self.spoiler_737(spoiler_4, spoiler_9)

        elif frame_name in ['737-4', '737-5', '737-5_NON-EIS', '737-6',
                            '737-6_NON-EIS']:
            self.array, self.offset = self.spoiler_737(spoiler_2, spoiler_7)

        else:
            raise DataFrameError(self.name, frame_name)


class SpeedbrakeSelected(MultistateDerivedParameterNode):
    '''
    Determines the selected state of the speedbrake.

    Speedbrake Selected Values:

    - 0 -- Stowed
    - 1 -- Armed / Commanded (Spoilers Down)
    - 2 -- Deployed / Commanded (Spoilers Up)
    '''

    values_mapping = {
        0: 'Stowed',
        1: 'Armed/Cmd Dn',
        2: 'Deployed/Cmd Up',
    }

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        x = available
        return 'Speedbrake Deployed' in x \
            or ('Frame' in x and 'Speedbrake Handle' in x)\
            or ('Frame' in x and 'Speedbrake' in x)
    
    def b737_speedbrake(self, spdbrk, handle):
        '''
        Speedbrake Handle Positions for 737-x:

            ========    ============
            Angle       Notes
            ========    ============
             0.0        Full Forward
             4.0        Armed
            24.0
            29.0
            38.0        In Flight
            40.0        Straight Up
            48.0        Full Up
            ========    ============
            
        Speedbrake Positions > 1 = Deployed
        '''
        if spdbrk and handle:
            # Speedbrake and Speedbrake Handle available
            '''
            Speedbrake status taken from surface position. This allows
            for aircraft where the handle is inoperative, overwriting
            whatever the handle position is when the brakes themselves
            have deployed.
            
            It's not possible to identify when the speedbrakes are just
            armed in this case, so we take any significant motion as
            deployed.
            
            If there is no handle position recorded, the default 'Stowed' 
            value is retained.
            '''
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 35.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where((handle.array >= 35.0) | (spdbrk.array > 1.0),
                                'Deployed/Cmd Up', armed)
        elif spdbrk and not handle:
            # Speedbrake only
            array = np.ma.where(spdbrk.array > 1.0,
                                'Deployed/Cmd Up', 'Stowed')
        elif handle and not spdbrk:
            # Speedbrake Handle only
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 35.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where(handle.array >= 35.0,
                                'Deployed/Cmd Up', armed)
        else:
            raise ValueError("Can't work without either Speedbrake or Handle")
        return array

    def derive(self,
            deployed=M('Speedbrake Deployed'),
            armed=M('Speedbrake Armed'),
            handle=P('Speedbrake Handle'),
            spdbrk=P('Speedbrake'),
            frame=A('Frame')):
        frame_name = frame.value if frame else ''
        if deployed:
            # Speedbrake Deployed available, use this
            # set initial state to 'Stowed'
            array = np.ma.zeros(len(deployed.array))
            if armed:
                # If we have Armed, set this first
                array[armed.array == 'Armed'] = 1
            # deployed will overide some armed states
            array[deployed.array == 'Deployed'] = 2
            self.array = array # (only call __set_attr__ once)
            
        elif frame_name.startswith('737-'):
            self.array = self.b737_speedbrake(spdbrk, handle)
            
        else:
            # Not implemented for this frame
            raise DataFrameError(self.name, frame_name)


################################################################################
# Stick Shaker


class StickShaker(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stick shaker system. Where two
    systems are recorded the results are OR'd to make a single parameter which
    operates in response to either system triggering. Hence the removal of
    automatic alignment of the signals.
    '''

    align = False
    values_mapping = {
        0: 'No_Shake',
        1: 'Shake',
    }

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # NOTE: Does not take into account which parameter for which frame!
        return 'Frame' in available and (
            'Stick Shaker (L)' in available or \
            'Stick Shaker (R)' in available or \
            'Shaker Activation' in available \
        )

    def derive(self, frame=A('Frame'),
            shake_l=M('Stick Shaker (L)'),
            shake_r=M('Stick Shaker (R)'),
            shake_act=M('Shaker Activation')):
        '''
        '''
        frame_name = frame.value if frame else ''

        if frame_name in ['CRJ-700-900'] and shake_act:
            self.array, self.frequency, self.offset = \
                shake_act.array, shake_act.frequency, shake_act.offset

        elif frame_name in ['737-1', '737-3A', '737-3B', '737-3C', '737-4',
                            '737-i', '737-300_PTI', '757-DHL']:
            self.array = np.ma.logical_or(shake_l.array, shake_r.array)
            self.frequency , self.offset = shake_l.frequency, shake_l.offset

        elif frame_name in ['737-5', '737-5_NON-EIS'] and shake_l:
            # Named (L) but in fact (L) and (R) are or'd together at the DAU.
            self.array, self.frequency, self.offset = \
                shake_l.array, shake_l.frequency, shake_l.offset

        # Stick shaker not found in 737-6 frame.
        else:
            raise DataFrameError(self.name, frame_name)


class ApproachRange(DerivedParameterNode):
    """
    This is the range to the touchdown point for both ILS and visual
    approaches including go-arounds. The reference point is the ILS Localizer
    antenna where the runway is so equipped, or the end of the runway where
    no ILS is available.
    
    The array is masked where no data has been computed, and provides
    measurements in metres from the reference point where the aircraft is on
    an approach.
    """
    
    unit = 'm'

    @classmethod
    def can_operate(cls, available):
        a = set(['Heading True Continuous','Airspeed True','Altitude AAL',
                 'FDR Approaches'])
        x = set(available)
        return not (a - x)
    
    def derive(self, gspd=P('Groundspeed'),
               drift=P('Drift'),
               glide=P('ILS Glideslope'),
               head=P('Heading True Continuous'),
               tas=P('Airspeed True'),
               alt_aal=P('Altitude AAL'),
               approaches=A('FDR Approaches'),
               ):
        
        app_range = np_ma_masked_zeros_like(head.array)
        freq = head.frequency
        
        for approach in approaches.value:
            # We are going to reference the approach to a runway touchdown
            # point. Without that it's pretty meaningless, so give up now.
            if 'runway' not in approach or approach['runway'] == None:
                continue
            
            # Retrieve the approach slice
            this_app_slice = slice_multiply(slice(approach['slice_start'], 
                                                  approach['slice_stop']),
                                            freq)

            # What is the heading with respect to the runway centreline for this approach?
            off_cl = (head.array[this_app_slice] - \
                      runway_heading(approach['runway'])) % 360.0
                
            # Use recorded groundspeed where available, otherwise
            # estimate range using true airspeed. This is because there
            # are aircraft which record ILS but not groundspeed data. In
            # either case the speed is referenced to the runway heading
            # in case of large deviations on the approach or runway.
            if gspd and drift:
                speed = gspd.array.data[this_app_slice] * \
                    np.cos(np.radians(off_cl+drift.array[this_app_slice]))
                freq = gspd.frequency
            elif gspd:
                speed = gspd.array.data[this_app_slice] * \
                    np.cos(np.radians(off_cl))
                freq = gspd.frequency
            else:
                speed = tas.array.data[this_app_slice] * \
                    np.cos(np.radians(off_cl))
                freq = tas.frequency
                
            # Estimate range by integrating back from zero at the end of the
            # phase to high range values at the start of the phase.
            spd_repaired = repair_mask(speed, extrapolate=True)
            app_range[this_app_slice] = integrate(spd_repaired, freq, 
                                                  scale=KTS_TO_MPS, 
                                                  direction='reverse')

            if 'ILS glideslope established' in approach:
                # reg_slice is the slice of data over which we will apply a
                # regression process to identify the touchdown point from the
                # height and distance arrays.
                reg_slice = slice_multiply(
                    approach['ILS glideslope established'], freq)
                # Compute best fit glidepath. The term (1-0.13 x glideslope
                # deviation) caters for the aircraft deviating from the
                # planned flightpath. 1 dot low is about 7% of a 3 degree
                # glidepath. Not precise, but adequate accuracy for the small
                # error we are correcting for here, and empyrically checked.
                corr, slope, offset = coreg(app_range[reg_slice],
                    alt_aal.array[reg_slice] * (1-0.13*glide.array[reg_slice]))
                # This should correlate very well, and any drop in this is a
                # sign of problems elsewhere.
                if corr < 0.995:
                    self.warning('Low convergence in computing ILS '
                                 'glideslope offset.')
            else:
                _,app_slices = slices_between(alt_aal.array[this_app_slice],
                                              100, 500)
                # Computed locally, so app_slices do not need rescaling.
                if len(app_slices) != 1:
                    self.info(
                        'Altitude AAL is not between 100-500 ft during an '
                        'approach slice. %s will not be calculated for this '
                        'section.', self.name)
                    continue
                reg_slice = shift_slice(app_slices[0], this_app_slice.start)         
                
                corr, slope, offset = coreg(app_range[reg_slice],
                                            alt_aal.array[reg_slice])
                # This should still correlate pretty well.
                if corr < 0.990:
                    self.warning('Low convergence in computing visual '
                                 'approach path offset.')
            
            ##import matplotlib.pyplot as plt
            ##x1=app_range[reg_slice]
            ##y1=alt_aal.array[reg_slice]
            ##x2=app_range[reg_slice]
            ##y2=alt_aal.array[reg_slice] * (1-0.13*glide.array[reg_slice])            
            ##xnew = np.linspace(np.min(x1),np.max(x1),num=2)
            ##ynew = (xnew - offset)/slope 
            ##plt.plot(x1,y1,'-',x2,y2,'-',xnew,ynew,'-')
            ##plt.show()
    
            # Touchdown point nominally 1000ft from start of runway but
            # use glideslope antenna position if we know it.
            runway = approach['runway']
            try:
                start_2_loc, gs_2_loc, end_2_loc, pgs_lat, pgs_lon = \
                    runway_distances(runway)
                extend = gs_2_loc
            except (KeyError, TypeError):
                extend = runway_length(runway) - 1000 / METRES_TO_FEET 

            # Shift the values in this approach so that the range = 0 at
            # 0ft on the projected ILS or approach slope.
            app_range[this_app_slice] += extend - offset

        self.array = app_range

################################################################################


class VOR1Frequency(DerivedParameterNode):
    """
    Extraction of VOR tuned frequencies from receiver (1).
    """

    name = "VOR (1) Frequency"
    units = 'MHz'

    def derive(self, f=P('ILS-VOR (1) Frequency')):
        self.array = filter_vor_ils_frequencies(f.array, 'VOR')


class VOR2Frequency(DerivedParameterNode):
    """
    Extraction of VOR tuned frequencies from receiver (1).
    """

    name = "VOR (2) Frequency"
    units = 'MHz'

    def derive(self, f=P('ILS-VOR (2) Frequency')):
        self.array = filter_vor_ils_frequencies(f.array, 'VOR')

class WindSpeed(DerivedParameterNode):
    '''
    Required for Embraer 135-145 Data Frame
    '''
    
    units = 'kts'
    
    def derive(self, wind_1=P('Wind Speed (1)'), wind_2=P('Wind Speed (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(wind_1, wind_2)
        
class WindDirection(DerivedParameterNode):
    '''
    Many aircraft have wind direction stored in true (not magnetic)
    coordinates. Rather than making a distinction, we merge true and magnetic
    wind into a single (albeit slightly ambiguous) parameter.
    
    The Embraer 135-145 data frame includes two sources, hence the
    alternative form.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Wind Direction True' in available or \
               ('Wind Direction (1)' in available and\
                'Wind Direction (2)' in available)
    
    units = 'deg'

    def derive(self, wind_true=P('Wind Direction True'),
               wind_1=P('Wind Direction (1)'), 
               wind_2=P('Wind Direction (2)')):
        if wind_true:
            self.array = wind_true.array
        else:
            self.array, self.frequency, self.offset = \
                blend_two_parameters(wind_1, wind_2)
        
        
class WheelSpeedInboard(DerivedParameterNode):
    '''
    Required for Embraer 135-145 Data Frame
    '''
    def derive(self, ws_1=P('Wheel Speed Inboard (1)'), ws_2=P('Wheel Speed Inboard (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(ws_1, ws_2)
        
        
class WheelSpeedOutboard(DerivedParameterNode):
    '''
    Required for Embraer 135-145 Data Frame
    '''
    def derive(self, ws_1=P('Wheel Speed Outboard (1)'), ws_2=P('Wheel Speed Outboard (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(ws_1, ws_2)
        
        
class WheelSpeed(DerivedParameterNode):
    '''
    Required for Embraer 135-145 Data Frame
    '''
    def derive(self, ws_in=P('Wheel Speed Inboard'), ws_out=P('Wheel Speed Outboard')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(ws_in, ws_out)
        


class HeadingTrack(DerivedParameterNode):
    '''
    Currently works only for 737-NG which record IRU Track Angle True
    '''
    units = 'deg'
    
    def derive(self, heading=P('Heading Continuous'), drift=P('Drift')):
        self.array = heading.array - drift.array
        
        
class HeadingDeviationFromRunway(DerivedParameterNode):
    
    def derive(heading_track=P('Heading Track'),
               takeoff=S('Takeoff'),
               rwy=A('FDR Takeoff Runway'),
               apps=S('Approaches')):

        for approach in apps:
            try:
                rwy_hdg = approach['runway']['magnetic_heading']
            except KeyError:
                # no runway identified in approach
                continue
            
            self.array[approach.slice] = runway_deviation(heading_track[approach.slice], rwy_hdg)


        
    
class StableApproach(MultistateDerivedParameterNode):
    '''
    During the Approach, the following steps are assessed for stability:
    
    1. Gear is down
    2. Landing Flap is set
    3. Heading aligned to Runway within 10 degrees 
    4. Approach Airspeed minus Reference speed within 10 knots
    5. Glideslope deviation within 1 dot
    6. Localizer deviation within 1 dot
    7. Engine Power greater than 75% and not Cycling within last 5 seconds
    
    if all the above steps are met, the result is the declaration of:
    8. "Stable"
    
    TODO:
    =====
    
    * Q: Fill masked values of parameters with False (unstable: stop here) or True (stable, carry on)
    * Declare that if there is no GS est < 1000ft non-precision (500ft)
    * CREATE Track Heading parameter (using recorded if available, else worked out from drift from Heading Mag). New Param Runway Deviation is Track Heading - RUNWAY HEADING for both takeoff phase and Approach and Landing phase of flight.
    * CREATE KPVs for height first stable and height last unstable    
    '''
    
    values_mapping = {
        0: '-',  # All values should be masked anyway, this helps align values
        1: 'Gear Not Down',
        2: 'Not Landing Flap',
        3: 'Not Aligned to Runway',   # Rename with heading?
        4: 'Airspeed Not Stable',  # Q: Split into two Airspeed High/Low?
        5: 'Glideslope Not Stable',
        6: 'Localizer Not Stable',
        7: 'Vertical Speed Not Stable',
        8: 'Engine Power Not Stable',
        9: 'Stable',
    }
    
    align_frequency = 1  # force to 1Hz
    
    @classmethod
    def can_operate(cls, available):
        return True
    
    def derive(self, 
               apps=S('Approach'),
               gear=M('Gear Down'),
               flap=M('Flap'),
               head=P('Heading Track'),
               aspd=P('Airspeed Relative'),
               vert_spd=P('Vertical Speed'),
               glide=P('ILS Glideslope'),
               loc=P('ILS Localizer'),
               eng=P('Eng (*) N1 Min'),
               alt=P('Altitude AAL'),
               ):
        # find point first stabalised
        # % stable after first_stable to landing
        
        #Last Unstable due to
        # simple yes/no indicators to identify the cause of the last unstable point
        # options are FLAP, GEAR GS HI/LO, LOC, SPD HI/LO and VSI HI/LO
        
        #Ht AAL due to
        # the altitude above airfield level corresponding to each cause
        # options are FLAP, GEAR GS HI/LO, LOC, SPD HI/LO and VSI HI/LO

        # An alternative to this discrete would be to create a multistate
        # with each point being the reason it's unstable
        
        
        

        # create an empty fuly masked array
        self.array = np_ma_masked_zeros_like(gear.array)
        self.array.mask = True
        
        for approach in apps:
    
            # do we have a heading??
            
            #TODO: Ensure loc/glide are established too
            ##loc_est = slices_and([approach['ILS localizer established']], approach.slice)
            ##glide_est = slices_and([approach['ILS Glideslope established']], approach.slice)
            glide_est = loc_est = approach.slice
            
            
            
            # prepare data for this appproach:
            gear_down = repair_mask(gear.array[approach.slice])
            flap_lever = repair_mask(flap.array[approach.slice])
            heading = repair_mask(head.array[approach.slice])
            airspeed = repair_mask(aspd.array[approach.slice])
            glideslope = repair_mask(glide.array[glide_est])
            localizer = repair_mask(loc.array[loc_est])
            vertical_speed = repair_mask(vert_spd.array[approach.slice])
            engine = repair_mask(eng.array[approach.slice])
            altitude = repair_mask(alt.array[approach.slice])
            
            
            # Assume unstable due to Gear Down at first
            self.array[approach.slice] = 1

            #== 1. Gear Down ==
            landing_gear_set = (gear_down == 'Down')
            stable = landing_gear_set.filled(False) # replace masked with false
            self.array[approach.slice][stable] = 2 # not due to landing gear - pass the blame on!
            
            #== 2. Landing Flap ==
            landing_flap_set = (flap_lever == flap_lever[-1])
            stable &= landing_flap_set.filled(False)
            self.array[approach.slice][stable] = 3
            
            #== 3. Heading ==
            STABLE_HEADING = 10  # degrees
            stable_heading = abs(heading) < STABLE_HEADING
            stable &= stable_heading.filled(False)  #Q: Should masked values assumed on track ???
            self.array[approach.slice][stable] = 4
            
            #== 4. Airspeed Relative ==
            STABLE_AIRSPEED_ABOVE_REF = 20
            stable_airspeed = (airspeed < STABLE_AIRSPEED_ABOVE_REF) | (altitude < 100)
            stable &= stable_airspeed.filled(False)  # if no V Ref speed, values are masked so consider unstable
            self.array[approach.slice][stable] = 5
                                                             
            #== 5. Glideslope Deviation ==
            ## has altitude cutoff too and 3 second gliding window
            STABLE_GLIDESLOPE = 1.0  # dots
            stable_gs = (abs(glideslope) < STABLE_GLIDESLOPE) | (altitude < 100)
            ##stable_gs = (glideslope > -STABLE_GLIDESLOPE) & (glideslope < STABLE_GLIDESLOPE)
            stable &= stable_gs.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
            self.array[approach.slice][stable] = 6
            
            #== 6. Localizer Deviation ==
            ## has altitude cutoff too and 3 second gliding window
            STABLE_LOCALIZER = 1.0  # dots
            stable_loc = (abs(localizer) < STABLE_LOCALIZER) | (altitude < 100)
            ##stable_loc = (localizer > -STABLE_LOCALIZER) & (localizer < STABLE_LOCALIZER)
            stable &= stable_loc.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
            stable[altitude < 100] = 1  # Assume stable below cutoff
            self.array[approach.slice][stable] = 7
            
            #== 7. Vertical Speed ==
            ## has altitude cutoff too and 3 second gliding window
            STABLE_VERTICAL_SPEED_MIN = -1000
            STABLE_VERTICAL_SPEED_MAX = -200
            stable_vert = (vertical_speed > STABLE_VERTICAL_SPEED_MIN) & (vertical_speed < STABLE_VERTICAL_SPEED_MAX) 
            stable_vert |= altitude < 50
            stable &= stable_vert.filled(False)  #Q: make True?
            self.array[approach.slice][stable] = 8
            
            #== 8. Engine Power (N1) ==
            STABLE_N1_MIN = 75  # %
            stable_engine = (engine > STABLE_N1_MIN)
            stable_engine |= (altitude > 1000) | (altitude < 50)  # Only use in altitude band 1000-50 feet
            stable &= stable_engine.filled(True)  #Q: make True?
            self.array[approach.slice][stable] = 9  # Woop, you're stable!
            
        #endfor
        
        pass
        
        
        
        
