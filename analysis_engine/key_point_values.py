import numpy as np

from operator import itemgetter

from utilities.geometry import midpoint

from analysis_engine.settings import (ACCEL_LAT_OFFSET_LIMIT,
                                      ACCEL_NORM_OFFSET_LIMIT,
                                      CLIMB_OR_DESCENT_MIN_DURATION,
                                      CONTROL_FORCE_THRESHOLD,
                                      FEET_PER_NM,
                                      GRAVITY_METRIC,
                                      HYSTERESIS_FPALT,
                                      KTS_TO_MPS,
                                      LEVEL_FLIGHT_MIN_DURATION,
                                      NAME_VALUES_DESCENT,
                                      NAME_VALUES_ENGINE,
                                      NAME_VALUES_FLAP)

from analysis_engine.node import KeyPointValueNode, KPV, KTI, P, S, A, M

from analysis_engine.library import (ambiguous_runway,
                                     any_of,
                                     bearings_and_distances,
                                     bump,
                                     clip, 
                                     clump_multistate,
                                     coreg, 
                                     cycle_counter,
                                     cycle_finder,
                                     find_edges,
                                     find_edges_on_state_change,
                                     hysteresis,
                                     index_at_value,
                                     integrate,
                                     is_index_within_slice,
                                     is_index_within_sections,
                                     mask_inside_slices,
                                     mask_outside_slices,
                                     max_abs_value,
                                     max_continuous_unmasked, 
                                     max_value,
                                     min_value, 
                                     repair_mask,
                                     np_ma_masked_zeros_like,
                                     peak_curvature,
                                     rate_of_change,
                                     runway_deviation,
                                     runway_distance_from_end,
                                     shift_slice,
                                     shift_slices,
                                     slice_samples,
                                     slices_and_not,
                                     slices_from_to,
                                     slices_not,
                                     slices_overlap,
                                     slices_and,
                                     touchdown_inertial,
                                     value_at_index)


################################################################################
# Superclasses

class FlapOrConfigurationMaxOrMin(object):
    '''
    Abstract superclass.
    '''
    def flap_or_conf_max_or_min(self, conflap, airspeed, function, scope=None, include_zero=False):
        '''
        Generic flap and conf event creation process.
        :param conflap: Conf or Flap data, restricted to detent settings.
        :type conflap: Numpy masked array, in conf values (floating point) or flap (degrees or %).
        :param airspeed: airspeed parameter
        :type airspeed: Numpy masked array
        :param function: function to be applied to the airspeed values
        :type function: 'max_value' or 'min_value'
        :param scope: Periods to restrict period to be monitored. Essential for minimum speed checks, otherwise all the results relate to taxi periods!
        :type scope: optional list of slices.
        :param include_zero: option to include zero flap settings. Used for monitoring AOA with clean configuration.
        :type include_zero: boolean, default = False.
        
        :returns: Nothing. KPVs are created within the routine.
        '''
        if scope == []:
            return # Can't have an event if the scope is empty.
        
        if scope:
            scope_array = np_ma_masked_zeros_like(airspeed.array)
            for valid in scope:
                scope_array.mask[
                    int(valid.slice.start or 0):
                    int(valid.slice.stop or len(scope_array)) + 1] = False
                
        for conflap_setting in np.ma.unique(conflap.array):
            if np.ma.is_masked(conflap_setting):
                # ignore masked values
                continue
            if conflap_setting == 0.0 and \
               include_zero == False:
                continue
            
            spd_with_conflap = np.ma.copy(airspeed.array)
            # apply flap mask
            spd_with_conflap.mask = np.ma.mask_or(airspeed.array.mask,
                                                  conflap.array.mask)
            spd_with_conflap[conflap.array != conflap_setting] = np.ma.masked
            if scope:
                spd_with_conflap.mask = np.ma.mask_or(spd_with_conflap.mask,
                                                      scope_array.mask)
            #TODO: Check logical OR is sensible for all values (probably ok as
            #airspeed will always be higher than max flap setting!)
            index, value = function(spd_with_conflap)
            
            # Check we have a result to record. Note that most flap setting will
            # not be used in the climb, hence this is normal operation.
            if index and value:
                if conflap.name == 'Flap':
                    self.create_kpv(index, value, flap=conflap_setting)
                else:
                    self.create_kpv(index, value, conf=conflap_setting)


################################################################################
# Acceleration

class AccelerationLateralAtTouchdown(KeyPointValueNode):
    '''
    Programmed at Goodyear office as a demonstration.
    '''
    def derive(self, acc=P('Acceleration Lateral Offset Removed'),
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            self.create_kpv(*bump(acc, tdwn))
        
            
class AccelerationLateralDuringLanding(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral)."
    '''
    
    def derive(self, acc=P('Acceleration Lateral Offset Removed'), 
               land_rolls=S('Landing Roll'), rwy=A('FDR Landing Runway')):
        if ambiguous_runway(rwy):
            return
        self.create_kpv_from_slices(acc.array, land_rolls, max_abs_value)

    
class AccelerationLateralMax(KeyPointValueNode):
    @classmethod
    def can_operate(cls, available):
        '''
        This KPV has no inherent flight phase associated with it, but we can
        reasonably say that we are not interested in anything while the
        aircraft is stationary.
        '''
        return 'Acceleration Lateral Offset Removed' in available
    
    def derive(self, acc_lat=P('Acceleration Lateral Offset Removed'),
               gspd=P('Groundspeed')):
        if gspd:
            self.create_kpvs_within_slices(
                acc_lat.array, gspd.slices_above(5), max_abs_value)
        else:
            index, value = max_value(acc_lat.array)
            self.create_kpv(index, value)
    

class AccelerationLateralOffset(KeyPointValueNode):
    """
    This KPV computes the lateral accelerometer datum offset, as for
    AccelerationNormalOffset. The more complex slicing statement ensures we
    only accumulate error estimates when taxiing in a straight line.
    """
    def derive(self, acc=P('Acceleration Lateral'), 
               taxis=S('Taxiing'), turns=S('Turning On Ground')):
        total_sum = 0.0
        total_count = 0
        straights = slices_and([s.slice for s in list(taxis)],
                               slices_not([s.slice for s in list(turns)]))
        for straight in straights:
            unmasked_data = np.ma.compressed(acc.array[straight])
            count = len(unmasked_data)
            if count:
                total_count += count
                total_sum += np.sum(unmasked_data)
        if total_count>20:
            delta = total_sum/float(total_count)
            if abs(delta) < ACCEL_LAT_OFFSET_LIMIT:
                self.create_kpv(0, delta)
    
    
class AccelerationLateralTakeoffMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Lateral)"
    '''
    def derive(self, acc_lat=P('Acceleration Lateral Offset Removed'), 
               to_rolls=S('Takeoff Roll')):
        self.create_kpvs_within_slices(acc_lat.array, to_rolls, max_abs_value)


class AccelerationLateralTaxiingStraightMax(KeyPointValueNode):
    '''
    Lateral acceleration while not turning is rarely an issue, so we compute
    only one KPV for taxi out and one for taxi in. The straight sections are
    identified by masking the turning phases and then testing the resulting
    data.
    '''
    def derive(self, acc_lat=P('Acceleration Lateral Offset Removed'),
               taxis=S('Taxiing'), turns=S('Turning On Ground')):
        accel = np.ma.copy(acc_lat.array) # Prepare to change mask here.
        for turn in turns:
            accel[turn.slice] = np.ma.masked
        self.create_kpv_from_slices(accel, taxis, max_abs_value)
    

class AccelerationLateralTaxiingTurnsMax(KeyPointValueNode):
    '''
    Lateral acceleration while taxiing normally occurs in turns, and leads to
    wear on the undercarriage and discomfort for passengers. In extremis this
    can lead to taxiway excursions. Lateral acceleration is used in
    preference to groundspeed as this parameter is available on older
    aircraft and is directly related to comfort.
    '''
    def derive(self, acc_lat=P('Acceleration Lateral Offset Removed'), 
               turns=S('Turning On Ground')):
        self.create_kpvs_within_slices(acc_lat.array, turns, max_abs_value)


class AccelerationLongitudinalPeakTakeoff(KeyPointValueNode):
    '''
    This may be of interest where takeoff performance is an issue, though not
    normally monitored as a safety event.
    '''
    def derive(self, accel=P('Acceleration Longitudinal'),
               takeoff=S('Takeoff')):
        self.create_kpv_from_slices(accel.array, takeoff, max_value)


class AccelerationLongitudinalPeakLanding(KeyPointValueNode):
    '''
    This is an indication of severe braking and/or use of reverse thrust or
    reverse pitch.
    '''
    def derive(self, accel=P('Acceleration Longitudinal'),
               landing=S('Landing')):
        self.create_kpv_from_slices(accel.array, landing, max_value)

        
class AccelerationNormal20FtToFlareMax(KeyPointValueNode):
    def derive(self, acc_normal=P('Acceleration Normal Offset Removed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(acc_normal.array,
                                       alt_aal.slices_from_to(20, 5),
                                       max_value)

        
class AccelerationNormalAirborneFlapsUpMax(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'), airborne=S('Airborne')):
        # Mask data where the flaps are down
        acc_flap_up = np.ma.masked_where(flap.array>0.0, accel.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalAirborneFlapsUpMin(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'), airborne=S('Airborne')):
        # Mask data where the flaps are down
        acc_flap_up = np.ma.masked_where(flap.array>0.0, accel.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, min_value)


class AccelerationNormalAirborneFlapsDownMax(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'), airborne=S('Airborne')):
        # Mask data where the flaps are up
        acc_flap_up = np.ma.masked_where(flap.array==0.0, accel.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalAirborneFlapsDownMin(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'), airborne=S('Airborne')):
        # Mask data where the flaps are up
        acc_flap_up = np.ma.masked_where(flap.array==0.0, accel.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, min_value)


class AccelerationNormalAtLiftoff(KeyPointValueNode):
    '''
    This is a measure of the normal acceleration at the point of liftoff, and
    is related to the pitch rate at takeoff.
    '''
    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               lifts=KTI('Liftoff')):
        for lift in lifts:
            self.create_kpv(*bump(acc, lift))


class AccelerationNormalAtTouchdown(KeyPointValueNode):
    '''
    This is the peak acceleration at landing, often used to identify hard
    landings for maintenance purposes.
    '''
    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            self.create_kpv(*bump(acc, tdwn))


class AccelerationNormalLiftoffTo35FtMax(KeyPointValueNode):
    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(acc.array, takeoffs, max_value)


class AccelerationNormalMax(KeyPointValueNode):
    
    def derive(self, acc_norm=P('Acceleration Normal Offset Removed'),
               moves=S('Mobile')):
        '''
        This KPV has no inherent flight phase associated with it, but we can
        reasonably say that we are not interested in anything while the
        aircraft is stationary.
        '''        
        self.create_kpv_from_slices(acc_norm.array, moves, max_value)


class AccelerationNormalOffset(KeyPointValueNode):
    """
    This KPV computes the normal accelerometer datum offset. This allows for
    offsets that are sometimes found in these sensors which remain in service
    although outside the permitted accuracy of the signal.
    """
    def derive(self, acc=P('Acceleration Normal'), taxis=S('Taxiing')):
        total_sum = 0.0
        total_count = 0
        for taxi in taxis:
            unmasked_data = np.ma.compressed(acc.array[taxi.slice])
            count = len(unmasked_data)
            if count:
                total_count += count
                total_sum += np.sum(unmasked_data)
        if total_count>20:
            delta = total_sum/float(total_count) - 1.0
            if abs(delta) < ACCEL_NORM_OFFSET_LIMIT:
                self.create_kpv(0, delta + 1.0)


################################################################################
# Airspeed


########################################
# Airspeed: General


class AirspeedMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), airborne=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            airborne,
            max_value,
        )


# FIXME: Rename class to 'AirspeedFor3SecMax' to keep with naming convention!
class AirspeedMax3Sec(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), airborne=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(
            clip(airspeed.array, 3.0, airspeed.hz),
            airborne,
            max_value,
        )


class AirspeedCruiseMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), cruises=S('Cruise')):
        '''
        '''
        self.create_kpv_from_slices(airspeed.array, cruises, max_value)


class AirspeedCruiseMin(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), cruises=S('Cruise')):
        '''
        '''
        self.create_kpv_from_slices(airspeed.array, cruises, min_value)


class AirspeedGustsDuringFinalApproach(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    Excursions - Landing (Lateral). Gusts during flare/final approach. This
    is tricky. Try Speed variation >15kt 30RA to 10RA. KPV looks at peak to
    peak values to get change in airspeed. Event uses interpolated RALT
    samples and looks at the airspeed samples that fall between RALT = 30ft
    and 10ft. DW suggested that the airspeed samples should also be
    interpolated in order to be able to estimate airspeed as to close to the
    ends of the RALT range as possible.
    '''
    def derive(self, aspd=P('Airspeed'), gspd=P('Groundspeed'),
               alt_rad=P('Altitude Radio'), airs=S('Airborne')):
        _, fin_apps = slices_from_to(alt_rad.array, 30, 10)
        descents = slices_and([s.slice for s in airs], fin_apps)
        for descent in descents:
            # Ensure we encompass the range of interest.
            scope = slice(descent.start-5, descent.stop+5)
            # We'd like to use groundspeed to compute the wind gust, but
            # variations in airspeed are a suitable backstop.
            if gspd:
                speed = aspd.array[scope]-gspd.array[scope]
            else:
                speed = aspd.array[scope]-aspd.array[scope][0]
            
            # Precise indexing is used as this is only a short segment. Note
            # that the _idx values are floating point interpolations of the
            # radio altimeter signal, and the speed array is also
            # interpolated.
            start_idx = index_at_value(alt_rad.array, 30.0, scope)
            stop_idx = index_at_value(alt_rad.array, 10.0, scope)
            new_app = shift_slice(descent, -scope.start)
            peak = max_value(speed, new_app, 
                             start_edge=start_idx-scope.start, 
                             stop_edge=stop_idx-scope.start)
            trough = min_value(speed, new_app,
                               start_edge=start_idx-scope.start, 
                               stop_edge=stop_idx-scope.start)
            if peak.value and trough.value:
                value = peak.value - trough.value
                index = ((peak.index + trough.index) / 2.0) + scope.start
                self.create_kpv(index, value)


########################################
# Airspeed: Climbing


class AirspeedAtLiftoff(KeyPointValueNode):
    '''
    A 'Tailwind At Liftoff' KPV would complement this KPV when used for 'Speed
    high at takeoff' events.
    '''

    def derive(self, airspeed=P('Airspeed'), liftoffs=KTI('Liftoff')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, liftoffs)


# FIXME: Do we want to try and lose the 'In Takeoff' suffix?
class AirspeedAt35FtInTakeoff(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), takeoff=S('Takeoff')):
        '''
        '''
        first_takeoff = takeoff.get_first()
        if first_takeoff:
            # NOTE: stop_edge is the precise endpoint of the takeoff at 35ft,
            #       not rounded to the nearest integer index.
            index = first_takeoff.stop_edge
            self.create_kpv(index, value_at_index(airspeed.array, index))


class Airspeed35To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class Airspeed35To1000FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


class Airspeed1000To8000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(1000, 8000),
            max_value,
        )


class Airspeed8000To10000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(8000, 10000),
            max_value,
        )


########################################
# Airspeed: Descending


class Airspeed10000To8000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(10000, 8000),
            max_value,
        )


class Airspeed8000To5000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(8000, 5000),
            max_value,
        )


class Airspeed5000To3000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(5000, 3000),
            max_value,
        )


class Airspeed3000To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpv_from_slices(
            airspeed.array,
            alt_aal.slices_from_to(3000, 1000),
            max_value,
        )


class Airspeed1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class Airspeed1000To500FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class Airspeed500To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class Airspeed500To20FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            airspeed.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedAtTouchdown(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), touchdowns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, touchdowns)


class AirspeedTrueAtTouchdown(KeyPointValueNode):
    '''
    This KPV relates to groundspeed at touchdown to illustrate headwinds and
    tailwinds. We also have 'Tailwind 100 Ft To Touchdown Max' to cater for
    safety event triggers.
    '''

    def derive(self, airspeed=P('Airspeed True'), touchdowns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, touchdowns)


########################################
# Airspeed: Minus V2


class AirspeedMinusV2AtLiftoff(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 At Liftoff'

    def derive(self, spd_v2=P('Airspeed Minus V2'), liftoffs=KTI('Liftoff')):
        '''
        '''
        self.create_kpvs_at_ktis(spd_v2.array, liftoffs)


class AirspeedMinusV2At35Ft(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 At 35 Ft'

    def derive(self, spd_v2=P('Airspeed Minus V2'), takeoffs=S('Takeoff')):
        '''
        '''
        for takeoff in takeoffs:
            index = takeoff.stop_edge  # Takeoff ends at 35ft!
            value = spd_v2.array[index]
            self.create_kpv(index, value)


class AirspeedMinusV235To1000FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Max'

    def derive(self, spd_v2=P('Airspeed Minus V2'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class AirspeedMinusV235To1000FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Min'

    def derive(self, spd_v2=P('Airspeed Minus V2'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


########################################
# Airspeed: Relative


class AirspeedRelativeAtTouchdown(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               touchdowns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(spd_rel.array, touchdowns)


class AirspeedRelative1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelative1000To500FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelative500To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelative500To20FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelative20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            max_value,
        )


class AirspeedRelative20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            min_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            max_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            min_value,
        )


class AirspeedRelativeFor3Sec500To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelativeFor3Sec500To20FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelativeFor5Sec1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelativeFor5Sec1000To500FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelativeFor5Sec500To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelativeFor5Sec500To20FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelativeFor5Sec20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            max_value,
        )


class AirspeedRelativeFor5Sec20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    def derive(self, spd_rel=P('Airspeed Relative For 5 Sec'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(20, 0),
            min_value,
        )


########################################
# Airspeed: (Other)


class AirspeedVacatingRunway(KeyPointValueNode):
    '''
    Airspeed vacating runway uses true airspeed, which is extended below the
    minimum range of the indicated airspeed specifically for this type of
    event. See the derived parameter for details of how groundspeed or
    acceleration data is used to cover the landing phase.
    '''

    def derive(self, airspeed=P('Airspeed True'),
            off_rwys=KTI('Landing Turn Off Runway')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, off_rwys)


class AirspeedRTOMax(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed RTO Max'

    def derive(self, airspeed=P('Airspeed'), rtos=S('Rejected Takeoff')):
        '''
        '''
        self.create_kpvs_within_slices(airspeed.array, rtos, max_value)


class Airspeed10000ToLandMax(KeyPointValueNode):
    '''
    Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.
        
    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 10,000ft boundary.
    '''
    
    name = 'Airspeed Below 10000 Ft In Descent Max'

    def derive(self, airspeed=P('Airspeed'),
            alt_std=P('Altitude STD Smoothed'),
            alt_qnh=P('Altitude QNH'),
            destination=A('FDR Landing Airport'), 
            descent=S('Descent')):
        '''
        '''
        country = None
        if destination.value:
            country = destination.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        alt = hysteresis(alt, HYSTERESIS_FPALT)

        height_bands = np.ma.clump_unmasked(np.ma.masked_greater(alt, 10000))
        descent_bands = slices_and(height_bands, descent.get_slices())
        self.create_kpvs_within_slices(airspeed.array, descent_bands, max_value)


class AirspeedTODTo10000Max(KeyPointValueNode):
    '''
    Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.
        
    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 10,000ft boundary.
    '''

    name = 'Airspeed Top Of Descent To 10000 Ft Max'

    def derive(self, airspeed=P('Airspeed'),
            alt_std=P('Altitude STD Smoothed'),
            alt_qnh=P('Altitude QNH'),
            destination=A('FDR Landing Airport'),
            descent=S('Descent')):
        '''
        '''
        country = None
        if destination.value:
            country = destination.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        alt = hysteresis(alt, HYSTERESIS_FPALT)

        height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt), 10000))
        descent_bands = slices_and(height_bands, descent.get_slices())
        self.create_kpvs_within_slices(airspeed.array, descent_bands, max_value)


class AirspeedBetween90SecToTouchdownAndTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            sec_to_touchdown=KTI('Secs To Touchdown')):
        '''
        '''
        for _90_sec in sec_to_touchdown.get(name='90 Secs To Touchdown'):
            tdwn = _90_sec.index + 90 * self.frequency
            index, value = max_value(airspeed.array, slice(_90_sec.index, tdwn))
            self.create_kpv(index, value)


class AirspeedLevelFlightMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), level_flight=S('Level Flight')):
        '''
        '''
        for sect in level_flight:
            # TODO: Move LEVEL_FLIGHT_MIN_DURATION to LevelFlight
            #       FlightPhaseNode so that only stable level flights are
            #       reported.
            duration = (sect.slice.stop - sect.slice.start) / self.frequency
            if duration > LEVEL_FLIGHT_MIN_DURATION:
                # We're in stable level flight...
                index, value = max_value(airspeed.array, sect.slice)
                self.create_kpv(index, value)
            else:
                self.debug('Uanble to create KPV: Level flight duration too short')


class AirspeedBelowAltitudeMax(KeyPointValueNode):
    '''
    '''

    NAME_FORMAT = 'Airspeed Below %(altitude)d Ft Max'
    NAME_VALUES = {'altitude': [10000, 8000, 5000, 3000]}
    
    def derive(self, airspeed=P('Airspeed'), 
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for altitude in self.NAME_VALUES['altitude']:
            self.create_kpv_from_slices(
                airspeed.array,
                alt_aal.slices_between(0, altitude),
                max_value,
                altitude=altitude,
            )

################################################################################
# Angle of Attack

class AOAWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control. Pitch/Angle of Attack vs stall angles"
    
    This is an adaptation of the airspeed algorithm, used to determine peak
    AOA vs flap. It may not be possible to obtain stalling angle of attack
    figures to set event thresholds, but a threshold based on in-service data
    may suffice.
    '''

    NAME_FORMAT = 'AOA With Flap %(flap)d Max'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), aoa=P('AOA'), scope=S('Fast')):
        '''
        '''
        # Fast scope traps flap changes very late on the approach and raising
        # flaps before 80kn on the landing run.
        self.flap_or_conf_max_or_min(flap, aoa, max_value, 
                                     scope=scope, include_zero=True)


################################################################################
# Airspeed With Flap

# NOTE: It is essential that Flap is the first parameter here to prevent the
#       flap values, which match the detent settings, from being interpolated.

class AirspeedWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d Max'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Fast')):
        '''
        '''
        # Fast scope traps flap changes very late on the approach and raising
        # flaps before 80kn on the landing run.
        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d Min'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Airborne')):
        '''
        '''
        # Airborne scope avoids deceleration on the runway "corrupting" the
        # minimum airspeed with landing flap.
        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapClimbMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d In Climb Max'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Climb')):
        '''
        '''
        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapClimbMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d In Climb Min'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Climb')):
        '''
        '''
        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapDescentMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d In Descent Max'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Descent')):
        '''
        '''
        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d In Descent Min'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Descent To Flare')):
        '''
        '''
        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedRelativeWithFlapDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed Relative With Flap %(flap)d In Descent Min'
    NAME_VALUES = NAME_VALUES_FLAP

    def derive(self, flap=P('Flap'), airspeed=P('Airspeed Relative'), scope=S('Descent To Flare')):
        '''
        '''
        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


################################################################################
# Thrust Reversers


def thrust_reversers_working(land, pwr, tr):
    '''
    Thrust reversers are deployed and average N1 over 65%.
    '''
    high_power = np.ma.clump_unmasked(np.ma.masked_less(pwr.array[land.slice],
                                                        65.0))
    return clump_multistate(tr.array, 'Deployed', high_power)


class AirspeedThrustReverseDeployedMin(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed With Thrust Reversers Deployed (Over 65% N1) Min'

    def derive(self, speed=P('Airspeed True'), tr=M('Thrust Reversers'),
               pwr=P('Eng (*) N1 Avg'), lands=S('Landing')):
        '''
        '''
        for land in lands:
            high_rev = thrust_reversers_working(land, pwr, tr)
            self.create_kpvs_within_slices(speed.array, high_rev, min_value)


class GroundspeedThrustReverseDeployedMin(KeyPointValueNode):
    '''
    '''

    name = 'Groundspeed With Thrust Reversers Deployed (Over 65% N1) Min'

    def derive(self, speed=P('Groundspeed'), tr=P('Thrust Reversers'),
               pwr=P('Eng (*) N1 Max'), lands=S('Landing')):
        '''
        '''
        for land in lands:
            high_rev = thrust_reversers_working(land, pwr, tr)
            self.create_kpvs_within_slices(speed.array, high_rev, min_value)


class AirspeedThrustReverseSelected(KeyPointValueNode):
    '''
    This gives the indicated airspeed where the thrust reversers were selected.
    '''
    def derive(self, speed=P('Airspeed'), tr=M('Thrust Reversers'), lands=S('Landing')):
        to_scan = clump_multistate(tr.array, 'Stowed', 
                                   [s.slice for s in lands],
                                   condition=False)
        self.create_kpv_from_slices(speed.array, to_scan, max_value)


class ThrustAsymmetryWithThrustReverse(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Asymmetric reverse thrust."
    '''
    def derive(self, ta=P('Thrust Asymmetry'), tr=M('Thrust Reversers'), 
               lands=S('Landing')):
        to_scan = clump_multistate(tr.array, 'Stowed', 
                                   [s.slice for s in lands],
                                   condition=False)
        self.create_kpv_from_slices(ta.array, to_scan, max_value)


class ThrustWithThrustReverseInTransit(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Asymmetric selection or achieved."
    '''
    def derive(self, pwr=P('Eng (*) N1 Avg'), tr=M('Thrust Reversers'), 
               lands=S('Landing')):
        to_scan = clump_multistate(tr.array, 'In Transit', 
                                   [s.slice for s in lands])
        self.create_kpv_from_slices(pwr.array, to_scan, max_value)


class TouchdownToThrustReverseDeployedDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Reverse thrust delay - time delay. 
    Selection more than 3sec after main wheel t/d."
    
    Note: 3 second threshold may be applied to derive an event from this KPV.
    '''
    def derive(self, tr=M('Thrust Reversers'),
               lands = S('Landing'), tdwns=KTI('Touchdown')):
        for land in lands:
            deploys = clump_multistate(tr.array, 'Deployed', land.slice)
            if deploys == []:
                continue
            deploy = deploys[0].start # Only interested in first opening of reversers on this landing.
            for tdwn in tdwns:
                if not is_index_within_slice(tdwn.index, land.slice):
                    continue
                self.create_kpv(deploy, (deploy-tdwn.index)/tr.hz)


class TouchdownToSpoilersDeployedDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Late spoiler deployment - time delay".
    '''
    def derive(self, brake=M('Speedbrake Selected'),
               lands = S('Landing'), tdwns=KTI('Touchdown')):
        '''
        '''
        deploys = find_edges_on_state_change('Deployed/Cmd Up', brake.array, phase=lands)
        for land in lands:
            for deploy in deploys:
                if not is_index_within_slice(deploy, land.slice):
                    continue
                for tdwn in tdwns:
                    if not is_index_within_slice(tdwn.index, land.slice):
                        continue
                    self.create_kpv(deploy, (deploy-tdwn.index)/brake.hz)


################################################################################
# Takeoff and Use of TOGA

class GroundspeedAtTOGA(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Selection of TOGA late in take-off
    roll."
    
    This KPV measures the groundspeed at the point of TOGA selection,
    irrespective of whether this is late (or early!).
    
    [Note: Takeoff phase is used as this includes turning onto the runway
    whereas Takeoff Roll only starts after the aircraft is accelerating.]
    '''
    
    def derive(self, gspd=P('Groundspeed'), toga=M('Takeoff And Go Around'),
               takeoff=S('Takeoff')):
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            speed = value_at_index(gspd.array, index) # interpolates as required
            self.create_kpv(index, speed)


class TOGASelectedInFlightNotGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control - Unexpected TOGA power selection in flight (except for
    a go-around)"
    '''
    def derive(self, toga=M('Takeoff And Go Around'), gas=S('Go Around And Climbout'),
               airs=S('Airborne')):

        to_scan=slices_and([s.slice for s in airs],
                           slices_not([s.slice for s in gas], 
                                       begin_at=airs[0].slice.start, 
                                       end_at=airs[-1].slice.stop))

        # The elegant create_kpvs_where_state function requires the phase
        # information as a section object, hence a couple of lines of glue.
        # TODO: Make create_kpvs_where_state accept list of slices.
        not_ga=S()
        not_ga.create_sections(to_scan, 'Airborne Not Go Around')
        
        self.create_kpvs_where_state('TOGA', toga.array, toga.hz, 
                                     phase=not_ga, exclude_leading_edge=True)
               
                           

class LiftoffToClimbPitchDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Slow climb out after rotation and
    slow rotation."
    
    This KPV originally used a threshold of 12.5 deg nose up, as suggested by
    the CAA, however it was found that some corporate operators do not
    achieve this attitude, so a lower threshold of 10deg was adopted.
    
    An endpoint of a minute after liftoff was added to avoid triggering well
    after the period of interest, and a pre-liftoff extension included for
    cases which rotate quickly and reach 10deg before liftoff !
    '''
    
    def derive(self, pitch=P('Pitch'),lifts=KTI('Liftoff')):
        for lift in lifts:
            pitch_up_idx = index_at_value(pitch.array, 10.0, 
                                          _slice=slice(lift.index-5*pitch.hz, 
                                                       lift.index+60.0*pitch.hz))
            if pitch_up_idx:
                duration = (pitch_up_idx - lift.index)/pitch.hz
                self.create_kpv(pitch_up_idx, duration)


################################################################################
# Landing Gear


########################################
# 'Gear Down' Multistate


class AirspeedWithGearDownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), gear=M('Gear Down'),
               airs=S('Airborne')):
        '''
        '''
        state = gear.array.state['Up']
        for air in airs:
            air_start = int(air.slice.start or 0)
            downs = np.ma.masked_equal(gear.array.raw[air.slice], state)
            downs = np.ma.clump_unmasked(downs)
            for down in downs:
                chunk = airspeed.array[air.slice][down]
                index = np.ma.argmax(chunk)
                value = chunk[index]
                self.create_kpv(air_start + down.start + index, value)


class MachWithGearDownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, mach=P('Mach'), gear=M('Gear Down'), airs=S('Airborne')):
        '''
        '''
        state = gear.array.state['Up']
        for air in airs:
            air_start = int(air.slice.start or 0)
            downs = np.ma.masked_equal(gear.array.raw[air.slice], state)
            downs = np.ma.clump_unmasked(downs)
            for down in downs:
                chunk = mach.array[air.slice][down]
                index = np.ma.argmax(chunk)
                value = chunk[index]
                self.create_kpv(air_start + down.start + index, value)


########################################
# Gear Retracting/Extending Section


class AirspeedAsGearRetractingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), gear_ret=S('Gear Retracting')):
        '''
        '''
        self.create_kpvs_within_slices(airspeed.array, gear_ret, max_value)


class AirspeedAsGearExtendingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'), gear_ext=S('Gear Extending')):
        '''
        '''
        self.create_kpvs_within_slices(airspeed.array, gear_ext, max_value)


class MachAsGearRetractingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, mach=P('Mach'), gear_ret=S('Gear Retracting')):
        '''
        '''
        self.create_kpvs_within_slices(mach.array, gear_ret, max_value)


class MachAsGearExtendingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, mach=P('Mach'), gear_ext=S('Gear Extending')):
        '''
        '''
        self.create_kpvs_within_slices(mach.array, gear_ext, max_value)


########################################
# Gear Up/Down Selection KTI


class AirspeedAtGearUpSelection(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            gear_up_sel=KTI('Gear Up Selection')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, gear_up_sel)


class AirspeedAtGearDownSelection(KeyPointValueNode):
    '''
    '''

    def derive(self, airspeed=P('Airspeed'),
            gear_dn_sel=KTI('Gear Down Selection')):
        '''
        '''
        self.create_kpvs_at_ktis(airspeed.array, gear_dn_sel)


##################################
# Braking


class BrakePressureInTakeoffRollMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral)". Primary Brake pressure during ground
    roll. Could also be applicable to longitudinal excursions on take-off.
    This is to capture scenarios where the brake is accidentally used when
    using the rudder (dragging toes on pedals)."
    '''
    def derive(self, bp=P('Brake Pressure'), rolls=S('Takeoff Roll')):
        self.create_kpvs_within_slices(bp.array, rolls, max_value)
        

# XXX: Can minus_60 fall outside end of landing slice? Fix if needed.
class DelayedBrakingAfterTouchdown(KeyPointValueNode):
    '''
    This parameter was requested by one customer, who asked us to adopt the
    Airbus AFPS implementation.
    '''
    def derive(self,
               lands=S('Landing'),
               gs=P('Groundspeed'),
               tdwns=KTI('Touchdown')):
        '''
        '''
        for land in lands:
            for tdwn in tdwns:
                if not is_index_within_slice(tdwn.index, land.slice):
                    continue
                gs_td = value_at_index(gs.array, tdwn.index)
                if gs_td is None:
                    continue
                minus_10 = index_at_value(gs.array, gs_td - 10.0, land.slice)
                minus_60 = index_at_value(gs.array, gs_td - 60.0, land.slice)
                if minus_10 is None or minus_60 is None:
                    continue
                dt = (minus_60 - minus_10) / gs.frequency
                self.create_kpv((minus_10 + minus_60) / 2.0, dt)

    
################################################################################


class GenericDescent(KeyPointValueNode):
    '''
    '''

    NAME_FORMAT = '%(parameter)s At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {
        'parameter': ['Airspeed', 'Airspeed Relative', 'Vertical Speed',
            'Slope To Landing', 'Flap', 'Gear Down', 'Speedbrake',
            'ILS Glideslope', 'ILS Localizer', 'Power', 'Pitch', 'Roll',
            'Heading'],
    }
    NAME_VALUES.update(NAME_VALUES_DESCENT)

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        return 'Descent' in available and 'Altitude AAL' in available

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               slope=P('Slope To Landing'), flap=P('Flap'),
               glide=P('ILS Glideslope'),  airspeed=P('Airspeed'),
               vert_spd=P('Vertical Speed'), gear=M('Gear Down'),
               loc=P('ILS Localizer'),  power=P('Eng (*) N1 Avg'),
               pitch=P('Pitch'),  brake=M('Speedbrake Selected'),
               roll=P('Roll'),  head=P('Heading Continuous'), descent=S('Descent')):
        '''
        '''
        for this_descent in descent.get_slices():
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, _slice=this_descent)
                if index:
                    self.create_kpv(index, value_at_index(slope.array, index),
                        parameter='Slope To Landing', altitude=alt)
                    self.create_kpv(index, value_at_index(flap.array, index),
                        parameter='Flap', altitude=alt)
                    self.create_kpv(index, value_at_index(glide.array, index),
                        parameter='ILS Glideslope', altitude=alt)
                    self.create_kpv(index, value_at_index(airspeed.array,
                                                          index),
                        parameter='Airspeed', altitude=alt)
                    self.create_kpv(index, value_at_index(vert_spd.array,
                                                          index),
                        parameter='Rate Of Descent', altitude=alt)
                    self.create_kpv(index, value_at_index(gear.array.raw,
                                                          index),
                        parameter='Gear Down', altitude=alt)
                    self.create_kpv(index, value_at_index(loc.array, index),
                        parameter='ILS Localizer', altitude=alt)
                    self.create_kpv(index, value_at_index(power.array, index),
                        parameter='Power', altitude=alt)
                    self.create_kpv(index, value_at_index(pitch.array, index),
                        parameter='Pitch', altitude=alt)
                    self.create_kpv(index, value_at_index(brake.array.raw,
                                                          index),
                        parameter='Speedbrake', altitude=alt)
                    self.create_kpv(index, value_at_index(roll.array, index),
                        parameter='Roll', altitude=alt)
                    self.create_kpv(index, value_at_index(head.array, index),
                        parameter='Heading', altitude=alt)


class AltitudeAtTouchdown(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(alt_std.array, touchdowns)



class AltitudeAtLiftoff(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(alt_std.array, liftoffs)


class AltitudeAtFirstFlapChangeAfterLiftoff(KeyPointValueNode):
    '''
    '''

    name = 'Altitude AAL At First Flap Change After Liftoff'

    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),
               airs=S('Airborne')):
        '''
        '''
        for air in airs:
            # Find where flap changes:
            change_indexes = np.ma.where(np.ma.diff(flap.array[air.slice]))[0]
            if len(change_indexes):
                # Create at first change:
                index = (air.slice.start or 0) + change_indexes[0]
                self.create_kpv(index, value_at_index(alt_aal.array, index))


class AltitudeAtLastFlapChangeBeforeLanding(KeyPointValueNode):
    '''
    '''

    name = 'Altitude AAL At Last Flap Change Before Landing'

    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),
               tdwns=KTI('Touchdown')):
        '''
        '''
        for tdwn in tdwns:
            land_flap = flap.array[tdwn.index]
            flap_move = abs(flap.array-land_flap)
            rough_index = index_at_value(flap_move, 0.5, slice(tdwn.index, 0,
                                                               -1))
            # index_at_value tries to be precise, but in this case we really
            # just want the index at the new flap setting.
            if rough_index:
                last_index = np.round(rough_index) 
                alt_last = value_at_index(alt_aal.array, last_index)
                self.create_kpv(last_index, alt_last)


class AltitudeAtGearUpSelection(KeyPointValueNode):
    '''
    '''

    name = 'Altitude AAL At Gear Up Selection'

    def derive(self, alt_aal=P('Altitude AAL'),
            gear_up_sel=KTI('Gear Up Selection')):
        '''
        Gear up selections after takeoff, not following a go-around (when it
        is normal to retract gear at significant height).
        '''
        self.create_kpvs_at_ktis(alt_aal.array, gear_up_sel)



class AltitudeAtGearDownSelection(KeyPointValueNode):
    '''
    '''

    name = 'Altitude AAL At Gear Down Selection'

    def derive(self, alt_aal=P('Altitude AAL'),
            gear_dn_sel=KTI('Gear Down Selection')):
        '''
        '''
        self.create_kpvs_at_ktis(alt_aal.array, gear_dn_sel)


class AltitudeAtMachMax(KeyPointValueNode):
    name = 'Altitude At Mach Max'
    def derive(self, alt_std=P('Altitude STD Smoothed'), max_mach=KPV('Mach Max')):
        # Aligns Altitude to Mach to ensure we have the most accurate
        # altitude reading at the point of Maximum Mach
        self.create_kpvs_at_kpvs(alt_std.array, max_mach)


class AltitudeWithFlapsMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), alt_std=P('Altitude STD Smoothed'),
               airs=S('Airborne')):
        '''
        The exceedance being detected here is the altitude reached with flaps
        not stowed, hence any flap value greater than zero is applicable and
        we're not really interested (for the purpose of identifying the
        event) what flap setting was reached.
        '''
        alt_flap = alt_std.array * np.ma.minimum(flap.array,1.0)
        self.create_kpvs_within_slices(alt_flap, airs, max_value)
       
       
class AltitudeFlapExtensionMax(KeyPointValueNode):
    name='Altitude AAL Flap Extension Max'
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),
               airs=S('Airborne')):
        # Restricted to avoid triggering on flap extension for takeoff.
        for air in airs:
            extends = find_edges(flap.array, air.slice)
            if extends:
                index=extends[0]
                value=alt_aal.array[index]
                self.create_kpv(index, value)

        
class AltitudeMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'), airs=S('Airborne')):
        self.create_kpvs_within_slices(alt_std.array, airs, max_value)


class AltitudeAutopilotEngaged(KeyPointValueNode):
    name = 'Altitude AAL AP Engaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'),
               ap_eng=KTI('AP Engaged Selection')):
        self.create_kpvs_at_ktis(alt_aal.array, ap_eng)
        
        
class AltitudeAutopilotDisengaged(KeyPointValueNode):
    name = 'Altitude AAL AP Disengaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'),
               ap_dis=KTI('AP Disengaged Selection')):
        self.create_kpvs_at_ktis(alt_aal.array, ap_dis)
        
        
class AltitudeAutothrottleEngaged(KeyPointValueNode):
    name = 'Altitude AAL AT Engaged In Flight'
    # Note: Autothrottle is normally engaged prior to takeoff, so will not trigger this event.
    def derive(self, alt_aal=P('Altitude AAL'),
               at_eng=KTI('AT Engaged Selection')):
        self.create_kpvs_at_ktis(alt_aal.array, at_eng)
        
        
class AltitudeAutothrottleDisengaged(KeyPointValueNode):
    name = 'Altitude AAL AT Disengaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'),
               at_dis=KTI('AT Disengaged Selection')):
        self.create_kpvs_at_ktis(alt_aal.array, at_dis)
        

class AutopilotOffInCruiseDuration(KeyPointValueNode):
    '''
    This monitors the duration for which all autopilot channels are
    disengaged in the cruise.
    '''
    def derive(self, autopilot=M('AP Engaged'), cruise=S('Cruise')):
        self.create_kpvs_where_state('-', autopilot.array, 
                                     autopilot.hz, phase=cruise)
               
class ControlColumnStiffness(KeyPointValueNode):
    """
    The control force and displacement of the flying controls should follow a
    predictable relationship. This parameter is included to identify
    stiffness in the controls in flight.
    """
    def derive(self,
               force=P('Control Column Force'),
               disp=P('Control Column'),
               fast=S('Fast')):
        # We only test during high speed operation to avoid "testing" the
        # full and free movements before flight.
        for speedy in fast:
            # We look for forces above a threshold to detect manual input.
            # This is better than looking for movement, as in the case of
            # stiff controls there is more force but less movement, hence
            # using a movement threshold will tend to be suppressed in the
            # cases we are looking to detect.
            push = force.array[speedy.slice]
            column = disp.array[speedy.slice]
            
            moves = np.ma.clump_unmasked(
                np.ma.masked_less(np.ma.abs(push),
                                  CONTROL_FORCE_THRESHOLD))
            for move in moves:
                if slice_samples(move) < 10:
                    continue
                corr, slope, off = \
                    coreg(push[move], indep_var=column[move], force_zero=True)
                if corr>0.85:  # This checks the data looks sound.
                    when = np.ma.argmax(np.ma.abs(push[move]))
                    self.create_kpv(
                        (speedy.slice.start or 0) + move.start + when, slope)

################################################################################
# Runway Distances at Takeoff

class DistanceFromLiftoffToRunwayEnd(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Runway remaining at rotation"
    '''
    
    units = 'm'
    
    def derive(self, lat_lift=KPV('Latitude At Liftoff'),
               lon_lift=KPV('Longitude At Liftoff'),
               rwy=A('FDR Takeoff Runway')):
        if ambiguous_runway(rwy) or not lat_lift:
            return
        toff_end = runway_distance_from_end(rwy.value, 
                                            lat_lift[0].value, 
                                            lon_lift[0].value)
        self.create_kpv(lat_lift[0].index, toff_end)


class DistanceFromRotationToRunwayEnd(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Runway remaining at rotation"
    '''
    
    def derive(self, lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):
          
        if ambiguous_runway(rwy):
            return
        for roll in toff_rolls:
            rot_idx = roll.stop_edge
            rot_end = runway_distance_from_end(rwy.value, 
                                                lat.array[rot_idx], 
                                                lon.array[rot_idx])
            self.create_kpv(rot_idx, rot_end)

class DecelerationToAbortTakeoffAtRotation(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Runway remaining at rotation"
    '''
    
    units = 'g'
    
    def derive(self, lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               gspd=P('Groundspeed'),
               aspd=P('Airspeed True'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):
          
        if ambiguous_runway(rwy):
            return
        if gspd:
            speed=gspd.array
        else:
            speed=aspd.array
        for roll in toff_rolls:
            rot_idx = roll.stop_edge 
            rot_end = runway_distance_from_end(rwy.value, 
                                               lat.array[rot_idx], 
                                               lon.array[rot_idx])
            
            lift_speed = value_at_index(speed, rot_idx) * KTS_TO_MPS
            mu = (lift_speed**2.0) / (2.0 * GRAVITY_METRIC * rot_end)
            self.create_kpv(rot_idx, mu)
            
            
################################################################################
# Runway Distances at Landing

class DistancePastGlideslopeAntennaToTouchdown(KeyPointValueNode):
    units = 'm'
    def derive(self, lat_tdn=KPV('Latitude At Touchdown'),
               lon_tdn=KPV('Longitude At Touchdown'),
               tdwns=KTI('Touchdown'),rwy=A('FDR Landing Runway'),
               ils_ldgs=S('ILS Localizer Established')):

        if ambiguous_runway(rwy) or not lat_tdn or not lon_tdn:
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        land_idx = last_tdwn.index
        # Check we did do an ILS approach (i.e. the ILS frequency was correct etc).
        if is_index_within_sections(land_idx, ils_ldgs):
            # OK, now do the geometry...
            gs = runway_distance_from_end(rwy.value, point='glideslope')
            td = runway_distance_from_end(rwy.value, lat_tdn.get_last().value,
                                          lon_tdn.get_last().value)
            if gs and td:
                distance = gs - td
                self.create_kpv(land_idx, distance)


class DistanceFromRunwayStartToTouchdown(KeyPointValueNode):
    '''
    Finds the distance from the runway start location to the touchdown point.
    This only operates for the last landing, and previous touch and goes will
    not be recorded.
    '''
    units = 'm'
    def derive(self, lat_tdn=KPV('Latitude At Touchdown'),
               lon_tdn=KPV('Longitude At Touchdown'),
               tdwns=KTI('Touchdown'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy) or not lat_tdn or not lon_tdn:
            return

        distance_to_start = runway_distance_from_end(rwy.value, point='start')
        distance_to_tdn = runway_distance_from_end(rwy.value,
                                                   lat_tdn.get_last().value,
                                                   lon_tdn.get_last().value)
        if distance_to_tdn < distance_to_start: # sanity check
            self.create_kpv(tdwns.get_last().index,
                            distance_to_start-distance_to_tdn)


class DistanceFromTouchdownToRunwayEnd(KeyPointValueNode):
    '''
    Finds the distance from the touchdown point to the end of the runway
    hardstanding. This only operates for the last landing, and previous touch
    and goes will not be recorded.
    '''
    units = 'm'
    def derive(self, lat_tdn=KPV('Latitude At Touchdown'),
               lon_tdn=KPV('Longitude At Touchdown'),
               tdwns=KTI('Touchdown'),
               rwy=A('FDR Landing Runway')):
        
        if ambiguous_runway(rwy) or not lat_tdn or not tdwns:
            return

        distance_to_tdn = runway_distance_from_end(rwy.value, 
                                                   lat_tdn.get_last().value, 
                                                   lon_tdn.get_last().value)
        self.create_kpv(tdwns.get_last().index, distance_to_tdn)
    

class DecelerationFromTouchdownToStopOnRunway(KeyPointValueNode):
    '''
    This determines the average level of deceleration required to stop the
    aircraft before reaching the end of the runway. It takes into account the
    length of the runway, the point of touchdown and the groundspeed at the
    point of touchdown.
    
    The numerical value is in units of g, and can be compared with surface
    conditions or autobrake settings. For example, if the value is 0.14 and
    the braking is "medium" (typically 0.1g) it is likely that the aircraft
    will overrun the runway if the pilot relies upon wheel brakes alone.
    
    The value will vary during the deceleration phase, but the highest value
    was found to arise at or very shortly after touchdown, as the aerodynamic
    and rolling drag at high speed normally exceed this level. Therefore for
    simplicity we just use the value at touchdown.
    '''
    def derive(self, gspd=P('Groundspeed'), tdwns=S('Touchdown'), landings=S('Landing'),
               lat_tdn=KPV('Latitude At Touchdown'),
               lon_tdn=KPV('Longitude At Touchdown'),
               rwy=A('FDR Landing Runway'),
               ils_gs_apps=S('ILS Glideslope Established'),
               ils_loc_apps=S('ILS Localizer Established'),
               precise=A('Precise Positioning')):
        if ambiguous_runway(rwy):
            return
        index = tdwns.get_last().index
        for landing in landings:
            if not is_index_within_slice(index, landing.slice):
                continue

            # Was this an ILS approach where the glideslope was captured?
            ils_approach = False
            for ils_loc_app in ils_loc_apps:
                if not slices_overlap(ils_loc_app.slice, landing.slice):
                    continue
                for ils_gs_app in ils_gs_apps:
                    if slices_overlap(ils_loc_app.slice, ils_gs_app.slice):
                        ils_approach = True

            # So for captured ILS approaches or aircraft with precision location we can compute the deceleration required.
            if (precise.value or ils_approach) and lat_tdn != []:
                distance_at_tdn = \
                    runway_distance_from_end(rwy.value, 
                                             lat_tdn.get_last().value, 
                                             lon_tdn.get_last().value)
                speed = value_at_index(gspd.array,index) * KTS_TO_MPS
                mu = (speed*speed) / (2.0 * GRAVITY_METRIC * (distance_at_tdn))
                self.create_kpv(index, mu)


class RunwayOverrunWithoutSlowingDuration(KeyPointValueNode):
    '''
    This determines the minimum time that the aircraft will take to reach the
    end of the runway without further braking. It takes into account the
    reducing groundspeed and distance to the end of the runway.
    
    The numerical value is in units of seconds.
    
    The value will decrease if the aircraft is not decelerating
    progressively. Therefore the lower values arise if the pilot allows the
    aircraft to roll down the runway without reducing speed. It will reflect
    the reduction in margins where aircraft roll at high speed towards
    taxiways near the end of the runway, and the value relates to the time
    available to the pilot.
    '''
    def derive(self, gspd=P('Groundspeed'), tdwns=S('Touchdown'),
               landings=S('Landing'), lat = P('Latitude Smoothed'),
               lon = P('Longitude Smoothed'),
               lat_tdn=KPV('Latitude At Touchdown'),
               lon_tdn=KPV('Longitude At Touchdown'),
               rwy=A('FDR Landing Runway'),
               ils_gs_apps=S('ILS Glideslope Established'),
               ils_loc_apps=S('ILS Localizer Established'),
               precise=A('Precise Positioning'),
               turnoff=KTI('Landing Turn Off Runway')):
        if ambiguous_runway(rwy):
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        for landing in landings:
            if not is_index_within_slice(last_tdwn.index, landing.slice):
                continue
            # Was this an ILS approach where the glideslope was captured?
            ils_approach = False
            for ils_loc_app in ils_loc_apps:
                if not slices_overlap(ils_loc_app.slice, landing.slice):
                    continue
                for ils_gs_app in ils_gs_apps:
                    if slices_overlap(ils_loc_app.slice, ils_gs_app.slice):
                        ils_approach = True
            # When did we turn off the runway?
            last_turnoff = turnoff.get_last()
            if not is_index_within_slice(last_turnoff.index, landing.slice):
                continue
            # So the period of interest is...
            land_roll = slice(last_tdwn.index, last_turnoff.index)
            # So for captured ILS approaches or aircraft with precision location we can compute the deceleration required.
            if precise.value or ils_approach:
                speed = gspd.array[land_roll] * KTS_TO_MPS
                if precise.value:
                    _, dist_to_end = bearings_and_distances(
                        lat.array[land_roll],
                        lon.array[land_roll],
                        rwy.value['end'])
                    time_to_end = dist_to_end / speed
                else:
                    distance_at_tdn = runway_distance_from_end(
                        rwy.value, lat_tdn.get_last().value,
                        lon_tdn.get_last().value)
                    dist_from_td = integrate(gspd.array[land_roll], 
                                             gspd.hz, scale=KTS_TO_MPS)
                    time_to_end = (distance_at_tdn - dist_from_td) / speed
                limit_point = np.ma.argmin(time_to_end)
                if limit_point < 0.0: # Some error conditions lead to rogue negative results.
                    continue
                limit_time = time_to_end[limit_point]
                self.create_kpv(limit_point + last_tdwn.index, limit_time)

    
class DistanceOnLandingFrom60KtToRunwayEnd(KeyPointValueNode):
    units = 'm'
    def derive(self, gspd=P('Groundspeed'),
               lat=P('Latitude Smoothed'),lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown'),rwy=A('FDR Landing Runway')):
        if ambiguous_runway(rwy):
            return
        last_tdwn = tdwns.get_last()
        if not last_tdwn:
            return
        land_idx = last_tdwn.index
        idx_60 = index_at_value(gspd.array, 60.0, slice(land_idx, None))
        if idx_60 and rwy.value and 'start' in rwy.value:
            # Only work out the distance if we have a reading at 60kts...
            distance = runway_distance_from_end(rwy.value,
                                                lat.array[idx_60],
                                                lon.array[idx_60])
            self.create_kpv(idx_60, distance) # Metres


class HeadingAtTakeoff(KeyPointValueNode):
    """
    We take the median heading, as this avoids problems with drift just
    after liftoff and turning onto the runway. The value is "assigned" to a
    time midway through the landing phase.
    """
    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff')):
        for toff in toffs:
            toff_head = np.ma.median(head.array[toff.slice])
            toff_index = (toff.slice.start + toff.slice.stop)/2.0
            self.create_kpv(toff_index, toff_head%360.0)


class HeadingAtLanding(KeyPointValueNode):
    """
    We take the median heading, as this avoids problems with drift just
    before touchdown and turning off the runway. The value is "assigned" to a
    time midway through the landing phase.
    """
    def derive(self, head=P('Heading Continuous'),
               lands=S('Landing')):
        for land in lands:
            # Check the landing slice is robust.
            if land.slice.start and land.slice.stop:
                land_head = np.ma.median(head.array[land.slice])
                land_index = (land.slice.start + land.slice.stop) / 2.0
                self.create_kpv(land_index, land_head % 360.0)


class HeadingAtLowestPointOnApproach(KeyPointValueNode):
    """
    The approach phase has been found already. Here we take the heading at
    the lowest point reached in the approach.
    """
    def derive(self, head=P('Heading Continuous'), 
               low_points=KTI('Lowest Point On Approach')):
        self.create_kpvs_at_ktis(head.array%360.0, low_points)


################################################################################
# Height Lost


# FIXME: Ensure that this uses .slices_from_takeoff_to(35)?
class HeightLostTakeoffTo35Ft(KeyPointValueNode):
    '''
    '''

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for climb in alt_aal.slices_from_to(0, 35):
            deltas = np.ma.ediff1d(alt_aal.array[climb], to_begin=0.0)
            downs = np.ma.masked_greater(deltas,0.0)
            index = np.ma.argmin(downs)
            drop = np.ma.sum(downs)
            if index:
                self.create_kpv(climb.start + index, drop)


class HeightLost35To1000Ft(KeyPointValueNode):
    '''
    '''

    def derive(self, ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for climb in alt_aal.slices_from_to(35, 1000):
            index = np.ma.argmin(ht_loss.array[climb])
            index += climb.start
            value = ht_loss.array[index]
            if value:
                self.create_kpv(index, value)


class HeightLost1000To2000Ft(KeyPointValueNode):
    '''
    '''

    def derive(self, ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for climb in alt_aal.slices_from_to(1000, 2000):
            index = np.ma.argmin(ht_loss.array[climb])
            index += climb.start
            value = ht_loss.array[index]
            if value:
                self.create_kpv(index, value)


################################################################################
# ILS


class ILSFrequencyOnApproach(KeyPointValueNode):
    '''
    Determine the ILS frequency on approach.

    The period when the aircraft was continuously established on the ILS and
    descending to the minimum point on the approach is already defined as a
    flight phase. This KPV just picks up the frequency tuned at that point.
    '''

    name = 'ILS Frequency On Approach'

    def derive(self,
               ils_frq=P('ILS Frequency'),
               loc_ests=S('ILS Localizer Established')):

        for loc_est in loc_ests:
            # Find the ILS frequency for the final period of operation of the
            # ILS during this approach. Note that median picks the value most
            # commonly recorded, so allows for some masked values and perhaps
            # one or two rogue values. If, however, all the ILS frequency data
            # is masked, no KPV is created.
            freq = np.ma.median(ils_frq.array[loc_est.slice])
            if freq:
                # Set the KPV index to the start of this ILS approach:
                self.create_kpv(loc_est.slice.start, freq)


class ILSGlideslopeDeviation1500To1000FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 1500 and 1000 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Glideslope Deviation 1500 To 1000 Ft Max'

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Glideslope Established')):

        alt_bands = alt_aal.slices_from_to(1500, 1000)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_glideslope.array,
            ils_bands,
            max_abs_value,
        )


class ILSGlideslopeDeviation1000To500FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 1000 and 500 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Glideslope Deviation 1000 To 500 Ft Max'

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Glideslope Established')):

        alt_bands = alt_aal.slices_from_to(1000, 500)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_glideslope.array,
            ils_bands,
            max_abs_value,
        )


class ILSGlideslopeDeviation500To200FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 500 and 200 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Glideslope Deviation 500 To 200 Ft Max'

    def derive(self,
               ils_glideslope=P('ILS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Glideslope Established')):

        alt_bands = alt_aal.slices_from_to(500, 200)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_glideslope.array,
            ils_bands,
            max_abs_value,
        )


class ILSLocalizerDeviation1500To1000FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 1500 and 1000 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Localizer Deviation 1500 To 1000 Ft Max'

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Localizer Established')):

        alt_bands = alt_aal.slices_from_to(1500, 1000)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_localizer.array,
            ils_bands,
            max_abs_value,
        )


class ILSLocalizerDeviation1000To500FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 1000 and 500 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Localizer Deviation 1000 To 500 Ft Max'

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Localizer Established')):

        alt_bands = alt_aal.slices_from_to(1000, 500)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_localizer.array,
            ils_bands,
            max_abs_value,
        )


class ILSLocalizerDeviation500To200FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the localizer between 500 and 200 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Localizer Deviation 500 To 200 Ft Max'

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Localizer Established')):

        alt_bands = alt_aal.slices_from_to(500, 200)
        ils_bands = slices_and(alt_bands, ils_ests.get_slices())
        self.create_kpvs_within_slices(
            ils_localizer.array,
            ils_bands,
            max_abs_value,
        )


class ILSLocalizerDeviationAtTouchdown(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    Excursions - Landing (Lateral) Lateral deviation at touchdown from
    Localiser Tricky to determine how close to runway edge using localiser
    parameter as there are variable runway widths and different localiser
    beam centreline error margins for different approach categories. ILS
    Localizer Deviation At Touchdown Measurements at <2 deg pitch after main
    gear TD."

    The ILS Established period may not last until touchdown, so it is
    artificially extended by a minute to ensure coverage of the touchdown
    instant.
    '''

    name = 'ILS Localizer Deviation At Touchdown'

    def derive(self,
               ils_localizer=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               tdwns=KTI('Touchdown')):

        for ils_est in ils_ests:
            for tdwn in tdwns:
                ext_end = ils_est.slice.stop + ils_localizer.frequency * 60.0
                ils_est_ext = slice(ils_est.slice.start, ext_end)
                if not is_index_within_slice(tdwn.index, ils_est_ext):
                    continue
                deviation = value_at_index(ils_localizer.array, tdwn.index)
                self.create_kpv(tdwn.index, deviation)


################################################################################


class IsolationValveOpenAtLiftoff(KeyPointValueNode):
    def derive(self, isol=P('Isolation Valve Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(isol.array, lifts, suppress_zeros=True)


class PackValvesOpenAtLiftoff(KeyPointValueNode):
    def derive(self, pack=M('Pack Valves Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(pack.array.raw, lifts, suppress_zeros=True)


################################################################################
# Latitude/Longitude


########################################
# Helpers


def calculate_runway_midpoint(rwy):
    '''
    Attempts to calculate the runway midpoint data provided in the AFR.

    1. If there are no runway start coordinates, use the runway end coordinates
    2. If there are no runway end coordinates, use the runway start coordinates
    3. Attempt to calculate the midpoint of the great circle path between them.
    '''
    rwy_s = rwy.get('start', {})
    rwy_e = rwy.get('end', {})
    lat_s = rwy_s.get('latitude')
    lat_e = rwy_e.get('latitude')
    lon_s = rwy_s.get('longitude')
    lon_e = rwy_e.get('longitude')
    if lat_s is None or lon_s is None:
        return (lat_e, lon_e)
    if lat_e is None or lon_e is None:
        return (lat_s, lon_s)
    return midpoint(lat_s, lon_s, lat_e, lon_e)


########################################
# Latitude/Longitude @ Takeoff/Landing


class LatitudeAtLanding(KeyPointValueNode):
    '''
    Latitude and Longitude at Landing and Touchdown.

    The position of the landing is recorded in the form of KPVs as this is
    used in a number of places. From the touchdown moments, the raw latitude
    and longitude data is used to create the *AtLanding parameters, and these
    are in turn used to compute the landing attributes.

    Once the landing attributes (especially the runway details) are known,
    the positional data can be smoothed using ILS data or (if this is a
    non-precision approach) the known touchdown aiming point. With more
    accurate positional data the touchdown point can be computed more
    accurately.

    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        one_of = lambda *names: any(name in available for name in names)
        return 'Touchdown' in available and any((
            'Latitude' in available,
            one_of('AFR Landing Runway', 'AFR Landing Airport'),
        ))

    def derive(self,
            lat=P('Latitude'),
            tdwns=KTI('Touchdown'),
            land_afr_apt=A('AFR Landing Airport'),
            land_afr_rwy=A('AFR Landing Runway')):
        '''
        '''
        # 1. Attempt to use latitude parameter if available:
        if lat:
            self.create_kpvs_at_ktis(lat.array, tdwns)
            return

        value = None

        # 2a. Attempt to use latitude of runway midpoint:
        if value is None and land_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(land_afr_rwy.value)
            value = lat_m

        # 2b. Attempt to use latitude of airport:
        if value is None and land_afr_apt:
            value = land_afr_apt.value.get('latitude')

        if value is not None:
            self.create_kpv(tdwns[-1].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a latitude at landing.')


class LongitudeAtLanding(KeyPointValueNode):
    '''
    Latitude and Longitude at Landing and Touchdown.

    The position of the landing is recorded in the form of KPVs as this is
    used in a number of places. From the touchdown moments, the raw latitude
    and longitude data is used to create the *AtLanding parameters, and these
    are in turn used to compute the landing attributes.

    Once the landing attributes (especially the runway details) are known,
    the positional data can be smoothed using ILS data or (if this is a
    non-precision approach) the known touchdown aiming point. With more
    accurate positional data the touchdown point can be computed more
    accurately.

    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        one_of = lambda *names: any(name in available for name in names)
        return 'Touchdown' in available and any((
            'Longitude' in available,
            one_of('AFR Landing Runway', 'AFR Landing Airport'),
        ))

    def derive(self,
            lon=P('Longitude'),
            tdwns=KTI('Touchdown'),
            land_afr_apt=A('AFR Landing Airport'),
            land_afr_rwy=A('AFR Landing Runway')):
        '''
        '''
        # 1. Attempt to use longitude parameter if available:
        if lon:
            self.create_kpvs_at_ktis(lon.array, tdwns)
            return

        value = None

        # 2a. Attempt to use longitude of runway midpoint:
        if value is None and land_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(land_afr_rwy.value)
            value = lon_m

        # 2b. Attempt to use longitude of airport:
        if value is None and land_afr_apt:
            value = land_afr_apt.value.get('longitude')

        if value is not None:
            self.create_kpv(tdwns[-1].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a longitude at landing.')


class LatitudeAtTakeoff(KeyPointValueNode):
    '''
    Latitude and Longitude at Takeoff and Liftoff.

    The position of the takeoff is recorded in the form of KPVs as this is
    used in a number of places. From the liftoff moments, the raw latitude
    and longitude data is used to create the *AtTakeoff parameters, and these
    are in turn used to compute the takeoff attributes.

    Once the takeoff attributes (especially the runway details) are known,
    the positional data can be smoothed the known liftoff point. With more
    accurate positional data the liftoff point can be computed more accurately.

    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        one_of = lambda *names: any(name in available for name in names)
        return 'Liftoff' in available and any((
            'Latitude' in available,
            one_of('AFR Takeoff Runway', 'AFR Takeoff Airport'),
        ))

    def derive(self,
            lat=P('Latitude'),
            liftoffs=KTI('Liftoff'),
            toff_afr_apt=A('AFR Takeoff Airport'),
            toff_afr_rwy=A('AFR Takeoff Runway')):
        '''
        '''
        # 1. Attempt to use latitude parameter if available:
        if lat:
            self.create_kpvs_at_ktis(lat.array, liftoffs)
            return

        value = None

        # 2a. Attempt to use latitude of runway midpoint:
        if value is None and toff_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(toff_afr_rwy.value)
            value = lat_m

        # 2b. Attempt to use latitude of airport:
        if value is None and toff_afr_apt:
            value = toff_afr_apt.value.get('latitude')

        if value is not None:
            self.create_kpv(liftoffs[0].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a latitude at takeoff.')


class LongitudeAtTakeoff(KeyPointValueNode):
    '''
    Latitude and Longitude at Takeoff and Liftoff.

    The position of the takeoff is recorded in the form of KPVs as this is
    used in a number of places. From the liftoff moments, the raw latitude
    and longitude data is used to create the *AtTakeoff parameters, and these
    are in turn used to compute the takeoff attributes.

    Once the takeoff attributes (especially the runway details) are known,
    the positional data can be smoothed the known liftoff point. With more
    accurate positional data the liftoff point can be computed more accurately.

    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        one_of = lambda *names: any(name in available for name in names)
        return 'Liftoff' in available and any((
            'Longitude' in available,
            one_of('AFR Takeoff Runway', 'AFR Takeoff Airport'),
        ))

    def derive(self,
            lon=P('Longitude'),
            liftoffs=KTI('Liftoff'),
            toff_afr_apt=A('AFR Takeoff Airport'),
            toff_afr_rwy=A('AFR Takeoff Runway')):
        '''
        '''
        # 1. Attempt to use longitude parameter if available:
        if lon:
            self.create_kpvs_at_ktis(lon.array, liftoffs)
            return

        value = None

        # 2a. Attempt to use longitude of runway midpoint:
        if value is None and toff_afr_rwy:
            lat_m, lon_m = calculate_runway_midpoint(toff_afr_rwy.value)
            value = lon_m

        # 2b. Attempt to use longitude of airport:
        if value is None and toff_afr_apt:
            value = toff_afr_apt.value.get('longitude')

        if value is not None:
            self.create_kpv(liftoffs[0].index, value)
            return

        # XXX: Is there something else we can do here other than fail?
        raise Exception('Unable to determine a longitude at takeoff.')


########################################
# Latitude/Longitude @ Liftoff/Touchdown


class LatitudeAtTouchdown(KeyPointValueNode):
    '''
    Latitude and Longitude at Landing and Touchdown.

    The position of the landing is recorded in the form of KPVs as this is
    used in a number of places. From the touchdown moments, the raw latitude
    and longitude data is used to create the *AtLanding parameters, and these
    are in turn used to compute the landing attributes.

    Once the landing attributes (especially the runway details) are known,
    the positional data can be smoothed using ILS data or (if this is a
    non-precision approach) the known touchdown aiming point. With more
    accurate positional data the touchdown point can be computed more
    accurately.
    '''

    def derive(self, lat=P('Latitude Smoothed'), tdwns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(lat.array, tdwns)


class LongitudeAtTouchdown(KeyPointValueNode):
    '''
    Latitude and Longitude at Landing and Touchdown.

    The position of the landing is recorded in the form of KPVs as this is
    used in a number of places. From the touchdown moments, the raw latitude
    and longitude data is used to create the *AtLanding parameters, and these
    are in turn used to compute the landing attributes.

    Once the landing attributes (especially the runway details) are known,
    the positional data can be smoothed using ILS data or (if this is a
    non-precision approach) the known touchdown aiming point. With more
    accurate positional data the touchdown point can be computed more
    accurately.
    '''

    def derive(self, lon=P('Longitude Smoothed'), tdwns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(lon.array, tdwns)


class LatitudeAtLiftoff(KeyPointValueNode):
    '''
    Latitude and Longitude at Takeoff and Liftoff.

    The position of the takeoff is recorded in the form of KPVs as this is
    used in a number of places. From the liftoff moments, the raw latitude
    and longitude data is used to create the *AtTakeoff parameters, and these
    are in turn used to compute the takeoff attributes.

    Once the takeoff attributes (especially the runway details) are known,
    the positional data can be smoothed the known liftoff point. With more
    accurate positional data the liftoff point can be computed more accurately.
    '''

    def derive(self, lat=P('Latitude Smoothed'), liftoffs=KTI('Liftoff')):
        '''
        '''
        self.create_kpvs_at_ktis(lat.array, liftoffs)


class LongitudeAtLiftoff(KeyPointValueNode):
    '''
    Latitude and Longitude at Takeoff and Liftoff.

    The position of the takeoff is recorded in the form of KPVs as this is
    used in a number of places. From the liftoff moments, the raw latitude
    and longitude data is used to create the *AtTakeoff parameters, and these
    are in turn used to compute the takeoff attributes.

    Once the takeoff attributes (especially the runway details) are known,
    the positional data can be smoothed the known liftoff point. With more
    accurate positional data the liftoff point can be computed more accurately.
    '''

    def derive(self, lon=P('Longitude Smoothed'), liftoffs=KTI('Liftoff')):
        '''
        '''
        self.create_kpvs_at_ktis(lon.array, liftoffs)


#########################################
# Latitude/Longitude @ Lowest Point on approach. Used to identify airport
# and runway, so that this works for both landings and aborted approaches /
# go-srounds.

class LatitudeAtLowestPointOnApproach(KeyPointValueNode):
    '''
    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    def derive(self, lat=P('Latitude Prepared'),
            low_points=KTI('Lowest Point On Approach')):
        '''
        '''
        self.create_kpvs_at_ktis(lat.array, low_points)


class LongitudeAtLowestPointOnApproach(KeyPointValueNode):
    '''
    Note: Cannot use smoothed position as this causes circular dependancy.
    '''

    def derive(self, lon=P('Longitude Prepared'),
            low_points=KTI('Lowest Point On Approach')):
        '''
        '''
        self.create_kpvs_at_ktis(lon.array, low_points)


################################################################################
# Mach


class MachMax(KeyPointValueNode):
    '''
    '''

    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(mach.array, airs, max_value)


# FIXME: Rename class to 'MachFor3SecMax' to keep with naming convention!
class MachMax3Sec(KeyPointValueNode):
    '''
    '''

    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(clip(mach.array, 3.0, mach.hz),
                                       airs, max_value)


################################################################################
# Magnetic Variation


class MagneticVariationAtTakeoff(KeyPointValueNode):
    '''
    '''

    def derive(self, var=P('Magnetic Variation'),
            toff=KTI('Takeoff Turn Onto Runway')):
        '''
        '''
        self.create_kpvs_at_ktis(var.array, toff)


class MagneticVariationAtLanding(KeyPointValueNode):
    '''
    '''

    def derive(self, var=P('Magnetic Variation'),
            land=KTI('Landing Turn Off Runway')):
        '''
        '''
        self.create_kpvs_at_ktis(var.array, land)


################################################################################
# Engine Bleed


# FIXME: Need to handle at least four engines here!
# Alignment should be resolved by align method, not use of integers.
class EngBleedValvesAtLiftoff(KeyPointValueNode):
    '''
    '''

    def derive(self, lifts=KTI('Liftoff'),
               b1=M('Eng (1) Bleed'), b2=M('Eng (2) Bleed')):
        '''
        '''
        # b1 & b2 are integer arrays, but to index them correctly we need to
        # align the KTI to match these arrays. The alignment will cause the
        # integer arrays to blur at transitions, so int(b1 + b2) is used to
        # remove this effect as the bleeds are changing state.
        bleeds = np.ma.array(b1.array + b2.array, dtype=int)
        for lift in lifts:
            valves = bleeds[lift.index]
            if valves:
                self.create_kpv(lift.index, valves)


################################################################################
# Engine EPR


# TODO: Write some unit tests!
class EngEPRToFL100Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR Up To FL100 Max'

    def derive(self, eng_epr_max=P('Eng (*) EPR Max'), 
               alt_std=P('Altitude STD Smoothed')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_epr_max.array,
            alt_std.slices_below(10000),
            max_value,
        )


# TODO: Write some unit tests!
class EngEPRAboveFL100Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR Above FL100 Max'

    def derive(self, eng_epr_max=P('Eng (*) EPR Max'), 
               alt_std=P('Altitude STD Smoothed')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_epr_max.array,
            alt_std.slices_above(10000),
            max_value,
        )


class EngEPR500FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR 500 Ft To Touchdown Min'

    def derive(self, eng_epr_min=P('Eng (*) EPR Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_epr_min.array,
            alt_aal.slices_from_to(500, 0),
            min_value,
        )


################################################################################
# Engine Gas Temperature


class EngGasTempTakeoffMax(KeyPointValueNode):
    '''
    '''

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempGoAroundMax(KeyPointValueNode):
    '''
    '''

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Go Around 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    We assume maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. So, by collecting all the high power
    periods and inverting these from the start of the first airborne section to
    the end of the last, we have the required periods of flight.
    '''

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               airs=S('Airborne')):
        '''
        '''
        if not airs:
            return
        high_power_ratings = to_ratings.get_slices() + ga_ratings.get_slices()
        max_cont_rating = slices_not(
            high_power_ratings,
            begin_at=min(air.slice.start for air in airs),
            end_at=max(air.slice.stop for air in airs),
        )
        self.create_kpvs_within_slices(eng_egt_max.array, max_cont_rating,
                                       max_value)


class EngGasTempStartMax(KeyPointValueNode):
    '''
    One key point value for maximum engine gas temperature at engine start for
    all engines. The value is taken from the engine with the largest value.
    '''

    @classmethod
    def can_operate(self, available):
        '''
        '''
        return all((
            any_of(('Eng (%d) Gas Temp' % n for n in range(1, 5)), available),
            any_of(('Eng (%d) N2' % n for n in range(1, 5)), available),
            'Takeoff Turn Onto Runway' in available,
        ))

    def peak_start_egt(self, egt, n2, idx):
        '''
        '''
        # We can't have started less than 20 seconds before takeoff:
        if idx < 20:
            return None, None
        # Ideally we'd use a shorter timebase, e.g. 2 seconds, but N2 is only
        # sampled at 1/4Hz on some aircraft:
        n2_rate = rate_of_change(n2, 4)
        # The engine only accelerates through 30% when starting. Let's find the
        # last time it did this before taking off:
        passing_30 = index_at_value(n2.array, 30.0, slice(idx, 0, -1))
        if not passing_30:
            return None, None
        # After which it will peak and the rate will fall below zero at the
        # overswing, which must happen within 30 seconds:
        started_up = index_at_value(n2_rate, 0.0,
                                    slice(passing_30, passing_30 + 30))
        # Track back to where the temperature started to increase:
        ignition = peak_curvature(egt.array, slice(passing_30, 0, -1))
        return ignition, started_up

    def derive(self,
               eng_1_egt=P('Eng (1) Gas Temp'),
               eng_2_egt=P('Eng (2) Gas Temp'),
               eng_3_egt=P('Eng (3) Gas Temp'),
               eng_4_egt=P('Eng (4) Gas Temp'),
               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2'),
               toff_turn_rwy=KTI('Takeoff Turn Onto Runway')):
        '''
        '''
        # We never see engine start if data started after aircraft is airborne:
        if not toff_turn_rwy:
            return
        # Extract the index for the first turn onto the runway:
        fto_idx = toff_turn_rwy.get_first().index
        # Determine the value of interest for each engine:
        data = []
        for n in range(1, 5):
            # Determine which engine parameters we are looking at:
            egt = locals().get('eng_%d_egt' % n)
            n2 = locals().get('eng_%d_n2' % n)
            # Skip this engine if missing any of the parameters:
            if not egt or not n2:
                continue
            # Determine the peak start engine gas temperature:
            ignition, started_up = self.peak_start_egt(egt, n2, fto_idx)
            if started_up:
                data.append(max_value(egt.array, slice(ignition, started_up)))
        # Create the KPV:
        data = filter(lambda t: None not in t, data)
        if data:
            self.create_kpv(*max(data, key=itemgetter(1)))


class EngGasTempInFlightMin(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control. In flight engine shut down."

    To detect a possible engine shutdown in flight, we look for the minimum
    gas temperature recorded during the flight. The event will then be computed
    later, testing against a suitable minimum value for a running engine.
    '''

    def derive(self,
               eng_temp_min=P('Eng (*) Gas Temp Min'),
               airs=S('Airborne')):
        '''
        '''
        for air in airs:
            index = np.ma.argmin(eng_temp_min.array[air.slice])
            index += air.slice.start
            value = eng_temp_min.array[index]
            self.create_kpv(index, value)


################################################################################
# Engine N1


class EngN1TaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Taxi Max'

    def derive(self, eng_n1_max=P('Eng (*) N1 Max'), taxi=S('Taxiing')):
        '''
        '''
        self.create_kpv_from_slices(eng_n1_max.array, taxi, max_value)


class EngN1TakeoffMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Takeoff Max'

    def derive(self, eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1GoAroundMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Go Around Max'

    def derive(self, eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Go Around 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1MaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Maximum Continuous Power Max'

    def derive(self, eng_n1_max=P('Eng (*) N1 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               gnd = S('Grounded')):
        '''
        '''
        ratings = to_ratings + ga_ratings + gnd
        self.create_kpv_outside_slices(eng_n1_max.array, ratings, max_value)


class EngN1CyclesInFinalApproach(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Cycles In Final Approach'

    def derive(self, eng_n1_avg=P('Eng (*) N1 Avg'), fapps=S('Final Approach')):
        '''
        '''
        for fapp in fapps:
            self.create_kpv(*cycle_counter(eng_n1_avg.array[fapp.slice], 5.0,
                                           10.0, eng_n1_avg.hz,
                                           fapp.slice.start))


# NOTE: Was named 'Eng N1 Cooldown Duration'.
# TODO: Similar KPV for duration between engine under 60 percent and engine shutdown
class Eng_N1MaxDurationUnder60PercentAfterTouchdown(KeyPointValueNode):
    '''
    Max duration N1 below 60% after Touchdown for engine cooldown. Using 60%
    allows for cooldown after use of Reverse Thrust.

    Evaluated for each engine to account for single engine taxi-in.

    Note: Assumes that all Engines are recorded at the same frequency.
    '''

    NAME_FORMAT = 'Eng (%(number)d) N1 Max Duration Under 60 Percent After Touchdown'
    NAME_VALUES = NAME_VALUES_ENGINE

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        return 'Touchdown' in available and 'Eng (*) Stop' in available and (
            'Eng (1) N1' in available or 'Eng (2) N1' in available or \
            'Eng (3) N1' in available or 'Eng (4) N1' in available)

    def derive(self, engines_stop=KTI('Eng (*) Stop'),
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1'),
               tdwn=KTI('Touchdown')):
        '''
        '''
        for eng_num, eng in enumerate((eng1, eng2, eng3, eng4), start=1):
            if eng is None:
                continue  # Engine is not available on this aircraft.
            eng_stop = engines_stop.get(name='Eng (%d) Stop' % eng_num)
            if not eng_stop:
                # XXX: Should we measure until the end of the flight anyway?
                # (Probably not.)
                self.debug('Engine %d did not stop on this flight, cannot '
                           'measure KPV', eng_num)
                continue
            last_tdwn_idx = tdwn.get_last().index
            last_eng_stop_idx = eng_stop[-1].index
            if last_tdwn_idx > last_eng_stop_idx:
                self.debug('Engine %d was stopped before last touchdown', eng_num)
                continue
            eng_array = repair_mask(eng.array)
            eng_below_60 = np.ma.masked_greater(eng_array, 60)
            # Measure duration between final touchdown and engine stop:
            touchdown_to_stop_slice = max_continuous_unmasked(
                eng_below_60, slice(last_tdwn_idx, last_eng_stop_idx))
            if touchdown_to_stop_slice:
                # TODO: Future storage of slice: self.slice = touchdown_to_stop_slice
                touchdown_to_stop_duration = (touchdown_to_stop_slice.stop - \
                                        touchdown_to_stop_slice.start) / self.hz
                self.create_kpv(touchdown_to_stop_slice.start,
                                touchdown_to_stop_duration, number=eng_num)
            else:
                # Create KPV of 0 seconds:
                self.create_kpv(last_eng_stop_idx, 0.0, number=eng_num)


class EngN1500To20FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 500 To 20 Ft Max'

    def derive(self, eng_n1_max=P('Eng (*) N1 Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_n1_max.array,
            alt_aal.slices_from_to(500, 20),
            max_value)


class EngN1500To20FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 500 To 20 Ft Min'

    def derive(self, eng_n1_min=P('Eng (*) N1 Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_n1_min.array,
            alt_aal.slices_from_to(500, 20),
            min_value)


################################################################################
# Engine N2


class EngN2TaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Taxi Max'

    def derive(self, eng_n2_max=P('Eng (*) N2 Max'), taxi=S('Taxiing')):
        '''
        '''
        self.create_kpv_from_slices(eng_n2_max.array, taxi, max_value)


class EngN2TakeoffMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Takeoff Max'

    def derive(self, eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2GoAroundMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Go Around Max'

    def derive(self, eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Go Around 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2MaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Maximum Continuous Power Max'

    def derive(self, eng_n2_max=P('Eng (*) N2 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               gnd = S('Grounded')):
        '''
        '''
        ratings = to_ratings + ga_ratings + gnd
        self.create_kpv_outside_slices(eng_n2_max.array, ratings, max_value)


class EngN2CyclesInFinalApproach(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Cycles In Final Approach'

    def derive(self, eng_n2_avg=P('Eng (*) N2 Avg'),
               fapps=S('Final Approach')):
        '''
        '''
        for fapp in fapps:
            self.create_kpv(*cycle_counter(eng_n2_avg.array[fapp.slice], 10.0,
                                           10.0, eng_n2_avg.hz,
                                           fapp.slice.start))


################################################################################
# Engine N3


class EngN3TaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 Taxi Max'

    def derive(self, eng_n3_max=P('Eng (*) N3 Max'), taxi=S('Taxiing')):
        '''
        '''
        self.create_kpv_from_slices(eng_n3_max.array, taxi, max_value)


class EngN3TakeoffMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 Takeoff Max'

    def derive(self, eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3GoAroundMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 Go Around Max'

    def derive(self, eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Go Around 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3MaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 Maximum Continuous Power Max'

    def derive(self, eng_n3_max=P('Eng (*) N3 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               gnd = S('Grounded')):
        '''
        '''
        ratings = to_ratings + ga_ratings + gnd
        self.create_kpv_outside_slices(eng_n3_max.array, ratings, max_value)


################################################################################
# Engine Throttles

class ThrustReductionOnLanding(KeyPointValueNode):
    '''
    This is a strange parameter with units of positive height and negative
    time, designed to suit legacy events.
    
    The original algorithm used reduction through 18deg throttle angle, but
    in cases where little power is being applied it was found that the
    throttle lever may not reach this setting. Also, this implies an
    aircraft-dependent threshold which would be difficult to maintain, and
    requires consistent throttle lever sensor rigging which may not be
    reliable on some types.
    
    For these reasons the algorithm has been adapted to use the peak
    curvature technique, scanning from 5 seconds before the start of the
    landing (passing 50ft) to the minimum throttle setting prior to
    application of reverse thrust.
    '''
    
    units=''
    
    def derive(self, alt=P('Altitude AAL'), tla=P('Throttle Levers'), 
               lands=S('Landing'), tdwns=KTI('Touchdown')):
        
        for land in lands:
            for tdwn in tdwns:
                if not is_index_within_slice(tdwn.index, land.slice):
                    continue
                
                # Seek the throttle lowpoint before thrust reverse is applied.
                retard_idx = index_at_value(tla.array, 0.0, 
                                            land.slice, 
                                            endpoint='closing')
                
                # the range of interest is therefore...
                scan=slice(land.slice.start - 5/alt.hz, 
                           retard_idx)

                # Now see where the power is reduced.
                reduce_idx = peak_curvature(tla.array, scan,
                                            curve_sense='Convex',
                                            gap=1,ttp=3)
                
                if reduce_idx:
                    dt = (reduce_idx - tdwn.index) / alt.hz
                
                    if dt<0:
                        # If before touchdown, measure the height at this moment
                        value = value_at_index(alt.array, reduce_idx)
                    else:
                        # If after, measure the time. Negative values allow the KPV to discriminate the two phases.
                        value = -dt
    
                    self.create_kpv(reduce_idx, value)


################################################################################
# Engine Oil Pressure



# TODO: Write some unit tests!
class EngOilPressMax(KeyPointValueNode):
    '''
    '''

    def derive(self, oil_press=P('Eng (*) Oil Press Max')):
        '''
        '''
        self.create_kpv(*max_value(oil_press.array))


# TODO: Write some unit tests!
class EngOilPressMin(KeyPointValueNode):
    '''
    '''

    def derive(self, oil_press=P('Eng (*) Oil Press Min'), airs=S('Airborne')):
        '''
        '''
        # Only check in flight to avoid zero pressure readings for stationary engines.
        for air in airs:
            self.create_kpv(*min_value(oil_press.array, air.slice))


################################################################################
# Engine Oil Quantity


class EngOilQtyMax(KeyPointValueNode):
    '''
    '''

    def derive(self, oil_qty=P('Eng (*) Oil Qty Max'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(oil_qty.array, airs, max_value)


class EngOilQtyMin(KeyPointValueNode):
    '''
    '''

    def derive(self, oil_qty=P('Eng (*) Oil Qty Min'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(oil_qty.array, airs, min_value)


################################################################################
# Engine Oil Temperature


class EngOilTempMax(KeyPointValueNode):
    '''
    '''

    def derive(self, oil_temp=P('Eng (*) Oil Temp Max'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(oil_temp.array, airs, max_value)


class EngOilTemp15MinuteMax(KeyPointValueNode):
    '''
    Maximum oil temperature sustained for 15 minutes.
    '''

    name = 'Eng Oil Temp 15 Minutes Max'

    def derive(self, oil_temp=P('Eng (*) Oil Temp Max')):
        '''
        Some aircraft don't have oil temp sensors fitted. This trap may be
        superceded by masking the Eng (*) Oil Temp Max parameter in future.
        '''
        if np.ma.count(oil_temp.array) == 0:
            return
        
        oil_15 = clip(oil_temp.array, 15 * 60, oil_temp.hz)
        # There have been cases where there were no valid oil temperature
        # measurements throughout the flight, in which case there's no point
        # testing for a maximum.
        if oil_15 is not None:
            self.create_kpv(*max_value(oil_15))


################################################################################
# Engine Torque


class EngTorqueTakeoffMax(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Takeoff 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_trq_max.array, ratings, max_value)


class EngTorqueGoAroundMax(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Go Around 5 Min Rating')):
        '''
        '''
        self.create_kpvs_within_slices(eng_trq_max.array, ratings, max_value)


class EngTorqueMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng Torque Maximum Continuous Power Max'

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               gnd = S('Grounded')):
        '''
        '''
        ratings = to_ratings + ga_ratings + gnd
        self.create_kpv_outside_slices(eng_trq_max.array, ratings, max_value)


# TODO: Write some unit tests!
class EngTorqueToFL100Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng Torque Up To FL100 Max'

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               alt_std=P('Altitude STD Smoothed')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_std.slices_below(10000),
            max_value,
        )


# TODO: Write some unit tests!
class EngTorqueAboveFL100Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng Torque Above FL100 Max'

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               alt_std=P('Altitude STD Smoothed')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_std.slices_above(10000),
            max_value,
        )


class EngTorqueAbove10000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_aal.slices_above(10000),
            max_value,
        )


class EngTorqueAbove10000FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_min=P('Eng (*) Torque Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_min.array,
            alt_aal.slices_above(10000),
            min_value,
        )


class EngTorque500FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_max=P('Eng (*) Torque Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_aal.slices_from_to(500, 0),
            max_value,
        )


class EngTorque500FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    def derive(self, eng_trq_min=P('Eng (*) Torque Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            eng_trq_min.array,
            alt_aal.slices_from_to(500, 0),
            min_value,
        )


################################################################################
# Engine Vibrations


class EngVibN1Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng Vib N1 Max'

    ####def derive(self, eng=P('Eng (*) Vib N1 Max'), fast=S('Fast')):
    ####    '''
    ####    '''
    ####    for sect in fast:
    ####        self.create_kpv(*max_value(eng.array, sect.slice))

    def derive(self, eng=P('Eng (*) Vib N1 Max'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(eng.array, airs, max_value)


class EngVibN2Max(KeyPointValueNode):
    name = 'Eng Vib N2 Max'

    ####def derive(self, eng=P('Eng (*) Vib N2 Max'), fast=S('Fast')):
    ####    '''
    ####    '''
    ####    for sect in fast:
    ####        self.create_kpv(*max_value(eng.array, sect.slice))

    def derive(self, eng=P('Eng (*) Vib N2 Max'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_within_slices(eng.array, airs, max_value)


################################################################################


class EventMarkerPressed(KeyPointValueNode):
    def derive(self, event=P('Event Marker'), airs=S('Airborne')):
        pushed = np.ma.clump_unmasked(np.ma.masked_equal(event.array, 0))
        events_in_air = slices_and(pushed, airs.get_slices())
        for event_in_air in events_in_air:        
            if event_in_air:
                duration = \
                    (event_in_air.stop - event_in_air.start) / event.frequency
                index = (event_in_air.stop + event_in_air.start) / 2.0
                self.create_kpv(index, duration)


class HeightOfBouncedLanding(KeyPointValueNode):
    '''
    This measures the peak height of the bounced landing
    '''
    def derive(self, alt = P('Altitude AAL'),
               bounced_landing=S('Bounced Landing')):
        self.create_kpvs_within_slices(alt.array, bounced_landing, max_value)
        

class HeadingDeviationOnTakeoffAbove80Kts(KeyPointValueNode):
    
    name = 'Heading Deviation From CL On Takeoff Above 80 Kts'
    
    """
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). Heading changes on runway before rotation
    commenced. During rotation on some types, the a/c may be allowed to
    weathercock into wind."

    The heading deviation is measured as the largest deviation from the
    runway centreline between 80kts airspeed and 5 deg nose pitch up, at
    which time the weight is clearly coming off the mainwheels (we avoid
    using weight on nosewheel as this is often not recorded).
    """
    def derive(self, head=P('Heading True Continuous'), airspeed=P('Airspeed'),
               pitch=P('Pitch'), toffs=S('Takeoff'), rwy=A('FDR Takeoff Runway')):
        
        if ambiguous_runway(rwy):
            return
        for toff in toffs:
            start = index_at_value(airspeed.array, 80.0, _slice=toff.slice)
            if not start:
                self.warning("'%s' did not transition through 80 kts in '%s' "
                             "slice '%s'.", airspeed.name, toffs.name,
                             toff.slice)
                continue
            stop = index_at_value(pitch.array, 5.0, _slice=toff.slice)
            if not stop:
                self.warning("'%s' did not transition through 5 deg in '%s' "
                             "slice '%s'.", pitch.name, toffs.name,
                             toff.slice)
                continue
            scan=slice(start, stop)
            dev=runway_deviation(head.array[scan], rwy.value)
            arg_max_dev = np.ma.argmax(np.ma.abs(dev))
            val_max_dev = dev[arg_max_dev]
            self.create_kpv(arg_max_dev+start, val_max_dev)


class HeadingDeviationAtTOGA(KeyPointValueNode):
    
    name = 'Heading Deviation From CL On Takeoff At TOGA'
    
    """
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). TOGA pressed before a/c aligned."
    """
    def derive(self, head=P('Heading True Continuous'), toga=M('Takeoff And Go Around'),
               takeoff=S('Takeoff'), rwy=A('FDR Takeoff Runway')):

        if ambiguous_runway(rwy):
            return

        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            brg=value_at_index(head.array, index)
            dev = runway_deviation(brg, rwy.value)
            self.create_kpv(index, dev)


class HeadingExcursion500To20Ft(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        for band in alt_aal.slices_from_to(500, 20):
            dev = np.ma.ptp(head.array[band])
            self.create_kpv(band.stop, dev)
            
            
class HeadingDeviationOnLandingAt50Ft(KeyPointValueNode):
    
    name = 'Heading Deviation From CL On Landing at 50 Ft'
    
    """
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). Crosswind. Could look at the difference
    between a/c heading and R/W heading at 50ft."
    """
    def derive(self, head=P('Heading True Continuous'), landings=S('Landing'), 
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy):
            return

        land=landings[-1] # Only have runway details for final landing.
        
        brg=value_at_index(head.array, land.start_edge) # By definition, landing starts at 50ft.
        dev = runway_deviation(brg, rwy.value)
        self.create_kpv(land.start_edge, dev)
        

class HeadingExcursionOnLandingAbove100Kts(KeyPointValueNode):
    """
    See heading excursion on takeoff comments. For landing the Altitude AAL
    is used to detect start of landing, again to avoid variation from the use
    of different aircraft recording configurations.
    """
    def derive(self, head=P('Heading Continuous'), airspeed=P('Airspeed'),
               alt=P('Altitude AAL For Flight Phases'), lands=S('Landing')):
        for land in lands:
            begin = index_at_value(alt.array, 1.0, _slice=land.slice)
            end = index_at_value(airspeed.array, 100.0, _slice=land.slice)
            if begin is None or begin > end:
                break # Corrupt landing slices or landed below 100kts. Can happen!
            else:
                head_dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv((begin+end)/2, head_dev)
            
            
class HeadingExcursionTouchdownPlus4SecTo60Kts(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'), tdwns=KTI('Touchdown'),
               airspeed=P('Airspeed')):
        for tdwn in tdwns:
            begin = tdwn.index + 4.0*head.frequency
            end = index_at_value(airspeed.array, 60.0, slice(begin,None))
            if end:
                # We found a suitable endpoint, so create a KPV...
                dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv(end, dev)


class HeadingDeviation2DegPitchTo60Kts(KeyPointValueNode):
    """
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Heading changes on runways."
    """
    
    name = 'Heading Deviation From CL On Landing 2 Deg Pitch To 60 Kts'

    def derive(self, head=P('Heading True Continuous'), land_rolls=S('Landing Roll'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy):
            return

        final_landing=land_rolls[-1].slice
        dev = runway_deviation(head.array, rwy.value)
        self.create_kpv_from_slices(dev, [final_landing], max_abs_value)


class HeadingVacatingRunway(KeyPointValueNode):
    '''
    Heading vacating runway is only used to try to identify handed
    runways in the absence of better information. See the
    flight_attribute Approaches and its _create_approach method.
    '''
    def derive(self, head=P('Heading Continuous'), 
               off_rwys=KTI('Landing Turn Off Runway')):
        # To save taking modulus of the entire array, we'll do this in stages.
        for off_rwy in off_rwys:
            # We try to extend the index by five seconds to make a clear
            # heading change. The KTI is at the point of turnoff at which
            # moment the heading change can be very small.
            index = min(off_rwy.index+5, len(head.array) - 1)
            value = head.array[index]%360.0
            self.create_kpv(index, value)
            

class AltitudeMinsToTouchdown(KeyPointValueNode):
    # TODO: TESTS
    # Q: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown
    NAME_FORMAT = "Altitude AAL " + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES
    
    def derive(self, alt_aal=P('Altitude AAL'),
               t_tdwns=KTI('Mins To Touchdown')):
        for t_tdwn in t_tdwns:
            # WARNING: This assumes Mins time will be the first value and only
            # two digit
            # TODO: Confirm *.array is correct (DJ)
            self.create_kpv(t_tdwn.index, alt_aal.array[t_tdwn.index],
                            time=int(t_tdwn.name[:2]))
            

class FlapAtGearDownSelection(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_sel_down=KTI('Gear Down Selection')):
        self.create_kpvs_at_ktis(flap.array, gear_sel_down)


class FlapWithGearUpMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear=M('Gear Down')):
        state = gear.array.state['Down']
        gear_up = np.ma.masked_equal(gear.array.raw, state)
        gear_up_slices = np.ma.clump_unmasked(gear_up)
        self.create_kpvs_within_slices(flap.array, gear_up_slices, max_value)


class FlapAtTouchdown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(flap.array, touchdowns)

   
class FlapAtLiftoff(KeyPointValueNode):
    def derive(self, flap=P('Flap'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(flap.array, liftoffs)
  

# TODO: Write some unit tests!
class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    '''
    '''

    def derive(self, flap=P('Flap'), speedbrake=M('Speedbrake Selected'),
               airs=S('Airborne'), lands=S('Landing')):
        '''
        '''
        # Mask all values where speedbrake isn't deployed:
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        array = np.ma.masked_where(speedbrake.array.raw != deployed, flap.array, copy=True)
        # Mask all values where the aircraft isn't airborne:
        array = mask_outside_slices(array, airs.get_slices())
        # Mask all values where the aircraft is landing (as we expect speedbrake to be deployed):
        array = mask_inside_slices(array, lands.get_slices())
        # Determine the maximum flap value when the speedbrake is deployed:
        index, value = max_value(array)
        # It is normal for flights to be flown without speedbrake and flap
        # together, so trap this case to avoid nuisance warnings:
        if index and value:
            self.create_kpv(index, value)


class FlareDuration20FtToTouchdown(KeyPointValueNode):
    #TODO: Tests
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               tdowns=KTI('Touchdown'), lands=S('Landing')):
        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)
            if this_landing:
                # Scan backwards from touchdown to the start of the landing
                # which is defined as 50ft, so will include passing through
                # 20ft AAL.
                idx_20 = index_at_value(alt_aal.array, 20.0,
                                        _slice=slice(tdown.index,
                                                     this_landing[0].start_edge,
                                                     -1))
                self.create_kpv(tdown.index,
                                (tdown.index - idx_20) / alt_aal.frequency)


class FlareDistance20FtToTouchdown(KeyPointValueNode):
    #TODO: Tests
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               tdowns=KTI('Touchdown'), lands=S('Landing'),
               gspd=P('Groundspeed')):
        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)
            if this_landing:
                idx_20 = index_at_value(
                    alt_aal.array, 20.0,
                    _slice=slice(tdown.index, this_landing[0].slice.start - 1, -1))
                # Integrate returns an array, so we need to take the max
                # value to yield the KTP value.
                if idx_20:
                    dist = max(integrate(gspd.array[idx_20:tdown.index],
                                         gspd.hz))
                    self.create_kpv(tdown.index, dist)


class AltitudeOvershootAtSuspectedLevelBust(KeyPointValueNode):
    '''
    FDS refined this KPV as part of the UK CAA Significant Seven programme.

    "Airborne Conflict (Mid-Air Collision) Level Busts (>300ft from an
    assigned level) It would be useful if this included overshoots of cleared
    level, i.e. a reversal of more than 300ft".
    '''
    def derive(self, alt_std=P('Altitude STD Smoothed')):
        bust = 300 # ft
        bust_time = 3 * 60 # 3 mins
        bust_length = bust_time * alt_std.frequency
        
        idxs, peaks = cycle_finder(alt_std.array, min_step=bust)

        if idxs is None:
            return
        for num, idx in enumerate(idxs[1:-1]):
            begin = index_at_value(np.ma.abs(alt_std.array - peaks[num + 1]),
                                   bust, _slice=slice(idx, None, -1))
            end = index_at_value(np.ma.abs(alt_std.array-peaks[num + 1]), bust,
                                 _slice=slice(idx, None))
            if begin and end:
                duration = (end - begin) / alt_std.frequency
                if duration < bust_time:
                    a=alt_std.array[idxs[num]] # One before the peak of interest
                    b=alt_std.array[idxs[num + 1]] # The peak of interest
                    # The next one (index reduced to avoid running beyond end of
                    # data)
                    c=alt_std.array[idxs[num + 2] - 1] 
                    idx_from = max(0, idxs[num + 1]-bust_length)
                    idx_to = min(len(alt_std.array), idxs[num + 1]+bust_length)
                    if b>(a+c)/2:
                        # Include a scan over the preceding and following
                        # bust_time in case the preceding or following peaks
                        # were outside this timespan.
                        alt_a = np.ma.min(alt_std.array[idx_from:idxs[num + 1]])
                        alt_c = np.ma.min(alt_std.array[idxs[num + 1]:idx_to])
                        overshoot = min(b-a, b-alt_a, b-alt_c, b-c)
                        if overshoot > 5000:
                            # This happens normally on short sectors
                            continue
                        self.create_kpv(idx, overshoot)
                    else:
                        alt_a = np.ma.max(alt_std.array[idx_from:idxs[num + 1]])
                        alt_c = np.ma.max(alt_std.array[idxs[num + 1]:idx_to])
                        undershoot = max(b-a, b-alt_a, b-alt_c, b-c)
                        self.create_kpv(idx, undershoot)


class FuelQtyAtLiftoff(KeyPointValueNode):
    def derive(self, fuel_qty=P('Fuel Qty'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(fuel_qty.array, liftoffs)


class FuelQtyAtTouchdown(KeyPointValueNode):
    def derive(self, fuel_qty=P('Fuel Qty'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(fuel_qty.array, touchdowns)


class GrossWeightAtLiftoff(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight Smoothed'), 
               liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(gross_weight.array, liftoffs)


class GrossWeightAtTouchdown(KeyPointValueNode):
    def derive(self, gross_weight=P('Gross Weight Smoothed'),
               touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(gross_weight.array, touchdowns)


class GroundspeedTaxiingStraightMax(KeyPointValueNode):
    '''
    Groundspeed while not turning is rarely an issue, so we compute only one
    KPV for taxi out and one for taxi in. The straight sections are
    identified by masking the turning phases and then testing the resulting
    data.
    '''
    def derive(self, gspeed=P('Groundspeed'), taxis=S('Taxiing'), 
            turns=S('Turning On Ground')):
        gspd = np.ma.copy(gspeed.array)  # Prepare to change mask.
        for turn in turns:
            gspd[turn.slice] = np.ma.masked
        self.create_kpv_from_slices(gspd, taxis, max_value)


class GroundspeedTaxiingTurnsMax(KeyPointValueNode):
    '''
    '''

    def derive(self, gspeed=P('Groundspeed'), taxis=S('Taxiing'),
            turns=S('Turning On Ground')):
        '''
        '''
        gspd = np.ma.copy(gspeed.array)  # Prepare to change mask.
        gspd = mask_outside_slices(gspd, turns.get_slices())
        self.create_kpvs_within_slices(gspd, taxis, max_value)

    
class GroundspeedRTOMax(KeyPointValueNode):
    name = 'Groundspeed RTO Max'
    def derive(self, gndspd=P('Groundspeed'),
               rejected_takeoffs=S('Rejected Takeoff')):
        self.create_kpvs_within_slices(gndspd.array, rejected_takeoffs,
                                       max_value)


class GroundspeedAtTouchdown(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(gspd.array, touchdowns)


class GroundspeedOnGroundMax(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), grounds=S('Grounded')):
        self.create_kpv_from_slices(gspd.array, grounds, max_value)


class GroundspeedVacatingRunway(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'),
               off_rwys=KTI('Landing Turn Off Runway')):
        self.create_kpvs_at_ktis(gspd.array, off_rwys)
        

################################################################################
# Pitch

class PitchMaxAfterFlapRetraction(KeyPointValueNode):
    '''
    FDS added this KPV during the UK CAA Significant Seven programme. "Loss
    of Control Pitch. FDS recommend addition of a maximum pitch attitude KPV,
    as this will make a good backstop to identify a number of events, such as
    control malfunctions, which from experience are often not detected by
    'normal' event algorithms. 
    
    Normal pitch maxima occur during takeoff and in some cases over 2,000ft
    but flap retraction is a good condition to apply to avoid these normal
    maxima.
    '''
    def derive(self, flap=P('Flap'), pitch=P('Pitch'), airs=S('Airborne')):
        scope=[]
        for air in airs:
            clean = np.ma.clump_unmasked(np.ma.masked_greater(flap.array[air.slice],0.0))
            if clean:
                scope.append(slice(air.slice.start+clean[0].start, air.slice.stop))
        self.create_kpvs_within_slices(pitch.array, scope, max_value)
        

class PitchAtLiftoff(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'), liftoffs=KTI('Liftoff')):
        '''
        '''
        self.create_kpvs_at_ktis(pitch.array, liftoffs)



class PitchAtTouchdown(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'), touchdowns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_at_ktis(pitch.array, touchdowns)


class PitchAt35FtInClimb(KeyPointValueNode):
    '''
    '''
    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL')):
        '''
        '''
        # Q: Should we create a KPV method for this?
        for climb in alt_aal.slices_from_to(1, 100):
            index = index_at_value(alt_aal.array, 35.0, climb)
            if index:
                value = value_at_index(pitch.array, index)
                self.create_kpv(index, value)


class PitchTakeoffTo35FtMax(KeyPointValueNode):
    '''
    '''
    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(0, 35),
            max_value,
        )


class Pitch35To400FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(35, 400),
            max_value,
        )


class Pitch35To400FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(35, 400),
            min_value,
        )


class Pitch400To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(400, 1000),
            max_value,
        )


class Pitch400To1000FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(400, 1000),
            min_value,
        )


class Pitch1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class Pitch1000To500FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class Pitch500To50FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class Pitch500To20FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class Pitch50FtToLandingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(50, 1),  # TODO: Implement .slices_to_landing_from(50)
            max_value,
        )


class Pitch20FtToLandingMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(20, 1),  # TODO: Implement .slices_to_landing_from(20)
            min_value,
        )


class Pitch7FtToLandingMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch=P('Pitch'), tdwns=KTI('Touchdown'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(7, tdwns),  # TODO: Implement .slices_to_landing_from(5)
            min_value,
        )


class PitchCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of pitch attitude that exceed 3 deg in
    pitch from peak to peak and with a maximum cycle period of 10 seconds
    during the final approach phase.
    '''

    def derive(self, pitch=P('Pitch'), fapps=S('Final Approach')):
        '''
        '''
        for fapp in fapps:
            self.create_kpv(*cycle_counter(pitch.array[fapp.slice], 3.0, 10.0,
                                           pitch.hz, fapp.slice.start))


################################################################################
# Pitch Rate


class PitchRate35To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class PitchRate20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_from_to(20, 0),
            max_value,
        )


class PitchRate20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    def derive(self, pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_from_to(20, 0),
            min_value,
        )


# TODO: Write some unit tests!
class PitchRate2DegPitchTo35FtMax(KeyPointValueNode):
    '''
    '''
    def derive(self, pitch_rate=P('Pitch Rate'),
               lifts=S('2 Deg Pitch To 35 Ft')):
        '''
        '''
        self.create_kpvs_within_slices(pitch_rate.array, lifts, max_value)


# TODO: Write some unit tests!
class PitchRate2DegPitchTo35FtMin(KeyPointValueNode):
    '''
    '''
    def derive(self, pitch_rate=P('Pitch Rate'),
               lifts=S('2 Deg Pitch To 35 Ft')):
        '''
        '''
        self.create_kpvs_within_slices(pitch_rate.array, lifts, min_value)


# TODO: Write some unit tests!
# TODO: Remove this KPV?  Not a dependency, not used in event definitions.
class PitchRate2DegPitchTo35FtAverage(KeyPointValueNode):
    '''
    '''
    def derive(self, pitch=P('Pitch'), lifts=S('2 Deg Pitch To 35 Ft')):
        '''
        '''
        for lift in lifts:
            pitch_max = value_at_index(pitch.array, lift.stop_edge)
            pitch_rate_avg = (pitch_max - 2.0)/ \
                ((lift.stop_edge - lift.start_edge) / pitch.frequency)
            mid_index = (lift.stop_edge + lift.start_edge) / 2.0
            self.create_kpv(mid_index, pitch_rate_avg)


# TODO: Write some unit tests!
# TODO: Remove this KPV?  Not a dependency, not used in event definitions.
# NOTE: Python class name restriction: '2 Deg Pitch To 35 Ft Duration'
class TwoDegPitchTo35FtDuration(KeyPointValueNode):
    '''
    '''
    name = '2 Deg Pitch To 35 Ft Duration'

    def derive(self, pitch=P('Pitch'), lifts=S('2 Deg Pitch To 35 Ft')):
        '''
        '''
        for lift in lifts:
            begin = lift.start_edge
            end = lift.stop_edge
            value = (end - begin) / pitch.frequency
            index = (begin + end) / 2.0
            self.create_kpv(index, value)


################################################################################
# Vertical Speed (Rate of Climb/Descent) Helpers


def vert_spd_phase_max_or_min(obj, vert_spd, phases, function):
    '''
    '''
    for phase in phases:
        duration = phase.slice.stop - phase.slice.start
        if duration > CLIMB_OR_DESCENT_MIN_DURATION:
            index, value = function(vert_spd.array, phase.slice)
            obj.create_kpv(index, value)


################################################################################
# Rate of Climb


# TODO: Write some unit tests!
class RateOfClimbMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'), climbing=S('Climbing')):
        '''
        In cases where the aircraft does not leave the ground, we get a
        descending phase that equates to an empty list, which is not
        iterable.

        '''
        vert_spd_phase_max_or_min(self, vert_spd, climbing, max_value)


class RateOfClimb35To1000FtMin(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )

class RateOfClimbBelow10000FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Airborne Conflict (Mid-Air Collision) Excessive rates of climb/descent
    (>3,000FPM) within a TMA (defined as < 10,000ft)"
    '''
    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(0, 10000),
            max_value,
        )


################################################################################
# Rate of Descent


# FIXME: Should rate of descent KPVs should occur for 3+ seconds?


# TODO: Write some unit tests!
class RateOfDescentMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'), descending=S('Descending')):
        '''
        In cases where the aircraft does not leave the ground, we get a
        descending phase that equates to an empty list, which is not
        iterable.
        '''
        vert_spd_phase_max_or_min(self, vert_spd, descending, min_value)


class RateOfDescentTopOfDescentTo10000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
            vert_spd=P('Vertical Speed'), descents=S('Descent')):
        '''
        '''
        for descent in descents:
            above_10k = np.ma.masked_less(alt_aal.array, 10000)
            drops = np.ma.clump_unmasked(above_10k)
            self.create_kpvs_within_slices(vert_spd.array, drops, min_value)


class RateOfDescentBelow10000FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Airborne Conflict (Mid-Air Collision) Excessive rates of climb/descent
    (>3,000FPM) within a TMA (defined as < 10,000ft)"
    '''
    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(10000, 0),
            min_value,
        )


class RateOfDescent10000To5000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(10000, 5000),
            min_value,
        )

class RateOfDescent5000To3000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(5000, 3000),
            min_value,
        )

class RateOfDescent3000To2000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(3000, 2000),
            min_value,
        )


class RateOfDescent2000To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(2000, 1000),
            min_value,
        )


class RateOfDescent1000To500FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class RateOfDescent500To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class RateOfDescent500FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    def derive(self, vert_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               tdwns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_to_kti(500, tdwns),
            min_value,
        )


class RateOfDescent20ToTouchdownMax(KeyPointValueNode):
    '''
    We use the inertial vertical speed to avoid ground effects this low to the
    runway.
    '''

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               vert_spd=P('Vertical Speed Inertial'), tdwns=KTI('Touchdown')):
        '''
        '''
        self.create_kpvs_within_slices(
            vert_spd.array,
            alt_aal.slices_to_kti(20, tdwns),
            min_value,
        )


class RateOfDescentAtTouchdown(KeyPointValueNode):
    '''
    We use the inertial vertical speed to avoid ground effects and give an
    accurate value at the point of touchdown.
    '''

    def derive(self, vert_spd=P('Vertical Speed Inertial'),
               lands = S('Landing'), alt = P('Altitude AAL')):
        '''
        '''
        for land in lands:
            index, rod = touchdown_inertial(land, vert_spd, alt)
            if index:
                self.create_kpv(index, rod)


##### TODO: Implement!
####class RateOfDescentOverGrossWeightLimitAtTouchdown(KeyPointValueNode):
####    '''
####    '''
####
####    def derive(self, x=P('Not Yet')):
####        '''
####        '''
####        return NotImplemented


################################################################################
# Roll


class RollTakeoffTo20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(1, 20),  # TODO: Implement .slices_from_takeoff_to(20)
            max_abs_value,
        )


class Roll20To400FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(20, 400),
            max_abs_value,
        )


class Roll400To1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(400, 1000),
            max_abs_value,
        )


class RollAbove1000FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_above(1000),
            max_abs_value,
        )


class Roll1000To300FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(1000, 300),
            max_abs_value,
        )


class Roll300To20FtMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(300, 20),
            max_abs_value,
        )


class Roll20FtToLandingMax(KeyPointValueNode):
    '''
    '''

    def derive(self, roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(20, 1),  # TODO: Implement .slices_to_landing_from(20)
            max_abs_value,
        )


class RollCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 5 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    final approach phase.
    
    The algorithm counts each half-cycle, so an "N" figure would give a value
    of 1.5 cycles.
    '''

    def derive(self, roll=P('Roll'), fapps=S('Final Approach')):
        '''
        '''
        for fapp in fapps:
            self.create_kpv(*cycle_counter(roll.array[fapp.slice], 5.0, 10.0,
                                           roll.hz, fapp.slice.start))


class RollCyclesNotInFinalApproach(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control - PIO. CAA limit > 20 deg total variation side to side".
    
    FDS cautioned 20 deg was excessive and evaluated different levels over 10
    sec time period with a view to settling the levels for production use.
    Having run a hundred sample flights using thresholds from 2 to 20 deg, 5
    deg was selected on the basis that this balanced enough data for trend
    analysis (a KPV was recorded for about one flight in three) without
    excessive counting of minor cycles. It was also convenient that this
    matched the existing threshold used by FDS for final approach analysis.

    Note: The algorithm counts each half-cycle, so an "N" figure would give a
    value of 1.5 cycles.
    '''

    def derive(self, roll=P('Roll'), airs=S('Airborne'), 
               fapps=S('Final Approach'), lands=S('Landing')):
        '''
        '''
        not_fas = slices_and_not(airs, fapps)
        # TODO: Fix this
        # not_fas = slices_and_not(not_fas, lands)
        for not_fa in not_fas:
            self.create_kpv(*cycle_counter(roll.array[not_fa], 5.0, 10.0,
                                           roll.hz, not_fa.start))


################################################################################
# Rudder

class RudderExcursionDuringTakeoff(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Lateral) Rudder kick/oscillations. Difficult due
    gusts and effect of buildings."    
    '''
    def derive(self, rudder=P('Rudder'), to_rolls=S('Takeoff Roll')):
        self.create_kpvs_within_slices(rudder.array, to_rolls, max_abs_value)


class RudderReversalAbove50Ft(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Rudder kick/oscillations Often there
    during landing, therefore need to determine what is abnormal, which may
    be difficult."

    Looks for sharp rudder reversal. Excludes operation below 50ft as this is
    normal use of the rudder to kick off drift. Uses the standard cycle
    counting process but looking for only one pair of half-cycles.
    '''

    def derive(self, rudder=P('Rudder'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        The threshold used to be 6.5 deg, derived from a manufacturer's
        document, but this did not provide meaningful results in routine
        operations, so the threshold was reduced to 2 deg over 2 seconds.
        '''
        for above_50 in alt_aal.slices_above(50.0):
            self.create_kpv(*cycle_counter(rudder.array[above_50], 2.0, 2.0,
                                           rudder.hz, above_50.start))


################################################################################
# Speedbrake


# TODO: Write some unit tests!
class SpeedbrakesDeployed1000To20FtDuration(KeyPointValueNode):
    '''
    '''

    def derive(self, speedbrake=M('Speedbrake Selected'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        for descent in alt_aal.slices_from_to(1000, 20):
            event = np.ma.masked_not_equal(speedbrake.array.raw[descent],
                                           deployed)
            value = np.ma.count(event) / speedbrake.frequency
            if value:
                index = descent.stop
                self.create_kpv(index, value)


# TODO: Write some unit tests!
class SpeedbrakesDeployedWithPowerOnDuration(KeyPointValueNode):
    '''
    Each time the aircraft is flown with high power and the speedbrakes open,
    something unusual is happening. We record the duration this happened for,
    and allow the analyst to find out the cause.

    The threshold for high power is 50% N1 for most aircraft, but 60% for
    Airbus types, to align with the Airbus AFPS.
    '''

    def derive(self, speedbrake=M('Speedbrake Selected'),
            power=P('Eng (*) N1 Avg'), airs=S('Airborne'),
            manufacturer=A('Manufacturer')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        speedbrake_in_flight = mask_outside_slices(speedbrake.array.raw,
                                                   airs.get_slices())
        speedbrakes_applied_in_flight = \
            np.ma.clump_unmasked(np.ma.masked_not_equal(speedbrake_in_flight,
                                                        deployed))
        percent = 60.0 if manufacturer == 'Airbus' else 50.0
        high_power = np.ma.clump_unmasked(np.ma.masked_less(power.array,
                                                            percent))
        # Speedbrake and Power => s_and_p
        s_and_ps = slices_and(speedbrakes_applied_in_flight, high_power)
        for s_and_p in s_and_ps:
            index = s_and_p.start + np.ma.argmax(power.array[s_and_p])
            value = (s_and_p.stop - s_and_p.start - 1) / speedbrake.hz
            if value:
                self.create_kpv(index, value)


# TODO: Write some unit tests!
class SpeedbrakesDeployedWithFlapDuration(KeyPointValueNode):
    '''
    '''

    def derive(self, speedbrake=M('Speedbrake Selected'), flap=P('Flap'),
            airs=S('Airborne')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        for air in airs:
            brakes = np.ma.clump_unmasked(np.ma.masked_not_equal(
                speedbrake.array.raw[air.slice], deployed))
            with_flap = np.ma.clump_unmasked(np.ma.masked_less(
                flap.array[air.slice], 0.5))
            # Speedbrake and Flap => s_and_f
            s_and_fs = slices_and(brakes, with_flap)
            for s_and_f in s_and_fs:
                index = s_and_f.start + (airs.get_first().slice.start or 0)
                value = (s_and_f.stop - s_and_f.start) / speedbrake.hz
                if value:
                    self.create_kpv(index, value)


# TODO: Write some unit tests!
class SpeedbrakesDeployedWithConfDuration(KeyPointValueNode):
    '''
    Conf used here, but not tried or tested. Presuming conf 2 / conf 3 should
    not be used with speedbrakes.
    '''

    def derive(self, speedbrake=M('Speedbrake Selected'),
               conf=P('Configuration')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        pos = np.ma.masked_where(speedbrake.array != deployed, conf.array,
                                 copy=True)
        pos = np.ma.masked_where(conf.array >= 2.0, pos)
        clumps = np.ma.clump_unmasked(pos)
        for clump in clumps:
            index = clump.start
            value = (clump.stop - clump.start) / speedbrake.hz
            if value:
                self.create_kpv(index, value)


# TODO: Write some unit tests!
class SpeedbrakesDeployedWithPowerOnInHeightBandsDuration(KeyPointValueNode):
    '''
    Specific to certain operators.
    '''

    NAME_FORMAT = 'Speedbrake Deployed With N1 Over %(eng_n1)d Between %(upper)d And %(lower)d Ft Duration'
    NAME_VALUES = {
        'eng_n1': [50, 60],
        'upper': [35000, 20000, 6000],
        'lower': [20000, 6000, 0],
    }

    def derive(self, speedbrake=M('Speedbrake Selected'),
               power=P('Eng (*) N1 Avg'),
               alt_aal=P('Altitude AAL For Flight Phases'), airs=S('Airborne')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        for eng_speed in self.NAME_VALUES['eng_n1']:
            for up in self.NAME_VALUES['upper']:
                for low in self.NAME_VALUES['lower']:
                    if up <= low:
                        break
                    speedbrake_in_band = \
                        mask_outside_slices(speedbrake.array.raw,
                                            alt_aal.slices_between(up, low))
                    speedbrakes_applied_in_flight = \
                        np.ma.clump_unmasked(
                            np.ma.masked_not_equal(speedbrake_in_band,
                                                   deployed))
                    high_power = \
                        np.ma.clump_unmasked(np.ma.masked_less(power.array,
                                                               eng_speed))
                    # Speedbrake and Power => s_and_p
                    s_and_ps = slices_and(speedbrakes_applied_in_flight,
                                          high_power)
                    for s_and_p in s_and_ps:
                        # Mark the point at highest power applied
                        index = s_and_p.start + \
                            np.ma.argmax(power.array[s_and_p])
                        value = \
                            (s_and_p.stop - s_and_p.start - 1) / speedbrake.hz
                        if value:
                            self.create_kpv(index, value, eng_n1=eng_speed,
                                            upper=up, lower=low)


################################################################################
# Warnings: Stick Pusher/Shaker


# TODO: Check that this triggers correctly as stick push events are probably
#       single samples.
class StickPusherActivatedDuration(KeyPointValueNode):
    '''
    We annotate the stick pusher event with the duration of the event.
    '''

    def derive(self, stick_pusher=M('Stick Pusher'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_where_state(
            'Push',
            stick_pusher.array,
            stick_pusher.hz,
            airs,
        )

        ##### TODO: Remove this old code?
        ####pushes = np.ma.clump_unmasked(
        ####    np.ma.masked_equal(stick_pusher.array, 0.0))
        ####for push in pushes:
        ####    index = push.start
        ####    value = (push.stop - push.start) / stick_pusher.hz
        ####    self.create_kpv(index, value)


class StickShakerActivatedDuration(KeyPointValueNode):
    '''
    We annotate the stick shaker event with the duration of the event.
    '''

    def derive(self, stick_shaker=M('Stick Shaker'), airs=S('Airborne')):
        '''
        '''
        self.create_kpvs_where_state(
            'Shake',
            stick_shaker.array,
            stick_shaker.hz,
            airs,
        )


################################################################################
# Tail Clearance


class TailClearanceOnTakeoffMin(KeyPointValueNode):
    '''
    '''

    def derive(self, alt_tail=P('Altitude Tail'), toffs=S('Takeoff')):
        '''
        '''
        self.create_kpvs_within_slices(alt_tail.array, toffs, min_value)


class TailClearanceOnLandingMin(KeyPointValueNode):
    '''
    '''

    def derive(self, alt_tail=P('Altitude Tail'), lands=S('Landing')):
        '''
        '''
        self.create_kpvs_within_slices(alt_tail.array, lands, min_value)


class TailClearanceOnApproach(KeyPointValueNode):
    '''
    This finds abnormally low tail clearance during the approach down to 100ft.
    It searches for the minimum angular separation between the flightpath and
    the terrain, so a 500ft clearance at 2500ft AAL is considered more
    significant than 500ft at 1500ft AAL. The value stored is the tail
    clearance. A matching KTI will allow these to be located on the approach
    chart.
    '''

    def derive(self,
            alt_aal=P('Altitude AAL'),
            alt_tail=P('Altitude Tail'),
            dtl=P('Distance To Landing')):
        '''
        '''
        for desc_slice in alt_aal.slices_from_to(3000, 100):
            angle_array = alt_tail.array[desc_slice] \
                / (dtl.array[desc_slice] * FEET_PER_NM)
            index, value = min_value(angle_array)
            if index:
                sample = index + desc_slice.start
                self.create_kpv(sample, alt_tail.array[sample])


class TerrainClearanceAbove3000FtMin(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Controlled Flight Into Terrain (CFIT) At/Below Minimum terrain clearance
    on approach/departure >3000ft AFE and <1000ft AGL"

    Solution: Compute minimum terrain clearance while Altitude AAL over
    3000ft. Note: For most flights, Altitude Radio will be over 2,500ft at
    this time, so masked, hence no kpv will be created.
    '''
    def derive(self, alt_rad=P('Altitude Radio'),
               alt_aal=P('Altitude AAL For Flight Phases'),):
        self.create_kpvs_within_slices(alt_rad.array,
                                       alt_aal.slices_above(3000.0),
                                       min_value)


################################################################################
# Tailwind

class TailwindLiftoffTo100FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Tailwind - Needs to be recorded
    just after take-off.
    
    CAA comment: Some operators will have purchased (AFM) a 15kt tailwind limit for
    take-off. But this should only be altered to 15 kt if it has been
    purchased.
    
    This event masks the tailwind array to that headwind conditions do not
    raise any KPV.
    '''

    def derive(self, tailwind=P('Tailwind'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(np.ma.masked_less_equal(tailwind.array, 0.0),
                                       alt_aal.slices_from_to(0, 100),
                                       max_value)


class Tailwind100FtToTouchdownMax(KeyPointValueNode):
    '''
    This event uses a masked tailwind array to that headwind conditions do
    not raise any KPV.
    '''

    def derive(self, tailwind=P('Tailwind'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(np.ma.masked_less_equal(tailwind.array, 0.0),
                                       alt_aal.slices_from_to(100, 0),
                                       max_value)


################################################################################
# Warnings: Takeoff Configuration Warning

class TakeoffConfigWarningDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Take-off config warning during
    takeoff roll."
    '''
    def derive (self, conf_warn=M('Takeoff Config Warning'),
                toff_rolls = S('Takeoff Roll')):
        
        self.create_kpvs_where_state(
            'Warning',
            conf_warn.array,
            conf_warn.hz,
            phase=toff_rolls
        )


class MasterWarningInTakeoffDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Master Caution or Master Warning
    triggered during takeoff. The idea of this is to inform the analyst of
    any possible distractions to the pilot"
    '''
    def derive (self, warn=M('Master Warning'), 
                toff_rolls = S('Takeoff Roll')):
        
        self.create_kpvs_where_state('Warning', warn.array, 
                                     warn.hz, phase=toff_rolls)


class MasterCautionInTakeoffDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Master Warning In Takeoff Duration".
    '''
    def derive (self, caution=M('Master Caution'), 
                toff_rolls=S('Takeoff Roll')):
        
        self.create_kpvs_where_state('Caution', caution.array, 
                                     caution.hz, phase=toff_rolls)


################################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TAWSGeneralDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS General Duration'

    def derive(self,
               taws_general=M('TAWS General'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_general.array,
            taws_general.hz,
            phase=airborne
        )


class TAWSAlertDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Alert Duration'

    def derive(self,
               taws_alert=M('TAWS Alert'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Alert',
            taws_alert.array,
            taws_alert.hz,
            phase=airborne
        )


class TAWSSinkRateWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Sink Rate Warning Duration'

    def derive(self,
               taws_sink_rate=M('TAWS Sink Rate'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_sink_rate.array,
            taws_sink_rate.hz,
            phase=airborne
        )


class TAWSTooLowFlapWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Too Low Flap Warning Duration'

    def derive(self,
               taws_too_low_flap=M('TAWS Too Low Flap'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_too_low_flap.array,
            taws_too_low_flap.hz,
            phase=airborne,
        )


class TAWSTerrainWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Terrain Warning Duration'

    def derive(self,
               taws_terrain=M('TAWS Terrain'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_terrain.array,
            taws_terrain.hz,
            phase=airborne
        )


class TAWSTerrainPullUpWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Terrain Pull Up Warning Duration'

    def derive(self,
               taws_terrain_pull_up=M('TAWS Terrain Ahead Pull Up'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_terrain_pull_up.array,
            taws_terrain_pull_up.hz,
            phase=airborne,
        )


class TAWSGlideslopeWarning1500To1000FtDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Glideslope Warning 1500 To 1000 Ft Duration'

    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_where_state(
            'Warning',
            taws_glideslope.array,
            taws_glideslope.hz,
            phase=alt_aal.slices_from_to(1500, 1000),
        )


class TAWSGlideslopeWarning1000To500FtDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Glideslope Warning 1000 To 500 Ft Duration'

    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_where_state(
            'Warning',
            taws_glideslope.array,
            taws_glideslope.hz,
            phase=alt_aal.slices_from_to(1000, 500),
        )


class TAWSGlideslopeWarning500To200FtDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Glideslope Warning 500 To 200 Ft Duration'

    def derive(self,
               taws_glideslope=M('TAWS Glideslope'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_where_state(
            'Warning',
            taws_glideslope.array,
            taws_glideslope.hz,
            phase=alt_aal.slices_from_to(500, 200),
        )


class TAWSTooLowTerrainWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Too Low Terrain Warning Duration'

    def derive(self,
               taws_too_low_terrain=M('TAWS Too Low Terrain'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_too_low_terrain.array,
            taws_too_low_terrain.hz,
            phase=airborne,
        )


class TAWSTooLowGearWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Too Low Gear Warning Duration'

    def derive(self,
               taws_too_low_gear=M('TAWS Too Low Gear'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_too_low_gear.array,
            taws_too_low_gear.hz,
            phase=airborne,
        )


class TAWSPullUpWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Pull Up Warning Duration'

    def derive(self,
               taws_pull_up=M('TAWS Pull Up'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_pull_up.array,
            taws_pull_up.hz,
            phase=airborne,
        )


class TAWSDontSinkWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Dont Sink Warning Duration'

    def derive(self,
               taws_dont_sink=M('TAWS Dont Sink'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Warning',
            taws_dont_sink.array,
            taws_dont_sink.hz,
            phase=airborne,
        )


class TAWSWindshearWarningBelow1500FtDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Windshear Warning Below 1500 Ft Duration'

    def derive(self,
               taws_windshear=M('TAWS Windshear Warning'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for descent in alt_aal.slices_from_to(1500, 0):
            self.create_kpvs_where_state(
                'Warning',
                taws_windshear.array[descent],
                taws_windshear.hz,
            )


################################################################################
# Warnings: Traffic Collision Avoidance System (TCAS)


class TCASRAWarningDuration(KeyPointValueNode):
    '''
    This is simply the number of seconds during which the TCAS RA was set.
    '''

    name = 'TCAS RA Warning Duration'

    def derive(self, tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        '''
        **Note:** We would like to do this but numpy can't handle text strings::

            ups = np.ma.masked_not_equal(tcas.array, 'Up Advisory Corrective')
            ups = np.ma.clump_unmasked(ups)
        
        hence the unweildy code below, used in all TCAS KPVs...
        
        ras_local = np.ma.clump_unmasked(np.ma.masked_outside(tcas.array, 4, 5))[air.slice]

        '''
        for air in airs:
            ras_local = np.ma.clump_unmasked(np.ma.masked_outside(tcas.array, 4, 5))[air.slice]
            self.create_kpvs_from_slice_durations(ras_local)


##### TODO: Implement!
####class TCASTAWarningDuration(KeyPointValueNode):
####    '''
####    '''
####
####    name = 'TCAS TA Warning Duration'
####
####    def derive(self, tcas=M('TCAS Combined Control'), airs=S('Airborne')):
####        '''
####        '''
####        pass


class TCASRAReactionDelay(KeyPointValueNode):
    '''
    '''

    name = 'TCAS RA Reaction Delay'

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
            tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        '''
        '''
        for air in airs:
            ras_local = np.ma.clump_unmasked(np.ma.masked_outside(tcas.array, 4, 5))[air.slice]
            ras = shift_slices(ras_local, air.slice.start)
            # Assume that the reaction takes place during the TCAS RA period:
            for ra in ras:
                if np.ma.count(acc.array[ra]) == 0:
                    continue
                i, p = cycle_finder(acc.array[ra] - 1.0, 0.15)
                # i, p will be None if the data is too short or invalid and so
                # no cycles can be found.
                if i is None:
                    continue
                indexes = np.array(i)
                peaks = np.array(p)
                # Look beyond 2 seconds to find slope from point of initiation.
                slopes = np.ma.where(indexes > 17, abs(peaks / indexes), 0.0)
                start_to_peak = slice(ra.start, ra.start + i[np.argmax(slopes)])
                react_index = peak_curvature(acc.array, _slice=start_to_peak,
                                             curve_sense='Bipolar') - ra.start
                self.create_kpv(ra.start + react_index,
                                react_index / acc.frequency)


class TCASRAInitialReaction(KeyPointValueNode):
    '''
    Here we calculate the strength of initial reaction, in terms of the rate of
    onset of g. When this is in the correct sense, it is positive while an
    initial movement in the wrong sense will be negative.
    '''

    name = 'TCAS RA Initial Reaction'

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
            tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        '''
        '''
        for air in airs:
            ras_local = np.ma.clump_unmasked(np.ma.masked_outside(tcas.array, 4, 5))[air.slice]
            ras = shift_slices(ras_local, air.slice.start)
            # We assume that the reaction takes place during the TCAS RA
            # period.
            for ra in ras:
                if np.ma.count(acc.array[ra]) == 0:
                    continue
                i, p = cycle_finder(acc.array[ra] - 1.0, 0.1)
                if i is None:
                    continue
                # Convert to Numpy arrays for ease of arithmetic
                indexes = np.array(i)
                peaks = np.array(p)
                slopes = np.ma.where(indexes > 17, abs(peaks / indexes), 0.0)
                s_max = np.argmax(slopes)

                # So we look for the steepest slope to the peak, which
                # ignores little early peaks or slightly high later peaks.
                # From inspection of many traces, this is the best way to
                # distinguish the peak of interest.
                if s_max == 0:
                    slope = peaks[0] / indexes[0]
                else:
                    slope = (peaks[s_max] - peaks[s_max - 1]) / \
                        (indexes[s_max] - indexes[s_max - 1])
                # Units of g/sec:
                slope *= acc.frequency

                if tcas.array[ra.start] == 5:
                    # Down advisory, so negative is good.
                    slope = -slope
                self.create_kpv(ra.start, slope)


class TCASRAToAPDisengageDuration(KeyPointValueNode):
    '''
    Here we calculate the time between the onset of the RA and disconnection of
    the autopilot.
    '''

    name = 'TCAS RA To AP Disengaged Duration'

    def derive(self, ap_offs=KTI('AP Disengaged Selection'),
            tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        '''
        '''
        for air in airs:
            ras_local = np.ma.clump_unmasked(np.ma.masked_outside(tcas.array, 4, 5))[air.slice]
            ras = shift_slices(ras_local, air.slice.start)
            # Assume that the reaction takes place during the TCAS RA period:
            for ra in ras:
                for ap_off in ap_offs:
                    if is_index_within_slice(ap_off.index, ra):
                        index = ap_off.index
                        onset = ra.slice.start
                        duration = (index - onset) / ap_offs.frequency
                        self.create_kpv(index, duration)


################################################################################
# Warnings: Alpha Floor, Alternate Law, Direct Law


##### TODO: Implement!
####class AlphaFloorWarningDuration(KeyPointValueNode):
####    '''
####    '''
####
####    def derive(self, alpha_floor=M('Alpha Floor Warning'),
####            airborne=S('Airborne')):
####        '''
####        '''
####        self.create_kpvs_where_state(
####            'Warning',
####            alpha_floor.array,
####            alpha_floor.hz,
####            phase=airborne,
####        )


##### TODO: Implement!
####class AlternateLawActivatedDuration(KeyPointValueNode):
####    '''
####    '''
####
####    def derive(self, alternate_law=M('Alternate Law Warning')):
####            airborne=S('Airborne')):
####        '''
####        '''
####        self.create_kpvs_where_state(
####            'Warning',
####            alternate_law.array,
####            alternate_law.hz,
####            phase=airborne,
####        )


##### TODO: Implement!
####class DirectLawActivatedDuration(KeyPointValueNode):
####    '''
####    '''
####
####    def derive(self, direct_law=M('Direct Law Warning')):
####            airborne=S('Airborne')):
####        '''
####        '''
####        self.create_kpvs_where_state(
####            'Warning',
####            direct_law.array,
####            direct_law.hz,
####            phase=airborne,
####        )


################################################################################


class ThrottleCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of throttle lever movement that exceed
    10 deg peak to peak and with a maximum cycle period of 14 seconds during
    the final approach phase.
    '''
    def derive(self, lever=P('Throttle Levers'), fapps=S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(lever.array[fapp.slice], 10.0, 10.0, 
                                           lever.hz, fapp.slice.start))

################################################################################
# Thrust Asymmetry in different conditions

class ThrustAsymmetryOnTakeoff(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral)" & "Loss of Control Significant torque
    or thrust split during T/O or G/A"
    '''
    def derive(self, ta=P('Thrust Asymmetry'), rolls=S('Takeoff Roll')):
        self.create_kpvs_within_slices(ta.array, rolls, max_value)
        

class ThrustAsymmetryInFlight(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Asymmetric thrust - may be due to an a/t fault"
    '''
    def derive(self, ta=P('Thrust Asymmetry'),
               airs=S('Airborne')):
        self.create_kpvs_within_slices(ta.array, airs, max_value)


class ThrustAsymmetryWithReverseThrustMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) - Asymmetric reverse thrust".
    
    A good KPV for providing measures on every flight, and preferred to the
    ThrustAsymmetryWithReverseThrustDuration which will normally not record
    any value.
    '''
    def derive(self, ta=P('Thrust Asymmetry'), rev_th=M('Thrust Reversers')):
        revs = np.ma.clump_unmasked(np.ma.masked_where(rev_th == 'Deployed',
                                                       ta.array))
        for rev in revs:
            idx = np.ma.argmax(ta.array[rev]) + rev.start
            self.create_kpv(idx, ta.array[idx])


class ThrustAsymmetryWithReverseThrustDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10% with reverse thrust operating.
    Included for customers with existing events using this approach.
    '''
    def derive(self, ta=P('Thrust Asymmetry'), rev_th=M('Thrust Reversers'), 
               mobile=S('Mobile')):
        #Note: Inclusion of the 'Mobile' phase ensures use of thrust reverse late
        #on the landing run is included, but corrupt data at engine start etc.
        #should be rejected.
        revs = clump_multistate(rev_th.array, 'Stowed', 
                                [s.slice for s in mobile], condition=False)
        for rev in revs:
            big_asym = shift_slices(
                np.ma.clump_unmasked(
                    np.ma.masked_less(ta.array[rev], 10.0)),
                rev.start)
            self.create_kpvs_from_slice_durations(big_asym)


class ThrustAsymmetryOnApproachMax(KeyPointValueNode):
    '''
    Peak thrust asymmetry on approach. A good KPV for providing measures on
    every flight, and preferred to the ThrustAsymmetryOnApproachDuration
    which will normally not record any value.
    '''
    def derive(self, ta=P('Thrust Asymmetry'), apps=S('Approach')):
        for app in apps:
            idx = np.ma.argmax(ta.array[app.slice]) + app.slice.start
            self.create_kpv(idx, ta.array[idx])


class ThrustAsymmetryOnApproachDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10%. Included for customers with
    existing events using this approach.
    '''
    def derive(self, ta=P('Thrust Asymmetry'), apps=S('Approach')):
        for app in apps:
            big_asym = shift_slices(
                np.ma.clump_unmasked(
                    np.ma.masked_less(ta.array[app.slice], 10.0)),
                app.slice.start)
            self.create_kpvs_from_slice_durations(big_asym)
            

class TouchdownToElevatorDownDuration(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), elevator=P('Elevator'),
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            index_elev = index_at_value(elevator.array, -14.0,
                                        slice(tdwn.index,None))
            if index_elev:
                e_14 = (index_elev - tdwn.index) / elevator.frequency
                self.create_kpv(index_elev, e_14)


class TouchdownTo60KtsDuration(KeyPointValueNode):
    """
    Ideally compute using groundspeed, otherwise use airspeed.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Airspeed' in available and 'Touchdown' in available

    def derive(self, airspeed=P('Airspeed'), groundspeed=P('Groundspeed'), 
               tdwns=KTI('Touchdown')):
        
        if groundspeed:
            speed=groundspeed.array
            freq=groundspeed.frequency
        else:
            speed=airspeed.array
            freq=airspeed.frequency
            
        for tdwn in tdwns:
            index_60kt = index_at_value(speed, 60.0, slice(tdwn.index,None))
            if index_60kt:
                t__60kt = (index_60kt - tdwn.index) / freq
                self.create_kpv(index_60kt, t__60kt)


################################################################################
# Turbulence


class TurbulenceInApproachMax(KeyPointValueNode):
    '''
    '''

    def derive(self, turb=P('Turbulence RMS g'), apps=S('Approach')):
        '''
        '''
        for app in apps:
            index = np.ma.argmax(turb.array[app.slice]) + app.slice.start
            value = turb.array[index]
            self.create_kpv(index, value)


class TurbulenceInCruiseMax(KeyPointValueNode):
    '''
    '''

    def derive(self, turb=P('Turbulence RMS g'), cruises=S('Cruise')):
        '''
        '''
        for cruise in cruises:
            index = np.ma.argmax(turb.array[cruise.slice]) + cruise.slice.start
            value = turb.array[index]
            self.create_kpv(index, value)


class TurbulenceInFlightMax(KeyPointValueNode):
    '''
    '''

    def derive(self, turb=P('Turbulence RMS g'), airs=S('Airborne')):
        '''
        '''
        for air in airs:
            index = np.ma.argmax(turb.array[air.slice]) + air.slice.start
            value = turb.array[index]
            self.create_kpv(index, value)


################################################################################


class WindSpeedInDescent(KeyPointValueNode):
    NAME_FORMAT = 'Wind Speed At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               wspd=P('Wind Speed')):
        # Aligned to alt_aal for cosmetic reasons; alignment to wind speed
        # leads to slightly misaligned KPVs for wind speed and wind
        # direction, which looks wrong although is arithmetically "correct".
        for this_descent_slice in alt_aal.slices_from_to(2100, 0):
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, this_descent_slice)
                if index:
                    speed = value_at_index(wspd.array, index)
                    if speed:
                        self.create_kpv(index, speed, altitude=alt)
                    

class WindDirectionInDescent(KeyPointValueNode):
    NAME_FORMAT = 'Wind Direction At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               wdir=P('Wind Direction Continuous')):
        for this_descent_slice in alt_aal.slices_from_to(2100, 0):
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, this_descent_slice)
                if index:
                    # We check that the direction is not masked at this point
                    # before 'risking' the %360 function.
                    direction = value_at_index(wdir.array, index)
                    if direction:
                        self.create_kpv(index, direction % 360.0, altitude=alt)


class WindAcrossLandingRunwayAt50Ft(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral). Crosswind - needs to be recorded just
    before landing, say at 50ft.
    '''
    def derive(self, walr = P('Wind Across Landing Runway'),
               lands = S('Landing')):
        for land in lands:
            index = land.slice.start # Because by definition landings start at 50ft
            value = walr.array[index]
            if value: # Trap None condition as this is common for this parameter
                self.create_kpv(index, value)
    
    
class ZeroFuelWeight(KeyPointValueNode):
    """
    The aircraft zero fuel weight is computed from the recorded gross weight
    and fuel data.
    
    See also the GrossWeightSmoothed calculation which uses fuel flow data to
    obtain a higher sample rate solution to the aircraft weight calculation,
    with a best fit to the available weight data.
    """
    def derive(self, fuel=P('Fuel Qty'), gw=P('Gross Weight')):
        zfw = np.ma.median(gw.array - fuel.array)
        self.create_kpv(0, zfw)


class HoldingDuration(KeyPointValueNode):
    """
    Identify time spent in the hold.
    """
    def derive(self, holds=S('Holding')):
        self.create_kpvs_from_slice_durations(holds, mark='end')


##### TODO: Implement!
####class DualStickInput(KeyPointValueNode):
####    def derive(self, x=P('Not Yet')):
####        return NotImplemented
####
####
##### TODO: Implement!
####class ControlForcesTimesThree(KeyPointValueNode):
####    def derive(self, x=P('Not Yet')):
####        return NotImplemented

################################################################################
# Go Around Related KPVs 

#See also: EngGasTempGoAroundMax, EngN1GoAroundMax, EngN2GoAroundMax,
#EngN3GoAroundMax, EngTorqueGoAroundMax

class TOGASelectedInGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control - TOGA power selection in flight (Go-arounds need to be
    kept as a separate case)."
    '''
    def derive(self, toga=M('Takeoff And Go Around'),
               gas=S('Go Around And Climbout')):
        self.create_kpvs_where_state('TOGA', toga.array, toga.hz, phase=gas)
               
                           
class AltitudeAtGoAroundMin(KeyPointValueNode):
    '''
    The altitude above the local airfield level at the minimum altitude point
    of the go-around.
    
    Note: This may be less than the radio altimeter reading at this point if
    there is higher ground in the area of the go-around minimum point.
    '''
    
    def derive(self, alt_aal=P('Altitude AAL'), gas=KTI('Go Around')):
        self.create_kpvs_at_ktis(alt_aal.array, gas)
            
 
class AltitudeGoAroundFlapRetracted(KeyPointValueNode):
    '''
    Go Around Flap Retracted pinpoints the flap retraction instance within the
    500ft go-around window. Create a single KPV for the first flap retraction
    within a Go Around And Climbout phase.
    '''
    def derive(self, alt_aal=P('Altitude AAL'),
               flap_retracteds=KTI('Go Around Flap Retracted'),
               go_arounds=S('Go Around And Climbout')):
        for go_around in go_arounds:
            flap_retracted = flap_retracteds.get_first(
                within_slice=go_around.slice)
            if flap_retracted:
                self.create_kpv(flap_retracted.index,
                                alt_aal.array[flap_retracted.index])


class AltitudeAtGoAroundGearUpSelection(KeyPointValueNode):
    
    name = 'Altitude Above Go Around Minimum At Gear Up Selection'
    
    # gagr pinpoints the gear retraction instance within the 500ft go-around window.

    def derive(self, alt_aal=P('Altitude AAL'), 
               gas=S('Go Around And Climbout'),
               gear_ups = KTI('Go Around Gear Selected Up')):
        for ga in gas:
            # Find the index and height at this go-around minimum.
            pit_index = np.ma.argmin(alt_aal.array[ga.slice])
            pit = alt_aal.array[ga.slice.start + pit_index]
            for gear_up in gear_ups:
                # Check this gear selected up matches the go-around in question
                if is_index_within_slice(gear_up.index, ga.slice):
                    # Did we raise the gear after the minimum height?
                    if gear_up.index > pit_index:
                        gear_up_ht = alt_aal.array[gear_up.index] - pit
                    else:
                        # Show zero if selected up before minimum height
                        gear_up_ht = 0.0
                    self.create_kpv(gear_up.index,gear_up_ht)

# TODO: Write some unit tests!
class SpeedbrakesDeployedInGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A - ...Speedbrake retraction."
    '''
    def derive(self, speedbrake=M('Speedbrake Selected'),
               gas=S('Go Around And Climbout')):
        '''
        '''
        deployed = speedbrake.array.state['Deployed/Cmd Up']
        for ga in gas:
            event = np.ma.masked_not_equal(speedbrake.array.raw[ga.slice],
                                           deployed)
            value = np.ma.count(event) / speedbrake.frequency
            if value:
                # Probably open at the start of the go-around, so when were
                # they closed?
                when = np.ma.clump_unmasked(event)
                index = when[-1].stop
                self.create_kpv(index, value)


class ThrustAsymmetryInGoAround(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Significant torque or thrust split during T/O or G/A"
    '''
    def derive(self, ta=P('Thrust Asymmetry'),
               gas=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(ta.array, gas, max_value)


class PitchInGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A - ...Rotation to 12 deg pitch..."
    '''
    def derive(self, pitch=P('Pitch'),
               gas=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(pitch.array, gas, max_value)


class VerticalSpeedInGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A." Concern here is excessive rates of
    climb following enthusiastic application of power and pitch up.
    '''
    def derive(self, vs=P('Vertical Speed'),
               gas=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(vs.array, gas, max_value)


class AOAInGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A"
    '''
    name = 'AOA In Go Around Max'
    def derive(self, aoa=P('AOA'), gas=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(aoa.array, gas, max_value)



