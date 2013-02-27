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
                                      NAME_VALUES_ENGINE,
                                      NAME_VALUES_FLAP)

from analysis_engine.node import KeyPointValueNode, KPV, KTI, P, S, A, M

from analysis_engine.library import (ambiguous_runway,
                                     all_of,
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
                                     index_of_first_start,
                                     index_of_last_stop,
                                     integrate,
                                     is_index_within_slice,
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
                                     runs_of_ones,
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
                                     value_at_index,
                                     vstack_params)


##############################################################################
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


##############################################################################
# Helpers


def thrust_reversers_working(landing, pwr, tr):
    '''
    Thrust reversers are deployed and average N1 over 65%.
    '''
    high_power = np.ma.masked_less(pwr.array[landing.slice], 65.0)
    high_power_slices = np.ma.clump_unmasked(high_power)
    return clump_multistate(tr.array, 'Deployed', high_power_slices)


##############################################################################
# Acceleration


########################################
# Acceleration: Lateral


class AccelerationLateralMax(KeyPointValueNode):
    '''
    This KPV has no inherent flight phase associated with it, but we can
    reasonably say that we are not interested in anything while the aircraft is
    stationary.
    '''

    units = 'g'

    @classmethod
    def can_operate(cls, available):
        return 'Acceleration Lateral Offset Removed' in available

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               gnd_spd=P('Groundspeed')):

        if gnd_spd:
            self.create_kpvs_within_slices(
                acc_lat.array,
                gnd_spd.slices_above(5),
                max_abs_value,
            )
        else:
            self.create_kpv(*max_value(acc_lat.array))


class AccelerationLateralAtTouchdown(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               touchdowns=KTI('Touchdown')):

        for touchdown in touchdowns:
            self.create_kpv(*bump(acc_lat, touchdown))


class AccelerationLateralDuringTakeoffMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Lateral)"
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_within_slices(
            acc_lat.array,
            takeoff_rolls,
            max_abs_value,
        )


class AccelerationLateralDuringLandingMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral)."
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               landing_rolls=S('Landing Roll'),
               ldg_rwy=A('FDR Landing Runway')):

        if ambiguous_runway(ldg_rwy):
            return
        self.create_kpv_from_slices(
            acc_lat.array,
            landing_rolls,
            max_abs_value,
        )


class AccelerationLateralWhileTaxiingStraightMax(KeyPointValueNode):
    '''
    Lateral acceleration while not turning is rarely an issue, so we compute
    only one KPV for taxi out and one for taxi in. The straight sections are
    identified by masking the turning phases and then testing the resulting
    data.
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        acc_lat_array = mask_inside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralWhileTaxiingTurnMax(KeyPointValueNode):
    '''
    Lateral acceleration while taxiing normally occurs in turns, and leads to
    wear on the undercarriage and discomfort for passengers. In extremis this
    can lead to taxiway excursions. Lateral acceleration is used in preference
    to groundspeed as this parameter is available on older aircraft and is
    directly related to comfort.
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral Offset Removed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        acc_lat_array = mask_outside_slices(acc_lat.array, turns.get_slices())
        self.create_kpvs_within_slices(acc_lat_array, taxiing, max_abs_value)


class AccelerationLateralOffset(KeyPointValueNode):
    '''
    This KPV computes the lateral accelerometer datum offset, as for
    AccelerationNormalOffset. The more complex slicing statement ensures we
    only accumulate error estimates when taxiing in a straight line.
    '''

    units = 'g'

    def derive(self,
               acc_lat=P('Acceleration Lateral'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        total_sum = 0.0
        total_count = 0
        straights = slices_and(
            [s.slice for s in list(taxiing)],
            slices_not([s.slice for s in list(turns)]),
        )
        for straight in straights:
            unmasked_data = np.ma.compressed(acc_lat.array[straight])
            count = len(unmasked_data)
            if count:
                total_count += count
                total_sum += np.sum(unmasked_data)
        if total_count > 20:
            delta = total_sum / float(total_count)
            if abs(delta) < ACCEL_LAT_OFFSET_LIMIT:
                self.create_kpv(0, delta)


########################################
# Acceleration: Longitudinal


class AccelerationLongitudinalDuringTakeoffMax(KeyPointValueNode):
    '''
    This may be of interest where takeoff performance is an issue, though not
    normally monitored as a safety event.
    '''

    units = 'g'

    def derive(self,
               acc_lon=P('Acceleration Longitudinal'),
               takeoff=S('Takeoff')):

        self.create_kpv_from_slices(acc_lon.array, takeoff, max_value)


class AccelerationLongitudinalDuringLandingMax(KeyPointValueNode):
    '''
    This is an indication of severe braking and/or use of reverse thrust or
    reverse pitch.
    '''

    units = 'g'

    def derive(self,
               acc_lon=P('Acceleration Longitudinal'),
               landing=S('Landing')):

        self.create_kpv_from_slices(acc_lon.array, landing, max_value)


########################################
# Acceleration: Normal


class AccelerationNormalMax(KeyPointValueNode):
    '''
    This KPV has no inherent flight phase associated with it, but we can
    reasonably say that we are not interested in anything while the aircraft is
    stationary.
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               mobile=S('Mobile')):

        self.create_kpv_from_slices(acc_norm.array, mobile, max_value)


class AccelerationNormal20FtToFlareMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            acc_norm.array,
            alt_aal.slices_from_to(20, 5),
            max_value,
        )


class AccelerationNormalWithFlapUpWhileAirborneMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'),
               airborne=S('Airborne')):

        # Mask data where the flaps are down:
        acc_flap_up = np.ma.masked_where(flap.array > 0.0, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalWithFlapUpWhileAirborneMin(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'),
               airborne=S('Airborne')):

        # Mask data where the flaps are down:
        acc_flap_up = np.ma.masked_where(flap.array > 0.0, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, min_value)


class AccelerationNormalWithFlapDownWhileAirborneMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'),
               airborne=S('Airborne')):

        # Mask data where the flaps are up:
        acc_flap_up = np.ma.masked_where(flap.array == 0.0, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalWithFlapDownWhileAirborneMin(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               flap=P('Flap'),
               airborne=S('Airborne')):

        # Mask data where the flaps are up:
        acc_flap_up = np.ma.masked_where(flap.array == 0.0, acc_norm.array)
        self.create_kpv_from_slices(acc_flap_up, airborne, min_value)


class AccelerationNormalAtLiftoff(KeyPointValueNode):
    '''
    This is a measure of the normal acceleration at the point of liftoff, and
    is related to the pitch rate at takeoff.
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               liftoffs=KTI('Liftoff')):

        for liftoff in liftoffs:
            self.create_kpv(*bump(acc_norm, liftoff))


class AccelerationNormalAtTouchdown(KeyPointValueNode):
    '''
    This is the peak acceleration at landing, often used to identify hard
    landings for maintenance purposes.
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               touchdowns=KTI('Touchdown')):

        for touchdown in touchdowns:
            self.create_kpv(*bump(acc_norm, touchdown))


class AccelerationNormalLiftoffTo35FtMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal Offset Removed'),
               takeoffs=S('Takeoff')):

        self.create_kpvs_within_slices(acc_norm.array, takeoffs, max_value)


class AccelerationNormalOffset(KeyPointValueNode):
    '''
    This KPV computes the normal accelerometer datum offset. This allows for
    offsets that are sometimes found in these sensors which remain in service
    although outside the permitted accuracy of the signal.
    '''

    units = 'g'

    def derive(self,
               acc_norm=P('Acceleration Normal'),
               taxiing=S('Taxiing')):

        total_sum = 0.0
        total_count = 0
        for taxi in taxiing:
            unmasked_data = np.ma.compressed(acc_norm.array[taxi.slice])
            count = len(unmasked_data)
            if count:
                total_count += count
                total_sum += np.sum(unmasked_data)
        if total_count > 20:
            delta = total_sum / float(total_count) - 1.0
            if abs(delta) < ACCEL_NORM_OFFSET_LIMIT:
                self.create_kpv(0, delta + 1.0)


##############################################################################
# Airspeed


########################################
# Airspeed: General


class AirspeedMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(air_spd.array, airborne, max_value)


class AirspeedDuringCruiseMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(air_spd.array, cruises, max_value)


class AirspeedDuringCruiseMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(air_spd.array, cruises, min_value)


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

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gnd_spd=P('Groundspeed'),
               alt_rad=P('Altitude Radio'),
               airborne=S('Airborne')):

        _, fin_apps = slices_from_to(alt_rad.array, 30, 10)
        descents = slices_and([s.slice for s in airborne], fin_apps)
        for descent in descents:
            # Ensure we encompass the range of interest.
            scope = slice(descent.start - 5, descent.stop + 5)
            # We'd like to use groundspeed to compute the wind gust, but
            # variations in airspeed are a suitable backstop.
            if gnd_spd:
                speed = air_spd.array[scope] - gnd_spd.array[scope]
            else:
                speed = air_spd.array[scope] - air_spd.array[scope][0]
            # Precise indexing is used as this is only a short segment. Note
            # that the _idx values are floating point interpolations of the
            # radio altimeter signal, and the speed array is also
            # interpolated.
            idx_start = index_at_value(alt_rad.array, 30.0, scope)
            idx_stop = index_at_value(alt_rad.array, 10.0, scope)
            new_app = shift_slice(descent, -scope.start)
            peak = max_value(speed, new_app,
                    start_edge=idx_start - scope.start,
                    stop_edge=idx_stop - scope.start)
            trough = min_value(speed, new_app,
                    start_edge=idx_start - scope.start,
                    stop_edge=idx_stop - scope.start)
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

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(air_spd.array, liftoffs)


class AirspeedAt35FtDuringTakeoff(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               takeoffs=S('Takeoff')):

        for takeoff in takeoffs:
            index = takeoff.stop_edge  # Takeoff ends at 35ft!
            value = value_at_index(air_spd.array, index)
            self.create_kpv(index, value)


class Airspeed35To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class Airspeed35To1000FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


class Airspeed1000To8000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(1000, 8000),
            max_value,
        )


class Airspeed8000To10000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(8000, 10000),
            max_value,
        )


########################################
# Airspeed: Descending


class Airspeed10000To8000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(10000, 8000),
            max_value,
        )


class Airspeed8000To5000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(8000, 5000),
            max_value,
        )


class Airspeed5000To3000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(5000, 3000),
            max_value,
        )


class Airspeed3000To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpv_from_slices(
            air_spd.array,
            alt_aal.slices_from_to(3000, 1000),
            max_value,
        )


class Airspeed1000To500FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class Airspeed1000To500FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class Airspeed500To20FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class Airspeed500To20FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedAtTouchdown(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(air_spd.array, touchdowns)


class AirspeedTrueAtTouchdown(KeyPointValueNode):
    '''
    This KPV relates to groundspeed at touchdown to illustrate headwinds and
    tailwinds. We also have 'Tailwind 100 Ft To Touchdown Max' to cater for
    safety event triggers.
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed True'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(air_spd.array, touchdowns)


########################################
# Airspeed: Minus V2


class AirspeedMinusV2AtLiftoff(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 At Liftoff'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(spd_v2.array, liftoffs)


class AirspeedMinusV2At35FtDuringTakeoff(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 At 35 Ft During Takeoff'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               takeoffs=S('Takeoff')):

        for takeoff in takeoffs:
            index = takeoff.stop_edge  # Takeoff ends at 35ft!
            value = spd_v2.array[index]
            self.create_kpv(index, value)


class AirspeedMinusV235To1000FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Max'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class AirspeedMinusV235To1000FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 35 To 1000 Ft Min'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


class AirspeedMinusV2For3Sec35To1000FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To 1000 Ft Max'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_v2.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class AirspeedMinusV2For3Sec35To1000FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Airspeed Minus V2 For 3 Sec 35 To 1000 Ft Min'
    units = 'kt'

    def derive(self,
               spd_v2=P('Airspeed Minus V2 For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

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

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(spd_rel.array, touchdowns)


class AirspeedRelative1000To500FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelative1000To500FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelative500To20FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelative500To20FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelative20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_value,
        )


class AirspeedRelative20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class AirspeedRelativeFor3Sec1000To500FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class AirspeedRelativeFor3Sec500To20FtMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            max_value,
        )


class AirspeedRelativeFor3Sec500To20FtMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_value,
        )


class AirspeedRelativeFor3Sec20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               spd_rel=P('Airspeed Relative For 3 Sec'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            spd_rel.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


########################################
# Airspeed: Flap


# NOTE: It is essential that Flap is the first parameter here to prevent the
#       flap values, which match the detent settings, from being interpolated.


class AirspeedWithFlapMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d Max'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Fast')):

        # Fast scope traps flap changes very late on the approach and raising
        # flaps before 80kn on the landing run.
        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d Min'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Airborne')):

        # Airborne scope avoids deceleration on the runway "corrupting" the
        # minimum airspeed with landing flap.
        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapDuringClimbMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d During Climb Max'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Climb')):

        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapDuringClimbMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d During Climb Min'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Climb')):

        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapDuringDescentMax(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d During Descent Max'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Descent')):

        self.flap_or_conf_max_or_min(flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapDuringDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed With Flap %(flap)d During Descent Min'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed'),
               scope=S('Descent To Flare')):

        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


class AirspeedRelativeWithFlapDuringDescentMin(KeyPointValueNode, FlapOrConfigurationMaxOrMin):
    '''
    '''

    NAME_FORMAT = 'Airspeed Relative With Flap %(flap)d During Descent Min'
    NAME_VALUES = NAME_VALUES_FLAP

    units = 'kt'

    def derive(self,
               flap=P('Flap'),
               airspeed=P('Airspeed Relative'),
               scope=S('Descent To Flare')):

        self.flap_or_conf_max_or_min(flap, airspeed, min_value, scope=scope)


########################################
# Airspeed: Landing Gear


class AirspeedWithGearDownMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gear=M('Gear Down'),
               airs=S('Airborne')):

        state = gear.array.state['Up']
        for air in airs:
            air_start = int(air.slice.start or 0)
            downs = np.ma.masked_equal(gear.array.raw[air.slice], state)
            downs = np.ma.clump_unmasked(downs)
            for down in downs:
                chunk = air_spd.array[air.slice][down]
                index = np.ma.argmax(chunk)
                value = chunk[index]
                self.create_kpv(air_start + down.start + index, value)


class AirspeedWhileGearRetractingMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gear_ret=S('Gear Retracting')):

        self.create_kpvs_within_slices(air_spd.array, gear_ret, max_value)


class AirspeedWhileGearExtendingMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gear_ext=S('Gear Extending')):

        self.create_kpvs_within_slices(air_spd.array, gear_ext, max_value)


class AirspeedAtGearUpSelection(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gear_up_sel=KTI('Gear Up Selection')):

        self.create_kpvs_at_ktis(air_spd.array, gear_up_sel)


class AirspeedAtGearDownSelection(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               gear_dn_sel=KTI('Gear Down Selection')):

        self.create_kpvs_at_ktis(air_spd.array, gear_dn_sel)


########################################
# Airspeed: Thrust Reversers


class AirspeedWithThrustReversersDeployedMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed True'),
               tr=M('Thrust Reversers'),
               power=P('Eng (*) N1 Avg'),
               landings=S('Landing')):

        for landing in landings:
            high_rev = thrust_reversers_working(landing, power, tr)
            self.create_kpvs_within_slices(air_spd.array, high_rev, min_value)


class AirspeedAtThrustReversersSelection(KeyPointValueNode):
    '''
    This gives the indicated airspeed where the thrust reversers were selected.
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               tr=M('Thrust Reversers'),
               landings=S('Landing')):

        slices = [s.slice for s in landings]
        to_scan = clump_multistate(tr.array, 'Stowed', slices, condition=False)
        self.create_kpv_from_slices(air_spd.array, to_scan, max_value)


########################################
# Airspeed: Other


class AirspeedVacatingRunway(KeyPointValueNode):
    '''
    Airspeed vacating runway uses true airspeed, which is extended below the
    minimum range of the indicated airspeed specifically for this type of
    event. See the derived parameter for details of how groundspeed or
    acceleration data is used to cover the landing phase.
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed True'),
               off_rwy=KTI('Landing Turn Off Runway')):

        self.create_kpvs_at_ktis(air_spd.array, off_rwy)


class AirspeedDuringRejectedTakeoffMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               rtos=S('Rejected Takeoff')):

        self.create_kpvs_within_slices(air_spd.array, rtos, max_value)


class AirspeedBelow10000FtDuringDescentMax(KeyPointValueNode):
    '''
    Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.

    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 10,000ft boundary.
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):

        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        alt = hysteresis(alt, HYSTERESIS_FPALT)

        height_bands = np.ma.clump_unmasked(np.ma.masked_greater(alt, 10000))
        descent_bands = slices_and(height_bands, descent.get_slices())
        self.create_kpvs_within_slices(air_spd.array, descent_bands, max_value)


class AirspeedTopOfDescentTo10000FtMax(KeyPointValueNode):
    '''
    Outside the USA 10,000 ft relates to flight levels, whereas FAA regulations
    (and possibly others we don't currently know about) relate to height above
    sea level (QNH) hence the options based on landing airport location.

    In either case, we apply some hysteresis to prevent nuisance retriggering
    which can arise if the aircraft is sitting on the 10,000ft boundary.
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               alt_std=P('Altitude STD Smoothed'),
               alt_qnh=P('Altitude QNH'),
               ldg_apt=A('FDR Landing Airport'),
               descent=S('Descent')):

        country = None
        if ldg_apt.value:
            country = ldg_apt.value.get('location', {}).get('country')

        alt = alt_qnh.array if country == 'United States' else alt_std.array
        alt = hysteresis(alt, HYSTERESIS_FPALT)

        height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt), 10000))
        descent_bands = slices_and(height_bands, descent.get_slices())
        self.create_kpvs_within_slices(air_spd.array, descent_bands, max_value)


class AirspeedDuringLevelFlightMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               air_spd=P('Airspeed'),
               lvl_flt=S('Level Flight')):

        for section in lvl_flt:
            self.create_kpv(*max_value(air_spd.array, section.slice))


##############################################################################
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
        # flaps before 80 kts on the landing run.
        self.flap_or_conf_max_or_min(flap, aoa, max_value,
                                     scope=scope, include_zero=True)


##############################################################################


class TouchdownToThrustReversersDeployedDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Reverse thrust delay - time delay.
    Selection more than 3sec after main wheel t/d."

    Note: 3 second threshold may be applied to derive an event from this KPV.
    '''

    units = 's'

    def derive(self,
               tr=M('Thrust Reversers'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown')):

        for landing in landings:
            # Only interested in first opening of reversers on this landing:
            deploys = clump_multistate(tr.array, 'Deployed', landing.slice)
            try:
                deployed = deploys[0].start
            except IndexError:
                continue
            touchdown = touchdowns.get_first(within_slice=landing.slice)
            if not touchdown:
                continue
            self.create_kpv(deployed, (deployed - touchdown.index) / tr.hz)


class TouchdownToSpoilersDeployedDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Late spoiler deployment - time delay".
    '''
    def derive(self, brake=M('Speedbrake Selected'),
               lands = S('Landing'), tdwns=KTI('Touchdown')):
        deploys = find_edges_on_state_change('Deployed/Cmd Up', brake.array, phase=lands)
        for land in lands:
            for deploy in deploys:
                if not is_index_within_slice(deploy, land.slice):
                    continue
                for tdwn in tdwns:
                    if not is_index_within_slice(tdwn.index, land.slice):
                        continue
                    self.create_kpv(deploy, (deploy-tdwn.index)/brake.hz)


class TrackDeviationFromRunway1000To500Ft(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 1000 to 500 feet.

    Helps establishing the stable criteria for IFR below 1000ft.

    Includes large deviations recoreded when aircraft turns onto runway at
    altitudes below 1000ft.
    '''
    def derive(self, track_dev=P('Track Deviation From Runway'),
               alt=P('Altitude AAL')):
        alt_bands = alt.slices_from_to(1000, 500)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )

class TrackDeviationFromRunway500To300Ft(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 500 to 300 feet.

    Helps establishing the stable criteria for VFR below 500ft.

    Includes large deviations recorded when aircraft turns onto runway at
    altitudes below 500ft, but should be stable by 300ft.
    '''
    def derive(self, track_dev=P('Track Deviation From Runway'),
               alt=P('Altitude AAL')):
        alt_bands = alt.slices_from_to(500, 300)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )


class TrackDeviationFromRunway300FtToTouchdown(KeyPointValueNode):
    '''
    Track deviation from the runway centreline from 300 to 0 feet.

    Helps establishing the FAA stable criteria for a late roll onto runway
    heading.

    There is almost no excuse for being unaligned with the runway at this
    altitude, so the distribution should have small variance.
    '''
    def derive(self, track_dev=P('Track Deviation From Runway'),
               alt=P('Altitude AAL')):
        alt_bands = alt.slices_from_to(300, 0)
        self.create_kpvs_within_slices(
            track_dev.array,
            alt_bands,
            max_abs_value,
        )
        
        
##############################################################################
# Takeoff and Use of TOGA


class TOGASelectedInFlightNotGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control - Unexpected TOGA power selection in flight (except for
    a go-around)"
    '''

    name = 'TOGA Selected In Flight Not Go Around Duration'

    def derive(self,
               toga=M('Takeoff And Go Around'),
               gas=S('Go Around And Climbout'),
               airs=S('Airborne')):

        to_scan = slices_and(
            [s.slice for s in airs],
            slices_not(
                [s.slice for s in gas],
                begin_at=airs[0].slice.start,
                end_at=airs[-1].slice.stop,
            ),
        )

        # The elegant create_kpvs_where_state function requires the phase
        # information as a section object, hence a couple of lines of glue.
        # TODO: Make create_kpvs_where_state accept list of slices.
        not_ga = S()
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


##############################################################################
# Landing Gear


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


##############################################################################
# Altitude


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
    name = 'Altitude AAL Flap Extension Max'
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


########################################
# Automated Systems


class AltitudeAtAPEngagedSelection(KeyPointValueNode):
    '''
    '''

    name = 'Altitude At AP Engaged Selection'
    units = 'ft'

    def derive(self,
               alt_aal=P('Altitude AAL'),
               ap_eng=KTI('AP Engaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, ap_eng)


class AltitudeAtAPDisengagedSelection(KeyPointValueNode):
    '''
    '''

    name = 'Altitude At AP Disengaged Selection'
    units = 'ft'

    def derive(self,
               alt_aal=P('Altitude AAL'),
               ap_dis=KTI('AP Disengaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, ap_dis)


class AltitudeAtATEngagedSelection(KeyPointValueNode):
    '''
    Note: Autothrottle is normally engaged prior to takeoff, so will not
          trigger this event.
    '''

    name = 'Altitude At AT Engaged Selection'
    units = 'ft'

    def derive(self,
               alt_aal=P('Altitude AAL'),
               at_eng=KTI('AT Engaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, at_eng)


class AltitudeAtATDisengagedSelection(KeyPointValueNode):
    '''
    '''

    name = 'Altitude At AT Disengaged Selection'
    units = 'ft'

    def derive(self,
               alt_aal=P('Altitude AAL'),
               at_dis=KTI('AT Disengaged Selection')):

        self.create_kpvs_at_ktis(alt_aal.array, at_dis)


########################################


class AltitudeFirstStableDuringApproach(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.

    Establish first point stable during approach.
    '''
    def derive(self, stable=P('Stable Approach'), alt=P('Altitude AAL')):
        # no need for approaches as we can assume each approach has no masked
        # values and inbetween there will be some
        apps = np.ma.clump_unmasked(stable.array)
        for app in apps:
            # iterate through approaches as only one KPV is to be created per
            # approach
            index = index_of_first_start(stable.array == 'Stable', app, min_dur=2)
            if index:
                self.create_kpv(index, value_at_index(alt.array, index))
            else:
                continue


class AltitudeLastUnStableDuringApproach(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.

    Establish last Unstable position during approach.
    '''
    def derive(self, stable=P('Stable Approach'), alt=P('Altitude AAL')):
        apps = np.ma.clump_unmasked(stable.array)
        for app in apps:
            # iterate through approaches as only one KPV is to be created per
            # approach
            index = index_of_last_stop(stable.array != 'Stable', app, min_dur=2)
            if index:
                self.create_kpv(index, value_at_index(alt.array, index))
            else:
                continue


class PercentApproachStableBelow(object):
    '''
    Abstract Class
    '''
    def percent_stable(self, stable, altitude, level=1000):
        stable[altitude > level] = np.ma.masked
        apps_under_level = np.ma.clump_unmasked(stable)
        for app in apps_under_level:
            is_stable = stable[app] == 'Stable'
            percent = sum(is_stable) / float(app.stop - app.start) * 100
            self.create_kpv(app.start, percent)


class PercentApproachStableBelow1000Ft(KeyPointValueNode, PercentApproachStableBelow):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.

    Creates a KPV around 1000 ft during the approach with the percent
    (0% to 100%) of the approach that was stable.
    '''
    def derive(self, stable=P('Stable Approach'), alt=P('Altitude AAL')):
        self.percent_stable(stable.array, alt.array, 1000)


class PercentApproachStableBelow500Ft(KeyPointValueNode, PercentApproachStableBelow):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.

    Creates a KPV around 1000 ft during the approach with the percent
    (0% to 100%) of the approach that was stable.
    '''
    def derive(self, stable=P('Stable Approach'), alt=P('Altitude AAL')):
        self.percent_stable(stable.array, alt.array, 500)


##############################################################################
# Autopilot


class APDisengagedDuringCruiseDuration(KeyPointValueNode):
    '''
    This monitors the duration for which all autopilot channels are disengaged
    in the cruise.
    '''

    name = 'AP Disengaged During Cruise Duration'
    units = ''

    def derive(self, ap=M('AP Engaged'), cruise=S('Cruise')):

        self.create_kpvs_where_state('-', ap.array, ap.hz, phase=cruise)


##############################################################################


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

##############################################################################
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

"""
This KPV was sketched out following Emirates' presentation, but requires a
value for V1 which is not currently set up as a derived (or recorded)
parameter.

class DecelerationToAbortTakeoffBeforeV1(KeyPointValueNode):
    '''
    FDS developed this KPV following the 2nd EOFDM conference.
    '''

    units = 'g'

    def derive(self, lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               gspd=P('Groundspeed'),
               aspd=P('Airspeed True'),
               v1=A('V1'),
               rwy=A('FDR Takeoff Runway'),
               toff_rolls=S('Takeoff Roll')):

        if ambiguous_runway(rwy):
            return
        if gspd:
            speed=gspd.array
        else:
            speed=aspd.array
        for roll in toff_rolls:
            v1_idx = v1.value
            rot_end = runway_distance_from_end(rwy.value,
                                               lat.array[v1_idx ],
                                               lon.array[v1_idx ])

            v1_mps = value_at_index(speed, v1.value) * KTS_TO_MPS
            mu = (v1_mps**2.0) / (2.0 * GRAVITY_METRIC * rot_end)
            self.create_kpv(vi_idx, mu)
"""
            
##############################################################################
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
        if ils_ldgs.get(containing_index=land_idx):
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
    We take the median heading during the takeoff roll only as this avoids
    problems when turning onto the runway or with drift just after liftoff.
    The value is "assigned" to a time midway through the takeoff roll.
    """
    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff Roll')):
        for toff in toffs:
            toff_head = np.ma.median(head.array[toff.slice])
            toff_index = (toff.slice.start + toff.slice.stop)/2.0
            self.create_kpv(toff_index, toff_head%360.0)


class HeadingAtLanding(KeyPointValueNode):
    """
    We take the median heading during the landing roll as this avoids
    problems with drift just before touchdown and heading changes when
    turning off the runway. The value is "assigned" to a time midway through
    the landing phase.
    """
    def derive(self, head=P('Heading Continuous'),
               lands=S('Landing Roll')):
        for land in lands:
            # Check the slice is robust.
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
        self.create_kpvs_at_ktis(head.array % 360.0, low_points)


##############################################################################
# Height Loss


class HeightLossLiftoffTo35Ft(KeyPointValueNode):
    '''
    At these low altitudes, the aircraft is in ground effect, so we use an
    inertial vertical speed to identify small height losses. This means that
    the algorithm will still work with low sample rate (or even missing)
    radio altimeters.
    '''

    def derive(self,
               vs=P('Vertical Speed Inertial'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for climb in alt_aal.slices_from_to(0, 35):
            array = np.ma.masked_greater_equal(vs.array[climb], 0.0)
            drops = np.ma.clump_unmasked(array)
            for drop in drops:
                ht_loss = integrate(vs.array[drop], vs.frequency)
                # Only interested in peak value - by definition the last value:
                if ht_loss[-1]:
                    self.create_kpv(drop.stop, abs(ht_loss[-1]))


class HeightLoss35To1000Ft(KeyPointValueNode):
    '''
    '''

    def derive(self,
               ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for climb in alt_aal.slices_from_to(35, 1000):
            index, value = min_value(ht_loss.array, climb)
            # Only report a positive value where height is lost:
            if index and value < 0:
                self.create_kpv(index, abs(value))


class HeightLoss1000To2000Ft(KeyPointValueNode):
    '''
    '''

    def derive(self,
               ht_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for climb in alt_aal.slices_from_to(1000, 2000):
            index, value = min_value(ht_loss.array, climb)
            # Only report a positive value where height is lost:
            if index and value < 0:
                self.create_kpv(index, abs(value))


##############################################################################
# ILS


class ILSFrequencyDuringApproach(KeyPointValueNode):
    '''
    Determine the ILS frequency during approach.

    The period when the aircraft was continuously established on the ILS and
    descending to the minimum point on the approach is already defined as a
    flight phase. This KPV just picks up the frequency tuned at that point.
    '''

    name = 'ILS Frequency During Approach'
    units = 'MHz'

    def derive(self,
               ils_frq=P('ILS Frequency'),
               loc_ests=S('ILS Localizer Established')):

        for loc_est in loc_ests:
            # Find the ILS frequency for the final period of operation of the
            # ILS during this approach. Note that median picks the value most
            # commonly recorded, so allows for some masked values and perhaps
            # one or two rogue values. If, however, all the ILS frequency data
            # is masked, no KPV is created.
            frequency = np.ma.median(ils_frq.array[loc_est.slice])
            if frequency:
                # Set the KPV index to the start of this ILS approach:
                self.create_kpv(loc_est.slice.start, frequency)


class ILSGlideslopeDeviation1500To1000FtMax(KeyPointValueNode):
    '''
    Determine maximum deviation from the glideslope between 1500 and 1000 ft.

    Find where the maximum (absolute) deviation occured and store the actual
    value. We can do abs on the statistics to normalise this, but retaining the
    sign will make it possible to look for direction of errors at specific
    airports.
    '''

    name = 'ILS Glideslope Deviation 1500 To 1000 Ft Max'
    units = 'dots'

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
    units = 'dots'

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
    units = 'dots'

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
    units = 'dots'

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
    units = 'dots'

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
    units = 'dots'

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
    units = 'dots'

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


##############################################################################


class IsolationValveOpenAtLiftoff(KeyPointValueNode):
    def derive(self, isol=P('Isolation Valve Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(isol.array, lifts, suppress_zeros=True)


class PackValvesOpenAtLiftoff(KeyPointValueNode):
    def derive(self, pack=M('Pack Valves Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(pack.array.raw, lifts, suppress_zeros=True)


##############################################################################
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
# go-arounds.

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


##############################################################################
# Mach


########################################
# Mach: General


class MachMax(KeyPointValueNode):
    '''
    '''

    units = 'Mach'

    def derive(self,
               mach=P('Mach'),
               airs=S('Airborne')):

        self.create_kpvs_within_slices(mach.array, airs, max_value)


########################################
# Mach: Landing Gear


class MachWithGearDownMax(KeyPointValueNode):
    '''
    '''

    units = 'Mach'

    def derive(self,
               mach=P('Mach'),
               gear=M('Gear Down'),
               airs=S('Airborne')):

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


class MachWhileGearRetractingMax(KeyPointValueNode):
    '''
    '''

    units = 'Mach'

    def derive(self,
               mach=P('Mach'),
               gear_ret=S('Gear Retracting')):

        self.create_kpvs_within_slices(mach.array, gear_ret, max_value)


class MachWhileGearExtendingMax(KeyPointValueNode):
    '''
    '''

    units = 'Mach'

    def derive(self,
               mach=P('Mach'),
               gear_ext=S('Gear Extending')):

        self.create_kpvs_within_slices(mach.array, gear_ext, max_value)


##############################################################################
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


##############################################################################
# Engine Bleed


# FIXME: Alignment should be resolved by align method, not use of integers.
class EngBleedValvesAtLiftoff(KeyPointValueNode):
    '''
    '''

    units = ''

    @classmethod
    def can_operate(cls, available):
        return all_of((
            'Eng (1) Bleed',
            'Eng (2) Bleed',
            'Liftoff',
        ), available)

    def derive(self,
               liftoffs=KTI('Liftoff'),
               b1=M('Eng (1) Bleed'),
               b2=M('Eng (2) Bleed'),
               b3=M('Eng (3) Bleed'),
               b4=M('Eng (4) Bleed')):

        # Note: The bleed arrays for each engine are integer arrays, but to
        # index them correctly we need to align the liftoff KTI to match these
        # arrays. The alignment will cause the integer arrays to blur at
        # transitions, so int(b1 + b2 + b3 + b4) is used to remove this effect
        # as the bleeds are changing state.
        bleeds = vstack_params(b1, b2, b3, b4).sum(axis=0).astype(int)
        for liftoff in liftoffs:
            valves = bleeds[liftoff.index]
            if valves:
                self.create_kpv(liftoff.index, valves)


##############################################################################
# Engine EPR


class EngEPRDuringTaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR During Taxi Max'
    units = '%'

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_epr_max.array, taxiing, max_value)


class EngEPRDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR During Takeoff 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_epr_max.array, ratings, max_value)


class EngEPRDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR During Go Around 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_epr_max.array, ratings, max_value)


class EngEPRDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR During Maximum Continuous Power Max'
    units = '%'

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               grounded=S('Grounded')):

        slices = to_ratings + ga_ratings + grounded
        self.create_kpv_outside_slices(eng_epr_max.array, slices, max_value)


class EngEPR500To50FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR 500 To 50 Ft Max'
    units = '%'

    def derive(self,
               eng_epr_max=P('Eng (*) EPR Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_epr_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngEPR500To50FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Eng EPR 500 To 50 Ft Min'
    units = '%'

    def derive(self,
               eng_epr_min=P('Eng (*) EPR Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_epr_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


##############################################################################
# Engine Gas Temperature


class EngGasTempDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    units = 'C'

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    units = 'C'

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngGasTempDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    We assume maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. So, by collecting all the high power
    periods and inverting these from the start of the first airborne section to
    the end of the last, we have the required periods of flight.
    '''

    units = 'C'

    def derive(self,
               eng_egt_max=P('Eng (*) Gas Temp Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               airborne=S('Airborne')):

        if not airborne:
            return
        high_power_ratings = to_ratings.get_slices() + ga_ratings.get_slices()
        max_cont_rating = slices_not(
            high_power_ratings,
            begin_at=min(air.slice.start for air in airborne),
            end_at=max(air.slice.stop for air in airborne),
        )
        self.create_kpvs_within_slices(
            eng_egt_max.array,
            max_cont_rating,
            max_value,
        )


# TODO: Extract out code to create 'Eng (*) Start' KTIs?
class EngGasTempDuringEngStartMax(KeyPointValueNode):
    '''
    One key point value for maximum engine gas temperature at engine start for
    all engines. The value is taken from the engine with the largest value.
    '''

    units = 'C'

    @classmethod
    def can_operate(self, available):
        return all((
            any_of(('Eng (%d) Gas Temp' % n for n in range(1, 5)), available),
            any_of(('Eng (%d) N2' % n for n in range(1, 5)), available),
            'Takeoff Turn Onto Runway' in available,
        ))

    def peak_start_egt(self, egt, n2, idx):
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


class EngGasTempDuringFlightMin(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control. In flight engine shut down."

    To detect a possible engine shutdown in flight, we look for the minimum
    gas temperature recorded during the flight. The event will then be computed
    later, testing against a suitable minimum value for a running engine.
    '''

    units = 'C'

    def derive(self,
               eng_egt_min=P('Eng (*) Gas Temp Min'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(
            eng_egt_min.array,
            airborne,
            min_value,
        )


##############################################################################
# Engine N1


class EngN1DuringTaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 During Taxi Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_n1_max.array, taxiing, max_value)


class EngN1DuringApproachMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 During Approach Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               approaches=S('Approach')):

        self.create_kpv_from_slices(eng_n1_max.array, approaches, max_value)


class EngN1DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 During Takeoff 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 During Go Around 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n1_max.array, ratings, max_value)


class EngN1DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 During Maximum Continuous Power Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               grounded=S('Grounded')):

        slices = to_ratings + ga_ratings + grounded
        self.create_kpv_outside_slices(eng_n1_max.array, slices, max_value)


class EngN1CyclesDuringFinalApproach(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 Cycles During Final Approach'
    units = 'cycles'

    def derive(self,
               eng_n1_avg=P('Eng (*) N1 Avg'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                eng_n1_avg.array[fin_app.slice],
                5.0, 10.0, eng_n1_avg.hz,
                fin_app.slice.start,
            ))


class EngN1500To50FtMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 500 To 50 Ft Max'
    units = '%'

    def derive(self,
               eng_n1_max=P('Eng (*) N1 Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_n1_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngN1500To50FtMin(KeyPointValueNode):
    '''
    '''

    name = 'Eng N1 500 To 50 Ft Min'
    units = '%'

    def derive(self,
               eng_n1_min=P('Eng (*) N1 Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_n1_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


class EngN1WithThrustReversersInTransitMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Asymmetric selection or achieved."
    '''

    name = 'Eng N1 With Thrust Reversers In Transit Max'
    units = '%'

    def derive(self,
               eng_n1_avg=P('Eng (*) N1 Avg'),
               tr=M('Thrust Reversers'),
               landings=S('Landing')):

        slices = [s.slice for s in landings]
        slices = clump_multistate(tr.array, 'In Transit', slices)
        self.create_kpv_from_slices(eng_n1_avg.array, slices, max_value)


# NOTE: Was named 'Eng N1 Cooldown Duration'.
# TODO: Similar KPV for duration between engine under 60 percent and engine shutdown
class EngN1Below60PercentAfterTouchdownDuration(KeyPointValueNode):
    '''
    Max duration N1 below 60% after Touchdown for engine cooldown. Using 60%
    allows for cooldown after use of Reverse Thrust.

    Evaluated for each engine to account for single engine taxi-in.

    Note: Assumes that all Engines are recorded at the same frequency.
    '''

    NAME_FORMAT = 'Eng (%(number)d) N1 Below 60 Percent After Touchdown Duration'
    NAME_VALUES = NAME_VALUES_ENGINE
    units = 's'

    @classmethod
    def can_operate(cls, available):
        return all((
            any_of(('Eng (%d) N1' % n for n in range(1, 5)), available),
            'Eng (*) Stop' in available,
            'Touchdown' in available,
        ))

    def derive(self,
               engines_stop=KTI('Eng (*) Stop'),
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1'),
               tdwn=KTI('Touchdown')):

        if not tdwn:
            return
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


##############################################################################
# Engine N2


class EngN2DuringTaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 During Taxi Max'
    units = '%'

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_n2_max.array, taxiing, max_value)


class EngN2DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 During Takeoff 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 During Go Around 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n2_max.array, ratings, max_value)


class EngN2DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 During Maximum Continuous Power Max'
    units = '%'

    def derive(self,
               eng_n2_max=P('Eng (*) N2 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               grounded=S('Grounded')):

        slices = to_ratings + ga_ratings + grounded
        self.create_kpv_outside_slices(eng_n2_max.array, slices, max_value)


class EngN2CyclesDuringFinalApproach(KeyPointValueNode):
    '''
    '''

    name = 'Eng N2 Cycles During Final Approach'

    def derive(self,
               eng_n2_avg=P('Eng (*) N2 Avg'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                eng_n2_avg.array[fin_app.slice],
                10.0, 10.0, eng_n2_avg.hz,
                fin_app.slice.start,
            ))


##############################################################################
# Engine N3


class EngN3DuringTaxiMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 During Taxi Max'
    units = '%'

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_n3_max.array, taxiing, max_value)


class EngN3DuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 During Takeoff 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3DuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 During Go Around 5 Min Rating Max'
    units = '%'

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_n3_max.array, ratings, max_value)


class EngN3DuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    name = 'Eng N3 During Maximum Continuous Power Max'
    units = '%'

    def derive(self,
               eng_n3_max=P('Eng (*) N3 Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               grounded=S('Grounded')):

        slices = to_ratings + ga_ratings + grounded
        self.create_kpv_outside_slices(eng_n3_max.array, slices, max_value)


##############################################################################
# Engine Throttles


class ThrottleReductionToTouchdownDuration(KeyPointValueNode):
    '''
    Records the duration from touchdown until Throttle leaver is reduced in
    seconds, negative seconds indicates throttle reduced before touchdown.

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

    units = 's'

    def derive(self,
               tla=P('Throttle Levers'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown')):

        for landing in landings:
            for touchdown in touchdowns.get(within_slice=landing.slice):
                # Seek the throttle lowpoint before thrust reverse is applied:
                retard_idx = index_at_value(tla.array, 0.0, landing.slice,
                                            endpoint='closing')
                # The range of interest is therefore...
                scan = slice(landing.slice.start - 5 / tla.hz, retard_idx)
                # Now see where the power is reduced:
                reduce_idx = peak_curvature(tla.array, scan,
                                            curve_sense='Convex', gap=1, ttp=3)
                if reduce_idx:
                    value = (reduce_idx - touchdown.index) / tla.hz
                    self.create_kpv(reduce_idx, value)


##############################################################################
# Engine Oil Pressure


class EngOilPressMax(KeyPointValueNode):
    '''
    '''

    units = 'psi'

    def derive(self,
               oil_press=P('Eng (*) Oil Press Max')):

        self.create_kpv(*max_value(oil_press.array))


class EngOilPressMin(KeyPointValueNode):
    '''
    '''

    units = 'psi'

    def derive(self,
               oil_press=P('Eng (*) Oil Press Min'),
               airborne=S('Airborne')):

        # Only in flight to avoid zero pressure readings for stationary engines.
        self.create_kpvs_within_slices(oil_press.array, airborne, min_value)


##############################################################################
# Engine Oil Quantity


class EngOilQtyMax(KeyPointValueNode):
    '''
    '''

    units = 'kg'

    def derive(self,
               oil_qty=P('Eng (*) Oil Qty Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_qty.array, airborne, max_value)


class EngOilQtyMin(KeyPointValueNode):
    '''
    '''

    units = 'kg'

    def derive(self,
               oil_qty=P('Eng (*) Oil Qty Min'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_qty.array, airborne, min_value)


##############################################################################
# Engine Oil Temperature


class EngOilTempMax(KeyPointValueNode):
    '''
    '''

    units = 'C'

    def derive(self,
               oil_temp=P('Eng (*) Oil Temp Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(oil_temp.array, airborne, max_value)


class EngOilTempFor15MinMax(KeyPointValueNode):
    '''
    Maximum oil temperature sustained for 15 minutes.
    '''

    units = 'C'

    def derive(self,
               oil_temp=P('Eng (*) Oil Temp Max')):

        # Some aircraft don't have oil temperature sensors fitted. This trap
        # may be superceded by masking the Eng (*) Oil Temp Max parameter in
        # future:
        if np.ma.count(oil_temp.array) == 0:
            return
        oil_15 = clip(oil_temp.array, 15 * 60, oil_temp.hz)
        # There have been cases where there were no valid oil temperature
        # measurements throughout the flight, in which case there's no point
        # testing for a maximum:
        if oil_15 is not None:
            self.create_kpv(*max_value(oil_15))


##############################################################################
# Engine Torque


class EngTorqueDuringTaxiMax(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               taxiing=S('Taxiing')):

        self.create_kpv_from_slices(eng_trq_max.array, taxiing, max_value)


class EngTorqueDuringTakeoff5MinRatingMax(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Takeoff 5 Min Rating')):

        self.create_kpvs_within_slices(eng_trq_max.array, ratings, max_value)


class EngTorqueDuringGoAround5MinRatingMax(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               ratings=S('Go Around 5 Min Rating')):

        self.create_kpvs_within_slices(eng_trq_max.array, ratings, max_value)


class EngTorqueDuringMaximumContinuousPowerMax(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating'),
               grounded=S('Grounded')):

        slices = to_ratings + ga_ratings + grounded
        self.create_kpv_outside_slices(eng_trq_max.array, slices, max_value)


class EngTorque500To50FtMax(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_trq_max.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class EngTorque500To50FtMin(KeyPointValueNode):
    '''
    '''

    units = '%'

    def derive(self,
               eng_trq_min=P('Eng (*) Torque Min'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            eng_trq_min.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


##############################################################################
# Engine Vibrations


class EngVibN1Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng Vib N1 Max'
    units = ''

    def derive(self,
               eng_vib_n1=P('Eng (*) Vib N1 Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_n1.array, airborne, max_value)


class EngVibN2Max(KeyPointValueNode):
    '''
    '''

    name = 'Eng Vib N2 Max'
    units = ''

    def derive(self,
               eng_vib_n2=P('Eng (*) Vib N2 Max'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(eng_vib_n2.array, airborne, max_value)


##############################################################################


class EventMarkerPressed(KeyPointValueNode):
    def derive(self, event=P('Event Marker'), airs=S('Airborne')):
        pushed = np.ma.clump_unmasked(np.ma.masked_equal(event.array, 0))
        events_in_air = slices_and(pushed, airs.get_slices())
        for event_in_air in events_in_air:
            if event_in_air:
                duration = (event_in_air.stop - event_in_air.start) / \
                    event.frequency
                index = (event_in_air.stop + event_in_air.start) / 2.0
                self.create_kpv(index, duration)


class HeightOfBouncedLanding(KeyPointValueNode):
    '''
    This measures the peak height of the bounced landing.

    Bounced landing phase is established by looking for the maximum height
    after touching the ground while still going fast.
    '''
    def derive(self, alt = P('Altitude AAL'),
               bounced_landing=S('Bounced Landing')):
        self.create_kpvs_within_slices(alt.array, bounced_landing, max_value)


##############################################################################
# Heading


class HeadingDeviationFromRunwayAbove80KtsAirspeedDuringTakeoff(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). Heading changes on runway before rotation
    commenced. During rotation on some types, the a/c may be allowed to
    weathercock into wind."

    The heading deviation is measured as the largest deviation from the runway
    centreline between 80kts airspeed and 5 deg nose pitch up, at which time
    the weight is clearly coming off the mainwheels (we avoid using weight on
    nosewheel as this is often not recorded).
    '''

    def derive(self,
               head=P('Heading True Continuous'),
               airspeed=P('Airspeed'),
               pitch=P('Pitch'),
               toffs=S('Takeoff'),
               rwy=A('FDR Takeoff Runway')):

        if ambiguous_runway(rwy):
            return
        # checks for multiple takeoffs from a single takeoff runway (rejected?)
        for toff in toffs:
            start = index_at_value(airspeed.array, 80.0, _slice=toff.slice)
            if not start:
                self.warning("'%s' did not transition through 80 kts in '%s' "
                        "slice '%s'.", airspeed.name, toffs.name, toff.slice)
                continue
            stop = index_at_value(pitch.array, 5.0, _slice=toff.slice)
            if not stop:
                self.warning("'%s' did not transition through 5 deg in '%s' "
                        "slice '%s'.", pitch.name, toffs.name, toff.slice)
                continue
            scan = slice(start, stop)
            dev = runway_deviation(head.array, rwy.value)
            index, value = max_abs_value(dev, scan)
            self.create_kpv(index, value)


class HeadingDeviationFromRunwayAtTOGADuringTakeoff(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). TOGA pressed before a/c aligned."
    '''

    name = 'Heading Deviation From From Runway At TOGA During Takeoff'

    def derive(self,
               head=P('Heading True Continuous'),
               toga=M('Takeoff And Go Around'),
               takeoff=S('Takeoff'),
               rwy=A('FDR Takeoff Runway')):

        if ambiguous_runway(rwy):
            return
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            brg = value_at_index(head.array, index)
            dev = runway_deviation(brg, rwy.value)
            self.create_kpv(index, dev)


class HeadingDeviationFromRunwayAt50FtDuringLanding(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral). Crosswind. Could look at the difference
    between a/c heading and R/W heading at 50ft."
    '''

    def derive(self,
               head=P('Heading True Continuous'),
               landings=S('Landing'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy):
            return
        # Only have runway details for final landing.
        land = landings[-1]
        # By definition, landing starts at 50ft.
        brg = value_at_index(head.array, land.start_edge)
        dev = runway_deviation(brg, rwy.value)
        self.create_kpv(land.start_edge, dev)


class HeadingDeviationFromRunwayDuringLandingRoll(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) Heading changes on runways."
    '''

    def derive(self,
               head=P('Heading True Continuous'),
               land_rolls=S('Landing Roll'),
               rwy=A('FDR Landing Runway')):

        if ambiguous_runway(rwy):
            return

        final_landing = land_rolls[-1].slice
        dev = runway_deviation(head.array, rwy.value)
        self.create_kpv_from_slices(dev, [final_landing], max_abs_value)


class HeadingVariation500To50Ft(KeyPointValueNode):
    '''
    '''

    def derive(self,
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for band in alt_aal.slices_from_to(500, 50):
            dev = np.ma.ptp(head.array[band])
            self.create_kpv(band.stop, dev)


class HeadingVariationAbove100KtsAirspeedDuringLanding(KeyPointValueNode):
    '''
    For landing the Altitude AAL is used to detect start of landing to avoid
    variation from the use of different aircraft recording configurations.
    '''

    def derive(self,
               head=P('Heading Continuous'),
               airspeed=P('Airspeed'),
               alt=P('Altitude AAL For Flight Phases'),
               lands=S('Landing')):

        for land in lands:
            begin = index_at_value(alt.array, 1.0, _slice=land.slice)
            end = index_at_value(airspeed.array, 100.0, _slice=land.slice)
            if begin is None or begin > end:
                # Corrupt landing slices or landed below 100kts. Can happen!
                break
            else:
                head_dev = np.ma.ptp(head.array[begin:end + 1])
                self.create_kpv((begin + end) / 2, head_dev)


class HeadingVariationTouchdownPlus4SecTo60KtsAirspeed(KeyPointValueNode):
    '''
    Maximum difference in Magnetic Heading.
    '''

    def derive(self,
               head=P('Heading Continuous'),
               airspeed=P('Airspeed'),
               tdwns=KTI('Touchdown')):

        for tdwn in tdwns:
            begin = tdwn.index + 4.0 * head.frequency
            end = index_at_value(airspeed.array, 60.0, slice(begin, None))
            if end:
                # We found a suitable endpoint, so create a KPV...
                dev = np.ma.ptp(head.array[begin:end + 1])
                self.create_kpv(end, dev)


class HeadingVacatingRunway(KeyPointValueNode):
    '''
    Heading vacating runway is only used to try to identify handed
    runways in the absence of better information. See Approaches node.
    '''

    def derive(self,
               head=P('Heading Continuous'),
               off_rwys=KTI('Landing Turn Off Runway')):

        # To save taking modulus of the entire array, we'll do this in stages.
        for off_rwy in off_rwys:
            # We try to extend the index by five seconds to make a clear
            # heading change. The KTI is at the point of turnoff at which
            # moment the heading change can be very small.
            index = min(off_rwy.index + 5, len(head.array) - 1)
            value = head.array[index] % 360.0
            self.create_kpv(index, value)


##############################################################################


class HeightMinsToTouchdown(KeyPointValueNode):
    '''
    '''

    # TODO: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown

    NAME_FORMAT = 'Height ' + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES

    def derive(self,
               alt_aal=P('Altitude AAL'),
               mtt_kti=KTI('Mins To Touchdown')):

        for mtt in mtt_kti:
            # XXX: Assumes that the number will be the first part of the name:
            time = int(mtt.name.split(' ')[0])
            self.create_kpv(mtt.index, alt_aal.array[mtt.index], time=time)


class FlapAtGearDownSelection(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_sel_down=KTI('Gear Down Selection')):
        self.create_kpvs_at_ktis(flap.array, gear_sel_down)


class FlapWithGearUpMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear=M('Gear Down')):
        #TODO: use self.create_kpvs_where_state instead
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


class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    '''
    '''
    def derive(self, flap=P('Flap'), speedbrake=M('Speedbrake Selected'),
               airs=S('Airborne'), lands=S('Landing')):
        deployed = speedbrake.array == 'Deployed/Cmd Up'
        # Mask all values where speedbrake isn't deployed:
        deployed = mask_outside_slices(deployed, airs.get_slices())
        deployed = mask_inside_slices(deployed, lands.get_slices())
        deployed_slices = runs_of_ones(deployed)
        self.create_kpv_from_slices(flap.array, deployed_slices, max_value)


class FlareDuration20FtToTouchdown(KeyPointValueNode):
    '''
    '''
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
                self.create_kpv(
                    tdown.index,
                    (tdown.index - idx_20) / alt_aal.frequency)


class FlareDistance20FtToTouchdown(KeyPointValueNode):
    '''
    '''
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


##############################################################################
# Groundspeed


class GroundspeedMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               grounded=S('Grounded')):

        self.create_kpvs_within_slices(gnd_spd.array, grounded, max_value)


class GroundspeedWhileTaxiingStraightMax(KeyPointValueNode):
    '''
    Groundspeed while not turning is rarely an issue, so we compute only one
    KPV for taxi out and one for taxi in. The straight sections are identified
    by masking the turning phases and then testing the resulting data.
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_inside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedWhileTaxiingTurnMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               taxiing=S('Taxiing'),
               turns=S('Turning On Ground')):

        gnd_spd_array = mask_outside_slices(gnd_spd.array, turns.get_slices())
        self.create_kpvs_within_slices(gnd_spd_array, taxiing, max_value)


class GroundspeedDuringRejectedTakeoffMax(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               rtos=S('Rejected Takeoff')):

        self.create_kpvs_within_slices(gnd_spd.array, rtos, max_value)


class GroundspeedAtTouchdown(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(gnd_spd.array, touchdowns)


class GroundspeedVacatingRunway(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               off_rwy=KTI('Landing Turn Off Runway')):

        self.create_kpvs_at_ktis(gnd_spd.array, off_rwy)


class GroundspeedAtTOGA(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Selection of TOGA late in take-off
    roll."

    This KPV measures the groundspeed at the point of TOGA selection,
    irrespective of whether this is late (or early!).

    Note: Takeoff phase is used as this includes turning onto the runway
          whereas Takeoff Roll only starts after the aircraft is accelerating.
    '''

    name = 'Groundspeed At TOGA'
    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               toga=M('Takeoff And Go Around'),
               takeoffs=S('Takeoff')):

        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoffs)
        for index in indexes:
            self.create_kpv(index, value_at_index(gnd_spd.array, index))


class GroundspeedWithThrustReversersDeployedMin(KeyPointValueNode):
    '''
    '''

    units = 'kt'

    def derive(self,
               gnd_spd=P('Groundspeed'),
               tr=M('Thrust Reversers'),
               power=P('Eng (*) N1 Max'),
               landings=S('Landing')):

        for landing in landings:
            high_rev = thrust_reversers_working(landing, power, tr)
            self.create_kpvs_within_slices(gnd_spd.array, high_rev, min_value)


##############################################################################
# Pitch


class PitchAfterFlapRetractionMax(KeyPointValueNode):
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

    units = 'deg'

    def derive(self,
               flap=P('Flap'),
               pitch=P('Pitch'),
               airborne=S('Airborne')):

        scope = []
        for air in airborne:
            clean = np.ma.masked_greater(flap.array[air.slice], 0.0)
            slices = np.ma.clump_unmasked(clean)
            if not slices:
                continue
            scope.append(slice(air.slice.start + slices[0].start, air.slice.stop))
        self.create_kpvs_within_slices(pitch.array, scope, max_value)


class PitchAtLiftoff(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(pitch.array, liftoffs)


class PitchAtTouchdown(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(pitch.array, touchdowns)


class PitchAt35FtDuringClimb(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL')):

        # Q: Should we create a KPV method for this?
        for climb in alt_aal.slices_from_to(1, 100):
            index = index_at_value(alt_aal.array, 35.0, climb)
            if index:
                value = value_at_index(pitch.array, index)
                self.create_kpv(index, value)


class PitchLiftoffTo35FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(0, 35),
            max_value,
        )


class Pitch35To400FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(35, 400),
            max_value,
        )


class Pitch35To400FtMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(35, 400),
            min_value,
        )


class Pitch400To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(400, 1000),
            max_value,
        )


class Pitch400To1000FtMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(400, 1000),
            min_value,
        )


class Pitch1000To500FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(1000, 500),
            max_value,
        )


class Pitch1000To500FtMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class Pitch500To50FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 50),
            max_value,
        )


class Pitch500To20FtMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_from_to(500, 20),
            min_value,
        )


class Pitch50FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(50, touchdowns),
            max_value,
        )


class Pitch20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


class Pitch7FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_aal.slices_to_kti(7, touchdowns),
            min_value,
        )


class PitchCyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of pitch attitude that exceed 3 deg in
    pitch from peak to peak and with a maximum cycle period of 10 seconds
    during the final approach phase.
    '''

    units = 'cycles'

    def derive(self,
               pitch=P('Pitch'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                pitch.array[fin_app.slice],
                3.0, 10.0, pitch.hz,
                fin_app.slice.start,
            ))


class PitchDuringGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A - ...Rotation to 12 deg pitch..."
    '''

    units = 'deg'

    def derive(self,
               pitch=P('Pitch'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(pitch.array, go_arounds, max_value)


##############################################################################
# Pitch Rate


class PitchRate35To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg/s'

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_from_to(35, 1000),
            max_value,
        )


class PitchRate20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    units = 'deg/s'

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_value,
        )


class PitchRate20FtToTouchdownMin(KeyPointValueNode):
    '''
    '''

    units = 'deg/s'

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            alt_aal.slices_to_kti(20, touchdowns),
            min_value,
        )


class PitchRate2DegPitchTo35FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg/s'

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            two_deg_pitch_to_35ft,
            max_value,
        )


class PitchRate2DegPitchTo35FtMin(KeyPointValueNode):
    '''
    '''

    units = 'deg/s'

    def derive(self,
               pitch_rate=P('Pitch Rate'),
               two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_within_slices(
            pitch_rate.array,
            two_deg_pitch_to_35ft,
            min_value,
        )


##############################################################################
# Vertical Speed (Rate of Climb/Descent) Helpers


def vert_spd_phase_max_or_min(obj, vrt_spd, phases, function):
    '''
    Vertical Speed (Rate of Climb/Descent) Helper
    '''
    for phase in phases:
        duration = phase.slice.stop - phase.slice.start
        if duration > CLIMB_OR_DESCENT_MIN_DURATION:
            index, value = function(vrt_spd.array, phase.slice)
            obj.create_kpv(index, value)


##############################################################################
# Rate of Climb


class RateOfClimbMax(KeyPointValueNode):
    '''
    In cases where the aircraft does not leave the ground, we get a descending
    phase that equates to an empty list, which is not iterable.
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               climbing=S('Climbing')):

        vert_spd_phase_max_or_min(self, vrt_spd, climbing, max_value)


class RateOfClimb35To1000FtMin(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(35, 1000),
            min_value,
        )


class RateOfClimbBelow10000FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Airborne Conflict (Mid-Air Collision) Excessive rates of climb/descent
    (>3,000FPM) within a TMA (defined as < 10,000ft)"
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(0, 10000),
            max_value,
        )


class RateOfClimbDuringGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A." Concern here is excessive rates of
    climb following enthusiastic application of power and pitch up.
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(vrt_spd.array, go_arounds, max_value)


##############################################################################
# Rate of Descent


# FIXME: Should rate of descent KPVs should occur for 3+ seconds?


class RateOfDescentMax(KeyPointValueNode):
    '''
    In cases where the aircraft does not leave the ground, we get a descending
    phase that equates to an empty list, which is not iterable.
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               descending=S('Descending')):

        vert_spd_phase_max_or_min(self, vrt_spd, descending, min_value)


class RateOfDescentTopOfDescentTo10000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               descents=S('Descent')):

        for descent in descents:
            above_10k = np.ma.masked_less(alt_aal.array, 10000)
            drops = np.ma.clump_unmasked(above_10k)
            self.create_kpvs_within_slices(vrt_spd.array, drops, min_value)


class RateOfDescentBelow10000FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Airborne Conflict (Mid-Air Collision) Excessive rates of climb/descent
    (>3,000FPM) within a TMA (defined as < 10,000ft)"
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(10000, 0),
            min_value,
        )


class RateOfDescent10000To5000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(10000, 5000),
            min_value,
        )

class RateOfDescent5000To3000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(5000, 3000),
            min_value,
        )

class RateOfDescent3000To2000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(3000, 2000),
            min_value,
        )


class RateOfDescent2000To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(2000, 1000),
            min_value,
        )


class RateOfDescent1000To500FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(1000, 500),
            min_value,
        )


class RateOfDescent500To50FtMax(KeyPointValueNode):
    '''
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_from_to(500, 50),
            min_value,
        )


class RateOfDescent50FtToTouchdownMax(KeyPointValueNode):
    '''
    Rate of descent between 50ft and touchdown.

    At this altitude, Altitude AAL is sourced from Altitude Radio where one
    is available, so this is effectively 50ft Radio to touchdown.

    The ground effect compressibility makes the normal pressure altitude
    based vertical speed meaningless, so we use the more complex inertial
    computation to give accurate measurements within ground effect.
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_aal.slices_to_kti(50, touchdowns),
            min_value,
        )


# FIXME: Should this use the KTI 'Touchdown'?
class RateOfDescentAtTouchdown(KeyPointValueNode):
    '''
    We use the inertial vertical speed to avoid ground effects and give an
    accurate value at the point of touchdown.
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_aal=P('Altitude AAL'),
               landings=S('Landing')):

        for landing in landings:
            index, rod = touchdown_inertial(landing, vrt_spd, alt_aal)
            if index:
                self.create_kpv(index, rod)


class RateOfDescentDuringGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A."
    '''

    units = 'fpm'

    def derive(self,
               vrt_spd=P('Vertical Speed'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(vrt_spd.array, go_arounds, min_value)


##### TODO: Implement!
####class RateOfDescentOverGrossWeightLimitAtTouchdown(KeyPointValueNode):
####    '''
####    '''
####
####    units = 'fpm'
####
####    def derive(self, x=P('Not Yet')):
####        '''
####        '''
####        return NotImplemented


##############################################################################
# Roll


class RollLiftoffTo20FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(1, 20),
            max_abs_value,
        )


class Roll20To400FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(20, 400),
            max_abs_value,
        )


class Roll400To1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(400, 1000),
            max_abs_value,
        )


class RollAbove1000FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_above(1000),
            max_abs_value,
        )


class Roll1000To300FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(1000, 300),
            max_abs_value,
        )


class Roll300To20FtMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_from_to(300, 20),
            max_abs_value,
        )


class Roll20FtToTouchdownMax(KeyPointValueNode):
    '''
    '''

    units = 'deg'

    def derive(self,
               roll=P('Roll'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            roll.array,
            alt_aal.slices_to_kti(20, touchdowns),
            max_abs_value,
        )


class RollCyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of cycles of roll attitude that exceed 5 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    final approach phase.

    The algorithm counts each half-cycle, so an "N" figure would give a value
    of 1.5 cycles.
    '''

    units = 'cycles'

    def derive(self,
               roll=P('Roll'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                roll.array[fin_app.slice],
                5.0, 10.0, roll.hz,
                fin_app.slice.start,
            ))


class RollCyclesNotDuringFinalApproach(KeyPointValueNode):
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

    units = 'cycles'

    def derive(self,
               roll=P('Roll'),
               airborne=S('Airborne'),
               fin_apps=S('Final Approach'),
               landings=S('Landing')):

        not_fas = slices_and_not(airborne, fin_apps)
        # TODO: Fix this:
        #not_fas = slices_and_not(not_fas, landings)
        for not_fa in not_fas:
            self.create_kpv(*cycle_counter(
                roll.array[not_fa],
                5.0, 10.0, roll.hz,
                not_fa.start,
            ))


##############################################################################
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


##############################################################################
# Speedbrake


class SpeedbrakeDeployed1000To20FtDuration(KeyPointValueNode):
    '''
    '''

    units = 's'

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for descent in alt_aal.slices_from_to(1000, 20):
            array = spd_brk.array[descent] == 'Deployed/Cmd Up'
            slices = shift_slices(runs_of_ones(array), descent.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


# TODO: Properly test this and compare with flap version above!
class SpeedbrakeDeployedWithConfDuration(KeyPointValueNode):
    '''
    Conf used here, but not tried or tested. Presuming conf 2 / conf 3 should
    not be used with speedbrakes.
    '''

    units = 's'

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               conf=P('Configuration'),
               airborne=S('Airborne')):

        for air in airborne:
            spd_brk_dep = spd_brk.array[air.slice] == 'Deployed/Cmd Up'
            conf_extend = conf.array[air.slice] >= 2.0
            array = spd_brk_dep & conf_extend
            slices = shift_slices(runs_of_ones(array), air.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class SpeedbrakeDeployedWithFlapDuration(KeyPointValueNode):
    '''
    '''

    units = 's'

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               flap=P('Flap'),
               airborne=S('Airborne')):

        for air in airborne:
            spd_brk_dep = spd_brk.array[air.slice] == 'Deployed/Cmd Up'
            flap_extend = flap.array[air.slice] > 0
            array = spd_brk_dep & flap_extend
            slices = shift_slices(runs_of_ones(array), air.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class SpeedbrakeDeployedWithPowerOnDuration(KeyPointValueNode):
    '''
    Each time the aircraft is flown with high power and the speedbrakes open,
    something unusual is happening. We record the duration this happened for,
    and allow the analyst to find out the cause.

    The threshold for high power is 50% N1 for most aircraft, but 60% for
    Airbus types, to align with the Airbus AFPS.
    '''

    units = 's'

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               power=P('Eng (*) N1 Avg'),
               airborne=S('Airborne'),
               manufacturer=A('Manufacturer')):

        percent = 60.0 if manufacturer == 'Airbus' else 50.0
        for air in airborne:
            spd_brk_dep = spd_brk.array[air.slice] == 'Deployed/Cmd Up'
            high_power = power.array[air.slice] >= percent
            array = spd_brk_dep & high_power
            slices = shift_slices(runs_of_ones(array), air.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


class SpeedbrakeDeployedDuringGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A - ...Speedbrake retraction."
    '''

    units = 's'

    def derive(self,
               spd_brk=M('Speedbrake Selected'),
               go_arounds=S('Go Around And Climbout')):

        deployed = spd_brk.array == 'Deployed/Cmd Up'
        for go_around in go_arounds:
            array = deployed[go_around.slice]
            slices = shift_slices(runs_of_ones(array),
                                  go_around.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency,
                                                  mark='start')


##############################################################################
# Warnings: Stick Pusher/Shaker


# TODO: Check that this triggers correctly as stick push events are probably
#       single samples.
class StickPusherActivatedDuration(KeyPointValueNode):
    '''
    We annotate the stick pusher event with the duration of the event.
    '''

    units = 's'

    def derive(self, stick_pusher=M('Stick Pusher'), airs=S('Airborne')):
        self.create_kpvs_where_state(
            'Push',
            stick_pusher.array,
            stick_pusher.hz,
            airs,
        )


class StickShakerActivatedDuration(KeyPointValueNode):
    '''
    We annotate the stick shaker event with the duration of the event.
    '''

    units = 's'

    def derive(self, stick_shaker=M('Stick Shaker'), airs=S('Airborne')):
        self.create_kpvs_where_state(
            'Shake',
            stick_shaker.array,
            stick_shaker.hz,
            airs,
        )


##############################################################################
# Tail Clearance


class TailClearanceDuringTakeoffMin(KeyPointValueNode):
    '''
    '''

    units = 'ft'

    def derive(self,
               alt_tail=P('Altitude Tail'),
               takeoffs=S('Takeoff')):

        self.create_kpvs_within_slices(alt_tail.array, takeoffs, min_value)


class TailClearanceDuringLandingMin(KeyPointValueNode):
    '''
    '''

    units = 'ft'

    def derive(self,
               alt_tail=P('Altitude Tail'),
               landings=S('Landing')):

        self.create_kpvs_within_slices(alt_tail.array, landings, min_value)


class TailClearanceDuringApproachMin(KeyPointValueNode):
    '''
    This finds abnormally low tail clearance during the approach down to 100ft.
    It searches for the minimum angular separation between the flightpath and
    the terrain, so a 500ft clearance at 2500ft AAL is considered more
    significant than 500ft at 1500ft AAL. The value stored is the tail
    clearance. A matching KTI will allow these to be located on the approach
    chart.
    '''

    units = 'ft'

    def derive(self,
               alt_aal=P('Altitude AAL'),
               alt_tail=P('Altitude Tail'),
               dtl=P('Distance To Landing')):

        for desc_slice in alt_aal.slices_from_to(3000, 100):
            angle_array = alt_tail.array[desc_slice] \
                / (dtl.array[desc_slice] * FEET_PER_NM)
            index, value = min_value(angle_array)
            if index:
                sample = index + desc_slice.start
                self.create_kpv(sample, alt_tail.array[sample])


##############################################################################
# Terrain Clearance


class TerrainClearanceAbove3000FtMin(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Controlled Flight Into Terrain (CFIT) At/Below Minimum terrain clearance
    on approach/departure >3000ft AFE and <1000ft AGL"

    Solution: Compute minimum terrain clearance while Altitude AAL over 3000ft.
    Note: For most flights, Altitude Radio will be over 2,500ft at this time,
    so masked, hence no kpv will be created.
    '''

    units = 'ft'

    def derive(self,
               alt_rad=P('Altitude Radio'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            alt_rad.array,
            alt_aal.slices_above(3000),
            min_value,
        )


##############################################################################
# Tailwind


class TailwindLiftoffTo100FtMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Tailwind - Needs to be recorded just
    after take-off.

    CAA comment: Some operators will have purchased (AFM) a 15kt tailwind limit
    for take-off. But this should only be altered to 15 kt if it has been
    purchased.

    This event masks the tailwind array so that headwind conditions do not
    raise any KPV.
    '''

    def derive(self,
               tailwind=P('Tailwind'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        self.create_kpvs_within_slices(
            np.ma.masked_less_equal(tailwind.array, 0.0),
            alt_aal.slices_from_to(0, 100),
            max_value,
        )


class Tailwind100FtToTouchdownMax(KeyPointValueNode):
    '''
    This event uses a masked tailwind array so that headwind conditions do
    not raise any KPV.
    '''

    def derive(self,
               tailwind=P('Tailwind'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            np.ma.masked_less_equal(tailwind.array, 0.0),
            alt_aal.slices_to_kti(100, touchdowns),
            max_value,
        )


##############################################################################
# Warnings: Takeoff Configuration Warning


class TakeoffConfigWarningDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Take-off config warning during
    takeoff roll."
    '''

    units = 's'

    def derive(self,
               config_warning=M('Takeoff Config Warning'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_where_state(
            'Warning',
            config_warning.array,
            config_warning.hz,
            phase=takeoff_rolls,
        )


class MasterWarningDuringTakeoffDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take-Off (Longitudinal), Master Caution or Master Warning
    triggered during takeoff. The idea of this is to inform the analyst of
    any possible distractions to the pilot"
    '''

    units = 's'

    def derive(self,
               warning=M('Master Warning'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_where_state(
            'Warning',
            warning.array,
            warning.hz,
            phase=takeoff_rolls,
        )


class MasterCautionDuringTakeoffDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Master Warning In Takeoff Duration".
    '''

    units = 's'

    def derive(self,
               caution=M('Master Caution'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_where_state(
            'Caution',
            caution.array,
            caution.hz,
            phase=takeoff_rolls,
        )


##############################################################################
# Warnings: Terrain Awareness & Warning System (TAWS)


class TAWSAlertDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS Alert Duration'
    units = 's'

    def derive(self,
               taws_alert=M('TAWS Alert'),
               airborne=S('Airborne')):

        self.create_kpvs_where_state(
            'Alert',
            taws_alert.array,
            taws_alert.hz,
            phase=airborne
        )


class TAWSGeneralWarningDuration(KeyPointValueNode):
    '''
    '''

    name = 'TAWS General Warning Duration'
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

    def derive(self,
               taws_terrain_pull_up=M('TAWS Terrain Pull Up'),
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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

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
    units = 's'

    def derive(self,
               taws_windshear=M('TAWS Windshear Warning'),
               alt_aal=P('Altitude AAL For Flight Phases')):

        for descent in alt_aal.slices_from_to(1500, 0):
            self.create_kpvs_where_state(
                'Warning',
                taws_windshear.array[descent],
                taws_windshear.hz,
            )


##############################################################################
# Warnings: Traffic Collision Avoidance System (TCAS)


class TCASTAWarningDuration(KeyPointValueNode):
    '''
    This is simply the number of seconds during which the TCAS TA was set.
    '''

    name = 'TCAS TA Warning Duration'
    units = 's'

    def derive(self, tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        
        for air in airs:
            ras_local = tcas.array[air.slice] == 'Preventive'
            ras_slices = shift_slices(runs_of_ones(ras_local), air.slice.start)
            self.create_kpvs_from_slice_durations(ras_slices, self.frequency,
                                                  mark='start')


class TCASRAWarningDuration(KeyPointValueNode):
    '''
    This is simply the number of seconds during which the TCAS RA was set.
    '''

    name = 'TCAS RA Warning Duration'
    units = 's'

    def derive(self, tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        
        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective')
            
            ras_slices = shift_slices(runs_of_ones(ras_local), air.slice.start)
            self.create_kpvs_from_slice_durations(ras_slices, self.frequency,
                                                  mark='start')


class TCASRAReactionDelay(KeyPointValueNode):
    '''
    This measures the time taken for the pilot to react, determined by the onset
    of the first major change in normal acceleration after the RA started.
    '''

    name = 'TCAS RA Reaction Delay'
    units = 's'

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        
        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective')
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
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


class TCASRAInitialReactionStrength(KeyPointValueNode):
    '''
    This measures the strength of the first reaction to the RA, in g per second.
    Most importantly, this is positive if the reaction is in the same sense as
    the Resolution Advisory (up for up or down for down) but negative in sign if
    the action is in the opposite direction to the RA.

    This is an ideal parameter for raising safety events when the pilot took
    the wrong initial action.
    '''

    name = 'TCAS RA Initial Reaction Strength'

    def derive(self, acc=P('Acceleration Normal Offset Removed'),
               tcas=M('TCAS Combined Control'), airs=S('Airborne')):
        
        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective')
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
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


class TCASRAToAPDisengagedDuration(KeyPointValueNode):
    '''
    Here we calculate the time between the onset of the RA and disconnection of
    the autopilot.

    Since the pilot's initial action should be to disengage the autopilot,
    this duration is another indication of pilot reaction time.
    '''

    name = 'TCAS RA To AP Disengaged Duration'
    units = 's'

    def derive(self,
               ap_offs=KTI('AP Disengaged Selection'),
               tcas=M('TCAS Combined Control'),
               airs=S('Airborne')):

        for air in airs:
            ras_local = tcas.array[air.slice].any_of('Drop Track',
                                                     'Altitude Lost',
                                                     'Up Advisory Corrective',
                                                     'Down Advisory Corrective')            
            ras = shift_slices(runs_of_ones(ras_local), air.slice.start)
            # Assume that the reaction takes place during the TCAS RA period:
            for ra in ras:
                ap_off = ap_offs.get_next(ra.start, within_slice=ra)
                if not ap_off:
                    continue
                index = ap_off.index
                duration = (index - ra.start) / self.frequency
                self.create_kpv(index, duration)


##############################################################################
# Warnings: Alpha Floor, Alternate Law, Direct Law


##### TODO: Implement!
####class AlphaFloorWarningDuration(KeyPointValueNode):
####    '''
####    '''
####
####    units = 's'
####
####    def derive(self,
####               alpha_floor=M('Alpha Floor Warning'),
####               airborne=S('Airborne')):
####
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
####    units = 's'
####
####    def derive(self,
####               alternate_law=M('Alternate Law Warning')):
####               airborne=S('Airborne')):
####
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
####    units = 's'
####
####    def derive(self,
####               direct_law=M('Direct Law Warning')):
####               airborne=S('Airborne')):
####
####        self.create_kpvs_where_state(
####            'Warning',
####            direct_law.array,
####            direct_law.hz,
####            phase=airborne,
####        )


##############################################################################
# Throttle


class ThrottleCyclesDuringFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of throttle lever movement that exceed
    10 deg peak to peak and with a maximum cycle period of 14 seconds during
    the final approach phase.
    '''

    units = 'cycles'

    def derive(self,
               levers=P('Throttle Levers'),
               fin_apps=S('Final Approach')):

        for fin_app in fin_apps:
            self.create_kpv(*cycle_counter(
                levers.array[fin_app.slice],
                10.0, 10.0, levers.hz,
                fin_app.slice.start,
            ))


##############################################################################
# Thrust Asymmetry


class ThrustAsymmetryDuringTakeoffMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Take off (Lateral)" & "Loss of Control Significant torque
    or thrust split during T/O or G/A"
    '''

    units = '%'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               takeoff_rolls=S('Takeoff Roll')):

        self.create_kpvs_within_slices(ta.array, takeoff_rolls, max_value)


class ThrustAsymmetryDuringFlightMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Asymmetric thrust - may be due to an a/t fault"
    '''

    units = '%'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(ta.array, airborne, max_value)


class ThrustAsymmetryDuringGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Significant torque or thrust split during T/O or G/A"
    '''

    units = '%'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               go_arounds=S('Go Around And Climbout')):

        self.create_kpvs_within_slices(ta.array, go_arounds, max_value)


class ThrustAsymmetryDuringApproachMax(KeyPointValueNode):
    '''
    Peak thrust asymmetry on approach. A good KPV for providing measures on
    every flight, and preferred to the ThrustAsymmetryOnApproachDuration
    which will normally not record any value.
    '''

    units = '%'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               approaches=S('Approach')):

        self.create_kpvs_within_slices(ta.array, approaches, max_value)


class ThrustAsymmetryWithThrustReversersDeployedMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral) - Asymmetric reverse thrust".

    A good KPV for providing measures on every flight, and preferred to the
    ThrustAsymmetryWithReverseThrustDuration which will normally not record
    any value.
    '''

    units = '%'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               tr=M('Thrust Reversers'),
               mobile=S('Mobile')):

        # Note: Inclusion of the 'Mobile' phase ensures use of thrust reverse
        #       late on the landing run is included, but corrupt data at engine
        #       start etc. should be rejected.
        # Note: Use not 'Stowed' as 'In Transit' implies partially 'Deployed':
        slices = clump_multistate(tr.array, 'Stowed', mobile.get_slices(),
                                  condition=False)
        self.create_kpvs_within_slices(ta.array, slices, max_value)


class ThrustAsymmetryDuringApproachDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10%. Included for customers with
    existing events using this approach.
    '''

    units = 's'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               approaches=S('Approach')):

        for approach in approaches:
            asymmetry = np.ma.masked_less(ta.array[approach.slice], 10.0)
            slices = np.ma.clump_unmasked(asymmetry)
            slices = shift_slices(slices, approach.slice.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency)


class ThrustAsymmetryWithThrustReversersDeployedDuration(KeyPointValueNode):
    '''
    Durations of thrust asymmetry over 10% with reverse thrust operating.
    Included for customers with existing events using this approach.
    '''

    units = 's'

    def derive(self,
               ta=P('Thrust Asymmetry'),
               tr=M('Thrust Reversers'),
               mobile=S('Mobile')):

        # Note: Inclusion of the 'Mobile' phase ensures use of thrust reverse
        #       late on the landing run is included, but corrupt data at engine
        #       start etc. should be rejected.
        slices = [s.slice for s in mobile]
        # Note: Use not 'Stowed' as 'In Transit' implies partially 'Deployed':
        slices = clump_multistate(tr.array, 'Stowed', slices, condition=False)
        for slice_ in slices:
            asymmetry = np.ma.masked_less(ta.array[slice_], 10.0)
            slices = np.ma.clump_unmasked(asymmetry)
            slices = shift_slices(slices, slice_.start)
            self.create_kpvs_from_slice_durations(slices, self.frequency)


##############################################################################


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


##############################################################################
# Turbulence


class TurbulenceDuringApproachMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               turbulence=P('Turbulence RMS g'),
               approaches=S('Approach')):

        self.create_kpvs_within_slices(turbulence.array, approaches, max_value)


class TurbulenceDuringCruiseMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               turbulence=P('Turbulence RMS g'),
               cruises=S('Cruise')):

        self.create_kpvs_within_slices(turbulence.array, cruises, max_value)


class TurbulenceDuringFlightMax(KeyPointValueNode):
    '''
    '''

    units = 'g'

    def derive(self,
               turbulence=P('Turbulence RMS g'),
               airborne=S('Airborne')):

        self.create_kpvs_within_slices(turbulence.array, airborne, max_value)


##############################################################################
# Wind


class WindSpeedAtAltitudeDuringDescent(KeyPointValueNode):
    '''
    Note: We align to Altitude AAL for cosmetic reasons; alignment to wind
          speed leads to slightly misaligned KPVs for wind speed, which looks
          wrong although is arithmetically "correct".
    '''

    NAME_FORMAT = 'Wind Speed At %(altitude)d Ft During Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}
    units = 'kt'

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               wind_spd=P('Wind Speed')):

        for descent in alt_aal.slices_from_to(2100, 0):
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, altitude, descent)
                if not index:
                    continue
                value = value_at_index(wind_spd.array, index)
                if value:
                    self.create_kpv(index, value, altitude=altitude)


class WindDirectionAtAltitudeDuringDescent(KeyPointValueNode):
    '''
    Note: We align to Altitude AAL for cosmetic reasons; alignment to wind
          direction leads to slightly misaligned KPVs for wind direction, which
          looks wrong although is arithmetically "correct".
    '''

    NAME_FORMAT = 'Wind Direction At %(altitude)d Ft During Descent'
    NAME_VALUES = {'altitude': [2000, 1500, 1000, 500, 100, 50]}
    units = 'kt'

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               wind_dir=P('Wind Direction Continuous')):

        for descent in alt_aal.slices_from_to(2100, 0):
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, altitude, descent)
                if not index:
                    continue
                # Check direction not masked before using % 360:
                value = value_at_index(wind_dir.array, index)
                if value:
                    self.create_kpv(index, value % 360.0, altitude=altitude)


class WindAcrossLandingRunwayAt50Ft(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Excursions - Landing (Lateral). Crosswind - needs to be recorded just
    before landing, say at 50ft.
    '''

    units = 'kt'

    def derive(self,
               walr=P('Wind Across Landing Runway'),
               landings=S('Landing')):
        for landing in landings:
            index = landing.slice.start  # Landing starts at 50ft!
            value = walr.array[index]
            if value is not None:
                self.create_kpv(index, value)


##############################################################################
# Weight


class GrossWeightAtLiftoff(KeyPointValueNode):
    '''
    '''

    units = 'kg'

    def derive(self,
               gross_wgt=P('Gross Weight Smoothed'),
               liftoffs=KTI('Liftoff')):

        self.create_kpvs_at_ktis(gross_wgt.array, liftoffs)


class GrossWeightAtTouchdown(KeyPointValueNode):
    '''
    '''

    units = 'kg'

    def derive(self,
               gross_wgt=P('Gross Weight Smoothed'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_at_ktis(gross_wgt.array, touchdowns)


class ZeroFuelWeight(KeyPointValueNode):
    '''
    The aircraft zero fuel weight is computed from the recorded gross weight
    and fuel data.

    See also the GrossWeightSmoothed calculation which uses fuel flow data to
    obtain a higher sample rate solution to the aircraft weight calculation,
    with a best fit to the available weight data.
    '''

    units = 'kg'

    def derive(self,
               fuel_qty=P('Fuel Qty'),
               gross_wgt=P('Gross Weight')):

        self.create_kpv(0, np.ma.median(gross_wgt.array - fuel_qty.array))


##############################################################################


class HoldingDuration(KeyPointValueNode):
    """
    Identify time spent in the hold.
    """
    def derive(self, holds=S('Holding')):
        self.create_kpvs_from_slice_durations(holds, self.frequency, mark='end')


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

##############################################################################
# Go Around Related KPVs 

#See also: EngGasTempGoAroundMax, EngN1GoAroundMax, EngN2GoAroundMax,
#EngN3GoAroundMax, EngTorqueGoAroundMax


class TOGASelectedInGoAroundDuration(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control - TOGA power selection in flight (Go-arounds need to be
    kept as a separate case)."
    '''

    name = 'TOGA Selected In Go Around Duration'

    def derive(self,
               toga=M('Takeoff And Go Around'),
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
    '''
    Finds the relative altitude at which gear up was selected from the point of
    minimum altitude in the go-around. If gear up was selected before that,
    just set the value to zero.
    '''

    def derive(self,
               alt_aal=P('Altitude AAL'),
               gas=S('Go Around And Climbout'),
               gear_ups=KTI('Go Around Gear Selected Up')):

        for ga in gas:
            # Find the index and height at this go-around minimum:
            pit_index, pit_value = min_value(alt_aal.array, ga.slice)
            for gear_up in gear_ups:
                # Ensure gear up selected matches for this go-around:
                if not is_index_within_slice(gear_up.index, ga.slice):
                    continue
                if gear_up.index > pit_index:
                    # Use height between go around minimum and gear up:
                    gear_up_ht = alt_aal.array[gear_up.index] - pit_value
                else:
                    # Use zero if gear up selected before minimum height:
                    gear_up_ht = 0.0
                self.create_kpv(gear_up.index, gear_up_ht)


class AOAInGoAroundMax(KeyPointValueNode):
    '''
    FDS developed this KPV to support the UK CAA Significant Seven programme.
    "Loss of Control Mis-handled G/A"
    '''
    name = 'AOA In Go Around Max'
    def derive(self, aoa=P('AOA'), gas=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(aoa.array, gas, max_value)


# NOTE: Python class name restriction: '2 Deg Pitch To 35 Ft Duration'
class TwoDegPitchTo35FtDuration(KeyPointValueNode):
    '''
    Time taken for aircraft to reach 35ft after rotating to 2 degrees pitch.
    '''

    name = '2 Deg Pitch To 35 Ft Duration'
    units = 's'

    def derive(self,
               two_deg_pitch_to_35ft=S('2 Deg Pitch To 35 Ft')):

        self.create_kpvs_from_slice_durations(
            two_deg_pitch_to_35ft,
            self.frequency,
            mark='midpoint',
        )


