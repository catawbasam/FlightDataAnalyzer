import numpy as np

from analysis_engine import settings
from analysis_engine.settings import (CONTROL_FORCE_THRESHOLD,
                                      FEET_PER_NM,
                                      NAME_VALUES_FLAP)

from analysis_engine.node import KeyPointValueNode, KPV, KTI, P, S, A

from analysis_engine.library import (clip, 
                                     coreg, 
                                     cycle_counter,
                                     cycle_finder,
                                     _dist,
                                     find_edges,
                                     hysteresis,
                                     ils_glideslope_align,
                                     index_at_value, 
                                     integrate,
                                     is_index_within_sections,
                                     mask_outside_slices,
                                     max_abs_value,
                                     max_continuous_unmasked, 
                                     max_value,
                                     min_value, 
                                     minimum_unmasked,
                                     repair_mask, 
                                     np_ma_masked_zeros_like,
                                     peak_curvature,
                                     rate_of_change,
                                     runway_distance_from_end,
                                     shift_slices,
                                     slice_samples, 
                                     slices_overlap,
                                     slices_and,
                                     value_at_index)


class AccelerationLateralMax(KeyPointValueNode):
    def derive(self, acc_lat=P('Acceleration Lateral')):
        index, value = max_abs_value(acc_lat.array)
        self.create_kpv(index, value)


class AccelerationLateralTaxiingStraightMax(KeyPointValueNode):
    '''
    Lateral acceleration while not turning is rarely an issue, so we compute
    only one KPV for taxi out and one for taxi in. The straight sections are
    identified by masking the turning phases and then testing the resulting
    data.
    '''
    def derive(self, acc_lat=P('Acceleration Lateral'), taxis=S('Taxiing'), 
               turns=S('Turning On Ground')):
        accel = np.ma.copy(acc_lat.array) # Prepare to change mask here.
        for turn in turns:
            accel[turn.slice]=np.ma.masked
        self.create_kpvs_within_slices(accel, taxis, max_abs_value)
    

class AccelerationLateralTaxiingTurnsMax(KeyPointValueNode):
    '''
    Lateral acceleration while taxiing normally occurs in turns, and leads to
    wear on the undercarriage and discomfort for passengers. In extremis this
    can lead to taxiway excursions. Lateral acceleration is used in
    preference to groundspeed as this parameter is available on older
    aircraft and is directly related to comfort.
    '''
    def derive(self, acc_lat=P('Acceleration Lateral'), 
               turns=S('Turning On Ground')):
        self.create_kpvs_within_slices(acc_lat.array, turns, max_abs_value)


class AccelerationLongitudinalPeakTakeoff(KeyPointValueNode):
    '''
    This may be of interest where takeoff performance is an issue, though not
    normally monitored as a safety event.
    '''
    def derive(self, takeoff=S('Takeoff'),
               accel=P('Acceleration Longitudinal')):
        self.create_kpvs_within_slices(accel.array, takeoff, max_value)


class DecelerationLongitudinalPeakLanding(KeyPointValueNode):
    '''
    This is an indication of severe braking and/or use of reverse thrust or
    reverse pitch.
    '''
    def derive(self, landing=S('Landing'),
               accel=P('Acceleration Longitudinal')):
        self.create_kpvs_within_slices(accel.array, landing, min_value)

        
class AccelerationNormal20FtToGroundMax(KeyPointValueNode):
    name = 'Acceleration Normal 20 Ft To Ground Max' # not required?
    def derive(self, acceleration_normal=P('Acceleration Normal'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(acceleration_normal.array,
                                       alt_aal.slices_from_to(20, 0),
                                       max_value)


class AccelerationNormalMax(KeyPointValueNode):
    def derive(self, acc_norm=P('Acceleration Normal')):
        index, value = max_value(acc_norm.array)
        self.create_kpv(index, value)
        
        
class AccelerationNormalAirborneFlapsUpMax(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), flap=P('Flap'), 
               airborne=S('Airborne')):
        # Mask data where the flaps are down
        acc_flap_up = np.ma.masked_where(flap.array>0.0, accel.array)
        self.create_kpvs_within_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalAirborneFlapsUpMin(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), flap=P('Flap'), 
               airborne=S('Airborne')):
        # Mask data where the flaps are down
        acc_flap_up = np.ma.masked_where(flap.array>0.0, accel.array)
        self.create_kpvs_within_slices(acc_flap_up, airborne, min_value)


class AccelerationNormalAirborneFlapsDownMax(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), flap=P('Flap'), 
               airborne=S('Airborne')):
        # Mask data where the flaps are up
        acc_flap_up = np.ma.masked_where(flap.array==0.0, accel.array)
        self.create_kpvs_within_slices(acc_flap_up, airborne, max_value)


class AccelerationNormalAirborneFlapsDownMin(KeyPointValueNode):
    def derive(self, accel=P('Acceleration Normal'), flap=P('Flap'), 
               airborne=S('Airborne')):
        # Mask data where the flaps are up
        acc_flap_up = np.ma.masked_where(flap.array==0.0, accel.array)
        self.create_kpvs_within_slices(acc_flap_up, airborne, min_value)

#-----------------------------------------------------------------------

def bump(acc, phase):
    # Scan the acceleration array for a short period either side of the
    # moment of interest.
    dt=2.0 # Half width of range to scan across for peak acceleration.
    from_index = max(int(phase.index-dt*acc.hz), 0)
    to_index = min(int(phase.index+dt*acc.hz)+1, len(acc.array))
    bump_accel = acc.array[from_index:to_index]
    bump_index = np.ma.argmax(bump_accel)
    peak = bump_accel[bump_index]
    return from_index + bump_index, peak
    
class AccelerationNormalAtLiftoff(KeyPointValueNode):
    '''
    This is a measure of the normal acceleration at the point of liftoff, and
    is related to the pitch rate at takeoff.
    '''
    def derive(self, acc=P('Acceleration Normal'), lifts=KTI('Liftoff')):
        for lift in lifts:
            self.create_kpv(*bump(acc, lift))

class AccelerationNormalAtTouchdown(KeyPointValueNode):
    '''
    This is the peak acceleration at landing, often used to identify hard
    landings for maintenance purposes.
    '''
    def derive(self, acc=P('Acceleration Normal'), tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            self.create_kpv(*bump(acc, tdwn))

#-----------------------------------------------------------------------


class Airspeed1000To500FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(1000, 500),
                                           max_value) 


class Airspeed1000To500FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(1000, 500),
                                           min_value) 


class Airspeed2000To30FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(2000, 30),
                                           min_value)


class AirspeedAt35FtInTakeoff(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), takeoff=S('Takeoff')):
        first_toff = takeoff.get_first()
        if first_toff:
            index = first_toff.slice.stop
            self.create_kpv(index, value_at_index(airspeed.array, index))


class Airspeed35To1000FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(35, 1000),
                                           max_value) 


class Airspeed35To1000FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(35, 1000),
                                           min_value) 

        
class Airspeed400To1500FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(400, 1500),
                                           min_value)


class Airspeed500To20FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 20),
                                           max_value) 


class Airspeed500To20FtMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 20),
                                           min_value) 


class AirspeedBelowAltitudeMax(KeyPointValueNode):
    NAME_FORMAT = 'Airspeed Below %(altitude)d Ft Max'
    NAME_VALUES = {'altitude': [500, 3000, 7000]}
    
    def derive(self, airspeed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
        for alt in self.NAME_VALUES['altitude']:
            self.create_kpvs_within_slices(airspeed.array,
                                           alt_aal.slices_between(0, alt),
                                           max_value,
                                           altitude=alt)


class AirspeedVacatingRunway(KeyPointValueNode):
    '''
    Airspeed vacating runway uses true airspeed, which is extended below the
    minimum range of the indicated airspeed specifically for this type of
    event. See the derived parameter for details of how groundspeed or
    acceleration data is used to cover the landing phase.
    '''
    def derive(self, airspeed=P('Airspeed True'), 
               off_rwy=KTI('Landing Turn Off Runway')):
        self.create_kpvs_at_ktis(airspeed.array, off_rwy)
        

class AirspeedMinusVrefFor5Sec1000To500FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed Minus Vref For 5 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(speed.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       max_value)


class AirspeedMinusVrefFor5Sec500To20FtMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed Minus Vref For 5 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
            self.create_kpvs_within_slices(speed.array,
                                           alt_aal.slices_from_to(500, 20),
                                           max_value) 


class AirspeedMinusVrefFor5Sec20FtToTouchdownMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed Minus Vref For 5 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(speed.array,
                                       alt_aal.slices_from_to(20, 0),
                                       max_value) 



def thrust_reverser_min_speed(land, pwr, tr):
    high_power = np.ma.clump_unmasked(np.ma.masked_less(pwr.array[land.slice],60.0))
    rev = np.ma.clump_unmasked(np.ma.masked_less(tr.array[land.slice],0.7))
    return shift_slices(slices_and(high_power, rev), land.slice.start)


class AirspeedThrustReversersDeployedMin(KeyPointValueNode):
    name = 'Airspeed With Thrust Reversers Deployed (Over 60% N1) Min'
    def derive(self, speed=P('Airspeed True'), tr=P('Thrust Reversers'), 
               pwr=P('Eng (*) N1 Avg'), lands=S('Landing')):
        for land in lands:
            high_rev = thrust_reverser_min_speed(land, pwr, tr)
            self.create_kpvs_within_slices(speed.array, high_rev, min_value)


class GroundspeedThrustReversersDeployedMin(KeyPointValueNode):
    name = 'Groundspeed With Thrust Reversers Deployed (Over 60% N1) Min'
    def derive(self, speed=P('Groundspeed'), tr=P('Thrust Reversers'), 
               pwr=P('Eng (*) N1 Max'), lands=S('Landing')):
        for land in lands:
            high_rev = thrust_reverser_min_speed(land, pwr, tr)
            self.create_kpvs_within_slices(speed.array, high_rev, min_value)


class AirspeedWithGearDownMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear=P('Gear Down'), 
               airs=S('Airborne')):
        for air in airs:
            downs = np.ma.clump_unmasked(np.ma.masked_equal(gear.array[air.slice], 0.0))
            for down in downs:
                index = np.ma.argmax(airspeed.array[air.slice][down])
                value = airspeed.array[air.slice][down][index]
                self.create_kpv(int((air.slice.start or 0))+down.start+index, value)


class MachWithGearDownMax(KeyPointValueNode):
    def derive(self, mach=P('Mach'), gear=P('Gear Down'), 
               airs=S('Airborne')):
        for air in airs:
            downs = np.ma.clump_unmasked(np.ma.masked_equal(gear.array[air.slice], 0.0))
            for down in downs:
                index = np.ma.argmax(mach.array[air.slice][down])
                value = mach.array[air.slice][down][index]
                self.create_kpv(int((air.slice.start or 0))+down.start+index, value)


class AirspeedAtGearSelectionUp(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_sel_up=KTI('Gear Selection Up')):
        self.create_kpvs_at_ktis(airspeed.array, gear_sel_up)


class AirspeedAtGearSelectionDown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_sel_down=KTI('Gear Selection Down')):
        self.create_kpvs_at_ktis(airspeed.array, gear_sel_down)


class AirspeedAsGearRetractingMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_ret=S('Gear Retracting')):
        self.create_kpvs_within_slices(airspeed.array, gear_ret, max_value)


class AirspeedAsGearExtendingMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), gear_ext=S('Gear Extending')):
        self.create_kpvs_within_slices(airspeed.array, gear_ext, max_value)


class MachAsGearRetractingMax(KeyPointValueNode):
    def derive(self, mach=P('Mach'), gear_ret=S('Gear Retracting')):
        self.create_kpvs_within_slices(mach.array, gear_ret, max_value)


class MachAsGearExtendingMax(KeyPointValueNode):
    def derive(self, mach=P('Mach'), gear_ext=S('Gear Extending')):
        self.create_kpvs_within_slices(mach.array, gear_ext, max_value)

        
class AltitudeAtGearSelectionUp(KeyPointValueNode):
    name = 'Altitude AAL At Gear Selection Up'
    def derive(self, alt_aal=P('Altitude AAL'), gear_sel_up=KTI('Gear Selection Up')):
        self.create_kpvs_at_ktis(alt_aal.array, gear_sel_up)


class AltitudeAtGearSelectionDown(KeyPointValueNode):
    name = 'Altitude AAL At Gear Selection Down'
    def derive(self, alt_aal=P('Altitude AAL'), gear_sel_down=KTI('Gear Selection Down')):
        self.create_kpvs_at_ktis(alt_aal.array, gear_sel_down)

#-------------------------------------------------------------------------------

class AirspeedAtLiftoff(KeyPointValueNode):
    '''
    DJ suggested TailWindAtLiftoff would complement this parameter when used
    for 'Speed high at liftoff' events.
    '''
    def derive(self, airspeed=P('Airspeed'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(airspeed.array, liftoffs)


'''
Redundant, as this will either be a go-around, with its minimum, or a landing

class AltitudeAtLowestPointOnApproach(KeyPointValueNode):
    """
    The approach phase has been found already. Here we take the height at
    the lowest point reached in the approach.
    """
    def derive(self, alt_aal=P('Altitude AAL'), alt_rad=P('Altitude Radio'), 
               low_points=KTI('Lowest Point On Approach')):
        height = minimum_unmasked(alt_aal.array, alt_rad.array)
        self.create_kpvs_at_ktis(height, low_points)
        '''


class AirspeedAtTouchdown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(airspeed.array, touchdowns)



class AirspeedMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), airs=S('Airborne')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        self.create_kpvs_within_slices(speed.array, airs, max_value)


class AirspeedMax3Sec(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), airs=S('Airborne')):
        self.create_kpvs_within_slices(clip(speed.array, 3.0, speed.hz), airs, max_value)

def flap_or_conf_max_or_min(self, conflap, airspeed, function, scope=None):
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
    
    :returns: Nothing. KPVs are created within the routine.
    '''
    if scope==[]:
        return # Can't have an event if the scope is empty.
    
    if scope:
        scope_array = np_ma_masked_zeros_like(airspeed.array)
        for valid in scope:
            scope_array.mask[int(valid.slice.start or 0):\
                             int(valid.slice.stop or len(scope_array))+1]=False
            
    for conflap_setting in np.ma.unique(conflap.array):
        if conflap_setting == 0.0 or \
           np.ma.is_masked(conflap_setting):
            # ignore masked values
            continue
        spd_with_conflap = np.ma.copy(airspeed.array)
        # apply flap mask
        spd_with_conflap.mask = np.ma.mask_or(airspeed.array.mask, conflap.array.mask)
        spd_with_conflap[conflap.array != conflap_setting] = np.ma.masked
        if scope:
            spd_with_conflap.mask = np.ma.mask_or(spd_with_conflap.mask, scope_array.mask)
        #TODO: Check logical OR is sensible for all values (probably ok as
        #airspeed will always be higher than max flap setting!)
        index, value = function(spd_with_conflap)
        
        # Check we have a result to record. Note that most flap setting will
        # not be used in the climb, hence this is normal operation.
        if index and value:
            if conflap.name=='Flap':
                self.create_kpv(index, value, flap=conflap_setting)
            else:
                self.create_kpv(index, value, conf=conflap_setting)


class AirspeedWithFlapMax(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d Max"
    NAME_VALUES = NAME_VALUES_FLAP
    # Allows for Hercules with 50% and 100% flap
    
    # Note: It is essential that Flap is the first paramter here to prevent
    # the flap values, which match the detent settings, from being
    # interpolated.
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), fast=S('Fast')):
        # Fast scope traps flap changes very late on the approach and raising
        # flaps before 80kn on the landing run.
        flap_or_conf_max_or_min(self, flap, airspeed, max_value, scope=fast)


class AirspeedWithFlapMin(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d Min"
    NAME_VALUES = NAME_VALUES_FLAP
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), airborne=S('Airborne')):
        # Airborne scope avoids deceleration on the runway "corrupting" the
        # minimum airspeed with landing flap.
        flap_or_conf_max_or_min(self, flap, airspeed, min_value, scope=airborne)


class AirspeedWithFlapClimbMin(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d in Climb Min"
    NAME_VALUES = NAME_VALUES_FLAP
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Climb')):
        flap_or_conf_max_or_min(self, flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapDescentMin(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d in Descent Min"
    NAME_VALUES = NAME_VALUES_FLAP
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Descent')):
        flap_or_conf_max_or_min(self, flap, airspeed, min_value, scope=scope)


class AirspeedWithFlapClimbMax(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d in Climb Max"
    NAME_VALUES = NAME_VALUES_FLAP
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Climb')):
        flap_or_conf_max_or_min(self, flap, airspeed, max_value, scope=scope)


class AirspeedWithFlapDescentMax(KeyPointValueNode):
    NAME_FORMAT = "Airspeed With Flap %(flap)d in Descent Max"
    NAME_VALUES = NAME_VALUES_FLAP
    def derive(self, flap=P('Flap'), airspeed=P('Airspeed'), scope=S('Descent')):
        flap_or_conf_max_or_min(self, flap, airspeed, max_value, scope=scope)


class AirspeedBelowAltitudeMax(KeyPointValueNode):
    NAME_FORMAT = 'Airspeed Below %(altitude)d Ft Max'
    NAME_VALUES = {'altitude': [10000, 8000, 7000, 5000, 3000]}
    
    def derive(self, airspeed=P('Airspeed'), alt_aal=P('Altitude AAL For Flight Phases')):
        for alt in self.NAME_VALUES['altitude']:
            self.create_kpvs_within_slices(airspeed.array,
                                           alt_aal.slices_between(0, alt),
                                           max_value,
                                           altitude=alt)

class Airspeed10000ToLandMax(KeyPointValueNode):
    '''
    TODO: Test.
    '''
    name = 'Airspeed Below 10000 Ft In Descent Max'
    def derive(self, airspeed=P('Airspeed'),
               alt_std=P('Altitude STD'),
               alt_qnh=P('Altitude QNH'),
               destination = A('FDR Landing Airport'), 
               descent=S('Descent')):
        # Outside the USA 10,000 ft relates to flight levels, whereas FAA
        # regulations (and possibly others we don't currently know about)
        # relate to height above sea level (QNH) hence the options based on
        # landing airport location.
        if destination and destination.value['location']['country'] == 'United States':
            alt=alt_qnh.array
        else:
            alt=alt_std.array
        height_bands = np.ma.clump_unmasked(np.ma.masked_greater(alt,10000))
        descent_bands = slices_and(height_bands, [s.slice for s in descent])
        self.create_kpvs_within_slices(airspeed.array, descent_bands, max_value)


class AirspeedRTOMax(KeyPointValueNode):
    name = 'Airspeed RTO Max'
    def derive(self, airspeed=P('Airspeed'),
               rejected_takeoffs=S('Rejected Takeoff')):
        for rejected_takeoff in rejected_takeoffs:
            self.create_kpvs_within_slices(airspeed.array, 
                                           rejected_takeoff.slice, max_value)


class AirspeedTODTo10000Max(KeyPointValueNode):
    '''
    TODO: Test.
    '''
    name = 'Airspeed Top Of Descent To 10000 Ft Max'
    def derive(self, airspeed=P('Airspeed'),
               alt_std=P('Altitude STD'), 
               alt_qnh=P('Altitude QNH'),
               destination = A('FDR Landing Airport'), 
               descent=S('Descent')):
        # See comments for Airspeed10000ToLandMax
        if destination and destination.value['location']['country'] == 'United States':
            alt=alt_qnh.array
        else:
            alt=alt_std.array
        height_bands = np.ma.clump_unmasked(np.ma.masked_less(repair_mask(alt),10000))
        descent_bands = slices_and(height_bands, [s.slice for s in descent])
        self.create_kpvs_within_slices(airspeed.array, descent_bands, max_value)


class AirspeedBetween90SecToTouchdownAndTouchdownMax(KeyPointValueNode):
    def derive(self, sec_to_touchdown=KTI('Secs To Touchdown'), airspeed=P('Airspeed')):
        for _90_sec in sec_to_touchdown.get(name='90 Secs To Touchdown'):
            # we're 90 seconds from touchdown
            tdwn = _90_sec.index + 90 * self.frequency
            index, value = max_value(airspeed.array, slice(_90_sec.index, tdwn))
            self.create_kpv(index, value)


class AirspeedCruiseMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), cruises=S('Cruise')):
        self.create_kpvs_within_slices(speed.array, cruises, max_value)
            
                
class AirspeedCruiseMin(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'), cruises=S('Cruise')):
        self.create_kpvs_within_slices(speed.array, cruises, min_value)
            
              
class GenericDescent(KeyPointValueNode):
    NAME_FORMAT = '%(parameter)s At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {'parameter':['Airspeed',
                                'Airspeed Minus Vref',
                                'Rate Of Descent',
                                'Slope To Landing',
                                'Flap',
                                'Gear Down',
                                'Speedbrake',
                                'ILS Glideslope',
                                'ILS Localizer',
                                'Power',
                                'Pitch',
                                'Roll',
                                'Heading'
                                ],
                   'altitude':[10000,9000,8000,7000,6000,5000,4000,3500,\
                               3000,2500,2000,1500,1000,750,500,400,300,\
                               200,150,100,75,50,35,20,10]}
    @classmethod
    def can_operate(cls, available):
        return 'Descent' in available and 'Altitude AAL' in available

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               slope=P('Slope To Landing'), flap=P('Flap'), 
               glide=P('ILS Glideslope'),  airspeed=P('Airspeed'), 
               roc=P('Rate Of Climb'), gear=P('Gear Down'), 
               loc=P('ILS Localizer'),  power=P('Eng (*) N1 Avg'),
               pitch=P('Pitch'),  brake=P('Speedbrake'),  
               roll=P('Roll'),  head=P('Heading'), descent=S('Descent')):
        descent_list = [s.slice for s in descent]
        for this_descent in descent_list:
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, _slice=this_descent)
                if index:
                    self.create_kpv(index, value_at_index(slope.array, index), 
                                    parameter='Slope To Landing', altitude=alt)
                    self.create_kpv(index, value_at_index(flap.array, index), 
                                        parameter='Flap', altitude=alt)
                    self.create_kpv(index, value_at_index(glide.array, index), 
                                    parameter='ILS Glideslope', altitude=alt)
                    self.create_kpv(index, value_at_index(airspeed.array, index), 
                                    parameter='Airspeed', altitude=alt)
                    self.create_kpv(index, value_at_index(roc.array, index), 
                                    parameter='Rate Of Descent', altitude=alt)
                    self.create_kpv(index, value_at_index(gear.array, index), 
                                    parameter='Gear Down', altitude=alt)
                    self.create_kpv(index, value_at_index(loc.array, index), 
                                    parameter='ILS Localizer', altitude=alt)
                    self.create_kpv(index, value_at_index(power.array, index), 
                                    parameter='Power', altitude=alt)
                    self.create_kpv(index, value_at_index(pitch.array, index), 
                                    parameter='Pitch', altitude=alt)
                    self.create_kpv(index, value_at_index(brake.array, index), 
                                    parameter='Speedbrake', altitude=alt)
                    self.create_kpv(index, value_at_index(roll.array, index), 
                                    parameter='Roll', altitude=alt)
                    self.create_kpv(index, value_at_index(head.array, index), 
                                    parameter='Heading', altitude=alt)
            
  
class AirspeedLevelFlightMax(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), level_flight=S('Level Flight')):
        for sect in level_flight:
            #TODO: Move LEVEL_FLIGHT_MIN_DURATION to LevelFlight
            #FlightPhaseNode so that only stable level flights are reported.
            duration = (sect.slice.stop - sect.slice.start)/self.frequency
            if duration > settings.LEVEL_FLIGHT_MIN_DURATION:
                # stable level flight
                index, value = max_value(airspeed.array, sect.slice)
                self.create_kpv(index, value)
            else:
                self.debug("Level flight duration too short to create KPV")


class AirspeedMinusV2AtLiftoff(KeyPointValueNode):
    name = 'Airspeed Minus V2 At Liftoff'
    def derive(self, airspeed=P('Airspeed Minus V2'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(airspeed.array, liftoffs)
        
        
class AirspeedMinusV2For3Sec35To400FtMin(KeyPointValueNode):
    name = 'Airspeed Minus V2 For 3 Sec 35 To 400 Ft Min'
    def derive(self, spd_v2=P('Airspeed Minus V2 For 3 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(spd_v2.array,
                                       alt_aal.slices_from_to(35, 400),
                                       min_value)


class AirspeedMinusV2For5Sec400To1500FtMin(KeyPointValueNode):
    name = 'Airspeed Minus V2 For 5 Sec 400 To 1500 Ft Min'
    def derive(self, spd_v2=P('Airspeed Minus V2 For 5 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(spd_v2.array,
                                       alt_aal.slices_from_to(400, 1500),
                                       min_value)


class AirspeedMinusVref500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, airspeed_minus_vref=P('Airspeed Minus Vref'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(airspeed_minus_vref.array,
                                       alt_aal.slices_from_to(500, 0),
                                       max_value)


class AirspeedMinusVrefAtTouchdown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed Minus Vref'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(airspeed.array, touchdowns)


class AirspeedMinusVrefFor5Sec20FtToTouchdownMax(KeyPointValueNode):
    def derive(self, speed=P('Airspeed Minus Vref For 5 Sec'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(speed.array,
                                       alt_aal.slices_from_to(20, 0),
                                       max_value)


class AirspeedMinusVrefBetween2MinutesToTouchdownAndTouchdownMin(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, mins_to_touchdown=KTI('Mins To Touchdown'), 
               airspeed=P('Airspeed Minus Vref')):
        for _2_min in mins_to_touchdown.get(name='2 Mins To Touchdown'):
            # add 2 mins to find touchdown
            tdwn = _2_min.index + 2 * 60 * self.frequency
            index, value = min_value(airspeed.array, slice(_2_min.index, tdwn))
            self.create_kpv(index, value)


class AltitudeAtTouchdown(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(alt_std.array, touchdowns)


class AltitudeAtGoAroundMin(KeyPointValueNode):
    @classmethod
    def can_operate(cls, available):
        return 'Go Around' in available and 'Altitude AAL' in available
    
    def derive(self, alt_rad=P('Altitude Radio'),
               alt_aal=P('Altitude AAL'),
               gas=KTI('Go Around')):
        for ga in gas:
            if alt_rad:
                pit = alt_rad.array[ga.index]
            else:
                pit = alt_aal.array[ga.index]
            self.create_kpv(ga.index, pit)


class AltitudeGoAroundFlapRetracted(KeyPointValueNode):
    # gafr pinpoints the flap retraction instance within the 500ft go-around window.
    def derive(self, alt_aal=P('Altitude AAL'), 
               gafr=KTI('Go Around Flap Retracted')):
        self.create_kpvs_at_ktis(alt_aal.array,gafr)


class AltitudeGoAroundGearRetracted(KeyPointValueNode):
    # gagr pinpoints the gear retraction instance within the 500ft go-around window.
    def derive(self, alt_aal=P('Altitude AAL'), 
               gagr=KTI('Go Around Gear Retracted')):
        self.create_kpvs_at_ktis(alt_aal.array,gagr)


class AltitudeAtLiftoff(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(alt_std.array, liftoffs)


class AltitudeAtLastFlapChangeBeforeLanding(KeyPointValueNode):
    name = 'Altitude AAL At Last Flap Change Before Landing'
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'), 
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            land_flap = flap.array[tdwn.index]
            last_index = index_at_value(flap.array-land_flap, -0.5, slice(tdwn.index, 0, -1))
            alt_last = value_at_index(alt_aal.array, last_index)
            self.create_kpv(last_index, alt_last)


class AltitudeAtMachMax(KeyPointValueNode):
    name = 'Altitude At Mach Max'
    def derive(self, alt_std=P('Altitude STD'), max_mach=KPV('Mach Max')):
        # Aligns Altitude to Mach to ensure we have the most accurate
        # altitude reading at the point of Maximum Mach
        self.create_kpvs_at_kpvs(alt_std.array, max_mach)


class AltitudeWithFlapsMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), alt_std=P('Altitude STD')):
        '''
        The exceedance being detected here is the altitude reached with flaps
        not stowed, hence any flap value greater than zero is applicable and
        we're not really interested (for the purpose of identifying the
        event) what flap setting was reached.
        '''
        alt_flap = alt_std.array * np.ma.minimum(flap.array,1.0)
        index = np.ma.argmax(alt_flap)
        self.create_kpv(index, value_at_index(alt_flap, index))
       
       
class AltitudeFlapExtensionMax(KeyPointValueNode):
    name='Altitude AAL Flap Extension Max'
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'), airs=S('Airborne')):
        # Restricted to avoid triggering on flap extension for takeoff.
        for air in airs:
            extends = find_edges(flap.array, air.slice)
            if extends:
                index=extends[0]
                value=alt_aal.array[index]
                self.create_kpv(index, value)

        
class AltitudeMax(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD'), airs=S('Airborne')):
        self.create_kpvs_within_slices(alt_std.array, airs, max_value)


class AltitudeAutopilotEngaged(KeyPointValueNode):
    name = 'Altitude AAL AP Engaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'), ap_eng=KTI('AP Selection Engaged')):
        self.create_kpvs_at_ktis(alt_aal.array, ap_eng)
        
        
class AltitudeAutopilotDisengaged(KeyPointValueNode):
    name = 'Altitude AAL AP Disengaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'), ap_dis=KTI('AP Selection Disengaged')):
        self.create_kpvs_at_ktis(alt_aal.array, ap_dis)
        
        
class AltitudeAutothrottleEngaged(KeyPointValueNode):
    name = 'Altitude AAL AT Engaged In Flight'
    # Note: Autothrottle is normally engaged prior to takeoff, so will not trigger this event.
    def derive(self, alt_aal=P('Altitude AAL'), at_eng=KTI('AT Selection Engaged')):
        self.create_kpvs_at_ktis(alt_aal.array, at_eng)
        
        
class AltitudeAutothrottleDisengaged(KeyPointValueNode):
    name = 'Altitude AAL AT Disengaged In Flight'
    def derive(self, alt_aal=P('Altitude AAL'), at_dis=KTI('AT Selection Disengaged')):
        self.create_kpvs_at_ktis(alt_aal.array, at_dis)
        

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

        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        import csv
        spam = csv.writer(open('cheese.csv', 'wb'))
        spam.writerow(['ref','column', 'force'])
        #-------------------------------------------------------------------
        # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
        # TODO: REMOVE THIS SECTION BEFORE RELEASE
        #-------------------------------------------------------------------
        

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
                    self.create_kpv(speedy.slice.start+move.start+when, slope)
                    
                    #-------------------------------------------------------------------
                    # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
                    # TODO: REMOVE THIS SECTION BEFORE RELEASE
                    #-------------------------------------------------------------------
                    n = speedy.slice.start+move.start
                    for showme in range(0, slice_samples(move)):
                        spam.writerow([n+showme,column[move][showme],push[move][showme]])
                     #-------------------------------------------------------------------
                    # TEST OUTPUT TO CSV FILE FOR DEBUGGING ONLY
                    # TODO: REMOVE THIS SECTION BEFORE RELEASE
                    #-------------------------------------------------------------------
                    


class DistancePastGlideslopeAntennaToTouchdown(KeyPointValueNode):
    units = 'm'
    def derive(self,  lat=P('Latitude Smoothed'),lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown'),rwy=A('FDR Landing Runway'),
               ils_ldgs=S('ILS Localizer Established')):

        land_idx=tdwns[-1].index
        # Check we did do an ILS approach (i.e. the ILS frequency was correct etc).
        if is_index_within_sections(land_idx, ils_ldgs):
            # Yes it was, so do the geometry...
            distance = runway_distance_from_end(rwy.value, point='glideslope')-\
                runway_distance_from_end(rwy.value, lat.array[land_idx], lon.array[land_idx])
                
            self.create_kpv(land_idx, distance)


class DistanceFromRunwayStartToTouchdown(KeyPointValueNode):
    units = 'm'
    def derive(self, lat=P('Latitude Smoothed'),lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown'),rwy=A('FDR Landing Runway')):
        
        land_idx=tdwns[-1].index
        distance = runway_distance_from_end(rwy.value, point='start')-\
            runway_distance_from_end(rwy.value, lat.array[land_idx], lon.array[land_idx])
        self.create_kpv(land_idx, distance)
    
    
class DistanceFrom60KtToRunwayEnd(KeyPointValueNode):
    units = 'm'
    def derive(self, gspd=P('Groundspeed'),
               lat=P('Latitude Smoothed'),lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown'),rwy=A('FDR Landing Runway')):

        land_idx=tdwns[-1].index
        idx_60 = index_at_value(gspd.array,60.0,slice(land_idx,None))
        if idx_60:
            # Only work out the distance if we have a reading at 60kts...
            distance = runway_distance_from_end(rwy.value, lat.array[idx_60], lon.array[idx_60])
            self.create_kpv(idx_60, distance) # Metres
        

class HeadingAtLanding(KeyPointValueNode):
    """
    We take the median heading, as this avoids problems with drift just
    before touchdown and turning off the runway. The value is "assigned" to a
    time midway through the landing phase.
    """
    def derive(self, head=P('Heading Continuous'),
               lands=S('Landing')):
        for land in lands:
            land_head = np.ma.median(head.array[land.slice])
            land_index = (land.slice.start + land.slice.stop)/2.0
            self.create_kpv(land_index, land_head%360.0)


class HeadingAtLowestPointOnApproach(KeyPointValueNode):
    """
    The approach phase has been found already. Here we take the heading at
    the lowest point reached in the approach.
    """
    def derive(self, head=P('Heading Continuous'), 
               low_points=KTI('Lowest Point On Approach')):
        self.create_kpvs_at_ktis(head.array%360.0, low_points)
    
    
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


class HeightLostInClimbTo35Ft(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        for climb in alt_aal.slices_from_to(0, 35):
            deltas = np.ma.ediff1d(alt_aal.array[climb], to_begin=0.0)
            downs = np.ma.masked_greater(deltas,0.0)
            index = np.ma.argmin(downs)
            drop = np.ma.sum(downs)
            if index:
                self.create_kpv(climb.start + index, drop)
                

class HeightLostInClimb35To1000Ft(KeyPointValueNode):
    def derive(self, height_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        for climb in alt_aal.slices_from_to(25, 1000):
            idx = np.ma.argmin(height_loss.array[climb])
            index = climb.start + idx
            value = height_loss.array[index]
            if value:
                self.create_kpv(index, value)
                
            
class HeightLostInClimb1000To2000Ft(KeyPointValueNode):
    def derive(self, height_loss=P('Descend For Flight Phases'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        for climb in alt_aal.slices_from_to(1000, 2000):
            idx = np.ma.argmin(height_loss.array[climb])
            index = climb.start + idx
            value = height_loss.array[index]
            if value:
                self.create_kpv(index, value)
                
class HoldingTime(KeyPointValueNode):
    """
    Identify time spent in the hold.
    """
    def derive(self, holds=S('Holding')):
        self.create_kpvs_from_slices(holds, mark='end')
        
        
class ILSFrequencyOnApproach(KeyPointValueNode):
    """
    The period when the aircraft was continuously established on the ILS and
    descending to the minimum point on the approach is already defined as a
    flight phase. This KPV just picks up the frequency tuned at that point.
    """
    name='ILS Frequency On Approach' #  Set here to ensure "ILS" in uppercase.
    def derive(self, ils_frq=P('ILS Frequency'),
               loc_ests=S('ILS Localizer Established')):
        
        for loc_est in loc_ests:
            # For the final period of operation of the ILS during this
            # approach, the ILS frequency was:
            freq=np.ma.median(ils_frq.array[loc_est.slice])
            # Note median picks the value most commonly recorded, so allows
            # for some masked values and perhaps one or two rogue values.

            # Identify the KPV as relating to the start of this ILS approach
            self.create_kpv(loc_est.slice.start, freq)


class ILSGlideslopeDeviation1500To1000FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation 1500 To 1000 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL For Flight Phases'),
               gs_ests=S('ILS Glideslope Established')):
        # Find where the maximum (absolute) deviation occured and
        # store the actual value. We can do abs on the statistics to
        # normalise this, but retaining the sign will make it
        # possible to look for direction of errors at specific
        # airports.
        alt_bands = alt_aal.slices_from_to(1500, 1000)
        ils_bands = slices_and(alt_bands, [s.slice for s in gs_ests])
        self.create_kpvs_within_slices(ils_glideslope.array,ils_bands,max_abs_value)  


class ILSGlideslopeDeviation1000To150FtMax(KeyPointValueNode):
    name = 'ILS Glideslope Deviation 1000 To 150 Ft Max'
    def derive(self, ils_glideslope=P('ILS Glideslope'),
               alt_aal = P('Altitude AAL For Flight Phases'),
               gs_ests=S('ILS Glideslope Established')):
        alt_bands = alt_aal.slices_from_to(1000, 150)
        ils_bands = slices_and(alt_bands, [s.slice for s in gs_ests])
        self.create_kpvs_within_slices(ils_glideslope.array,ils_bands,max_abs_value)  


class ILSLocalizerDeviation1500To1000FtMax(KeyPointValueNode):
    name = 'ILS Localizer Deviation 1500 To 1000 Ft Max'
    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal = P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Localizer Established')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        alt_bands = alt_aal.slices_from_to(1500, 1000)
        ils_bands = slices_and(alt_bands, [s.slice for s in ils_ests])
        self.create_kpvs_within_slices(ils_loc.array,ils_bands,max_abs_value)  


class ILSLocalizerDeviation1000To150FtMax(KeyPointValueNode):
    name = 'ILS Localizer Deviation 1000 To 150 Ft Max'
    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal = P('Altitude AAL For Flight Phases'),
               ils_ests=S('ILS Localizer Established')):
        # For commented version, see GlideslopeDeviation1500To1000FtMax
        alt_bands = alt_aal.slices_from_to(1000, 150)
        ils_bands = slices_and(alt_bands, [s.slice for s in ils_ests])
        self.create_kpvs_within_slices(ils_loc.array,ils_bands,max_abs_value)  

        
class IsolationValveOpenAtLiftoff(KeyPointValueNode):
    def derive(self, isol=P('Isolation Valve Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(isol.array, lifts, suppress_zeros=True)

        
class PackValvesOpenAtLiftoff(KeyPointValueNode):
    def derive(self, isol=P('Pack Valves Open'), lifts=KTI('Liftoff')):
        self.create_kpvs_at_ktis(isol.array, lifts, suppress_zeros=True)


class LatitudeAtLanding(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lat=P('Latitude'), tdwns=KTI('Touchdown')):
        '''
        While storing this is redundant due to geo-locating KeyPointValues, it is
        used in multiple Nodes to simplify their implementation.
        '''    
        self.create_kpvs_at_ktis(lat.array, tdwns)
            

class LongitudeAtLanding(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lon=P('Longitude'),tdwns=KTI('Touchdown')):
        '''
        While storing this is redundant due to geo-locating KeyPointValues, 
        it is used in multiple Nodes to simplify their implementation.
        '''       
        self.create_kpvs_at_ktis(lon.array, tdwns)


class LatitudeAtTakeoff(KeyPointValueNode):
    def derive(self, lat=P('Latitude'),
               liftoffs=KTI('Liftoff')):
        # OK, At the risk of causing confusion, we use the liftoff instant to
        # identify the takeoff airport. Strictly, takeoff is a process taking
        # time and distance, whereas liftoff is an instant in time and space.
        self.create_kpvs_at_ktis(lat.array, liftoffs)


class LongitudeAtTakeoff(KeyPointValueNode):
    '''
    While storing this is redundant due to geo-locating KeyPointValues, it is
    used in multiple Nodes to simplify their implementation.
    '''    
    def derive(self, lon=P('Longitude'),
               liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(lon.array, liftoffs)


class LatitudeAtLowestPointOnApproach(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lat=P('Latitude'), 
               low_points=KTI('Lowest Point On Approach')):
        self.create_kpvs_at_ktis(lat.array, low_points)
    """
    def derive(self, lat=P('Latitude'), 
               lands=KTI('Approach And Landing Lowest Point')):
        self.create_kpvs_at_ktis(lat.array, lands)
        """    


class LongitudeAtLowestPointOnApproach(KeyPointValueNode):
    # Cannot use smoothed position as this causes circular dependancy.
    def derive(self, lon=P('Longitude'), 
               low_points=KTI('Lowest Point On Approach')):
        self.create_kpvs_at_ktis(lon.array, low_points)
    """
    def derive(self, lon=P('Longitude'), 
               lands=KTI('Approach And Landing Lowest Point')):
        self.create_kpvs_at_ktis(lon.array, lands)
        """


class MachMax(KeyPointValueNode):
    name = 'Mach Max'
    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        self.create_kpvs_within_slices(mach.array, airs, max_value)


class MachMax3Sec(KeyPointValueNode):
    def derive(self, mach=P('Mach'), airs=S('Airborne')):
        self.create_kpvs_within_slices(clip(mach.array, 3.0, mach.hz),
                                       airs, max_value)


class MagneticVariationAtTakeoff(KeyPointValueNode):
    def derive(self, var=P('Magnetic Variation'), toff=KTI('Takeoff Turn Onto Runway')):
        self.create_kpvs_at_ktis(var.array, toff)


class MagneticVariationAtLanding(KeyPointValueNode):
    def derive(self, var=P('Magnetic Variation'), land=KTI('Landing Turn Off Runway')):
        self.create_kpvs_at_ktis(var.array, land)


#-------------------------------------------------------------------------------
# Engine start conditions
#-------------------------------------------------------------------------------

def peak_start_egt(egt, n2, idx):
    # Prepare to look for starting conditions.
    if idx < 20:
        # We can't have started less than 20 seconds before takeoff, so throw
        # this away.
        return None, None

    # Ideally we'd use a shorter timebase, say 2 sec, but N2 only sampled at
    # 1/4Hz on some aircraft.
    n2_rate = rate_of_change(n2,4)
    # The engine only accelerates through 30% when starting. Let's find the
    # last time it did this before taking off.
    passing_30 = index_at_value(n2.array,30.0, slice(idx,0,-1))
    # after which it will peak and the rate will fall below zero at the
    # overswing, which must happen within 30 seconds.
    if passing_30:
        started_up = index_at_value(n2_rate,0.0,slice(passing_30,passing_30+30))
        # Track back to where the temperature started to increase.
        ignition = peak_curvature(egt.array, slice(passing_30,0,-1))
        return started_up, ignition
    else:
        return None, None

class Eng1GasTempStartMax(KeyPointValueNode):
    name = 'Eng (1) Gas Temp Start Max'
    def derive(self, egt=P('Eng (1) Gas Temp'), n2=P('Eng (1) N2'),
               toffs=KTI('Takeoff Turn Onto Runway')):
        # Extract the index for the first turn onto the runway.
        if len(toffs) > 0:
            # Extract the index for the first turn onto the runway.
            fto_idx = [t.index for t in toffs][0]
            started_up, ignition = peak_start_egt(egt, n2, fto_idx)
            if started_up:
                self.create_kpvs_within_slices(egt.array,[slice(ignition,started_up)],max_value)

    
class Eng2GasTempStartMax(KeyPointValueNode):
    name = 'Eng (2) Gas Temp Start Max'
    def derive(self, egt=P('Eng (2) Gas Temp'), n2=P('Eng (2) N2'),
               toffs=KTI('Takeoff Turn Onto Runway')):
        # If the data started after the aircraft is airborne, we'll never see the engine start.
        if len(toffs) > 0:
            # Extract the index for the first turn onto the runway.
            fto_idx = [t.index for t in toffs][0]
            started_up, ignition = peak_start_egt(egt, n2, fto_idx)
            if started_up:
                self.create_kpvs_within_slices(egt.array,[slice(ignition,started_up)],max_value)

#-------------------------------------------------------------------------------
    
class EngGasTempAfterTakeoffMax(KeyPointValueNode):
    name = 'Eng Gas Temp After Takeoff Max'
    def derive(self, eng_egt=P('Eng (*) Gas Temp Max'), ratings=S('TOGA 5 Min Rating')):
        egt = np.ma.copy(eng_egt.array)
        for rating in ratings:
            egt[rating.slice] = np.ma.masked
        index, value = max_value(egt) # i.e. maximum excluding takeoff periods
        self.create_kpv(index, value)


class EngGasTempTakeoffMax(KeyPointValueNode):
    name = 'Eng Gas Temp Takeoff Max'
    def derive(self, eng_egt_max=P('Eng (*) Gas Temp Max'), ratings=S('TOGA 5 Min Rating')):
        self.create_kpvs_within_slices(eng_egt_max.array, ratings, max_value)


class EngN1ToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng N1 Up To FL100 Max'
    def derive(self, eng=P('Eng (*) N1 Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_from_to(-1000, 10000),
                                       max_value)


class EngEPRToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng EPR Up To FL100 Max'
    def derive(self, eng=P('Eng (*) EPR Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_from_to(-1000, 10000),
                                       max_value)


class EngTorqueToFL100Max(KeyPointValueNode):
    '''TODO: Test.'''
    name = 'Eng Torque Up To FL100 Max'
    def derive(self, eng=P('Eng (*) Torque Max'), alt_std=P('Altitude STD')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_std.slices_from_to(-1000, 10000),
                                       max_value)


class EngEPR500FtToTouchdownMin(KeyPointValueNode):
    name = 'Eng EPR 500 Ft To Touchdown Min'
    def derive(self, eng=P('Eng (*) EPR Min'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)

class EngN1500FtToTouchdownMin(KeyPointValueNode):
    name = 'Eng N1 500 Ft To Touchdown Min'
    def derive(self, eng=P('Eng (*) N1 Min'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)


class EngTorque500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, eng=P('Eng (*) Torque Max'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)


class EngTorque500FtToTouchdownMin(KeyPointValueNode):
    def derive(self, eng=P('Eng (*) Torque Min'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)

        
class EngN13000FtToTouchdownMax(KeyPointValueNode):
    name = 'Eng N1 3000 Ft To Touchdown Max'
    def derive(self, eng=P('Eng (*) N1 Max'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(3000, 0),
                                       max_value)


class EngN1500To20FtMax(KeyPointValueNode):
    name = 'Eng N1 500 To 20 Ft Max'
    def derive(self, eng=P('Eng (*) N1 Max'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(eng.array,
                                       alt_aal.slices_from_to(500, 20),
                                       max_value)


class Eng_N1MaxDurationUnder60PercentAfterTouchdown(KeyPointValueNode): ##was named: EngN1CooldownDuration
    """
    Max duration N1 below 60% after Touchdown for engine cooldown. Using 60%
    allows for cooldown after use of Reverse Thrust.
    
    Evaluated for each engine to account for single engine taxi-in.
    
    Note: Assumes that all Engines are recorded at the same frequency.
    
    TODO: Eng_Stop KTI
    TODO: Similar KPV for duration between engine under 60 percent and engine shutdown
    """
    NAME_FORMAT = "Eng (%(eng_num)d) N1 Max Duration Under 60 Percent After Touchdown"
    NAME_VALUES = {'eng_num': [1,2,3,4]}
    
    @classmethod
    def can_operate(cls, available):
        return 'Touchdown' in available and \
               'Eng (*) Stop' in available and \
               ('Eng (1) N1' in available or \
                'Eng (2) N1' in available or \
                'Eng (3) N1' in available or \
                'Eng (4) N1' in available)
    
    def derive(self, 
               eng1=P('Eng (1) N1'),
               eng2=P('Eng (2) N1'),
               eng3=P('Eng (3) N1'),
               eng4=P('Eng (4) N1'),
               tdwn=KTI('Touchdown'), engines_stop=KTI('Eng (*) Stop')):
                
        for eng_num, eng in enumerate((eng1,eng2,eng3,eng4), start=1):
            if eng is None:
                continue # engine not available on this aircraft
            eng_stop = engines_stop.get(name='Eng (%d) Stop' % eng_num)
            if not eng_stop:
                #Q: Should we measure until the end of the flight anyway? (probably not)
                self.debug("Engine %d did not stop on this flight, cannot measure KPV", eng_num)
                continue
            
            eng_array = repair_mask(eng.array)
            eng_below_60 = np.ma.masked_greater(eng_array, 60)
            # measure duration between final touchdown and engine stop
            touchdown_to_stop_slice = max_continuous_unmasked(
                eng_below_60, slice(tdwn[-1].index, eng_stop[0].index))
            if touchdown_to_stop_slice:
                #TODO future storage of slice: self.slice = touchdown_to_stop_slice
                touchdown_to_stop_duration = (touchdown_to_stop_slice.stop - \
                                        touchdown_to_stop_slice.start) / self.hz
                self.create_kpv(touchdown_to_stop_slice.start,
                                touchdown_to_stop_duration, eng_num=eng_num)
            else:
                # create KPV of 0 seconds
                self.create_kpv(eng_stop[0].index, 0.0, eng_num=eng_num)
        


"""
Replaced by Takeoff and After Takeoff bands.

class EngN1Max(KeyPointValueNode):
    name = 'Eng N1 Max'
    def derive(self, eng=P('Eng (*) N1 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)
        """


class EngN1TaxiMax(KeyPointValueNode):
    name = 'Eng N1 Taxi Max'
    def derive(self, eng_n1=P('Eng (*) N1 Max'), taxi=S('Taxiing')):
        self.create_kpvs_within_slices(eng_n1.array, taxi, max_value)


class EngN1TakeoffMax(KeyPointValueNode):
    name = 'Eng N1 Takeoff Max'
    def derive(self, eng_n1=P('Eng (*) N1 Max'), ratings=S('TOGA 5 Min Rating')):
        self.create_kpvs_within_slices(eng_n1.array, ratings, max_value)


class EngN1AfterTakeoffMax(KeyPointValueNode):
    name = 'Eng N1 After Takeoff Max'
    def derive(self, eng_n1=P('Eng (*) N1 Max'), ratings=S('TOGA 5 Min Rating')):
        eng_n1 = np.ma.copy(eng_n1.array)
        for rating in ratings:
            eng_n1[rating.slice] = np.ma.masked
        index, value = max_value(eng_n1) # i.e. maximum excluding takeoff periods
        self.create_kpv(index, value)


class EngN1GoAroundMax(KeyPointValueNode):
    name = 'Eng N1 Go Around Max'
    def derive(self, eng_n1=P('Eng (*) N1 Max'), ga=S('Go Around And Climbout')):
        self.create_kpvs_within_slices(eng_n1.array, ga, max_value)


class EngN1CyclesInFinalApproach(KeyPointValueNode):
    def derive(self, eng_n1=P('Eng (*) N1 Max'), fapps = S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(eng_n1.array[fapp.slice], 5.0, 10.0, eng_n1.hz, fapp.slice.start))


class EngN1WithThrustReversersDeployedMax(KeyPointValueNode):
    name = 'Eng N1 With Thrust Reversers Deployed Max'
    def derive(self, pwr=P('Eng (*) N1 Max'),tr=P('Thrust Reversers')):
        index = np.ma.argmax(np.ma.where(tr.array==1, pwr.array, np.ma.masked))
        if index:
            self.create_kpv(index, pwr.array[index])


class EngN2TakeoffMax(KeyPointValueNode):
    name = 'Eng N2 Takeoff Max'
    def derive(self, eng_n2=P('Eng (*) N2 Avg'), ratings=S('TOGA 5 Min Rating')):
        self.create_kpvs_within_slices(eng_n2.array, ratings, max_value)


class EngN2AfterTakeoffMax(KeyPointValueNode):
    name = 'Eng N2 After Takeoff Max'
    def derive(self, eng_n2=P('Eng (*) N2 Max'), ratings=S('TOGA 5 Min Rating')):
        eng_n2 = np.ma.copy(eng_n2.array)
        for rating in ratings:
            eng_n2[rating.slice] = np.ma.masked
        index, value = max_value(eng_n2) # i.e. maximum excluding takeoff periods
        self.create_kpv(index, value)


class EngN2CyclesInFinalApproach(KeyPointValueNode):
    def derive(self, eng_n2=P('Eng (*) N2 Avg'), fapps = S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(eng_n2.array[fapp.slice], 10.0, 10.0, eng_n2.hz, fapp.slice.start))


class EngN3TakeoffMax(KeyPointValueNode):
    name = 'Eng N3 Takeoff Max'
    def derive(self, eng_n3=P('Eng (*) N3 Avg'), ratings=S('TOGA 5 Min Rating')):
        self.create_kpvs_within_slices(eng_n3.array, ratings, max_value)


class EngN3AfterTakeoffMax(KeyPointValueNode):
    name = 'Eng N3 After Takeoff Max'
    def derive(self, eng_n3=P('Eng (*) N3 Max'), ratings=S('TOGA 5 Min Rating')):
        eng_n3 = np.ma.copy(eng_n3.array)
        for rating in ratings:
            eng_n3[rating.slice] = np.ma.masked
        index, value = max_value(eng_n3) # i.e. maximum excluding takeoff periods
        self.create_kpv(index, value)


class EngOilTempMax(KeyPointValueNode):
    name = 'Eng Oil Temp Max'
    def derive(self, oil_temp=P('Eng (*) Oil Temp Max')):
        index, value = max_value(oil_temp.array)
        self.create_kpv(index, value)


class EngOilTemp15MinuteMax(KeyPointValueNode):
    name = 'Eng Oil Temp 15 Minutes Max'
    def derive(self, oil_temp=P('Eng (*) Oil Temp Max')):
        oil_15 = clip(oil_temp.array, 15*60, oil_temp.hz)
        # There have been cases where there were no valid oil temperature
        # measurements throughout the flight, in which case there's no point
        # testing for a maximum.
        if oil_15 != None:
            self.create_kpv(*max_value(oil_15))


class EngVibN1Max(KeyPointValueNode):
    name = 'Eng Vib N1 Max'
    ##def derive(self, eng=P('Eng (*) Vib N1 Max'), fast=S('Fast')):
        ##for sect in fast:
            ##index, value = max_value(eng.array, sect.slice)
            ##self.create_kpv(index, value)
            
    def derive(self, eng=P('Eng (*) Vib N1 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)
            
            
class EngVibN2Max(KeyPointValueNode):
    name = 'Eng Vib N2 Max'
    ##def derive(self, eng=P('Eng (*) Vib N2 Max'), fast=S('Fast')):
        ##for sect in fast:
            ##index, value = max_value(eng.array, sect.slice)
            ##self.create_kpv(index, value)
            
    def derive(self, eng=P('Eng (*) Vib N2 Max')):
        index, value = max_value(eng.array)
        self.create_kpv(index, value)


class EventMarkerPressed(KeyPointValueNode):
    def derive(self, event=P('Event Marker'), airs=S('Airborne')):
        pushed = np.ma.clump_unmasked(np.ma.masked_equal(event.array, 0))
        events_in_air = slices_and(pushed, [air.slice for air in airs])
        for event_in_air in events_in_air:        
            if event_in_air:
                duration = (event_in_air.stop - event_in_air.start) / event.frequency
                index = (event_in_air.stop + event_in_air.start) / 2.0
                self.create_kpv(index, duration)


class RateOfClimbMax(KeyPointValueNode):
    '''
    .. TODO:: testcases
    '''
    def derive(self, rate_of_climb=P('Rate Of Climb'),
               climbing=S('Climbing')):
        #TODO: Merge with below RateOfDescentMax accepting a flightphase arg
        for climb in climbing:
            duration = climb.slice.stop - climb.slice.start
            if duration > settings.CLIMB_OR_DESCENT_MIN_DURATION:
                index, value = max_value(rate_of_climb.array, climb.slice)
                self.create_kpv(index, value)


class RateOfClimb35To1000FtMin(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(35, 1000),
                                       min_value)


class RateOfDescentMax(KeyPointValueNode):
    '''
    .. TODO:: testcases
    '''
    def derive(self, rate_of_climb=P('Rate Of Climb'),
               descending=S('Descending')):
        for descent in descending:
            duration = descent.slice.stop - descent.slice.start
            if duration > settings.CLIMB_OR_DESCENT_MIN_DURATION:
                index, value = min_value(rate_of_climb.array, descent.slice)
                self.create_kpv(index, value)


# FIXME: Bounced Landing name duplicated between KPV and Section!
class BouncedLanding(KeyPointValueNode):
    def derive(self, bounced_landing=S('Bounced Landing Section')):
        return NotImplemented


class AltitudeAtFirstConfigChangeAfterLiftoff(KeyPointValueNode):
    def derive(self, flap=P('Flap'), alt_aal=P('Altitude AAL'),airs=S('Airborne')):
        for air in airs:
            # find where flap changes
            change_indexes = np.ma.where(np.ma.diff(flap.array[air.slice]))[0]
            if len(change_indexes):
                # create at first change
                index = (air.slice.start or 0) + change_indexes[0]
                self.create_kpv(index, value_at_index(alt_aal.array, index))


class HeadingDeviationOnTakeoffAbove100Kts(KeyPointValueNode):
    """
    The heading deviation is measured as the peak-to-peak deviation between
    100kts and 5 deg nose pitch up, at which time the weight is clearly off
    the wheel (we avoid using weight on nosewheel as this is often not
    recorded).
    The value is annotated half way between the end conditions.
    """
    def derive(self, head=P('Heading Continuous'), airspeed=P('Airspeed'),
               pitch=P('Pitch'), toffs=S('Takeoff')):
        for toff in toffs:
            start = index_at_value(airspeed.array, 100.0, _slice=toff.slice)
            stop = index_at_value(pitch.array, 5.0, _slice=toff.slice)
            head_dev = np.ma.ptp(head.array[start:stop])
            self.create_kpv((start+stop)/2, head_dev)
    

class HeadingDeviation500To20Ft(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'), alt_aal=P('Altitude AAL For Flight Phases')):
        for band in alt_aal.slices_from_to(500, 20):
            dev = np.ma.ptp(head.array[band])
            self.create_kpv(band.stop, dev)
        

class HeadingDeviationTouchdownPlus4SecTo60Kts(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'), tdwns=KTI('Touchdown'), airspeed=P('Airspeed')):
        for tdwn in tdwns:
            begin = tdwn.index + 4.0*head.frequency
            end = index_at_value(airspeed.array, 60.0, slice(begin,None))
            if end:
                # We found a suitable endpoint, so create a KPV...
                dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv(end, dev)


class HeadingDeviationOnLandingAbove100Kts(KeyPointValueNode):
    """
    See heading deviation on takeoff comments. For landing the Altitude AAL
    is used to detect start of landing, again to avoid variation from the use
    of different aircraft recording configurations.
    """
    def derive(self, head=P('Heading Continuous'), airspeed=P('Airspeed'),
               alt=P('Altitude AAL For Flight Phases'), lands=S('Landing')):
        for land in lands:
            begin = index_at_value(alt.array, 1.0, _slice=land.slice)
            end = index_at_value(airspeed.array, 100.0, _slice=land.slice)
            if begin > end:
                break # Landed below 100kts. Can happen!
            else:
                head_dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv((begin+end)/2, head_dev)
            
            
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
            index = min(off_rwy.index+5, len(head.array))
            value = head.array[index]%360.0
            self.create_kpv(index, value)
            

class AltitudeMinsToTouchdown(KeyPointValueNode):
    #TODO: TESTS
    #Q: Review and improve this technique of building KPVs on KTIs.
    from analysis_engine.key_time_instances import MinsToTouchdown
    NAME_FORMAT = "Altitude AAL " + MinsToTouchdown.NAME_FORMAT
    NAME_VALUES = MinsToTouchdown.NAME_VALUES
    
    def derive(self, alt_aal=P('Altitude AAL'), t_tdwns=KTI('Mins To Touchdown')):
        for t_tdwn in t_tdwns:
            #WARNING: This assumes Mins time will be the first value and only two digit
            # TODO: Confirm *.array is correct (DJ)
            self.create_kpv(t_tdwn.index, alt_aal.array[t_tdwn.index], time=int(t_tdwn.name[:2]))
            

class HeightLost1000To2000FtMax(KeyPointValueNode):
    def derive(self, ht_loss=P('Descend For Flight Phases'), 
               alt_aal=P('Altitude AAL For Flight Phases')):
        climbs = alt_aal.slices_from_to(1000, 2000)
        for climb in climbs:
            index = np.ma.argmin(ht_loss.array[climb])
            if index == 0:
                break
            else:
                index += climb.start
                drop = ht_loss.array[index]
                self.create_kpv(index, drop)
                

class HeightLost50To1000Max(KeyPointValueNode):
    def derive(self, ht_loss=P('Descend For Flight Phases'), 
               alt_aal=P('Altitude AAL For Flight Phases')):
        climbs = alt_aal.slices_from_to(50, 1000)
        for climb in climbs:
            index = np.ma.argmin(ht_loss.array[climb])
            if index == 0:
                break
            else:
                index += climb.start
                drop = ht_loss.array[index]
                self.create_kpv(index, drop)


class FlapAtGearSelectionDown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_sel_down=KTI('Gear Selection Down')):
        self.create_kpvs_at_ktis(flap.array, gear_sel_down)


class FlapWithGearUpMax(KeyPointValueNode):
    def derive(self, flap=P('Flap'), gear_down=P('Gear Down')):
        gear_up = np.ma.masked_equal(gear_down.array, 1)
        gear_up_slices = np.ma.clump_unmasked(gear_up)
        self.create_kpvs_within_slices(flap.array, gear_up_slices, max_value)


class FlapAtTouchdown(KeyPointValueNode):
    def derive(self, flap=P('Flap'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(flap.array, touchdowns)

   
class FlapAtLiftoff(KeyPointValueNode):
    def derive(self, flap=P('Flap'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(flap.array, liftoffs)
  

class FlapWithSpeedbrakesDeployedMax(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, flap=P('Flap'), speedbrake=P('Speedbrake'), airs=S('Airborne')):
        # mask all values where speedbrake isn't deployed
        # Requires Speedbrake == 0 means not_deployed
        flap.array[speedbrake.array == 0] = np.ma.masked
        index, value = max_value(mask_outside_slices(flap.array, [s.slice for s in airs]))
        # It is normal for flights to be flown without speedbrake and flap
        # together, so trap this case to avoid nuisance warnings.
        if index and value:
            self.create_kpv(index, value)


class FlareTime20FtToTouchdown(KeyPointValueNode):
    #TODO: Tests
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'), tdowns=KTI('Touchdown'), lands=S('Landing')):
        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)[0]
            if this_landing:
                # Scan backwards from touchdown to the start of the landing
                # which is defined as 50ft, so will include passing through
                # 20ft AAL.
                idx_20 = index_at_value(alt_aal.array, 20.0, _slice=slice(tdown.index,this_landing.start_edge,-1))
                self.create_kpv(tdown.index, (tdown.index-idx_20)/alt_aal.frequency)


class FlareDistance20FtToTouchdown(KeyPointValueNode):
    #TODO: Tests
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'), tdowns=KTI('Touchdown'), lands=S('Landing'), gspd=P('Groundspeed')):
        for tdown in tdowns:
            this_landing = lands.get_surrounding(tdown.index)[0]
            if this_landing:
                idx_20 = index_at_value(alt_aal.array, 20.0, _slice=slice(tdown.index,this_landing.slice.start,-1))
                # Integrate returns an array, so we need to take the max
                # value to yield the KTP value.
                dist = max(integrate(gspd.array[idx_20:tdown.index], gspd.hz))
                self.create_kpv(tdown.index, dist)


class AltitudeAtSuspectedLevelBust(KeyPointValueNode):
    def derive(self, alt_std=P('Altitude STD')):
        bust = 300 # ft
        bust_time = 3*60 # 3 mins
        
        alt_hyst = hysteresis(alt_std.array, bust)
        hyst_rate = np.ma.ediff1d(alt_hyst, to_end=0.0)
        # Given application of hysteresis and taking differences, we can be
        # sure of zero values where data is constant.
        changes = np.ma.clump_unmasked(np.ma.masked_equal(hyst_rate, 0.0))
        
        if len(changes) < 3:
            return # You can't have a level bust if you just go up and down.
        
        for num in range(len(changes)-1):
            begin = changes[num].stop
            end = changes[num+1].start
            if hyst_rate[begin-1] * hyst_rate[end] < 0.0:
                duration = (end-begin)/alt_std.frequency
                
                if duration < bust_time:
                    alt_before = alt_std.array[changes[num].start]
                    alt_after = alt_std.array[changes[num+1].stop-1]
                    peak_idx = np.ma.argmax(
                        np.ma.abs(alt_std.array[begin:end]-
                                  alt_std.array[begin]))\
                        + begin
                    
                    alt_peak = alt_std.array[peak_idx]
                    
                    if alt_peak>(alt_before+alt_after)/2:
                        overshoot = min(alt_peak-alt_before,
                                        alt_peak-alt_after)
                        
                    else:
                        # Strictly this is an undershoot, but keeping the
                        # name the same saves a line of code
                        overshoot = max(alt_peak-alt_before,
                                        alt_peak-alt_after)
                        
                    self.create_kpv(peak_idx,overshoot)
                    
            

        """
        idxs, peaks = cycle_finder(alt_std.array, min_step=bust)

        if idxs == None:
            return
        for num, idx in enumerate(idxs[1:-1]):
            begin = index_at_value(np.ma.abs(alt_std.array-peaks[num+1]), bust, _slice=slice(idx,None,-1))
            end = index_at_value(np.ma.abs(alt_std.array-peaks[num+1]), bust, _slice=slice(idx,None))
            if begin and end:
                duration = (end-begin)/alt_std.frequency
                if duration < bust_time:
                    a=alt_std.array[idxs[num]] # One before the peak of interest
                    b=alt_std.array[idxs[num+1]] # The peak of interst
                    c=alt_std.array[idxs[num+2]] # The next one
                    if b>(a+c)/2:
                        overshoot = min(b-a,b-c)
                        self.create_kpv(idx,overshoot)
                    else:
                        undershoot = max(b-a,b-c)
                        self.create_kpv(idx,undershoot)
                        """
                        

        

class EngOilPressMax(KeyPointValueNode):
    #TODO: TESTS
    name = 'Eng Oil Press Max'
    def derive(self, eng_oil_press=P('Eng (*) Oil Press Max')):
        index, value = max_value(eng_oil_press.array)
        self.create_kpv(index, value)
        

class EngOilPressMin(KeyPointValueNode):
    #TODO: TESTS
    name = 'Eng Oil Press Min'
    def derive(self, eng_oil_press=P('Eng (*) Oil Press Min'), 
               airs=S('Airborne')):
        # Only check in flight to avoid zero pressure readings for stationary engines.
        for air in airs:
            index, value = min_value(eng_oil_press.array)
            self.create_kpv(index, value)


class EngOilQtyMin(KeyPointValueNode):
    def derive(self, q1=P('Eng (1) Oil Qty'), q2=P('Eng (2) Oil Qty'),
               q3=P('Eng (3) Oil Qty'), q4=P('Eng (4) Oil Qty'),
               airs=S('Airborne')):
        qmin = np.ma.minimum(q1.array,q2.array,q3.array,q4.array)
        self.create_kpvs_within_slices(qmin, airs, min_value)

class EngOilQtyMax(KeyPointValueNode):
    def derive(self, q1=P('Eng (1) Oil Qty'), q2=P('Eng (2) Oil Qty'), 
               q3=P('Eng (3) Oil Qty'), q4=P('Eng (4) Oil Qty'),
               airs=S('Airborne')):
        qmax = np.ma.maximum(q1.array,q2.array,q3.array,q4.array)
        self.create_kpvs_within_slices(qmax, airs, max_value)


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
        gspd = np.ma.copy(gspeed.array) # Prepare to change mask here.
        for turn in turns:
            gspd[turn.slice]=np.ma.masked
        self.create_kpvs_within_slices(gspd, taxis, max_value)


class GroundspeedTaxiingTurnsMax(KeyPointValueNode):
    def derive(self, gspeed=P('Groundspeed'), 
               turns=S('Turning On Ground')):
        self.create_kpvs_within_slices(gspeed.array, turns, max_value)

    
class GroundspeedRTOMax(KeyPointValueNode):
    name = 'Groundspeed RTO Max'
    def derive(self, gndspd=P('Groundspeed'),
               rejected_takeoffs=S('Rejected Takeoff')):
        for rejected_takeoff in rejected_takeoffs:
            self.create_kpvs_within_slices(gndspd.array, 
                                           rejected_takeoff.slice, max_value)


class GroundspeedAtTouchdown(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(gspd.array, touchdowns)


class GroundspeedOnGroundMax(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), grounds=S('On Ground')):
        self.create_kpvs_within_slices(gspd.array, grounds, max_value)


class GroundspeedVacatingRunway(KeyPointValueNode):
    def derive(self, gspd=P('Groundspeed'), off_rwy=KTI('Landing Turn Off Runway')):
        self.create_kpvs_at_ktis(gspd.array, off_rwy)
        



'''
# These funcitons fail because eng_clipped has only been computed
# for the period alt_slice, whereas min_value searches for
# alt_slice WITHIN eng_clipped. Need to be able to offset
# this function - better than computing eng_clipped for
# the whole flight.

class LowPowerInFinalApproachFor10Sec(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, eng_n1_avg=P('Eng (*) N1 Avg'), fin_apps=S('Final Approach')):
        for fin_app in fin_apps:
            eng_clipped = clip(eng_n1_avg.array[fin_app.slice], 10, eng_n1_avg.hz, remove='troughs')
            self.create_kpv(*min_value(eng_clipped, fin_app.slice))

    
class LowPowerBelow500FtFor10Sec(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, eng_n1_avg=P('Eng (*) N1 Avg'), alt=P('Altitude AAL'), fin_apps=S('Final Approach')):
        for alt_slice in alt.slices_from_to(500, 0):
            for fin_app in fin_apps:
                if slices_overlap(alt_slice, fin_app.slice):
                   eng_clipped = clip(eng_n1_avg.array[alt_slice], 10, eng_n1_avg.hz, remove='troughs')
                   self.create_kpv(*min_value(eng_clipped, alt_slice))
'''

class PitchCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of pitch attitude that exceed 3 deg in
    pitch from peak to peak and with a maximum cycle period of 10 seconds
    during the final approach phase.
    '''
    def derive(self, pitch=P('Pitch'), fapps = S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(pitch.array[fapp.slice], 3.0, 10.0, pitch.hz, fapp.slice.start))


class Pitch5FtToTouchdownMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(5, 0), max_value)


class PitchAtLiftoff(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), liftoffs=KTI('Liftoff')):
        self.create_kpvs_at_ktis(pitch.array, liftoffs)


class PitchAt35FtInClimb(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal = P('Altitude AAL For Flight Phases'), climbs=S('Climb')):
        for climb in climbs:
            index= index_at_value(alt_aal.array, 35.0, climb.slice)
            if index:
                value = value_at_index(pitch.array, index)
                self.create_kpv(index, value)


class Pitch35To1000FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(35, 1000),
                                       min_value)


class PitchUpTo1000FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(0, 1000),
                                       max_value)


class PitchLiftoffTo35FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(pitch.array, takeoffs, max_value)


class PitchAtTouchdown(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), touchdowns=KTI('Touchdown')):
        self.create_kpvs_at_ktis(pitch.array, touchdowns)


class Pitch1000To100FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 100),
                                       max_value)


class Pitch1000To100FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 100),
                                       min_value)


class Pitch1000To500FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       min_value)


class Pitch500To100FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(500, 100),
                                       min_value)


class Pitch500To20FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(500, 20),
                                       min_value)


class Pitch20FtToTouchdownMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(20, 0), min_value)


class Pitch35To400FtMax(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(35, 400),
                                       max_value)


class Pitch35To400FtMin(KeyPointValueNode):
    def derive(self, pitch=P('Pitch'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt_aal.slices_from_to(35, 400),
                                       min_value)


class PitchRate35To1000FtMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch_rate.array,
                                       alt_aal.slices_from_to(35, 1000),
                                       max_value)


class PitchRate20FtToTouchdownMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch_rate.array, 
                                       alt_aal.slices_from_to(20, 0),
                                       min_value)


class PitchRate20FtToTouchdownMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(pitch_rate.array, 
                                       alt_aal.slices_from_to(20, 0),
                                       max_value)


class PitchRateDuringTakeoffMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), takeoffs=S('Takeoff'), 
               lifts=KTI('Liftoff')):
        for toff in takeoffs:
            for lift in lifts:
                if slices_overlap(toff.slice, slice(lift.index)):
                    index, value = min_value(pitch_rate.array,
                                             _slice=slice(lift.index,
                                                          toff.slice.stop))
                    self.create_kpv(index, value)


class PitchRateDuringTakeoffMax(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), takeoffs=S('Takeoff')):
        self.create_kpvs_within_slices(pitch_rate.array, takeoffs, max_value)


class PitchRateDuringLandingMin(KeyPointValueNode):
    def derive(self, pitch_rate=P('Pitch Rate'), lands=S('Landing')):
        self.create_kpvs_within_slices(pitch_rate.array, lands, min_value)


class PitchRate2DegPitchTo35FtMin(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, pitch_rate=P('Pitch Rate'), pitch=P('Pitch'), 
               takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            pitch_2_deg_idx = index_at_value(pitch.array, 2, reversed_slice,
                                         endpoint='closing') #- takeoff.slice.start
            index, value = min_value(pitch_rate.array,
                                     slice(pitch_2_deg_idx, takeoff.slice.stop))
            self.create_kpv(index, value)


class PitchRate2DegPitchTo35FtMax(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, pitch_rate=P('Pitch Rate'), pitch=P('Pitch'), 
               takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            pitch_2_deg_idx = index_at_value(pitch.array, 2, reversed_slice,
                                         endpoint='closing') #- takeoff.slice.start
            index, value = max_value(pitch_rate.array,
                                     slice(pitch_2_deg_idx, takeoff.slice.stop))
            self.create_kpv(index, value)


class Time2DegPitchTo35Ft(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, pitch_rate=P('Pitch Rate'), pitch=P('Pitch'), 
               takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            pitch_2_deg_idx = index_at_value(pitch.array, 2, reversed_slice,
                                         endpoint='closing') #- takeoff.slice.start
            begin = pitch_2_deg_idx
            end = takeoff.slice.stop
            value = (end - begin) / pitch_rate.frequency
            index = (begin + end) / 2.0
            self.create_kpv(index, value)


class PitchRate2DegPitchTo35FtAverage(KeyPointValueNode):
    #TODO: TESTS
    def derive(self, pitch_rate=P('Pitch Rate'), pitch=P('Pitch'), 
               takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            pitch_2_deg_idx = index_at_value(pitch.array, 2, reversed_slice,
                                         endpoint='closing') #- takeoff.slice.start
            begin = pitch_2_deg_idx
            end = takeoff.slice.stop
            pitch_35_ft = value_at_index(pitch.array, end)
            value = (pitch_35_ft - 2.0) / (end - begin) * pitch_rate.frequency
            index = (begin + end) / 2.0
            self.create_kpv(index, value)


class RateOfDescentTopOfDescentTo10000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL For Flight Phases'),
               descents=S('Descent')):
        for descent in descents:
            above_10k = np.ma.masked_less(alt_aal.array, 10000)
            drops = np.ma.clump_unmasked(above_10k)
            self.create_kpvs_within_slices(roc.array, drops, min_value)


class RateOfDescent10000To5000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(10000, 5000),
                                       min_value)

class RateOfDescent5000To3000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(5000, 3000),
                                       min_value)
        
class RateOfDescent3000To2000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(3000, 2000),
                                       min_value)

class RateOfDescent2000To1000FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(2000, 1000),
                                       min_value)

class RateOfDescent1000To500FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'),alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array,
                                       alt_aal.slices_from_to(1000, 500),
                                       min_value)

class RateOfDescent500FtTo20FtMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array, 
                                       alt_aal.slices_from_to(500, 20),
                                       min_value)

class RateOfDescent500FtToTouchdownMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array, 
                                       alt_aal.slices_from_to(500, 0),
                                       min_value)

class RateOfDescent20ToTouchdownMax(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roc.array, 
                                       alt_aal.slices_from_to(20, 0),
                                       min_value)
        
        
class RateOfDescentAtTouchdown(KeyPointValueNode):
    def derive(self, roc=P('Rate Of Climb'), tdwn=S('Touchdown')):
        self.create_kpvs_at_ktis(roc.array, tdwn)


class RollCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of roll attitude that exceed 5 deg from
    peak to peak and with a maximum cycle period of 10 seconds during the
    final approach phase.
    '''
    def derive(self, roll=P('Roll'), fapps = S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(roll.array[fapp.slice], 5.0, 10.0, roll.hz, fapp.slice.start))
            

class RollFromTakeoffTo20FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_rad=P('Altitude Radio')):
        self.create_kpvs_within_slices(roll.array, alt_rad.slices_from_to(1,20),
                                       max_abs_value)


class RollFrom20FtToLandingMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_rad=P('Altitude Radio')):
        self.create_kpvs_within_slices(roll.array, alt_rad.slices_from_to(20,1),
                                       max_abs_value)


class RollBetween20And400FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(20, 400),
                                       max_abs_value)


class RollBetween400And1000FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(500, 1500),
                                       max_abs_value)


class RollAbove1000FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roll.array, alt_aal.slices_above(1000),
                                       max_abs_value)



class RollBetween1000And300FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(1000, 300),
                                       max_abs_value)


class RollBetween300And20FtMax(KeyPointValueNode):
    def derive(self, roll=P('Roll'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(roll.array,
                                       alt_aal.slices_between(300, 20),
                                       max_abs_value)


class RudderReversalAbove50Ft(KeyPointValueNode):
    '''
    Looks for sharp rudder reversal. Excludes operation below 50ft as this is
    normal use of the rudder to kick off drift. Uses the standard cycle
    counting process but looking for only one pair of half-cycles.
    '''
    def derive(self, rudder=P('Rudder'), alt_aal=P('Altitude AAL For Flight Phases')):
        # above_50s = np.ma.clump_unmasked(np.ma.masked_less(alt_aal.array, 50.0))
        for above_50 in alt_aal.slices_above(50.0):
            self.create_kpv(*cycle_counter(rudder.array[above_50], 6.25, 2.0, 
                                           rudder.hz, above_50.start))

            
class SpeedbrakesDeployed1000To20FtDuration(KeyPointValueNode):
    def derive(self, speedbrake=P('Speedbrake'), alt_aal=P('Altitude AAL For Flight Phases')):
        descents = alt_aal.slices_from_to(1000, 20)
        for descent in descents:
            event = np.ma.masked_less(speedbrake.array[descent],0.5) 
            # 0 = stowed, 1+ = deployed
            duration = np.ma.count(event) / speedbrake.frequency
            if duration:
                index = descent.stop
                self.create_kpv(index, duration)


class SpeedbrakesDeployedInGoAroundDuration(KeyPointValueNode):
    def derive(self, speedbrake=P('Speedbrake'), gas=S('Go Around And Climbout')):
        for ga in gas:
            event = np.ma.masked_less(speedbrake.array[ga.slice],0.5) 
            duration = np.ma.count(event) / speedbrake.frequency
            if duration:
                # Probably open at the start of the go-around, so when were they closed?
                when = np.ma.clump_unmasked(event)
                index = when[-1].stop 
                self.create_kpv(index, duration)


class SpeedbrakesDeployedWithPowerOnDuration(KeyPointValueNode):
    '''
    Each time the aircraft is flown with more than 50% N1 average power and
    the speedbrakes are open, something odd is going on! Let's record the
    duration this happened for, and allow the analyst to find out the cause.
    '''
    def derive(self, speedbrake=P('Speedbrake'), 
               power=P('Eng (*) N1 Avg'), airs=S('Airborne')):
        speedbrake_in_flight = mask_outside_slices(speedbrake.array, [s.slice for s in airs])
        speedbrakes_applied_in_flight = np.ma.clump_unmasked(np.ma.masked_less(speedbrake_in_flight,0.5))
        high_power = np.ma.clump_unmasked(np.ma.masked_less(power.array, 50.0))
        # Speedbrake and Power => s_and_p
        s_and_ps = slices_and(speedbrakes_applied_in_flight, high_power)
        for s_and_p in s_and_ps:
            index = s_and_p.start + np.ma.argmax(power.array[s_and_p])
            value = (s_and_p.stop - s_and_p.start - 1) / speedbrake.hz
            if value:
                self.create_kpv(index, value)


class SpeedbrakesDeployedWithFlapDuration(KeyPointValueNode):
    '''
    TODO: Check "With flap" equates to more than flap 0 - or get better suggestion.
    '''
    def derive(self, speedbrake=P('Speedbrake'), flap=P('Flap'), airs=S('Airborne')):
        for air in airs:
            brakes = np.ma.clump_unmasked(np.ma.masked_less(speedbrake.array[air.slice],0.5))
            with_flap = np.ma.clump_unmasked(np.ma.masked_less(flap.array[air.slice], 0.0))
            # Speedbrake and Flap => s_and_f
            s_and_fs = slices_and(brakes, with_flap)
            for s_and_f in s_and_fs:
                index = s_and_f.start + (airs.get_first().slice.start or 0)
                value = (s_and_f.stop - s_and_f.start) / speedbrake.hz
                if value:
                    self.create_kpv(index, value)


class SpeedbrakesDeployedWithConfDuration(KeyPointValueNode):
    '''
    Conf used here, but not tried or tested. Presuming conf 2 / conf 3 should not be used with speedbrakes.
    '''
    def derive(self, speedbrake=P('Speedbrake'), conf=P('Conf')):
        pos = np.ma.masked_where(speedbrake.array == 0, conf.array, copy=True)
        pos = np.ma.masked_where(conf.array >= 2.0, pos)
        clumps = np.ma.clump_unmasked(pos)
        for clump in clumps:
            index = clump.start
            value = (clump.stop - clump.start) / speedbrake.hz
            if value:
                self.create_kpv(index, value)


class SpeedbrakesDeployedWithPowerOnInHeightBandsDuration(KeyPointValueNode):
    '''
    Specific to certain operators.
    '''
    NAME_FORMAT = 'Speedbrake Deployed With N1>%(eng_n1)d Between %(upper)d And %(lower)d Ft Duration'
    NAME_VALUES = {'eng_n1': [50, 60],
                   'upper':[35000, 20000, 6000], 
                   'lower':[20000, 6000, 0]}
    def derive(self, speedbrake=P('Speedbrake'), power=P('Eng (*) N1 Avg'), 
               alt_aal=P('Altitude AAL For Flight Phases'), airs=S('Airborne')):
        
        for eng_speed in self.NAME_VALUES['eng_n1']:
            for up in self.NAME_VALUES['upper']:
                for low in self.NAME_VALUES['lower']:
                    if up <= low:
                        break
                    speedbrake_in_band = mask_outside_slices(speedbrake.array, alt_aal.slices_between(up, low))
                    speedbrakes_applied_in_flight = np.ma.clump_unmasked(np.ma.masked_less(speedbrake_in_band,0.5))
                    high_power = np.ma.clump_unmasked(np.ma.masked_less(power.array, eng_speed))
                    # Speedbrake and Power => s_and_p
                    s_and_ps = slices_and(speedbrakes_applied_in_flight, high_power)
                    for s_and_p in s_and_ps:
                        # Mark the point at highest power applied
                        index = s_and_p.start + np.ma.argmax(power.array[s_and_p])
                        value = (s_and_p.stop - s_and_p.start - 1) / speedbrake.hz
                        if value:
                            self.create_kpv(index, value, 
                                            eng_n1=eng_speed, upper=up, lower=low)


class TimeStickPusherActivated(KeyPointValueNode):
    '''
    We annotate the stick pusher event with the duration of the event.
    TODO: Check that this triggers correctly as stick push events are probably single samples.
    '''
    def derive(self, stick_push=P('Stick Pusher')):
        pushes = np.ma.clump_unmasked(
            np.ma.masked_equal(stick_push.array, 0.0))
        for push in pushes:
            index = push.start
            value = (push.stop - push.start) / stick_push.hz
            self.create_kpv(index, value)
            
            
class TimeStickShakerActivated(KeyPointValueNode):
    '''
    We annotate the stick shaker event with the duration of the event.
    '''
    def derive(self, stick_shaker=P('Stick Shaker')):
        shakes = np.ma.clump_unmasked(
            np.ma.masked_equal(stick_shaker.array, 0.0))
        for shake in shakes:
            index = shake.start
            value = (shake.stop - shake.start) / stick_shaker.hz
            self.create_kpv(index, value)


class TailClearanceOnTakeoffMin(KeyPointValueNode):
    def derive(self, alt_tail=P('Altitude Tail'), toffs=S('Takeoff')):
        self.create_kpvs_within_slices(alt_tail.array, toffs, min_value)


class TailClearanceOnLandingMin(KeyPointValueNode):
    def derive(self, alt_tail=P('Altitude Tail'), lands=S('Landing')):
        self.create_kpvs_within_slices(alt_tail.array, lands, min_value)


class TailClearanceOnApproach(KeyPointValueNode):
    def derive(self, alt_aal=P('Altitude AAL'), alt_tail=P('Altitude Tail'), 
                 dtl=P('Distance To Landing')):
        '''
        This finds abnormally low tail clearance during the approach down to
        100ft. It searches for the minimum angular separation between the
        flightpath and the terrain, so a 500ft clearance at 2500ft AAL is
        considered more significant than 500ft at 1500ft AAL. The value
        stored is the tail clearance. A matching KTI will allow these to be
        located on the approach chart.
        '''
        for desc_slice in alt_aal.slices_from_to(3000, 100):
            angle_array = alt_tail.array[desc_slice]/(dtl.array[desc_slice]*FEET_PER_NM)
            index, value = min_value(angle_array)
            if index:
                sample = index + desc_slice.start
                self.create_kpv(sample, alt_tail.array[sample])
    

class Tailwind100FtToTouchdownMax(KeyPointValueNode):
    """
    This event uses a masked tailwind array to that headwind conditions do
    not raise any KPV.
    """
    def derive(self, tailwind=P('Tailwind'), alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_kpvs_within_slices(np.ma.masked_less(tailwind.array,0.0),
                                       alt_aal.slices_from_to(100, 0),
                                       max_value)            
    

class TAWSAlert(KeyPointValueNode):
    name = 'TAWS Alert'
    def derive(self, taws_alert=P('TAWS Alert')):
        self.create_kpvs_from_discretes(taws_alert.array, taws_alert.hz, min_duration=2)

class TAWSSinkRateWarning(KeyPointValueNode):
    name = 'TAWS Sink Rate Warning'
    def derive(self, taws_sink_rate=P('TAWS Sink Rate')):
        self.create_kpvs_from_discretes(taws_sink_rate.array, taws_sink_rate.hz, min_duration=2)

class TAWSTooLowFlapWarning(KeyPointValueNode):
    name = 'TAWS Too Low Flap Warning'
    def derive(self, taws_too_low_flap=P('TAWS Terrain')):
        self.create_kpvs_from_discretes(taws_too_low_flap.array, taws_too_low_flap.hz, min_duration=2)

class TAWSTerrainWarning(KeyPointValueNode):
    name = 'TAWS Terrain Warning'
    def derive(self, taws_terrain=P('TAWS Terrain Ahead')):
        self.create_kpvs_from_discretes(taws_terrain.array, taws_terrain.hz, min_duration=2)

class TAWSTerrainPullUpWarning(KeyPointValueNode):
    name = 'TAWS Terrain Pull Up Warning'
    def derive(self, taws_terrain_pull_up=P('TAWS Terrain Ahead Pull Up')):
        self.create_kpvs_from_discretes(taws_terrain_pull_up.array, taws_terrain_pull_up.hz, min_duration=2)

class TAWSGlideslopeWarning(KeyPointValueNode):
    name = 'TAWS Glideslope Warning'
    def derive(self, taws_glideslope=P('TAWS Glideslope')):
        self.create_kpvs_from_discretes(taws_glideslope.array, taws_glideslope.hz, min_duration=2)

class TAWSTooLowTerrainWarning(KeyPointValueNode):
    name = 'TAWS Too Low Terrain Warning'
    def derive(self, taws_too_low_terrain=P('TAWS Too Low Terrain')):
        self.create_kpvs_from_discretes(taws_too_low_terrain.array, taws_too_low_terrain.hz, min_duration=2)

class TAWSTooLowGearWarning(KeyPointValueNode):
    name = 'TAWS Too Low Gear Warning'
    def derive(self, taws_too_low_gear=P('TAWS Too Low Gear')):
        self.create_kpvs_from_discretes(taws_too_low_gear.array, taws_too_low_gear.hz, min_duration=2)

class TAWSPullUpWarning(KeyPointValueNode):
    name='TAWS Pull Up Warning'
    def derive(self, taws_pull_up=P('TAWS Pull Up')):
        self.create_kpvs_from_discretes(taws_pull_up.array, taws_pull_up.hz, min_duration=2)

class TAWSDontSinkWarning(KeyPointValueNode):
    name='TAWS Dont Sink Warning'
    def derive(self, taws_dont_sink=P('TAWS Dont Sink')):
        self.create_kpvs_from_discretes(taws_dont_sink.array, taws_dont_sink.hz, min_duration=2)

class TAWSWindshearWarningBelow1500Ft(KeyPointValueNode):
    name='TAWS Windshear Warning Below 1500 Ft'
    def derive(self, taws_windshear=P('TAWS Windshear Warning'), 
               alt_aal=P('Altitude AAL For Flight Phases')):
        for descent in alt_aal.slices_from_to(1500,0):
            self.create_kpvs_from_discretes(taws_windshear.array[descent], taws_windshear.hz, min_duration=2)


class ThrottleCyclesInFinalApproach(KeyPointValueNode):
    '''
    Counts the number of half-cycles of throttle lever movement that exceed
    10 deg peak to peak and with a maximum cycle period of 14 seconds during
    the final approach phase.
    '''
    def derive(self, lever=P('Throttle Levers'), fapps = S('Final Approach')):
        for fapp in fapps:
            self.create_kpv(*cycle_counter(lever.array[fapp.slice], 10.0, 10.0, 
                                           lever.hz, fapp.slice.start))


class TimeTouchdownToElevatorDown(KeyPointValueNode):
    def derive(self, airspeed=P('Airspeed'), elevator=P('Elevator'),
               tdwns=KTI('Touchdown')):
        for tdwn in tdwns:
            index_elev = index_at_value(elevator.array, -14.0, slice(tdwn.index,None))
            if index_elev:
                e_14 = (index_elev - tdwn.index) / elevator.frequency
                self.create_kpv(index_elev, e_14)


class TimeTouchdownTo60Kts(KeyPointValueNode):
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
            


class WindSpeedInDescent(KeyPointValueNode):
    NAME_FORMAT = 'Windspeed At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {'parameter':['Wind Speed'],
                   'altitude':[2000,1500,1000,500,100,50]}
    def derive(self, wspd=P('Wind Speed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        for this_descent_slice in alt_aal.slices_from_to(2100, 0):
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, this_descent_slice)
                if index:
                    speed = value_at_index(wspd.array, index)
                    if speed:
                        self.create_kpv(index, speed, 
                                        parameter='Wind Speed', 
                                        altitude=alt)
                    

class WindDirectionInDescent(KeyPointValueNode):
    NAME_FORMAT = 'Wind Direction At %(altitude)d Ft AAL In Descent'
    NAME_VALUES = {'parameter':['Wind Direction Continuous'],
                   'altitude':[2000,1500,1000,500,100,50]}
    def derive(self, wdir=P('Wind Direction Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        for this_descent_slice in alt_aal.slices_from_to(2100, 0):
            for alt in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_aal.array, alt, this_descent_slice)
                if index:
                    # We check that the direction is not masked at this point
                    # before 'risking' the %360 function.
                    direction = value_at_index(wdir.array, index)
                    if direction:
                        self.create_kpv(index, direction%360.0,
                                        parameter='Wind Direction Continuous', 
                                        altitude=alt)


class WindAcrossLandingRunwayAt50Ft(KeyPointValueNode):
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
        zfw=np.ma.median(gw.array-fuel.array)
        self.create_kpv(0,zfw)
        
        
class DualStickInput(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented


class EngineBleedValvesAtLiftoff(KeyPointValueNode):
    def derive(self, b1=P('Eng (1) Bleed'), b2=P('Eng (2) Bleed'),
               lifts=KTI('Liftoff')):
        # b1 & b2 are integer arrays, but to index them correctly we need to
        # align the KTI to match these arrays. The alignment will cause the
        # integer arrays to blur at transitions, so int(b1+b2) is used to
        # remove this effect as the bleeds are changing state.
        bleeds = np.ma.array(b1.array+b2.array, dtype=int)
        for lift in lifts:
            valves = bleeds[lift.index]
            if valves:
                self.create_kpv(lift.index, valves)
        

class EngineTorqueTakeoffMax(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented


class EngineTorqueAbove10000FtMax(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented
        
        
class TimeHolding(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented

class ControlForcesTimesThree(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented

class StickPusher(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented

class AlphaFloorWarning(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented

class AlternateLawActivated(KeyPointValueNode):
    def derive(self, x=P('Not Yet')):
        return NotImplemented



