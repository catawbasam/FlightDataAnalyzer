import numpy as np

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
                                     bearings_and_distances,
                                     bump,
                                     clip, 
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
                                     runway_length,
                                     np_ma_masked_zeros_like,
                                     peak_curvature,
                                     rate_of_change,
                                     runway_distance_from_end,
                                     shift_slices,
                                     slice_samples,
                                     slices_not,
                                     slices_overlap,
                                     slices_and,
                                     touchdown_inertial,
                                     value_at_index)


################################################################################
# Superclasses


"""
Tried but doesn't work - Airspeed masked below 50kts so does not return a result.
class AirspeedAtTOGA(KeyPointValueNode):
    '''
    This KPV measures the airspeed at the point of TOGA selection.
    '''
    
    name = 'Airspeed At TOGA'
    
    def derive(self, airspeed=P('Airspeed'), toga=P('Takeoff And Go Around'),
               takeoff=S('Takeoff')):
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            speed = value_at_index(airspeed.array, index) # interpolates as required
            self.create_kpvs(index, speed)
"""            

class GroundspeedAtTOGA(KeyPointValueNode):
    '''
    This KPV measures the groundspeed at the point of TOGA selection.
    '''
    
    name = 'Groundspeed At TOGA'
    
    def derive(self, gspd=P('Groundspeed'), toga=P('Takeoff And Go Around'),
               takeoff=S('Takeoff')):
        indexes = find_edges_on_state_change('TOGA', toga.array, phase=takeoff)
        for index in indexes:
            speed = value_at_index(gspd.array, index) # interpolates as required
            self.create_kpv(index, speed)
            

class DistanceFromLiftoffToRunwayEnd(KeyPointValueNode):
    def derive(self, lat_lift=KPV('Latitude At Liftoff'),
               lon_lift=KPV('Longitude At Liftoff'),
               rwy=A('FDR Takeoff Runway')):
        if ambiguous_runway(rwy) or not lat_lift:
            return
        toff_end = runway_distance_from_end(rwy.value, 
                                            lat_lift[0].value, 
                                            lon_lift[0].value)
        length = runway_length(rwy.value)
        self.create_kpv(lat_lift[0].index, toff_end/length*100.0)


class DistanceFromRotationToRunwayEnd(KeyPointValueNode):
    def derive(self, lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               pitch=P('Pitch'),
               rwy=A('FDR Takeoff Runway'),
               toffs=S('Takeoff')):
        if ambiguous_runway(rwy):
            return
        for toff in toffs:
            rot_idx = index_at_value(pitch.array, 2.0, toff.slice)
            rot_end = runway_distance_from_end(rwy.value, 
                                                lat.array[rot_idx], 
                                                lon.array[rot_idx])
            length = runway_length(rwy.value)
            self.create_kpv(rot_idx, rot_end/length*100.0)



class ILSLocalizerDeviationAtTouchdown(KeyPointValueNode):
    name = 'ILS Localizer Deviation At Touchdown'
    def derive(self, ils_loc=P('ILS Localizer'),
               ils_ests=S('ILS Localizer Established'),
               tdns=KTI('Touchdown')):
        for ils_est in ils_ests:
            for tdn in tdns:
                if is_index_within_slice(tdn.index, ils_est.slice):
                    deviation = value_at_index(ils_loc.array, tdn.index)
                    self.create_kpv(tdn.index, deviation)

        
"""

Existing FDS event...

class HeadingDeviationTouchdownPlus4SecTo60Kts(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'), tdwns=KTI('Touchdown'),
               airspeed=P('Airspeed')):
        for tdwn in tdwns:
            begin = tdwn.index + 4.0*head.frequency
            end = index_at_value(airspeed.array, 60.0, slice(begin,None))
            if end:
                # We found a suitable endpoint, so create a KPV...
                dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv(end, dev)
"""

class HeadingDeviationTouchdownPlus2SecTo80Kts(KeyPointValueNode):
    def derive(self, head=P('Heading Continuous'), tdwns=KTI('Touchdown'),
               airspeed=P('Airspeed')):
        for tdwn in tdwns:
            begin = tdwn.index + 2.0*head.frequency
            end = index_at_value(airspeed.array, 80.0, slice(begin,None))
            if end:
                # We found a suitable endpoint, so create a KPV...
                dev = np.ma.ptp(head.array[begin:end+1])
                self.create_kpv(end, dev)


class TailwindLiftoffTo100FtMax(KeyPointValueNode):
    '''
    This event uses a masked tailwind array to that headwind conditions do
    not raise any KPV.
    '''

    def derive(self, tailwind=P('Tailwind'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        self.create_kpvs_within_slices(np.ma.masked_less(tailwind.array, 0.0),
                                       alt_aal.slices_from_to(0, 100),
                                       max_value)


