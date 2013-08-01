# -*- coding: utf-8 -*-

import numpy as np

from flightdatautilities.model_information import (
    #get_aileron_map,
    get_conf_map,
    get_flap_values_mapping,
    #get_slat_map
)

from hdfaccess.parameter import MappedArray

#from analysis_engine.exceptions import DataFrameError
from analysis_engine.node import (
    A, MultistateDerivedParameterNode, 
    #KPV, KTI, 
    M,
    P,
    #S
)
from analysis_engine.library import (#actuator_mismatch,
                                     #air_track,
                                     #align,
                                     all_of,
                                     #any_of,
                                     #alt2press,
                                     #alt2sat,
                                     #bearing_and_distance,
                                     #bearings_and_distances,
                                     #blend_parameters,
                                     #blend_two_parameters,
                                     #cas2dp,
                                     #coreg,
                                     #cycle_finder,
                                     #datetime_of_index,
                                     #dp2tas,
                                     #dp_over_p2mach,
                                     #filter_vor_ils_frequencies,
                                     #first_valid_sample,
                                     #first_order_lag,
                                     #first_order_washout,
                                     #ground_track,
                                     #ground_track_precise,
                                     #hysteresis,
                                     #index_at_value,
                                     #integrate,
                                     #ils_localizer_align,
                                     #index_closest_value,
                                     #interpolate,
                                     #is_day,
                                     #is_index_within_slice,
                                     #last_valid_sample,
                                     #latitudes_and_longitudes,
                                     #localizer_scale,
                                     #machtat2sat,
                                     #mask_inside_slices,
                                     #mask_outside_slices,
                                     #max_value,
                                     #merge_masks,
                                     #merge_two_parameters,
                                     #moving_average,
                                     #np_ma_ones_like,
                                     np_ma_masked_zeros_like,
                                     #np_ma_zeros_like,
                                     #offset_select,
                                     #peak_curvature,
                                     #rate_of_change,
                                     #repair_mask,
                                     #rms_noise,
                                     #round_to_nearest,
                                     #runway_deviation,
                                     #runway_distances,
                                     #runway_heading,
                                     #runway_length,
                                     #runway_snap_dict,
                                     #shift_slice,
                                     #slices_between,
                                     #slices_from_to,
                                     #slices_not,
                                     #slices_or,
                                     #smooth_track,
                                     step_values,
                                     #straighten_altitudes,
                                     #straighten_headings,
                                     #second_window,
                                     #track_linking,
                                     #value_at_index,
                                     vstack_params,
                                     #vstack_params_where_state
                                     )

#from settings import (AZ_WASHOUT_TC,
                      #FEET_PER_NM,
                      #HYSTERESIS_FPIAS,
                      #HYSTERESIS_FPROC,
                      #GRAVITY_IMPERIAL,
                      #KTS_TO_FPS,
                      #KTS_TO_MPS,
                      #METRES_TO_FEET,
                      #METRES_TO_NM,
                      #VERTICAL_SPEED_LAG_TC)


class Configuration(MultistateDerivedParameterNode):
    '''
    Parameter for aircraft that use configuration.

    Multi-state with the following mapping::

        {
            0 : '0',
            1 : '1',
            2 : '1+F',
            3 : '1*',
            4 : '2',
            5 : '2*',
            6 : '3',
            7 : '4',
            8 : '5',
            9 : 'Full',
        }

    Some values are based on footnotes in various pieces of documentation:

    - 2(a) corresponds to CONF 1*
    - 3(b) corresponds to CONF 2*

    Note: Does not use the Flap Lever position. This parameter reflects the
    actual configuration state of the aircraft rather than the intended state
    represented by the selected lever position.

    Note: Values that do not map directly to a required state are masked with
    the data being random (memory alocated)
    '''

    values_mapping = {
        0 : '0',
        1 : '1',
        2 : '1+F',
        3 : '1*',
        4 : '2',
        5 : '2*',
        6 : '3',
        7 : '4',
        8 : '5',
        9 : 'Full',
    }

    @classmethod
    def can_operate(cls, available):
        # TODO: Implement check for the value of Family for Airbus
        return all_of(('Slat', 'Flap', 'Series', 'Family'), available)

    def derive(self, slat=P('Slat'), flap=M('Flap'), flaperon=P('Flaperon'),
               series=A('Series'), family=A('Family'), manu=A('Manufacturer')):

        if manu and manu.value != 'Airbus':
            # TODO: remove check once we can check attributes in can_operate
            self.array = np_ma_masked_zeros_like(flap.array)
            return

        mapping = get_conf_map(series.value, family.value)
        qty_param = len(mapping.itervalues().next())
        if qty_param == 3 and not flaperon:
            # potential problem here!
            self.warning("Flaperon not available, so will calculate "
                         "Configuration using only slat and flap")
            qty_param = 2
        elif qty_param == 2 and flaperon:
            # only two items in values tuple
            self.debug("Flaperon available but not required for "
                       "Configuration calculation")
            pass

        #TODO: Scale each parameter individually to ensure uniqueness.
        
        # Sum the required parameters (creates a unique state value at present)
        summed = vstack_params(*(slat, flap, flaperon)[:qty_param]).sum(axis=0)

        # create a placeholder array fully masked
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array), 
                                 self.values_mapping)
        for state, values in mapping.iteritems():
            s = np.ma.sum(values[:qty_param])
            # unmask bits we know about
            self.array[summed == s] = state




class Flap(MultistateDerivedParameterNode):
    '''
    Steps raw Flap Angle surface measurements into detents rounding to the
    midpoint of the Flap Angle transition.
    '''
    units = 'deg'

    def derive(self, flap=P('Flap Angle'), 
               series=A('Series'), family=A('Family')):
        self.values_mapping = get_flap_values_mapping(series, family, flap)
        self.array = step_values(flap.array, flap.frequency, 
                                 self.values_mapping.keys(),
                                 step_at='midpoint')
        
        
class FlapExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the lower of the start and endpoints of the movement
    apply. This minimises the chance of needing a flap overspeed inspection.
    '''

    units = 'deg'

    def derive(self, flap=P('Flap Angle'), 
               series=A('Series'), family=A('Family')):
        self.values_mapping = get_flap_values_mapping(series, family, flap)
        self.array = step_values(flap.array, flap.frequency, 
                                 self.values_mapping.keys(),
                                 step_at='excluding_transition')


class FlapIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = 'deg'

    def derive(self, flap=P('Flap Angle'), 
               series=A('Series'), family=A('Family')):
        self.values_mapping = get_flap_values_mapping(series, family, flap)
        self.array = step_values(flap.array, flap.frequency, 
                                 self.values_mapping.keys(),
                                 step_at='including_transition')
            
            
class FlapLever(MultistateDerivedParameterNode):
    '''
    Rounds the Flap Lever Angle to the selected detent at the start of the
    angle movement.
    '''

    units = 'deg'

    ##@classmethod
    ##def can_operate(cls, available):
        ##return any_of(('Flap Angle'), available) \
            ##and all_of(('Series', 'Family'), available)

    def derive(self, flap_lever=P('Flap Lever Angle'),
               series=A('Series'), family=A('Family')):
        self.values_mapping = get_flap_values_mapping(series, family, flap_lever)
        # Take the moment the flap starts to move.        
        self.array = step_values(flap_lever.array, flap_lever.frequency, 
                                 self.values_mapping.keys(),
                                 step_at='move_start')
        # Q: Should we allow for flap angle if no flap lever angle?
        ## Use flap lever position where recorded, otherwise revert to flap surface.
        ###if flap_lvr:
            #### Take the moment the lever passes midway between two flap detents.
            ###self.array = step_values(flap_lvr.array, flap_lvr.frequency, 
                                     ###flap_steps, step_at='midpoint')
        ###else:
        
        
    