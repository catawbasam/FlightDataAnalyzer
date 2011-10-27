
#=============================
from hdf_access import hdf_file
from flight_phase import compute_flight_phases
from derived import Derived, 


def validate_and_derive_parameters(param_graph):
    for order in param_graph:
        if order.type == 'derived':
            derived_params = calculate_derived_params(order.parameter)
            hdf.append_table(derived_params, table='parameter')
        elif order.type == 'correlation':
            param_correlation_masks = validate_correlations(all_params) #Q: Before or after derived?
            hdf.append_table(param_correlation_masks, table='parameter_validity')
        else:
            raise ValueError, "type %s not implemented" % order.type
    

"""
Assumes storage of all LFL params in the HDF - uses this as the param set
"""
"""
NOTES
Can have multiple KPVs, but only one of each type is marked as "primary".


DerivedParams()
  dependancy
  ma = derive(params)
  
KeyPoints() # return mixed types?
  dependancy
  kpv or kpt or kpv/t_list = calc_kpv(params)
"""
def process_flight(hdf_path):
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
       
        # get list of KPV and standard parameters to be calculated
        app = top_level_params(aircraft)
        nodes = list_all_derived_params() + list_raw_params_in_hdf()
        # calculate dependency tree
        process_order = dependency_tree(nodes, app)
        # get limits for params to be processed
        operating_limits = get_validity_limits(aircraft, process_order)
        store_limits(hdf, operating_limits)
        # establish timebase for start of data
        start_datetime = calculate_timebase(hdf.years, hdf.months, hdf.days, hdf.hours, hdf.mins, hdf.seconds)
        
        params = validate_and_derive_parameters(process_order)
        
        # go get bonus info at time of KPVs
        ##kpv_info = get_geo_location_etc_for_kpv(kpv_list)
        
        # go get bonus info at time of KPVs
        kti_info = geo_locate(params['ktis'])
        
        downsampled_params = downsample_for_graphs(graph_params_list)
        store_flight_information(flight_info, kti_info, params['kpvs'])  # in DB (not HDF)
    '''
    
    ##if not force_analysis:
            ### ensure the aircraft's the same one as we're told it is
            ##aircraft_found = validate_aircraft(aircraft, segment['aircraft_ident']) # raises error? or just returns aircraft?
            ##segment['aircraft'] = aircraft #TODO: DO SOMETHING CLEVER!!!
            
    # no longer exists:    
    ##flight_phase_map1 = flight_phases_basic(altitude, airspeed, heading) # inc. runway turn on / off KPTs
    ##hdf.append_table(flight_phase_map1, table='flight_phase') # add to HDF    
    
    
    ##flight_phase_map2 = compute_flight_phases()
    ##hdf.append_table(flight_phase_map2, table='flight_phase') # add to HDF
    
    # establish Airports and lookup to DB for further information
    ## Takeoff, Approach(s), Landing
    ##if lat_long_available:
        ##airport_meta_data = establish_airports(hdf)
        ##hdf.store_meta(airports=airport_meta_data)
    
    # calculate more derived parameters from the params above once confirmed valid
    
    
    #TODO: put into derived params?
    # calculate timebase file or each entire dfc
    
    segment['data_start_time'] = data_seg_start_datetime + timedelta(seconds=segment_slice.start)
    # not required for analysis, used for partial flight matching.
    segment['data_end_time'] = data_seg_start_datetime + timedelta(seconds=segment_slice.stop)

    # generic technique for algorithms (derived params, validity, correlations, etc.
    for algorithm in dependency_ordered(algorithms):
        data = algorithm.function(hdf[algorithm['parents']])
        hdf.append_table(algorithm['type'], data)
    
    
    
    #TODO: Review the order of below - should KPT / KPV be moved up and into
    #derivations?
    
    downsampled_params = downsample_for_graphs(graph_params_list)
    hdf.append_table(downsampled_params, table='table')
    
    kpts = calculate_key_point_times() # KPT -> KTI (Key Time Instances)
    kpv_list = get_required_kpv(aircraft_info)
    kpvs = calculate_key_point_values(hdf, kpv_list)
    store_flight_information(flight_info, kpts, kpvs) # in DB (not HDF)
    
    
    #=============================
    
    # Request to the Database
    request_event_detection(flight)  # uses Nile/web/thresholds.py

    '''

#================================
if __name__ == '__main__':
    ##file_path = os.path.join('.', 'file.hdf')
    #open hdf
    process_flight(hdf)
