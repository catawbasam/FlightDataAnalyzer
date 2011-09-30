
#=============================

def process_flight():
    # open HDF for reading
    #with hdf_flight as hdf:
    
    param_validity_masks = validate_operating_limits(all_params)
    hdf.append_table(param_validity_masks, table='parameter_validity')
    
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
    
    flight_phase_map2 = flight_phases2()
    hdf.append_table(flight_phase_map2, table='flight_phase') # add to HDF
    
    # calculate more derived parameters from the params above once confirmed valid
    validate_and_derive_parameters(param_list)
    
    
    
    # generic technique for algorithms (derived params, validity, correlations, etc.
    for algorithm in dependency_ordered(algorithms):
        data = algorithm.function(hdf[algorithm['parents']])
        hdf.append_table(algorithm['type'], data)
    
    
    
    
    
    downsampled_params = downsample_for_graphs(graph_params_list)
    hdf.append_table(downsampled_params, table='table')
    
    kpts = calculate_key_point_times() # KPT -> KTI (Key Time Instances)
    kpv_list = get_required_kpv(aircraft_info)
    kpvs = calculate_key_point_values(hdf, kpv_list)
    store_flight_information(flight_info, kpts, kpvs) # in DB (not HDF)
    
    
    #=============================
    
    # Request to the Database
    request_event_detection(flight)  # uses Nile/web/thresholds.py

    

#================================
if __name__ == '__main__':
    ##file_path = os.path.join('.', 'file.hdf')
    #open hdf
    process_flight(hdf)
