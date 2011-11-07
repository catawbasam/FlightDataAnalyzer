
#=============================
from analysis.hdf_access import hdf_file
from analysis.flight_phase import compute_flight_phases

from analysis.dependency_graph import dependency_order
from analysis.nodes import GeoKeyTimeInstance
from analysis.validate import validate

"""

TODO:
=====

* Document and validate the return types from nodes. Could even add an
accessor for getting always a list of example.





"""


def get_required_params(aircraft):
    """
    """
    param_list = []
    return param_list


def get_and_store_validity_limits(hdf, aircraft, params):
    """
    Fetch from storage (server or file)
    Store in HDF as JSON
    """
    # Use REST to find the limits?
    url = '/aircraft/%{aircraft}/limit/' #fetch them all or just the ones needed?
    # filter out the ones we need?
    
    for name, value in parameter_limits:
        hdf.set_param_limits(name, value)
        
def geo_locate(hdf, kti_list):
    """ Translate KeyTimeInstance into GeoKeyTimeInstance namedtuples
    """
    lat_pos = hdf['Latitude Smoothed']
    long_pos = hdf['Longitude Smoothed']
    gkti_list = []
    for kti in kti_list:
        gkti = GeoKeyTimeInstance(kti.index, kti.state,
                                  lat_pos[kti.index], long_pos[kti.index])
        gkti_list.append(gkti)
    return gkti_list


def validate_and_derive_parameters(hdf, nodes, process_order):
    """
    Derives the parameter values and if limits are available, applies
    parameter validation upon each param before storing the resulting masked
    array back into the hdf file.
    
    :param hdf: Data file accessor used to get and save parameter data and attributes
    :type hdf: hdf_file
    :param nodes: Used to determine the type of node in the process_order
    :type nodes: NodeManager
    :param process_order: Parameter names in the required order to be processed
    :type process_order: list of strings
    """
    for param_name in process_order:
        if param_name in nodes.lfl:
            param_ma = hdf.get_param_data(param_name)
        elif param_name in nodes.derived:  
            node = nodes.derived[param_name]
            # retrieve dependencies what we have
            deps = hdf.get_params(node.dependencies)
            result = node.derive(deps)
            
            if isinstance(node, KeyPointValueNode):
                # expect a single KPV or a list of KPVs
                params['kpv'].extend(list(result))
                continue # no further processing required
            elif isinstance(node, KeyTimeInstanceNode):
                # expect a single KTI or a list of KTIs
                params['kti'].extend(list(result))
                continue # no further processing required
            elif isinstance(node, FlightPhaseNode):
                # expect a slice
                params['phase'].append(result)
                continue # no further processing required
            elif isinstance(node, DerivedParameterNode):
                param_ma = result
                pass # further processing required
            else:
                raise NotImplementedError("Unknown Type %s" % node.__class__)
        else:
            raise KeyError("Urr, Node does not exist as a known node!")
            
            
        # test validity
        limits = hdf.get_param_limits(param_name)
        if limits:
            vparam_ma = validate(param_ma, limits)
            hdf.set_param_data(param_name, vparam_ma)
        else:
            hdf.set_param_data(param_name, param_ma)
    #endfor
        
    ##if 'correlation':
        ##validate_correlations(all_params) #Q: Before or after derived?
    

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
def process_flight(hdf_path, aircraft):
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        # get list of KPV and standard parameters to be calculated
        required_params = get_required_params(aircraft)
        # assume that all params in HDF are from LFL(!)
        lfl_params = hdf.get_param_list()
        # calculate dependency tree
        nodes, process_order = dependency_order(lfl_params, required_params)
        # get limits for params to be processed
        get_and_store_validity_limits(hdf, aircraft, process_order)
        #hdf.set_operating_limits(operating_limits)
        
        
        # establish timebase for start of data -- Q: When will this be used? Can we do this later on?
        start_datetime = calculate_timebase(hdf.years, hdf.months, hdf.days, hdf.hours, hdf.mins, hdf.seconds)
        
        params = validate_and_derive_parameters(hdf, nodes, process_order)
        
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
