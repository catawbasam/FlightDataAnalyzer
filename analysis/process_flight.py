import sys

from utilities.dict_helpers import dict_filter

from hdfaccess.file import hdf_file

from analysis import settings
from analysis.dependency_graph import dependency_order
from analysis.library import calculate_timebase
from analysis.node import (DerivedParameterNode, 
    FlightAttributeNode, FlightPhaseNode, GeoKeyTimeInstance, KeyPointValue,
    KeyPointValueNode, KeyTimeInstance, KeyTimeInstanceNode, SectionNode)


def geo_locate(hdf, kti_list):
    """
    Translate KeyTimeInstance into GeoKeyTimeInstance namedtuples
    
    TODO: Account for different frequency kti indexes.
    """
    if 'Latitude Smoothed' not in hdf \
       or 'Longitude Smoothed' not in hdf:
        return kti_list
    
    lat_pos = hdf['Latitude Smoothed']
    long_pos = hdf['Longitude Smoothed']
    gkti_list = []
    for kti in kti_list:
        gkti = GeoKeyTimeInstance(kti.index, kti.name,
                                  lat_pos[kti.index], long_pos[kti.index])
        gkti_list.append(gkti)
    return gkti_list


def derive_parameters(hdf, node_mgr, process_order):
    """
    Derives the parameter values and if limits are available, applies
    parameter validation upon each param before storing the resulting masked
    array back into the hdf file.
    
    :param hdf: Data file accessor used to get and save parameter data and attributes
    :type hdf: hdf_file
    :param node_mgr: Used to determine the type of node in the process_order
    :type node_mgr: NodeManager
    :param process_order: Parameter / Node class names in the required order to be processed
    :type process_order: list of strings
    """
    params = {} # store all derived params that aren't masked arrays
    kpv_list = [] # duplicate storage, but maintaining types
    kti_list = []
    phase_list = []  # 'Node Name' : node()  pass in node.get_accessor()
    flight_attrs = []
    
    for param_name in process_order:
        if param_name in node_mgr.lfl:
            if settings.POST_LFL_PARAM_PROCESS:
                # perform any post_processing on LFL params
                param = hdf.get_param(param_name)
                _param = settings.POST_LFL_PARAM_PROCESS(hdf, param)
                if _param:
                    hdf.set_param(_param)
            continue
        
        elif node_mgr.get_attribute(param_name):
            # add attribute to dictionary of available params
            params[param_name] = node_mgr.get_attribute(param_name) #TODO: optimise with only one call to get_attribute
            continue
        
        node_class = node_mgr.derived_nodes[param_name]  #NB raises KeyError if Node is "unknown"
        
        # build ordered dependencies
        deps = []
        for dep_name in node_class.get_dependency_names():
            if dep_name in params:  # already calculated KPV/KTI/Phase
                deps.append(params[dep_name])
            elif dep_name in hdf:  # LFL/Derived parameter
                deps.append(hdf[dep_name])
            else:  # dependency not available
                deps.append(None)
        if not any(deps):
            raise RuntimeError("No dependencies available - Nodes cannot "
                               "operate without ANY dependencies available! "
                               "Node: %s" % node_class.__name__)
        first_dep = next((d for d in deps if d is not None))
        # initialise node
        node = node_class(frequency=first_dep.frequency,
                          offset=first_dep.offset)
        # Derive the resulting value
        result = node.get_derived(deps)
        
        if isinstance(node, KeyPointValueNode):
            #Q: track node instead of result here??
            params[param_name] = result
            kpv_list.extend(result)
        elif isinstance(node, KeyTimeInstanceNode):
            params[param_name] = result
            kti_list.extend(result)
        elif isinstance(node, FlightAttributeNode):
            params[param_name] = result
            flight_attrs.append(Attribute(result.name, result.value)) # only has one Attribute result
        elif isinstance(node, FlightPhaseNode):
            # expect a single slice
            params[param_name] = result
            phase_list.extend(result)
        elif isinstance(node, DerivedParameterNode):
            # perform any post_processing
            if settings.POST_DERIVED_PARAM_PROCESS:
                process_result = settings.POST_DERIVED_PARAM_PROCESS(hdf, result)
                if process_result:
                    result = process_result
            if hdf.duration:
                # check that the right number of results were returned
                assert len(result.array) == hdf.duration * result.frequency
            hdf.set_param(result)
        else:
            raise NotImplementedError("Unknown Type %s" % node.__class__)
        continue

    return kti_list, kpv_list, phase_list


def process_flight(hdf_path, aircraft_info, achieved_flight_record=None,
                   required_params=[], draw=False):
    """
    aircraft_info API:
    {
        'Tail Number':  # Aircraft Registration
        'Identifier':  # Aircraft Ident
        'Manufacturer': # e.g. Boeing
        'Manufacturer Serial Number': #MSN
        'Model': # e.g. 737-800
        'Frame': # e.g. 737-3C
        'Main Gear To Altitude Radio': # Distance in metres
        'Wing Span': # Distance in metres
    }
    
    achieved_flight_record API:
    {
        # TODO!
    }
    
    :param hdf_path: Path to HDF File
    :type hdf_pat: String
    
    :param aircraft: Aircraft specific attributes
    :type aircraft: dict
    
    :returns: See below:
    :rtype: Dict
    {
        'flight':[Attribute('name value')]  # sample: [Attirubte('Takeoff Airport', {'id':1234, 'name':'Int. Airport'}, Attribute('Approaches', [4567,7890]), ...], 
        'kti':[GeoKeyTimeInstance('index name latitude longitude')] if lat/long available else [KeyTimeInstance('index name')]
        'kpv':[KeyPointValue('index value name slice')]
    }
    
    """
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        # assume that all params in HDF are from LFL(!)
        lfl_params = hdf.keys()
        # calculate dependency tree
        node_mgr, process_order = dependency_order(
            lfl_params, required_params, aircraft_info, achieved_flight_record, 
            draw=sys.platform != 'win32' # False for Windows :-(
            ) 
        
        if settings.PRE_FLIGHT_ANALYSIS:
            settings.PRE_FLIGHT_ANALYSIS(hdf, aircraft_info, process_order)
        
        # derive parameters
        derived_results = derive_parameters(hdf, node_mgr, process_order)
        kti_list, kpv_list, phase_list, flight_attrs = derived_results

        #Q: Confirm aircraft tail here?
        ##validate_aircraft(aircraft_info['Identifier'], hdf['aircraft_ident'])
        
        # establish timebase for start of data
        #Q: Move to a Key Time Instance so that dependencies can be met appropriately?
        ##data_start_datetime = calculate_timebase(hdf.years, hdf.months, hdf.days, hdf.hours, hdf.mins, hdf.seconds)
                
        # go get bonus info at time of KPVs
        geo_kti_list = geo_locate(hdf, kti_list)
            
            
    if draw:
        from analysis.plot_flight import plot_flight
        plot_flight(kti_list, kpv_list, phase_list)
        
    return {'flight' : flight_attrs, 
            'kti' : geo_kti_list, 
            'kpv' : kpv_list}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-p', dest='plot', action='store_true',
                        default=False, help='Plot flight onto a graph.')
    args = parser.parse_args()
    process_flight(args.file, {'Tail Number': 'G-ABCD'}, {}, [], draw=args.plot)
