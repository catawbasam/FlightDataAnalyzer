import sys

from utilities.dict_helpers import dict_filter  #TODO: Mark utiltities as a dependency

from hdfaccess.file import hdf_file

from analysis import settings
from analysis.dependency_graph import dependency_order
from analysis.library import calculate_timebase
from analysis.node import (
    DerivedParameterNode, GeoKeyTimeInstance, KeyPointValue, KeyPointValueNode,
    KeyTimeInstance, KeyTimeInstanceNode, FlightPhaseNode)


def get_required_params(aircraft):
    """
    """
    param_list = [] ##['Rate Of Descent High', 'Top of Climb and Top of Descent']
    return param_list



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
        gkti = GeoKeyTimeInstance(kti.index, kti.state,
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
    phase_list = []
    
    for param_name in process_order:
        if param_name in node_mgr.lfl:
            if settings.POST_LFL_PARAM_PROCESS:
                # perform any post_processing on LFL params
                param = hdf.get_param(param_name)
                _param = settings.POST_LFL_PARAM_PROCESS(hdf, param)
                if _param:
                    hdf.set_param(_param)
            continue
        
        node_class = node_mgr.derived_nodes[param_name]  # raises KeyError if Node is "unknown"
        
        # build ordered dependencies
        deps = []
        for param in node_class.get_dependency_names():
            if param in params:  # already calculated KPV/KTI/Phase
                deps.append(params[param])
            elif param in hdf:  # LFL/Derived parameter
                deps.append(hdf[param])
            else:  # dependency not available
                deps.append(None)
        if not any(deps):
            raise RuntimeError("No dependencies available - Nodes cannot operate without ANY dependencies available! Node: %s" % node_class.__name__)
        # find first not-None dependency to use at the base param
        first_dep = next(x for x in deps if x is not None)
        # initialise node
        node = node_class(frequency=first_dep.frequency, offset=first_dep.offset)
        # Derive the resulting value
        result = node.get_derived(deps)
        
        if isinstance(node, KeyPointValueNode):
            #Q: track node instead of result here??
            params[param_name] = result  # keep track
            kpv_list.extend(result)
        elif isinstance(node, KeyTimeInstanceNode):
            params[param_name] = result  # keep track
            kti_list.extend(result)
        elif isinstance(node, FlightPhaseNode):
            # expect a single slice
            params[param_name] = result  # keep track
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


def process_flight(hdf_path, aircraft, achieved_flight_record=None, draw=False):
    """
    aircraft API:
    {
        'Tail Number':  # Aircraft Registration
        'Identifier':  # Aircraft Ident
        'Manufacturer': 
        'Manufacturer  aircraft.manufacturer_serial_number, #MSN
        'Model': 
        'Frame': 
        'Main Gear To Altitude Radio': ,
        'Wing Span': 
    }
    :param hdf_path: Path to HDF File
    :type hdf_pat: String
    
    :param aircraft: Aircraft specific attributes
    :type aircraft: dict
    
    """
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        # get list of KPV and standard parameters to be calculated
        required_params = get_required_params(aircraft)
        # assume that all params in HDF are from LFL(!)
        lfl_params = hdf.keys()
        # calculate dependency tree
        node_mgr, process_order = dependency_order(
            lfl_params, required_params, draw=sys.platform != 'win32') # False for Windows :-(
        
        if settings.PRE_FLIGHT_ANALYSIS:
            settings.PRE_FLIGHT_ANALYSIS(hdf, aircraft, process_order)
            
        kti_list, kpv_list, phase_list = derive_parameters(hdf, node_mgr, process_order)

        #Q: Confirm aircraft tail here?
        ##validate_aircraft(aircraft['Identifier'], hdf['aircraft_ident'])
        
        # establish timebase for start of data
        #Q: Move to a Key Time Instance so that dependencies can be met appropriately?
        ##data_start_datetime = calculate_timebase(hdf.years, hdf.months, hdf.days, hdf.hours, hdf.mins, hdf.seconds)
                
        # go get bonus info at time of KPVs
        kti_info = geo_locate(hdf, kti_list)
            
            
    if draw:
        from analysis.plot_flight import plot_flight
        plot_flight(kti_list, kpv_list, phase_list)
        
    flight_info = {'takeoff_airport':None,
                   'takeoff_runway':'',
                   'landing_airport':None,
                   'landing_runway':'',
                   # etc...
                   }
    return flight_info, kti_info, params['kpvs']



if __name__ == '__main__':
    import sys
    hdf_path = sys.argv[1]
    process_flight(hdf_path, None)
