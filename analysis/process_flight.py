import inspect
import logging

from datetime import datetime, timedelta

from analysis import settings
from analysis.dependency_graph import dependency_order
from analysis.node import (Attribute, DerivedParameterNode, 
    FlightAttributeNode, FlightPhaseNode, KeyPointValue,
    KeyPointValueNode, KeyTimeInstance, KeyTimeInstanceNode, 
    Node, NodeManager, P, SectionNode)
from hdfaccess.file import hdf_file
from utilities.dict_helpers import dict_filter


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
    for kti in kti_list:
        kti.latitude = lat_pos[kti.index]
        kti.longitude = long_pos[kti.index]
    return kti_list


def _timestamp(start_datetime, item_list):
    """
    Adds item.datetime (from timedelta of item.index + start_datetime)
    
    :param start_datetime: Origin timestamp used as a base to the index
    :type start_datetime: datetime
    :param item_list: list of objects with a .index attribute
    :type item_list: list
    """
    for item in item_list:
        item.datetime = start_datetime + timedelta(seconds=float(item.index))
    return item_list


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
    
    nodes_not_implemented = []
    
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
        node_deps = node_class.get_dependency_names()
        for dep_name in node_deps:
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
        try:
            first_dep = next((d for d in deps if d is not None and d.frequency))
            frequency = first_dep.frequency
            offset = first_dep.offset
        except StopIteration:
            frequency = None
            offset = None
        # initialise node
        node = node_class(frequency=first_dep.frequency,
                          offset=first_dep.offset)
        logging.info("Processing parameter %s", param_name)
        # Derive the resulting value
        try:
            result = node.get_derived(deps)
        except NotImplementedError:
            ##logging.error("Node %s not implemented", node_class.__name__)
            #TODO: remove hack below!!!
            params[param_name] = node # HACK!
            nodes_not_implemented.append(node_class.__name__)
            continue
        
        if isinstance(node, KeyPointValueNode):
            #Q: track node instead of result here??
            params[param_name] = result
            kpv_list.extend(result.get_aligned(P('for_aligned_storage',[],1,0)))
        elif isinstance(node, KeyTimeInstanceNode):
            params[param_name] = result
            kti_list.extend(result.get_aligned(P('for_aligned_storage',[],1,0)))
        elif isinstance(node, FlightAttributeNode):
            params[param_name] = result
            flight_attrs.append(Attribute(result.name, result.value)) # only has one Attribute result
        elif isinstance(node, FlightPhaseNode):
            # expect a single slice
            params[param_name] = result
            phase_list.extend(result.get_aligned(P('for_aligned_storage',[],1,0)))
        elif isinstance(node, DerivedParameterNode):
            # perform any post_processing
            if settings.POST_DERIVED_PARAM_PROCESS:
                process_result = settings.POST_DERIVED_PARAM_PROCESS(hdf, result)
                if process_result:
                    result = process_result
            if hdf.duration:
                # check that the right number of results were returned
                assert len(result.array) == hdf.duration * result.frequency, \
                       "Array lengths mismatch."
            hdf.set_param(result)
        else:
            raise NotImplementedError("Unknown Type %s" % node.__class__)
        continue
    if nodes_not_implemented:
        logging.error("Nodes not implemented: %s", nodes_not_implemented)
    return kti_list, kpv_list, phase_list, flight_attrs


def get_derived_nodes(module_names):
    """ Get all nodes into a dictionary
    """
    def isclassandsubclass(value, classinfo):
        return inspect.isclass(value) and issubclass(value, classinfo)

    nodes = {}
    for name in module_names:
        #Ref:
        #http://code.activestate.com/recipes/223972-import-package-modules-at-runtime/
        # You may notice something odd about the call to __import__(): why is
        # the last parameter a list whose only member is an empty string? This
        # hack stems from a quirk about __import__(): if the last parameter is
        # empty, loading class "A.B.C.D" actually only loads "A". If the last
        # parameter is defined, regardless of what its value is, we end up
        # loading "A.B.C"
        ##abstract_nodes = ['Node', 'Derived Parameter Node', 'Key Point Value Node', 'Flight Phase Node'
        module = __import__(name, globals(), locals(), [''])
        for c in vars(module).values():
            if isclassandsubclass(c, Node) and c.__module__ != 'analysis.node':
                try:
                    nodes[c.get_name()] = c
                except TypeError:
                    #TODO: Handle the expected error of top level classes
                    # Can't instantiate abstract class DerivedParameterNode
                    # - but don't know how to detect if we're at that level without resorting to 'if c.get_name() in 'derived parameter node',..
                    logging.exception('Failed to import class: %s' % c.get_name())
    return nodes


def process_flight(hdf_path, aircraft_info, start_datetime=datetime.now(), achieved_flight_record={},
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
        'flight':[Attribute('name value')]  # sample: [Attribute('Takeoff Airport', {'id':1234, 'name':'Int. Airport'}, Attribute('Approaches', [4567,7890]), ...], 
        'kti':[GeoKeyTimeInstance('index name latitude longitude')] if lat/long available else [KeyTimeInstance('index name')]
        'kpv':[KeyPointValue('index value name slice')]
    }
    
    """
    # go through modules to get derived nodes
    derived_nodes = get_derived_nodes(settings.NODE_MODULES)
    # if required_params isn't set, try using ALL derived_nodes!
    if not required_params:
        logging.warning("No required_params declared, using all derived nodes")
        required_params = derived_nodes.keys()
    
    # include all flight attributes as required
    required_params += get_derived_nodes(['analysis.flight_attribute']).keys()
        
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        # Track nodes. Assume that all params in HDF are from LFL(!)
        node_mgr = NodeManager(start_datetime, hdf.keys(), required_params, 
                               derived_nodes, aircraft_info, achieved_flight_record)
        
        # calculate dependency tree
        process_order = dependency_order(node_mgr, draw=draw) 
        
        if settings.PRE_FLIGHT_ANALYSIS:
            settings.PRE_FLIGHT_ANALYSIS(hdf, aircraft_info, process_order)
        
        # derive parameters
        kti_list, kpv_list, phase_list, flight_attrs = derive_parameters(
            hdf, node_mgr, process_order)
             
        # geo locate KTIs
        kti_list = geo_locate(hdf, kti_list)
        kti_list = _timestamp(start_datetime, kti_list)

        # timestamp KPVs
        kpv_list = _timestamp(start_datetime, kpv_list)
        
    if draw:
        from analysis.plot_flight import plot_flight
        plot_flight(hdf_path, kti_list, kpv_list, phase_list)
        
    return {'flight' : flight_attrs, 
            'kti' : kti_list, 
            'kpv' : kpv_list,
            'phases' : phase_list}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-p', dest='plot', action='store_true',
                        default=False, help='Plot flight onto a graph.')
    args = parser.parse_args()
    process_flight(args.file, {'Tail Number': 'G-ABCD'}, draw=args.plot)
