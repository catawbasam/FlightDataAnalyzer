import argparse
import logging
import numpy as np
import os
import sys

from datetime import datetime, timedelta
from inspect import isclass
from networkx.readwrite import json_graph

from flightdatautilities.filesystem_tools import copy_file

from hdfaccess.file import hdf_file

from analysis_engine import hooks, settings, __version__
from analysis_engine.api_handler import APIError, get_api_handler
from analysis_engine.dependency_graph import dependency_order, graph_adjacencies
from analysis_engine.library import np_ma_masked_zeros_like
from analysis_engine.node import (ApproachNode, Attribute,
                                  derived_param_from_hdf,
                                  DerivedParameterNode,
                                  FlightAttributeNode,
                                  KeyPointValueNode,
                                  KeyTimeInstanceNode, Node,
                                  NodeManager, P, Section, SectionNode)


logger = logging.getLogger(__name__)


def geo_locate(hdf, items):
    '''
    Translate KeyTimeInstance into GeoKeyTimeInstance namedtuples
    '''
    if 'Latitude Smoothed' not in hdf \
       or 'Longitude Smoothed' not in hdf:
        logger.warning("Could not geo-locate as either 'Latitude Smoothed' or "
                       "'Longitude Smoothed' were not found within the hdf.")
        return items
    
    lat_pos = derived_param_from_hdf(hdf['Latitude Smoothed'])
    long_pos = derived_param_from_hdf(hdf['Longitude Smoothed'])
    
    for item in items:
        item.latitude = lat_pos.at(item.index)
        item.longitude = long_pos.at(item.index)
    return items


def _timestamp(start_datetime, item_list):
    '''
    Adds item.datetime (from timedelta of item.index + start_datetime)
    
    :param start_datetime: Origin timestamp used as a base to the index
    :type start_datetime: datetime
    :param item_list: list of objects with a .index attribute
    :type item_list: list
    '''
    for item in item_list:
        item.datetime = start_datetime + timedelta(seconds=float(item.index))
    return item_list


def derive_parameters(hdf, node_mgr, process_order):
    '''
    Derives the parameter values and if limits are available, applies
    parameter validation upon each param before storing the resulting masked
    array back into the hdf file.
    
    :param hdf: Data file accessor used to get and save parameter data and attributes
    :type hdf: hdf_file
    :param node_mgr: Used to determine the type of node in the process_order
    :type node_mgr: NodeManager
    :param process_order: Parameter / Node class names in the required order to be processed
    :type process_order: list of strings
    '''
    params = {} # store all derived params that aren't masked arrays
    approach_list = ApproachNode(restrict_names=False)
    kpv_list = KeyPointValueNode(restrict_names=False) # duplicate storage, but maintaining types
    kti_list = KeyTimeInstanceNode(restrict_names=False)
    section_list = SectionNode()  # 'Node Name' : node()  pass in node.get_accessor()
    flight_attrs = []
    duration = hdf.duration
    
    for param_name in process_order:
        if param_name in node_mgr.hdf_keys:
            continue
        
        elif node_mgr.get_attribute(param_name) is not None:
            # add attribute to dictionary of available params
            ###params[param_name] = node_mgr.get_attribute(param_name) #TODO: optimise with only one call to get_attribute
            continue
        
        node_class = node_mgr.derived_nodes[param_name]  #NB raises KeyError if Node is "unknown"
        
        # build ordered dependencies
        deps = []
        node_deps = node_class.get_dependency_names()
        for dep_name in node_deps:
            if dep_name in params:  # already calculated KPV/KTI/Phase
                deps.append(params[dep_name])
            elif node_mgr.get_attribute(dep_name) is not None:
                deps.append(node_mgr.get_attribute(dep_name))
            elif dep_name in node_mgr.hdf_keys:  
                # LFL/Derived parameter
                # all parameters (LFL or other) need get_aligned which is
                # available on DerivedParameterNode
                try:
                    dp = derived_param_from_hdf(hdf.get_param(dep_name,
                                                              valid_only=True))
                except KeyError:
                    # Parameter is invalid.
                    dp = None
                deps.append(dp)
            else:  # dependency not available
                deps.append(None)
        if all([d is None for d in deps]):
            raise RuntimeError("No dependencies available - Nodes cannot "
                               "operate without ANY dependencies available! "
                               "Node: %s" % node_class.__name__)

        # initialise node
        node = node_class()
        logger.info("Processing parameter %s", param_name)
        # Derive the resulting value

        result = node.get_derived(deps)

        if node.node_type is KeyPointValueNode:
            #Q: track node instead of result here??
            params[param_name] = result
            for one_hz in result.get_aligned(P(frequency=1, offset=0)):
                if not (0 <= one_hz.index <= duration):
                    raise IndexError(
                        "KPV '%s' index %.2f is not between 0 and %d" %
                        (one_hz.name, one_hz.index, duration))
                kpv_list.append(one_hz)
        elif node.node_type is KeyTimeInstanceNode:
            params[param_name] = result
            for one_hz in result.get_aligned(P(frequency=1, offset=0)):
                if not (0 <= one_hz.index <= duration):
                    raise IndexError(
                        "KTI '%s' index %.2f is not between 0 and %d" %
                        (one_hz.name, one_hz.index, duration))
                kti_list.append(one_hz)
        elif node.node_type is FlightAttributeNode:
            params[param_name] = result
            try:
                flight_attrs.append(Attribute(result.name, result.value)) # only has one Attribute result
            except:
                logger.warning("Flight Attribute Node '%s' returned empty "
                               "handed.", param_name)
        elif issubclass(node.node_type, SectionNode):
            aligned_section = result.get_aligned(P(frequency=1, offset=0))
            for index, one_hz in enumerate(aligned_section):
                # SectionNodes allow slice starts and stops being None which
                # signifies the beginning and end of the data. To avoid TypeErrors
                # in subsequent derive methods which perform arithmetic on section
                # slice start and stops, replace with 0 or hdf.duration.
                fallback = lambda x, y: x if x is not None else y

                duration = fallback(duration, 0)

                start = fallback(one_hz.slice.start, 0)
                stop = fallback(one_hz.slice.stop, duration)
                start_edge = fallback(one_hz.start_edge, 0)
                stop_edge = fallback(one_hz.stop_edge, duration)

                slice_ = slice(start, stop)
                one_hz = Section(one_hz.name, slice_, start_edge, stop_edge)
                aligned_section[index] = one_hz
                
                if not (0 <= start <= duration and 0 <= stop <= duration + 1):
                    msg = "Section '%s' (%.2f, %.2f) not between 0 and %d"
                    raise IndexError(msg % (one_hz.name, start, stop, duration))
                if not 0 <= start_edge <= duration:
                    msg = "Section '%s' start_edge (%.2f) not between 0 and %d"
                    raise IndexError(msg % (one_hz.name, start_edge, duration))
                if not 0 <= stop_edge <= duration + 1:
                    msg = "Section '%s' stop_edge (%.2f) not between 0 and %d"
                    raise IndexError(msg % (one_hz.name, stop_edge, duration))
                section_list.append(one_hz)
            params[param_name] = aligned_section
        elif issubclass(node.node_type, DerivedParameterNode):
            if duration:
                # check that the right number of results were returned
                # Allow a small tolerance. For example if duration in seconds
                # is 2822, then there will be an array length of  1411 at 0.5Hz and 706
                # at 0.25Hz (rounded upwards). If we combine two 0.25Hz
                # parameters then we will have an array length of 1412.
                expected_length = duration * result.frequency
                if result.array is None:
                    logger.warning("No array set; creating a fully masked array for %s", param_name)
                    array_length = expected_length
                    # Where a parameter is wholly masked, we fill the HDF
                    # file with masked zeros to maintain structure.
                    result.array = \
                        np_ma_masked_zeros_like(np.ma.arange(expected_length))
                else:
                    array_length = len(result.array)
                length_diff = array_length - expected_length
                if length_diff == 0:
                    pass
                elif 0 < length_diff < 5:
                    logger.warning("Cutting excess data for parameter '%s'. "
                                   "Expected length was '%s' while resulting "
                                   "array length was '%s'.", param_name,
                                   expected_length, len(result.array))
                    result.array = result.array[:expected_length]
                else:
                    raise ValueError("Array length mismatch for parameter "
                                     "'%s'. Expected '%s', resulting array "
                                     "length '%s'." % (param_name,
                                                       expected_length,
                                                       array_length))
            
            hdf.set_param(result)
            # Keep hdf_keys up to date.
            node_mgr.hdf_keys.append(param_name)
        elif issubclass(node.node_type, ApproachNode):
            aligned_approach = result.get_aligned(P(frequency=1, offset=0))
            for approach in aligned_approach:
                # Does not allow slice start or stops to be None.
                valid_turnoff = (not approach.turnoff or
                                 (0 <= approach.turnoff <= duration))
                valid_slice = ((0 <= approach.slice.start <= duration) and
                               (0 <= approach.slice.stop <= duration))
                valid_gs_est = (not approach.gs_est or
                                ((0 <= approach.gs_est.start <= duration) and
                                 (0 <= approach.gs_est.stop <= duration)))
                valid_loc_est = (not approach.loc_est or
                                 ((0 <= approach.loc_est.start <= duration) and
                                  (0 <= approach.loc_est.stop <= duration)))
                if not all([valid_turnoff, valid_slice, valid_gs_est,
                            valid_loc_est]):
                    raise ValueError('ApproachItem contains index outside of '
                                     'flight data: %s' % approach)
                approach_list.append(approach)
            params[param_name] = aligned_approach
        else:
            raise NotImplementedError("Unknown Type %s" % node.__class__)
        continue
    return kti_list, kpv_list, section_list, approach_list, flight_attrs


def get_derived_nodes(module_names):
    '''
    Get all nodes into a dictionary.
    '''
    def isclassandsubclass(value, classinfo):
        return isclass(value) and issubclass(value, classinfo)

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
        ##print 'importing', name
        module = __import__(name, globals(), locals(), [''])
        for c in vars(module).values():
            if isclassandsubclass(c, Node) \
                    and c.__module__ != 'analysis_engine.node':
                try:
                    nodes[c.get_name()] = c
                except TypeError:
                    #TODO: Handle the expected error of top level classes
                    # Can't instantiate abstract class DerivedParameterNode
                    # - but don't know how to detect if we're at that level without resorting to 'if c.get_name() in 'derived parameter node',..
                    logger.exception('Failed to import class: %s' % c.get_name())
    return nodes


def process_flight(hdf_path, tail_number, aircraft_info={},
                   start_datetime=datetime.now(), achieved_flight_record={},
                   required_params=[], include_flight_attributes=True):
    '''
    Processes the HDF file (hdf_path) to derive the required_params (Nodes)
    within python modules (settings.NODE_MODULES).
    
    Note: For Flight Data Services, the definitive API is located here:
        "PolarisTaskManagement.test.tasks_mask.process_flight"
        
    :param hdf_path: Path to HDF File
    :type hdf_pat: String
    :param aircraft: Aircraft specific attributes
    :type aircraft: dict   
    :param start_datetime: Datetime of the origin of the data (at index 0)
    :type start_datetime: Datetime
    :param achieved_flight_record: See API Below
    :type achieved_flight_record: Dict
    :param required_params: Parameters to fetch (dependencies will also be evaluated)
    :type required_params: List of Strings
    :param include_flight_attributes: Whether to include all flight attributes
    :type include_flight_attributes: Boolean

    :returns: See below:
    :rtype: Dict
    
    
    Sample aircraft_info
    --------------------
    {
        'Tail Number':  # Aircraft Registration
        'Identifier':  # Aircraft Ident
        'Manufacturer': # e.g. Boeing
        'Manufacturer Serial Number': #MSN
        'Model': # e.g. 737-808-ER
        'Series': # e.g. 737-800
        'Family': # e.g. 737
        'Frame': # e.g. 737-3C
        'Main Gear To Altitude Radio': # Distance in metres
        'Wing Span': # Distance in metres
    }
    
    Sample achieved_flight_record
    -----------------------------
    {
        # Simple values first, e.g. string, int, float, etc.
        'AFR Flight ID': # e.g. 1
        'AFR Flight Number': # e.g. 1234
        'AFR Type': # 'POSITIONING'
        'AFR Off Blocks Datetime': # datetime(2015,01,01,13,00)
        'AFR Takeoff Datetime': # datetime(2015,01,01,13,15)
        'AFR Takeoff Pilot': # 'Joe Bloggs'
        'AFR Takeoff Gross Weight': # weight in kg
        'AFR Takeoff Fuel': # fuel in kg
        'AFR Landing Datetime': # datetime(2015,01,01,18,45)
        'AFR Landing Pilot': # 'Joe Bloggs'
        'AFR Landing Gross Weight': # weight in kg
        'AFR Landing Fuel': # weight in kg
        'AFR On Blocks Datetime': # datetime(2015,01,01,19,00)
        'AFR V2': # V2 used at takeoff in kts
        'AFR Vapp': # Vapp used in kts
        'AFR Vref': # Vref used in kts
        # More complex data that needs to be looked up next:
        'AFR Takeoff Airport':  {
            'id': 4904, # unique id
            'name': 'Athens Intl Airport Elefterios Venizel',
            'code': {'iata': 'ATH', 'icao': 'LGAV'},
            'latitude': 37.9364,
            'longitude': 23.9445,
            'location': {'city': u'Athens', 'country': u'Greece'},
            'elevation': 266, # ft
            'magnetic_variation': 'E003186 0106',
            }
           },
        'AFR Landing Aiport': {
            'id': 1, # unique id
            'name': 'Athens Intl Airport Elefterios Venizel',
            'code': {'iata': 'ATH', 'icao': 'LGAV'},
            'latitude': 37.9364,
            'longitude': 23.9445,
            'location': {'city': u'Athens', 'country': u'Greece'},
            'elevation': 266, # ft
            'magnetic_variation': 'E003186 0106',
            }
           },
        'AFR Destination Airport': None, # if not required, or exclude this key
        'AFR Takeoff Runway': {
            'id': 1,
            'identifier': '21L',
            'magnetic_heading': 212.6,
            'strip': {
                'id': 1, 
                'length': 13123, 
                'surface': 'ASP', 
                'width': 147},
            'start': {
                'elevation': 308, 
                'latitude': 37.952425, 
                'longitude': 23.970422},
            'end': {
                'elevation': 279, 
                'latitude': 37.923511, 
                'longitude': 23.943261},
            'glideslope': {
                'angle': 3.0,
                'elevation': 282,
                'latitude': 37.9473,
                'longitude': 23.9676,
                'threshold_distance': 999},
            'localizer': {
                'beam_width': 4.5,
                'elevation': 256,
                'frequency': 111100,
                'heading': 213,
                'latitude': 37.919281,
                'longitude': 23.939294},
            },
        'AFR Landing Runway': {
            'id': 1,
            'identifier': '21L',
            'magnetic_heading': 212.6,
            'strip': {
                'id': 1, 
                'length': 13123, 
                'surface': 'ASP', 
                'width': 147},
            'start': {
                'elevation': 308, 
                'latitude': 37.952425, 
                'longitude': 23.970422},
            'end': {
                'elevation': 279, 
                'latitude': 37.923511, 
                'longitude': 23.943261},
            'glideslope': {
                'angle': 3.0,
                'elevation': 282,
                'latitude': 37.9473,
                'longitude': 23.9676,
                'threshold_distance': 999},
            'localizer': {
                'beam_width': 4.5,
                'elevation': 256,
                'frequency': 111100,
                'heading': 213,
                'latitude': 37.919281,
                'longitude': 23.939294},
            },
    }
    
    Sample Return
    -------------
    {
        'flight':[Attribute('name value')]  # sample: [Attribute('Takeoff Airport', {'id':1234, 'name':'Int. Airport'}, Attribute('Approaches', [4567,7890]), ...], 
        'kti':[GeoKeyTimeInstance('index name latitude longitude')] if lat/long available else [KeyTimeInstance('index name')]
        'kpv':[KeyPointValue('index value name slice')]
    }
    
    '''
    logger.info("Processing: %s", hdf_path)
    
    
    if aircraft_info:
        # Aircraft info has already been provided.
        logger.info(
            "Using aircraft_info dictionary passed into process_flight '%s'." %
            aircraft_info)
    else:
        # Fetch aircraft info through the API.
        api_handler = get_api_handler(settings.API_HANDLER)
        
        try:
            aircraft_info = api_handler.get_aircraft(tail_number)
        except APIError:
            if settings.API_HANDLER == settings.LOCAL_API_HANDLER:
                raise
            # Fallback to the local API handler.
            logger.info(
                "Aircraft '%s' could not be found with '%s' API handler. "
                "Falling back to '%s'.", tail_number, settings.API_HANDLER,
                settings.LOCAL_API_HANDLER)
            api_handler = get_api_handler(settings.LOCAL_API_HANDLER)
            aircraft_info = api_handler.get_aircraft(tail_number)
        
        logger.info("Using aircraft_info provided by '%s' '%s'.",
                    api_handler.__class__.__name__, aircraft_info)
    
    aircraft_info['Tail Number'] = tail_number
    
    # go through modules to get derived nodes
    derived_nodes = get_derived_nodes(settings.NODE_MODULES)
    required_params = \
        list(set(required_params).intersection(set(derived_nodes)))
    # if required_params isn't set, try using ALL derived_nodes!
    if not required_params:
        logger.info("No required_params declared, using all derived nodes")
        required_params = derived_nodes.keys()
    
    # include all flight attributes as required
    if include_flight_attributes:
        required_params = list(set(
            required_params + get_derived_nodes(
                ['analysis_engine.flight_attribute']).keys()))
        
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        if hooks.PRE_FLIGHT_ANALYSIS:
            logger.info("Performing PRE_FLIGHT_ANALYSIS actions: %s", 
                         hooks.PRE_FLIGHT_ANALYSIS.func_name)
            hooks.PRE_FLIGHT_ANALYSIS(hdf, aircraft_info)
        else:
            logger.info("No PRE_FLIGHT_ANALYSIS actions to perform")
        
        # Track nodes. Assume that all params in HDF are from LFL(!)
        node_mgr = NodeManager(
            start_datetime, hdf.duration, hdf.valid_param_names(),
            required_params, derived_nodes, aircraft_info,
            achieved_flight_record)
        # calculate dependency tree
        process_order, gr_st = dependency_order(node_mgr, draw=False)
        if settings.CACHE_PARAMETER_MIN_USAGE:
            # find params used more than
            for node in gr_st.nodes():
                if node in node_mgr.derived_nodes:
                    # this includes KPV/KTIs but they'll be ignored by HDF
                    qty = len(gr_st.predecessors(node))
                    if qty > settings.CACHE_PARAMETER_MIN_USAGE:
                        hdf.cache_param_list.append(node)
            logging.info("HDF set to cache parameters: %s",
                         hdf.cache_param_list)
        
        # derive parameters
        kti_list, kpv_list, section_list, approach_list, flight_attrs = \
            derive_parameters(hdf, node_mgr, process_order)
             
        # geo locate KTIs
        kti_list = geo_locate(hdf, kti_list)
        kti_list = _timestamp(start_datetime, kti_list)
        
        # geo locate KPVs
        kpv_list = geo_locate(hdf, kpv_list)
        kpv_list = _timestamp(start_datetime, kpv_list)
        
        # Store version of FlightDataAnalyser
        hdf.analysis_version = __version__
        # Store dependency tree
        hdf.dependency_tree = json_graph.dumps(gr_st)
        # Store aircraft info
        hdf.set_attr('aircraft_info', aircraft_info)
        hdf.set_attr('achieved_flight_record', achieved_flight_record)
        
    return {
        'flight' : flight_attrs, 
        'kti' : kti_list, 
        'kpv' : kpv_list,
        'approach': approach_list,
        'phases' : section_list,
    }


def main():
    print 'FlightDataAnalyzer (c) Copyright 2013 Flight Data Services, Ltd.'
    print '  - Powered by POLARIS'
    print '  - http://www.flightdatacommunity.com'
    print ''
    from analysis_engine.plot_flight import csv_flight_details, track_to_kml
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    parser = argparse.ArgumentParser(description="Process a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    help = 'Write CSV of processing results. Set "False" to disable.'
    parser.add_argument('-csv', dest='write_csv', type=str, default='True', 
                        help=help)
    help = 'Write KML of flight track. Set "False" to disable.'
    parser.add_argument('-kml', dest='write_kml', type=str, default='True', 
                        help=help)
    parser.add_argument('-r', '--required', type=str, nargs='+', dest='required_params',
                       help='Required parameters.')

    parser.add_argument('-tail', dest='tail_number',
                        default='G-FDSL', # as per flightdatacommunity file
                        help='Aircraft tail number.')
    
    # Aircraft info
    parser.add_argument('-aircraft-family', dest='aircraft_family', type=str,
                        help='Aircraft family.')
    parser.add_argument('-aircraft-series', dest='aircraft_series', type=str,
                        help='Aircraft series.')
    parser.add_argument('-aircraft-model', dest='aircraft_model', type=str,
                        help='Aircraft model.')    
    parser.add_argument('-aircraft-manufacturer', dest='aircraft_manufacturer',
                        type=str, help='Aircraft manufacturer.')
    help = 'Whether or not the aircraft records precise positioning parameters.'
    parser.add_argument('-precise-positioning', dest='precise_positioning',
                        type=str, help=help)
    parser.add_argument('-frame', dest='frame', type=str, 
                        help='Data frame name.')
    parser.add_argument('-frame-qualifier', dest='frame_qualifier', type=str, 
                        help='Data frame qualifier.')
    parser.add_argument('-identifier', dest='identifier', type=str,
                        help='Aircraft identifier.')
    parser.add_argument('-manufacturer-serial-number',
                        dest='manufacturer_serial_number', type=str,
                        help="Manufacturer's serial number of the aircraft.")
    parser.add_argument('-qar-serial-number', dest='qar_serial_number',
                        type=str, help='QAR serial number.')
    help = 'Main gear to radio altimeter antenna in metres.'
    parser.add_argument('-main-gear-to-radio-altimeter-antenna',
                        dest='main_gear_to_alt_rad',
                        type=float, help=help)
    help = 'Main gear to lowest point of tail in metres.'
    parser.add_argument('-main-gear-to-lowest-point-of-tail',
                        dest='main_gear_to_tail',
                        type=float, help=help)
    help = 'Ground to lowest point of tail in metres.'
    parser.add_argument('-ground-to-lowest-point-of-tail',
                        dest='ground_to_tail',
                        type=float, help=help)
    parser.add_argument('-engine-count', dest='engine_count',
                        type=int, help='Number of engines.')
    parser.add_argument('-engine-manufacturer', dest='engine_manufacturer',
                        type=str, help='Engine manufacturer.')
    parser.add_argument('-engine-series', dest='engine_series', type=str,
                        help='Engine series.')
    parser.add_argument('-engine-type', dest='engine_type', type=str,
                        help='Engine type.')
    
    args = parser.parse_args()
    aircraft_info = {}
    if args.aircraft_model:
        aircraft_info['Model'] = args.aircraft_model
    if args.aircraft_family:
        aircraft_info['Series'] = args.aircraft_series
    if args.aircraft_manufacturer:
        aircraft_info['Manufacturer'] = args.aircraft_manufacturer
    if args.precise_positioning:
        aircraft_info['Precise Positioning'] = args.precise_positioning
    if args.frame:
        aircraft_info['Frame'] = args.frame
    if args.frame_qualifier:
        aircraft_info['Frame Qualifier'] = args.frame_qualifier
    if args.identifier:
        aircraft_info['Identifier'] = args.identifier
    if args.manufacturer_serial_number:
        aircraft_info['Manufacturer Serial Number'] = args.manufacturer_serial_number
    if args.qar_serial_number:
        aircraft_info['QAR Serial Number'] = args.qar_serial_number
    if args.main_gear_to_alt_rad:
        aircraft_info['Main Gear To Radio Altimeter Antenna'] = args.main_gear_to_alt_rad
    if args.main_gear_to_tail:
        aircraft_info['Main Gear To Lowest Point Of Tail'] = args.main_gear_to_tail
    if args.ground_to_tail:
        aircraft_info['Ground To Lowest Point Of Tail'] = args.ground_to_tail
    if args.engine_count:
        aircraft_info['Engine Count'] = args.engine_count
    if args.engine_series:
        aircraft_info['Engine Series'] = args.engine_series
    if args.engine_manufacturer:
        aircraft_info['Engine Manufacturer'] = args.engine_manufacturer
    if args.engine_series:
        aircraft_info['Engine Series'] = args.engine_series
    if args.engine_type:
        aircraft_info['Engine Type'] = args.engine_type
    
    # Derive parameters to new HDF
    hdf_copy = copy_file(args.file, postfix='_process')
    res = process_flight(
        hdf_copy, args.tail_number, aircraft_info=aircraft_info,
        required_params=args.required_params or [])
    logger.info("Derived parameters stored in hdf: %s", hdf_copy)
    # Write CSV file
    if args.write_csv.lower() == 'true':
        csv_dest = os.path.splitext(hdf_copy)[0] + '.csv'
        csv_flight_details(hdf_copy, res['kti'], res['kpv'], res['phases'], 
                           dest_path=csv_dest)
        logger.info("KPV, KTI and Phases writen to csv: %s", csv_dest)
    # Write KML file
    if args.write_kml.lower() == 'true':
        kml_dest = os.path.splitext(hdf_copy)[0] + '.kml'
        track_to_kml(hdf_copy, res['kti'], res['kpv'], res['approach'], 
                     plot_altitude='Altitude QNH', dest_path=kml_dest)
        logger.info("Flight Track with attributes writen to kml: %s", kml_dest)
    
    # - END -


if __name__ == '__main__':
    main()
