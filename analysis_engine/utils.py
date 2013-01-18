import argparse
import os

from datetime import datetime

from hdfaccess.file import hdf_file
from hdfaccess.utils import strip_hdf

from analysis_engine.dependency_graph import dependencies3, graph_nodes
from analysis_engine.node import NodeManager
from analysis_engine.process_flight import get_derived_nodes
from analysis_engine.settings import NODE_MODULES


def derived_trimmer(hdf_path, node_names, dest):
    '''
    Trims an HDF file of parameters which are not dependencies of nodes in
    node_names.
    
    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param node_names: A list of Node names which are required.
    :type node_names: list of str
    :param dest: destination path for trimmed output file
    :type dest: str
    :return: parameters in stripped hdf file
    :rtype: [str]
    '''
    params = []
    with hdf_file(hdf_path) as hdf:
        derived_nodes = get_derived_nodes(NODE_MODULES)
        node_mgr = NodeManager(datetime.now(), hdf.duration, hdf.keys(), [],
                               derived_nodes, {}, {})
        _graph = graph_nodes(node_mgr)
        for node_name in node_names:
            deps = dependencies3(_graph, node_name, node_mgr)
            params.extend(filter(lambda d: d in node_mgr.hdf_keys, deps))
    return strip_hdf(hdf_path, params, dest) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command',
                                      description="Utility command, currently "
                                      "only 'trimmer' is supported",
                                      help='Additional help')
    trimmer_parser = subparser.add_parser('trimmer')
    trimmer_parser.add_argument('input_file_path', help='Input hdf filename.')  
    trimmer_parser.add_argument('output_file_path', help='Output hdf filename.')
    trimmer_parser.add_argument('nodes', nargs='+',
                                help='Keep dependencies of the specified nodes '
                                'within the output hdf file. All other '
                                'parameters will be stripped.')
    
    args = parser.parse_args()
    if args.command == 'trimmer':
        if not os.path.isfile(args.input_file_path):
            parser.error("Input file path '%s' does not exist." %
                         args.input_file_path)
        if os.path.exists(args.output_file_path):
            parser.error("Output file path '%s' already exists." %
                         args.output_file_path)
        output_parameters = derived_trimmer(args.input_file_path, args.nodes,
                                            args.output_file_path)
        if output_parameters:
            print 'The following parameters are in the output hdf file:'
            for name in output_parameters:
                print ' * %s' % name
        else:
            print 'No matching parameters were found in the hdf file.'            
    else:
        parser.error("'%s' is not a known command." % args.command)

def _get_names(module_locations, fetch_names=True, fetch_dependencies=False):
    '''
    Get the names of Nodes and dependencies.
    
    :param module_locations: list of locations to fetch modules from
    :type module_locations: list of strings
    :param fetch_names: Return name of parameters etc. created by class
    :type fetch_names: Bool
    :param fetch_dependencies: Return names of the arguments in derive methods
    :type fetch_dependencies: Bool
    '''
    nodes = get_derived_nodes(module_locations)
    names = []
    for node in nodes.values():
        if fetch_names:
            if hasattr(node, 'names'):
                # FormattedNameNode (KPV/KTI) can have many names
                names.extend(node.names())
            else:
                names.append(node.get_name())
        if fetch_dependencies:
            names.extend(node.get_dependency_names())
    return sorted(names)
    
    
def list_parameters():
    '''
    Return an ordered list of 
    '''
    # exclude all KPV and KTIs and Phases?
    exclude = _get_names(['analysis_engine.key_point_values',
                         'analysis_engine.key_time_instances',
                         'analysis_engine.flight_attribute',
                         'analysis_engine.flight_phase'],
                        fetch_names=True, fetch_dependencies=False)
    # remove names of kpvs etc from parameters
    parameters = set(list_everything()) - set(exclude)
    return sorted(parameters)


def list_derived_parameters():
    '''
    Return an ordered list of the Derived Parameters which have been coded.
    '''
    return _get_names(['analysis_engine.derived_parameters'])


def list_lfl_parameter_dependencies():
    '''
    Return a list of the Parameters without Derived ones, therefore should be
    mostly LFL parameters!
    
    Note: a few Attributes will be in here too!
    '''
    parameters = set(list_parameters()) - set(list_derived_parameters())
    return sorted(parameters)


def list_everything():
    '''
    Return an ordered list of all parameters both derived and required.
    '''
    return _get_names(NODE_MODULES, True, True)


def list_kpvs():
    '''
    Return an ordered list of the Key Point Values which have been coded.
    '''
    return _get_names(['analysis_engine.key_point_values'])


def list_ktis():
    '''
    Return an ordered list of the Key Time Instances which have been coded.
    '''
    return _get_names(['analysis_engine.key_time_instances'])


def list_flight_attributes():
    '''
    Return an ordered list of the Flight Attributes which have been coded.
    '''
    return _get_names(['analysis_engine.flight_attribute'])


def list_flight_phases():
    '''
    Return an ordered list of the Flight Phases which have been coded.
    '''
    return _get_names(['analysis_engine.flight_phase'])