import sys
from datetime import datetime
import argparse

import h5py

from utilities.dict_helpers import dict_filter
from hdfaccess.utils import strip_hdf

from analysis_engine.node import NodeManager
from analysis_engine.process_flight import dependency_order, get_derived_nodes
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
    :return: path to output hdf file containing specified segment.
    :rtype: str
    '''
    with h5py.File(hdf_path, 'r') as hdf_file:
        derived_nodes = get_derived_nodes(NODE_MODULES)
        filtered_nodes = dict_filter(derived_nodes, keep=node_names)
        lfl_names = hdf_file['series'].keys()
        node_mgr = NodeManager(datetime.now(), lfl_names, filtered_nodes,
                               derived_nodes, {}, {})
        process_order, _ = dependency_order(node_mgr, draw=False)
    strip_hdf(hdf_path, process_order, dest)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command',
                                      description="Utility command, currently "
                                      "only 'trimmer' is supported",
                                      help='Additional help')
    trimmer_parser = subparser.add_parser('trimmer')
    trimmer_parser.add_argument('input_file_path', help='Input hdf filename.')  
    trimmer_parser.add_argument('output_file_path', help='Output hdf filename.')
    trimmer_parser.add_argument('nodes', nargs='+')    
    
    
    
    command_parser.add_argument('command',
                                help="utils command to run, options: 'trimmer'.")
    args = command_parser.parse_args()
    if args.command == 'trimmer':
        if not os.path.isfile(args.input_file_path):
            parser.error("Input file path '%s' does not exist." %
                         args.input_file_path)
        if os.path.exists(args.output_file_path):
            parser.error("Output file path '%s' already exists." %
                         args.output_file_path)
        derived_trimmer(trimmer_args.input_file_path, trimmer_args.nodes,
                        trimmer_args.output_file_path)
    else:
        trimmer_parser.error("'%s' is not a known command." % args.command)

