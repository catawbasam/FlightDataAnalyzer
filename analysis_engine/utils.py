from datetime import datetime

import h5py

from utilities.dict_helpers import dict_filter
from hdfaccess.utils import strip_hdf

from analysis_engine.node import NodeManager
from analysis_engine.process_flight import dependency_order, get_derived_nodes
from analysis_engine.settings import NODE_MODULES


def trimmer(hdf_path, node_names, dest):
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
        process_order = dependency_order(node_mgr, draw=False)
    strip_hdf(hdf_path, process_order, dest)
    return dest
        

if __name__ == '__main__':
    strip_hdf('/tmp/tmp_strip_hdf.hdf5', ['Altitude When Descending', 'Heading Continuous', 'Rate Of Climb For Flight Phases'],
              '/tmp/tmp_strip_hdf_out.hdf5')

