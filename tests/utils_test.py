import unittest

import h5py

from mock import Mock, patch

from analysis.utils import trimmer

class TestTrimmer(unittest.TestCase):
    '''
    TODO: Functional test with fewer mocks.
    '''
    @patch('analysis.utils.h5py.File')
    @patch('analysis.utils.datetime')
    @patch('analysis.utils.get_derived_nodes')
    @patch('analysis.utils.dependency_order')
    @patch('analysis.utils.strip_hdf')
    @patch('analysis.utils.NodeManager')
    @patch('analysis.utils.NODE_MODULES')
    def test_trimmer_mocked(self, node_modules, node_mgr, strip_hdf,
                            dependency_order, get_derived_nodes, datetime,
                            file_patched):
        '''
        Mocks the majority of inputs and outputs. 
        '''
        datetime.now = Mock()
        hdf_contents = {'series': {'IVV': Mock(), 'DME': Mock(), 'WOW': Mock()}}
        class hdf_file(dict):
            def __enter__(self, *args, **kwargs):
                return hdf_contents
            def __exit__(self, *args, **kwargs):
                return False
        file_patched.return_value = hdf_file()
        derived_nodes = {'IVV': Mock(), 'DME': Mock(), 'WOW': Mock()}
        get_derived_nodes.return_value = derived_nodes
        in_path = 'in.hdf5'
        out_path = 'out.hdf5'
        dest = trimmer(in_path, ['IVV', 'DME'], out_path)
        file_patched.assert_called_once_with(in_path, 'r')
        get_derived_nodes.assert_called_once_with(node_modules)
        filtered_nodes = {'IVV': derived_nodes['IVV'],
                          'DME': derived_nodes['DME']}
        node_mgr.assert_called_once_with(datetime.now.return_value,
                                         hdf_contents['series'].keys(),
                                         filtered_nodes, derived_nodes,
                                         {}, {})
        dependency_order.assert_called_once_with(node_mgr.return_value,
                                                 draw=False)
        strip_hdf.assert_called_once_with(in_path,
                                          dependency_order.return_value,
                                          out_path)
        self.assertEqual(dest, out_path)
        
        
        
        
        
        
