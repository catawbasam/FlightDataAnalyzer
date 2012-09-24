import unittest

from mock import Mock, patch

from analysis_engine.utils import derived_trimmer

class TestTrimmer(unittest.TestCase):
    '''
    TODO: Functional test with fewer mocks.
    '''
    @patch('analysis_engine.utils.h5py.File')
    @patch('analysis_engine.utils.datetime')
    @patch('analysis_engine.utils.get_derived_nodes')
    @patch('analysis_engine.utils.dependency_order')
    @patch('analysis_engine.utils.strip_hdf')
    @patch('analysis_engine.utils.NodeManager')
    @patch('analysis_engine.utils.NODE_MODULES')
    def test_derived_trimmer_mocked(self, node_modules, node_mgr, strip_hdf,
                                    dependency_order, get_derived_nodes,
                                    datetime, file_patched):
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
        dependency_order.return_value = (Mock(), Mock())
        dest = derived_trimmer(in_path, ['IVV', 'DME'], out_path)
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
                                          dependency_order.return_value[0],
                                          out_path)
        self.assertEqual(dest, out_path)
        
        
        
        
        
        
