import unittest

from mock import Mock, patch

from analysis_engine.utils import derived_trimmer

class TestTrimmer(unittest.TestCase):
    '''
    TODO: Functional test with fewer mocks.
    '''
    @patch('analysis_engine.utils.hdf_file')
    @patch('analysis_engine.utils.datetime')
    @patch('analysis_engine.utils.get_derived_nodes')
    @patch('analysis_engine.utils.strip_hdf')
    @patch('analysis_engine.utils.NODE_MODULES')
    def test_derived_trimmer_mocked(self, node_modules, strip_hdf,
                                    get_derived_nodes, datetime, file_patched):
        '''
        Mocks the majority of inputs and outputs. 
        '''
        datetime.now = Mock()
        hdf_contents = {'IVV': Mock(), 'DME': Mock(), 'WOW': Mock()}
        class hdf_file(dict):
            duration = 10
            def keys(self):
                return hdf_contents.keys()
            def __enter__(self, *args, **kwargs):
                return hdf_file(hdf_contents)
            def __exit__(self, *args, **kwargs):
                return False
        file_patched.return_value = hdf_file()
        strip_hdf.return_value = ['IVV', 'DME']
        derived_nodes = {'IVV': Mock(), 'DME': Mock(), 'WOW': Mock()}
        get_derived_nodes.return_value = derived_nodes
        in_path = 'in.hdf5'
        out_path = 'out.hdf5'
        dest = derived_trimmer(in_path, ['IVV', 'DME'], out_path)
        file_patched.assert_called_once_with(in_path)
        get_derived_nodes.assert_called_once_with(node_modules)
        filtered_nodes = {'IVV': derived_nodes['IVV'],
                          'DME': derived_nodes['DME']}
        strip_hdf.assert_called_once_with(in_path,
                                          ['IVV', 'DME'],
                                          out_path)
        self.assertEqual(dest, strip_hdf.return_value)
        
        
        
        
        
        
