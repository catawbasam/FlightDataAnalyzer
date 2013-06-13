import unittest

from mock import Mock, patch

from analysis_engine.utils import (
    derived_trimmer,
    list_derived_parameters,
    list_everything,
    list_flight_attributes,
    list_flight_phases,
    list_kpvs,
    list_ktis,
    list_lfl_parameter_dependencies,
    list_parameters,
    )

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
            def valid_param_names(self):
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
        strip_hdf.assert_called_once_with(
            in_path, ['IVV', 'DME'], out_path)
        self.assertEqual(dest, strip_hdf.return_value)



class TestGetNames(unittest.TestCase):
    def test_list_parameters(self):
        params = list_parameters()
        self.assertIn('Airspeed', params)  # LFL
        self.assertIn('Altitude AAL', params)  # Derived Node
        # ensure dependencies of other modules (not in derived) are included
        self.assertIn('TAWS General', params)
        self.assertIn('Key Satcom (1)', params)
        # ensure KPV and KTIs are excluded
        self.assertNotIn('AOA With Flap 15 Max', params)  # KPV
        self.assertNotIn('Landing Turn Off Runway', params)  # KTI
        self.assertNotIn('FDR Takeoff Runway', params)  # Attribute


    def test_list_derived_parameters(self):
        params = list_derived_parameters()
        self.assertIn('Altitude AAL', params)  # Derived Node
        self.assertNotIn('Airspeed', params)  # LFL

    def test_list_kpvs(self):
        kpvs = list_kpvs()
        self.assertIn('Airspeed Max', kpvs)
        # check the formatted name is there
        self.assertIn('AOA With Flap 15 Max', kpvs)
        # and that the actual node name is not
        self.assertNotIn('AOA With Flap Max', kpvs)
        # check dependencies are not included
        self.assertNotIn('Airspeed', kpvs)
        self.assertNotIn('Landing Turn Off Runway', kpvs)

    def test_list_ktis(self):
        ktis = list_ktis()
        self.assertIn('Landing Turn Off Runway', ktis)
        self.assertNotIn('Airspeed', ktis)
        self.assertNotIn('AOA With Flap 15 Max', ktis)
        self.assertNotIn('FDR Takeoff Runway', ktis)

    def test_list_lfl_parameters(self):
        params = list_lfl_parameter_dependencies()
        self.assertIn('Airspeed', params)  # LFL
        self.assertNotIn('Altitude AAL', params)  # Derived Node

    def test_list_everything(self):
        params = list_everything()
        self.assertIn('Airspeed', params)  # LFL
        self.assertIn('Altitude AAL', params)  # Derived Node
        self.assertIn('TAWS General', params)
        self.assertIn('Key Satcom (1)', params)
        self.assertIn('AOA With Flap 15 Max', params)  # KPV
        self.assertIn('Landing Turn Off Runway', params)  # KTI
        self.assertIn('FDR Takeoff Runway', params)  # Attribute

    def test_list_flight_attributes(self):
        atts = list_flight_attributes()
        self.assertIn('FDR Landing Runway', atts)

    def test_list_flight_phases(self):
        phases = list_flight_phases()
        self.assertIn('Bounced Landing', phases)


