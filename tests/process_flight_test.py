################################################################################
# Process Flight Tests
################################################################################


'''
Tests for processing flights using the POLARIS analysis engine.
'''


################################################################################
# Imports


import mock
import os
import shutil
import sys
import unittest

from datetime import datetime, timedelta
from timeit import Timer

from analysis_engine import hooks, ___version___
from analysis_engine.plot_flight import (
    csv_flight_details,
    track_to_kml,
)
from analysis_engine.process_flight import (
    get_derived_nodes,
    process_flight,
)

from test_params import PROCESS_PARAMETERS

debug = sys.gettrace() is not None
if debug:
    # Only import if we're going to use this as it's slow!
    from analysis_engine.plot_flight import plot_flight


################################################################################
# Test Cases


class TestProcessFlight(unittest.TestCase):
    '''
    '''

    ############################################################################
    # Test Set Up: Creates dictionaries of airport and runway data:

    def setUp(self):
        '''
        '''
        self.airports = {
            'BGO': {'magnetic_variation': 'W001185 0106', 'code': {'icao': 'ENBR', 'iata': 'BGO'}, 'name': 'Bergen Lufthavn Flesland', 'longitude': 5.21814, 'location': {'city': 'Bergen', 'country': 'Norway'}, 'latitude': 60.2934, 'id': 2455},
            'DUB': {'magnetic_variation': 'W005068 0106', 'code': {'icao': 'EIDW', 'iata': 'DUB'}, 'name': 'Dublin', 'longitude': -6.27007, 'location': {'city': 'Dublin', 'country': 'Ireland'}, 'latitude': 53.4213, 'id': 2429},
            'KRS': {'magnetic_variation': 'E000091 0106', 'code': {'icao': 'ENCN', 'iata': 'KRS'}, 'name': 'Kristiansand Lufthavn Kjevik', 'longitude': 8.08537, 'location': {'city': 'Kjevik', 'country': 'Norway'}, 'latitude': 58.2042, 'id': 2456},
            'ORY': {'magnetic_variation': 'W001038 0106', 'code': {'icao': 'LFPO', 'iata': 'ORY'}, 'name': 'Paris Orly', 'longitude': 2.35944, 'location': {'city': 'Paris/Orly', 'country': 'France'}, 'latitude': 48.7253, 'id': 3031},
            'OSL': {'magnetic_variation': 'E001226 0106', 'code': {'icao': 'ENGM', 'iata': 'OSL'}, 'name': 'Oslo Gardermoen', 'longitude': 11.1004, 'location': {'city': 'Oslo', 'country': 'Norway'}, 'latitude': 60.1939, 'id': 2461},
            'TRD': {'magnetic_variation': 'E001220 0706', 'code': {'icao': 'ENVA', 'iata': 'TRD'}, 'name': 'Vaernes', 'longitude': 10.9399, 'location': {'city': 'Trondheim', 'country': 'Norway'}, 'latitude': 63.4576, 'id': 2472},
            'VNO': {'magnetic_variation': 'E005552 0106', 'code': {'icao': 'EYVI', 'iata': 'VNO'}, 'name': 'Vilnius Intl', 'longitude': 25.2858, 'location': {'country': 'Lithuania'}, 'latitude': 54.6341, 'id': 2518},
        }

        self.runways = {
            'BGO': {
                '17': {'items': [{'end': {'latitude': 60.280151, 'longitude': 5.222579}, 'localizer': {'latitude': 60.2789, 'beam_width': 4.5, 'frequency': 109900.00, 'heading': 173, 'longitude': 5.223}, 'glideslope': {'latitude': 60.300981, 'angle': 3.1, 'longitude': 5.214092, 'threshold_distance': 1161}, 'start': {'latitude': 60.30662494, 'longitude': 5.21370074}, 'magnetic_heading': 170.0, 'strip': {'width': 147, 'length': 9810, 'id': 4097, 'surface': 'ASP'}, 'identifier': '17', 'id': 8193}], 'ident': '17'},
                '35': {'items': [{'end': {'latitude': 60.30662494, 'longitude': 5.21370074}, 'localizer': {'latitude': 60.307575, 'beam_width': 4.5, 'frequency': 110500.00, 'heading': 353, 'longitude': 5.213367}, 'glideslope': {'latitude': 60.285489, 'angle': 3.0, 'longitude': 5.219394, 'threshold_distance': 1133}, 'start': {'latitude': 60.280151, 'longitude': 5.222579}, 'magnetic_heading': 350.0, 'strip': {'width': 147, 'length': 9810, 'id': 4097, 'surface': 'ASP'}, 'identifier': '35', 'id': 8194}], 'ident': '35'},
            },
            'DUB': {
                '10': {'items': [{'end': {'latitude': 53.42026102, 'longitude': -6.25057848}, 'localizer': {'latitude': 53.420025, 'beam_width': 4.5, 'frequency': 108900.00, 'heading': 101, 'longitude': -6.246317}, 'glideslope': {'latitude': 53.420972, 'angle': 3.0, 'longitude': -6.284858, 'threshold_distance': 1182}, 'start': {'latitude': 53.42243102, 'longitude': -6.29007448}, 'magnetic_heading': 100.0, 'strip': {'width': 147, 'length': 8652, 'id': 2009, 'surface': 'CON'}, 'identifier': '10', 'id': 4017}], 'ident': '10'},
                '16': {'items': [{'end': {'latitude': 53.41990602, 'longitude': -6.24959343}, 'localizer': {'latitude': 53.418264, 'beam_width': 4.5, 'frequency': 111500.00, 'heading': 162, 'longitude': -6.248406}, 'glideslope': {'latitude': 53.434078, 'angle': 3.0, 'longitude': -6.262003, 'threshold_distance': 973}, 'start': {'latitude': 53.43699002, 'longitude': -6.26197743}, 'magnetic_heading': 160.0, 'strip': {'width': 200, 'length': 6798, 'id': 2011, 'surface': 'ASP'}, 'identifier': '16', 'id': 4021}], 'ident': '16'},
                '28': {'items': [{'end': {'latitude': 53.42243102, 'longitude': -6.29007448}, 'localizer': {'latitude': 53.422664, 'beam_width': 4.5, 'frequency': 108900.00, 'heading': 281, 'longitude': -6.294333}, 'glideslope': {'latitude': 53.419339, 'angle': 3.0, 'longitude': -6.255117, 'threshold_distance': 955}, 'start': {'latitude': 53.42026102, 'longitude': -6.25057848}, 'magnetic_heading': 280.0, 'strip': {'width': 147, 'length': 8652, 'id': 2009, 'surface': 'CON'}, 'identifier': '28', 'id': 4018}], 'ident': '28'},
                '34': {'items': [{'end': {'latitude': 53.43699002, 'longitude': -6.26197743}, 'start': {'latitude': 53.41990602, 'longitude': -6.24959343}, 'magnetic_heading': 340.0, 'strip': {'width': 200, 'length': 6798, 'id': 2011, 'surface': 'ASP'}, 'identifier': '34', 'id': 4022}], 'ident': '34'},
            },
            'KRS': {
                '04': {'items': [{'end': {'latitude': 58.211678, 'longitude': 8.095269}, 'localizer': {'latitude': 58.212397, 'beam_width': 4.5, 'frequency': 110300.00, 'heading': 36, 'longitude': 8.096228}, 'glideslope': {'latitude': 58.198664, 'angle': 3.4, 'longitude': 8.080164, 'threshold_distance': 720}, 'start': {'latitude': 58.196703, 'longitude': 8.075406}, 'magnetic_heading': 40.0, 'strip': {'width': 147, 'length': 6660, 'id': 4064, 'surface': 'ASP'}, 'identifier': '04', 'id': 8127}], 'ident': '04'},
                '22': {'items': [{'end': {'latitude': 58.196703, 'longitude': 8.075406}, 'localizer': {'latitude': 58.196164, 'beam_width': 4.5, 'frequency': 110900.00, 'heading': 216, 'longitude': 8.074692}, 'glideslope': {'latitude': 58.208922, 'angle': 3.6, 'longitude': 8.093275, 'threshold_distance': 422}, 'start': {'latitude': 58.211678, 'longitude': 8.095269}, 'magnetic_heading': 220.0, 'strip': {'width': 147, 'length': 6660, 'id': 4064, 'surface': 'ASP'}, 'identifier': '22', 'id': 8128}], 'ident': '22'},
            },
            'ORY': {
                '02': {'items': [{'end': {'latitude': 48.73820555, 'longitude': 2.38707369}, 'localizer': {'latitude': 48.739661, 'beam_width': 4.5, 'frequency': 110300.00, 'heading': 19, 'longitude': 2.387792}, 'glideslope': {'latitude': 48.719633, 'angle': 3.0, 'longitude': 2.380117, 'threshold_distance': 984}, 'start': {'latitude': 48.71772177, 'longitude': 2.37680629}, 'magnetic_heading': 20.0, 'strip': {'width': 197, 'length': 7874, 'id': 2426, 'surface': 'CON'}, 'identifier': '02', 'id': 4851}], 'ident': '02'},
                '06': {'items': [{'end': {'latitude': 48.7354471, 'longitude': 2.36068699}, 'localizer': {'latitude': 48.736389, 'beam_width': 4.5, 'frequency': 108500.00, 'heading': 63, 'longitude': 2.363325}, 'glideslope': {'latitude': 48.723856, 'angle': 3.0, 'longitude': 2.323992, 'threshold_distance': 1189}, 'start': {'latitude': 48.7199661, 'longitude': 2.31692799}, 'magnetic_heading': 60.0, 'strip': {'width': 147, 'length': 11975, 'id': 2427, 'surface': 'ASP'}, 'identifier': '06', 'id': 4853}], 'ident': '06'},
                '08': {'items': [{'end': {'latitude': 48.7274221, 'longitude': 2.40207799}, 'localizer': {'latitude': 48.727811, 'beam_width': 4.5, 'frequency': 108150.00, 'heading': 75, 'longitude': 2.404133}, 'glideslope': {'latitude': 2.36324, 'angle': 3.0, 'longitude': 48.72311}, 'start': {'latitude': 48.7193991, 'longitude': 2.35860099}, 'magnetic_heading': 80.0, 'strip': {'width': 147, 'length': 10892, 'id': 2428, 'surface': 'CON'}, 'identifier': '08', 'id': 4855}], 'ident': '08'},
                '20': {'items': [{'end': {'latitude': 48.71772177, 'longitude': 2.37680629}, 'start': {'latitude': 48.73820555, 'longitude': 2.38707369}, 'magnetic_heading': 200.0, 'strip': {'width': 197, 'length': 7874, 'id': 2426, 'surface': 'CON'}, 'identifier': '20', 'id': 4852}], 'ident': '20'},
                '24': {'items': [{'end': {'latitude': 48.7199661, 'longitude': 2.31692799}, 'localizer': {'latitude': 48.719344, 'beam_width': 4.5, 'frequency': 110900.00, 'heading': 243, 'longitude': 2.315142}, 'glideslope': {'latitude': 48.735189, 'angle': 3.2, 'longitude': 2.356069, 'threshold_distance': 1029}, 'start': {'latitude': 48.7354471, 'longitude': 2.36068699}, 'magnetic_heading': 240.0, 'strip': {'width': 147, 'length': 11975, 'id': 2427, 'surface': 'ASP'}, 'identifier': '24', 'id': 4854}], 'ident': '24'},
                '26': {'items': [{'end': {'latitude': 48.7193991, 'longitude': 2.35860099}, 'localizer': {'latitude': 48.718633, 'beam_width': 4.5, 'frequency': 111750.00, 'heading': 255, 'longitude': 2.354403}, 'glideslope': {'latitude': 48.724258, 'angle': 3.0, 'longitude': 2.393294, 'threshold_distance': 924}, 'start': {'latitude': 48.7274221, 'longitude': 2.40207799}, 'magnetic_heading': 260.0, 'strip': {'width': 147, 'length': 10892, 'id': 2428, 'surface': 'CON'}, 'identifier': '26', 'id': 4856}], 'ident': '26'},
            },
            'OSL': {
                '01*': {'items': [{'end': {'latitude': 60.201207999999994, 'longitude': 11.122486000000011}, 'localizer': {'latitude': 60.204968999999984, 'beam_width': 4.5, 'frequency': 111950.00, 'heading': 16, 'longitude': 11.12466100000001}, 'glideslope': {'latitude': 60.177935999999995, 'angle': 3.0, 'longitude': 11.111327999999986, 'threshold_distance': 945}, 'start': {'latitude': 60.17575600000001, 'longitude': 11.107781000000006}, 'magnetic_heading': 15.0, 'strip': {'width': 147, 'length': 9678, 'id': 4075, 'surface': 'ASP'}, 'identifier': '01R', 'id': 8149}, {'end': {'latitude': 60.216066999999995, 'longitude': 11.091663999999993}, 'localizer': {'latitude': 60.21982499999999, 'beam_width': 4.5, 'frequency': 110300.00, 'heading': 16, 'longitude': 11.093832999999997}, 'glideslope': {'latitude': 60.18778099999996, 'angle': 3.0, 'longitude': 11.073058000000012, 'threshold_distance': 943}, 'start': {'latitude': 60.18499999999998, 'longitude': 11.073744}, 'magnetic_heading': 15.0, 'strip': {'width': 147, 'length': 11811, 'id': 4076, 'surface': 'ASP'}, 'identifier': '01L', 'id': 8151}], 'ident': '01*'},
                '01L': {'items': [{'end': {'latitude': 60.216066999999995, 'longitude': 11.091663999999993}, 'localizer': {'latitude': 60.21982499999999, 'beam_width': 4.5, 'frequency': 110300.00, 'heading': 16, 'longitude': 11.093832999999997}, 'glideslope': {'latitude': 60.18778099999996, 'angle': 3.0, 'longitude': 11.073058000000012, 'threshold_distance': 943}, 'start': {'latitude': 60.18499999999998, 'longitude': 11.073744}, 'magnetic_heading': 15.0, 'strip': {'width': 147, 'length': 11811, 'id': 4076, 'surface': 'ASP'}, 'identifier': '01L', 'id': 8151}], 'ident': '01L'},
                '01R': {'items': [{'end': {'latitude': 60.201207999999994, 'longitude': 11.122486000000011}, 'localizer': {'latitude': 60.204968999999984, 'beam_width': 4.5, 'frequency': 111950.00, 'heading': 16, 'longitude': 11.12466100000001}, 'glideslope': {'latitude': 60.177935999999995, 'angle': 3.0, 'longitude': 11.111327999999986, 'threshold_distance': 945}, 'start': {'latitude': 60.17575600000001, 'longitude': 11.107781000000006}, 'magnetic_heading': 15.0, 'strip': {'width': 147, 'length': 9678, 'id': 4075, 'surface': 'ASP'}, 'identifier': '01R', 'id': 8149}], 'ident': '01R'},
                '19*': {'items': [{'end': {'latitude': 60.17575600000001, 'longitude': 11.107781000000006}, 'localizer': {'latitude': 60.17199699999999, 'beam_width': 4.5, 'frequency': 110550.00, 'heading': 196, 'longitude': 11.105611000000003}, 'glideslope': {'latitude': 60.198139, 'angle': 3.0, 'longitude': 11.123000000000006, 'threshold_distance': 1052}, 'start': {'latitude': 60.201207999999994, 'longitude': 11.122486000000011}, 'magnetic_heading': 195.0, 'strip': {'width': 147, 'length': 9678, 'id': 4075, 'surface': 'ASP'}, 'identifier': '19L', 'id': 8150}, {'end': {'latitude': 60.18499999999998, 'longitude': 11.073744}, 'localizer': {'latitude': 60.182103, 'beam_width': 4.5, 'frequency': 111300.00, 'heading': 196, 'longitude': 11.072074999999991}, 'glideslope': {'latitude': 60.213763999999976, 'angle': 3.0, 'longitude': 11.088044000000007, 'threshold_distance': 991}, 'start': {'latitude': 60.216066999999995, 'longitude': 11.091663999999993}, 'magnetic_heading': 195.0, 'strip': {'width': 147, 'length': 11811, 'id': 4076, 'surface': 'ASP'}, 'identifier': '19R', 'id': 8152}], 'ident': '19*'},
                '19L': {'items': [{'end': {'latitude': 60.17575600000001, 'longitude': 11.107781000000006}, 'localizer': {'latitude': 60.17199699999999, 'beam_width': 4.5, 'frequency': 110550.00, 'heading': 196, 'longitude': 11.105611000000003}, 'glideslope': {'latitude': 60.198139, 'angle': 3.0, 'longitude': 11.123000000000006, 'threshold_distance': 1052}, 'start': {'latitude': 60.201207999999994, 'longitude': 11.122486000000011}, 'magnetic_heading': 195.0, 'strip': {'width': 147, 'length': 9678, 'id': 4075, 'surface': 'ASP'}, 'identifier': '19L', 'id': 8150}], 'ident': '19L'},
                '19R': {'items': [{'end': {'latitude': 60.18499999999998, 'longitude': 11.073744}, 'localizer': {'latitude': 60.182103, 'beam_width': 4.5, 'frequency': 111300.00, 'heading': 196, 'longitude': 11.072074999999991}, 'glideslope': {'latitude': 60.213763999999976, 'angle': 3.0, 'longitude': 11.088044000000007, 'threshold_distance': 991}, 'start': {'latitude': 60.216066999999995, 'longitude': 11.091663999999993}, 'magnetic_heading': 195.0, 'strip': {'width': 147, 'length': 11811, 'id': 4076, 'surface': 'ASP'}, 'identifier': '19R', 'id': 8152}], 'ident': '19R'},
            },
            'TRD': {
                '09': {'items': [{'end': {'latitude': 63.45755277, 'longitude': 10.94666812}, 'localizer': {'latitude': 63.45755, 'beam_width': 4.5, 'frequency': 110300.00, 'heading': 89, 'longitude': 10.947803}, 'glideslope': {'latitude': 63.457086, 'angle': 3.0, 'longitude': 10.901011, 'threshold_distance': 1067}, 'start': {'latitude': 63.45765623, 'longitude': 10.88929278}, 'magnetic_heading': 90.0, 'strip': {'width': 147, 'length': 9347, 'id': 4065, 'surface': 'ASP'}, 'identifier': '09', 'id': 8129}], 'ident': '09'},
                '27': {'items': [{'end': {'latitude': 63.45765623, 'longitude': 10.88929278}, 'localizer': {'latitude': 63.457658, 'beam_width': 4.5, 'frequency': 109900.00, 'heading': 269, 'longitude': 10.887294}, 'glideslope': {'latitude': 63.458519, 'angle': 3.4, 'longitude': 10.934078, 'threshold_distance': 1287}, 'start': {'latitude': 63.45755277, 'longitude': 10.94666812}, 'magnetic_heading': 270.0, 'strip': {'width': 147, 'length': 9347, 'id': 4065, 'surface': 'ASP'}, 'identifier': '27', 'id': 8130}], 'ident': '27'},
            },
            'VNO': {
                '02': {'items': [{'end': {'latitude': 54.64460419, 'longitude': 25.29304551}, 'localizer': {'latitude': 54.648494, 'beam_width': 4.5, 'frequency': 110500.00, 'heading': 17, 'longitude': 25.295725}, 'glideslope': {'latitude': 54.626656, 'angle': 2.67, 'longitude': 25.282744, 'threshold_distance': 1328}, 'start': {'latitude': 54.6237288, 'longitude': 25.27854667}, 'magnetic_heading': 20.0, 'strip': {'width': 164, 'length': 8251, 'id': 3656, 'surface': 'ASP'}, 'identifier': '02', 'id': 7311}], 'ident': '02'},
                '20': {'items': [{'end': {'latitude': 54.6237288, 'longitude': 25.27854667}, 'localizer': {'latitude': 54.616497, 'beam_width': 4.5, 'frequency': 109100.00, 'heading': 197, 'longitude': 25.27355}, 'glideslope': {'latitude': 54.6414, 'angle': 3.0, 'longitude': 25.292828, 'threshold_distance': 1118}, 'start': {'latitude': 54.64460419, 'longitude': 25.29304551}, 'magnetic_heading': 200.0, 'strip': {'width': 164, 'length': 8251, 'id': 3656, 'surface': 'ASP'}, 'identifier': '20', 'id': 7312}], 'ident': '20'},
            },
        }

    ############################################################################
    # Helper: Creates a copy of an HDF file for testing.

    def _copy_hdf_file(self, hdf_orig, hdf_path):
        '''
        '''
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)

    ############################################################################
    # Helper: Generates a mock API handler:

    def _mock_api_handler(self, airports, runways):
        '''
        Generates a mock API handler so that we do not make HTTP requests.

        Will return the same airport and runway for each query, can be avoided
        with side effect.
        '''

        def mock_nearest_airport(latitude, longitude):
            '''
            '''
            return airports[(latitude, longitude)]

        def mock_nearest_runway(airport, heading, latitude=None,
                                longitude=None, ilsfreq=None):
            '''
            '''
            return runways[airport]

        api_handler = mock.Mock()

        api_handler.get_nearest_airport = mock.Mock()
        if airports.__class__.__name__ == 'function':
            api_handler.get_nearest_airport.side_effect = airports
        else:
            api_handler.get_nearest_airport.side_effect = mock_nearest_airport

        api_handler.get_nearest_runway = mock.Mock()
        if runways.__class__.__name__ == 'function':
            api_handler.get_nearest_runway.side_effect = runways
        else:
            api_handler.get_nearest_runway.side_effect = mock_nearest_runway

        return api_handler

    ############################################################################
    # Helper: Generates output data files post-processing:

    def _render_output_files(self, hdf_path, results):
        '''
        Renders output files from testing such as CSV and KML.

        :param hdf_path: The path to the HDF file.
        :type hdf_path: string
        :param results: The results dictionary from flight processing.
        :type results: dict
        '''
        track_to_kml(hdf_path, results['kti'], results['kpv'])
        csv_flight_details(hdf_path, results['kti'], results['kpv'], results['phases'])

        if debug:
            plot_flight(hdf_path, results['kti'], results['kpv'], results['phases'])

    ############################################################################
    # Test 1: 737-3C Frame

    ## This first test file holds data for 6 sectors without splitting and was
    ## used during early development. Retained for possible reprocessing of data
    ## used in development spreadsheets, but not part of the test file set.

    #@unittest.skipIf(not os.path.isfile('test_data/1_7295949_737-3C.hdf5'),
    #                 'Test file not present')
    #@mock.patch('analysis_engine.flight_attribute.get_api_handler')
    #def test_1_7295949_737_3C(self, get_api_handler):
    #    '''
    #    '''
    #    hdf_orig = 'test_data/1_7295949_737-3C.hdf5'
    #    hdf_path = 'test_data/1_7295949_737-3C_copy.hdf5'
    #
    #    self._copy_hdf_file(hdf_orig, hdf_path)
    #
    #    aircraft_info = {
    #        'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
    #        'Frame': '737-3C',
    #        'Identifier': '5',
    #        'Main Gear To Altitude Radio': 10,
    #        'Manufacturer': 'Boeing',
    #        'Model Series': '737',
    #        'Precise Positioning': True,
    #        'Tail Number': 'G-ABCD',
    #    }
    #
    #    airports = {
    #        (60.201646909117699, 11.083488464355469): self.airports['OSL'],
    #        (60.207918026368986, 11.087010689351679): self.airports['OSL'],
    #        (60.209332779049873, 11.08782559633255):  self.airports['OSL'],
    #        (60.292314738035202, 5.2184030413627625): self.airports['BGO'],
    #        (60.295075884447485, 5.2175367817352285): self.airports['BGO'],
    #        (60.297126313897756, 5.2168199977260254): self.airports['BGO'],
    #        (63.457546234130859, 10.920455589077017): self.airports['TRD'],
    #    }
    #
    #    runways = {
    #        2455: self.runways['BGO']['17'],
    #        2461: self.runways['OSL']['19R'],
    #        2472: self.runways['TRD']['09'],
    #    }
    #
    #    get_api_handler.return_value = self._mock_api_handler(airports, runways)
    #
    #    results = process_flight(hdf_path, aircraft_info, draw=False)
    #
    #    self.assertEqual(len(results), 4)
    #
    #    self._render_output_files(hdf_path, results)

    #def test_time_taken_test_1_7295949_737_3C(self):
    #    '''
    #    '''
    #    timer = Timer(self.test_1_7295949_737_3C)
    #    time_taken = min(timer.repeat(1, 1))
    #    print 'Time taken %s secs' % time_taken
    #    self.assertLess(time_taken, 1.0, msg='Took too long!')

    ############################################################################
    # Test 2: L382-Hercules Frame

    @unittest.skipIf(not os.path.isfile('test_data/2_6748957_L382-Hercules.hdf5'),
                     'Test file not present')
    def test_2_6748957_L382_Hercules(self):
        '''
        '''
        hdf_orig = 'test_data/2_6748957_L382-Hercules.hdf5'
        hdf_path = 'test_data/2_6748957_L382-Hercules_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Frame': u'L382-Hercules',
            'Identifier': u'',
            'Manufacturer Serial Number': u'',
            'Manufacturer': u'Lockheed',
            'Model Series': 'L382',
            'Model': u'L382',
            'Precise Positioning': False,
            'Tail Number': u'A-HERC',
        }

        afr = {
            'AFR Destination Airport': 3279,
            'AFR Flight ID': 4041843,
            'AFR Flight Number': u'ISF51VC',
            'AFR Landing Aiport': 3279,
            'AFR Landing Datetime': datetime(2011, 4, 4, 8, 7, 42),
            'AFR Landing Fuel': 0,
            'AFR Landing Gross Weight': 0,
            'AFR Landing Pilot': 'CAPTAIN',
            'AFR Landing Runway': '23*',
            'AFR Off Blocks Datetime': datetime(2011, 4, 4, 6, 48),
            'AFR On Blocks Datetime': datetime(2011, 4, 4, 8, 18),
            'AFR Takeoff Airport': 3282,
            'AFR Takeoff Datetime': datetime(2011, 4, 4, 6, 48, 59),
            'AFR Takeoff Fuel': 0,
            'AFR Takeoff Gross Weight': 0,
            'AFR Takeoff Pilot': 'FIRST_OFFICER',
            'AFR Takeoff Runway': '11*',
            'AFR Type': u'LINE_TRAINING',
            'AFR V2': 149,
            'AFR Vapp': 135,
            'AFR Vref': 120,
        }

        results = process_flight(hdf_path, aircraft_info, achieved_flight_record=afr, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        tdwn = results['kti'].get(name='Touchdown')[0]
        tdwn_minus_1 = results['kti'].get(name='1 Mins To Touchdown')[0]

        self.assertAlmostEqual(tdwn.index, 4967.0, places=0)
        self.assertAlmostEqual(tdwn_minus_1.index, 4907.0, places=0)
        self.assertEqual(tdwn.datetime - tdwn_minus_1.datetime, timedelta(minutes=1))

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 3: L382-Hercules Frame

    @unittest.skipIf(not os.path.isfile('test_data/3_6748984_L382-Hercules.hdf5'),
                     'Test file not present')
    def test_3_6748984_L382_Hercules(self):
        '''
        '''
        hdf_orig = 'test_data/3_6748984_L382-Hercules.hdf5'
        hdf_path = 'test_data/3_6748984_L382-Hercules_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Frame': u'L382-Hercules',
            'Identifier': u'',
            'Manufacturer Serial Number': u'',
            'Manufacturer': u'Lockheed',
            'Model Series': 'L382',
            'Model': u'L382',
            'Precise Positioning': False,
            'Tail Number': u'B-HERC',
        }

        # NOTE: Test copied from another so AFR may be inaccurate!
        afr = {
            'AFR Destination Airport': 3279, # TODO: Choose another airport.
            'AFR Flight ID': 4041843,
            'AFR Flight Number': u'ISF51VC',
            'AFR Landing Aiport': 3279,
            'AFR Landing Datetime': datetime(2011, 4, 4, 8, 7, 42),
            'AFR Landing Fuel': 0,
            'AFR Landing Gross Weight': 0,
            'AFR Landing Pilot': 'CAPTAIN',
            'AFR Landing Runway': '23*',
            'AFR Off Blocks Datetime': datetime(2011, 4, 4, 6, 48),
            'AFR On Blocks Datetime': datetime(2011, 4, 4, 8, 18),
            'AFR Takeoff Airport': 3282,
            'AFR Takeoff Datetime': datetime(2011, 4, 4, 6, 48, 59),
            'AFR Takeoff Fuel': 0,
            'AFR Takeoff Gross Weight': 0,
            'AFR Takeoff Pilot': 'FIRST_OFFICER',
            'AFR Takeoff Runway': '11*',
            'AFR Type': u'LINE_TRAINING',
            'AFR V2': 149,
            'AFR Vapp': 135,
            'AFR Vref': 120,
        }

        results = process_flight(hdf_path, aircraft_info, achieved_flight_record=afr, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 3A: L382-Hercules Frame

    @unittest.skipIf(not os.path.isfile('test_data/HERCDIP.hdf5'),
                     'Test file not present')
    def test_3A_L382_Hercules_NODIP(self):
        '''
        '''
        hdf_orig = 'test_data/HERCNODIP.hdf5'
        hdf_path = 'test_data/HERCNODIP_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Frame': u'L382-Hercules',
            'Identifier': u'',
            'Manufacturer': u'Lockheed',
            'Manufacturer Serial Number': u'',
            'Model': u'L382',
            'Tail Number': u'B-HERC',
            'Precise Positioning': False,
        }

        # NOTE: Test copied from another so AFR may be inaccurate!
        afr = {
            'AFR Destination Airport': 3279, # TODO: Choose another airport.
            'AFR Flight ID': 4041843,
            'AFR Flight Number': u'ISF51VC',
            'AFR Landing Aiport': 3279,
            'AFR Landing Datetime': datetime(2011, 4, 4, 8, 7, 42),
            'AFR Landing Fuel': 0,
            'AFR Landing Gross Weight': 0,
            'AFR Landing Pilot': 'CAPTAIN',
            'AFR Landing Runway': '23*',
            'AFR Off Blocks Datetime': datetime(2011, 4, 4, 6, 48),
            'AFR On Blocks Datetime': datetime(2011, 4, 4, 8, 18),
            'AFR Takeoff Airport': 3282,
            'AFR Takeoff Datetime': datetime(2011, 4, 4, 6, 48, 59),
            'AFR Takeoff Fuel': 0,
            'AFR Takeoff Gross Weight': 0,
            'AFR Takeoff Pilot': 'FIRST_OFFICER',
            'AFR Takeoff Runway': '11*',
            'AFR Type': u'LINE_TRAINING',
            'AFR V2': 149,
            'AFR Vapp': 135,
            'AFR Vref': 120,
        }

        results = process_flight(hdf_path, aircraft_info, achieved_flight_record=afr, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 3B: L382-Hercules Frame

    @unittest.skipIf(not os.path.isfile('test_data/HERCDIP.hdf5'),
                     'Test file not present')
    def test_3B_L382_Hercules_DIP(self):
        '''
        '''
        hdf_orig = 'test_data/HERCDIP.hdf5'
        hdf_path = 'test_data/HERCDIP_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Frame': u'L382-Hercules',
            'Identifier': u'',
            'Manufacturer Serial Number': u'',
            'Manufacturer': u'Lockheed',
            'Model': u'L382',
            'Precise Positioning': False,
            'Tail Number': u'B-HERC',
        }

        # NOTE: Test copied from another so AFR may be inaccurate!
        afr = {
            'AFR Destination Airport': 3279, # TODO: Choose another airport.
            'AFR Flight ID': 4041843,
            'AFR Flight Number': u'ISF51VC',
            'AFR Landing Aiport': 3279,
            'AFR Landing Datetime': datetime(2011, 4, 4, 8, 7, 42),
            'AFR Landing Fuel': 0,
            'AFR Landing Gross Weight': 0,
            'AFR Landing Pilot': 'CAPTAIN',
            'AFR Landing Runway': '23*',
            'AFR Off Blocks Datetime': datetime(2011, 4, 4, 6, 48),
            'AFR On Blocks Datetime': datetime(2011, 4, 4, 8, 18),
            'AFR Takeoff Airport': 3282,
            'AFR Takeoff Datetime': datetime(2011, 4, 4, 6, 48, 59),
            'AFR Takeoff Fuel': 0,
            'AFR Takeoff Gross Weight': 0,
            'AFR Takeoff Pilot': 'FIRST_OFFICER',
            'AFR Takeoff Runway': '11*',
            'AFR Type': u'LINE_TRAINING',
            'AFR V2': 149,
            'AFR Vapp': 135,
            'AFR Vref': 120,
        }

        results = process_flight(hdf_path, aircraft_info, achieved_flight_record=afr, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 4: 146_301 Frame

    @unittest.skipIf(not os.path.isfile('test_data/4_3377853_146_301.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_4_3377853_146_301(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/4_3377853_146_301.hdf5'
        hdf_path = 'test_data/4_3377853_146_301_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 18, 24, 30, 33],
            'Frame': '146-301',
            'Identifier': '1',
            'Manufacturer': 'BAE',
            'Model Series': '146',
            'Tail Number': 'G-ABCD',
        }

        afr = {
            'AFR Flight ID': 3377853,
            'AFR Landing Fuel': 500,
            'AFR Takeoff Fuel': 1000,
        }

        airport = {'id': 100, 'icao': 'EGLL'}
        runway  = {'identifier': '09L'}
        api_handler = mock.Mock()
        api_handler.get_nearest_airport = mock.Mock()
        api_handler.get_nearest_airport.return_value = airport
        api_handler.get_nearest_runway = mock.Mock()
        api_handler.get_nearest_runway.return_value = runway

        start_datetime = datetime.now()

        # Avoid side effects which may be caused by PRE_FLIGHT_ANALYSIS:
        hooks.PRE_FLIGHT_ANALYSIS = None

        results = process_flight(hdf_path, aircraft_info, start_datetime, afr)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)
        self.assertTrue('flight' in results)

        self._render_output_files(hdf_path, results)

        flight_attrs = {attr.name: attr for attr in results['flight']}

        # Assert FDR Flight ID matches AFR Flight ID:
        self.assertEqual(flight_attrs['FDR Flight ID'].value, 3377853)

        # Assert sensible analysis datetime:
        fdr_analysis_dt = flight_attrs['FDR Analysis Datetime']
        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)
        self.assertTrue(now > fdr_analysis_dt.value > five_minutes_ago)

        # Assert correct takeoff datetime:
        takeoff_datetime = flight_attrs['FDR Takeoff Datetime'].value
        self.assertEqual(takeoff_datetime - start_datetime,
                         timedelta(0, 427, 250000))

        # Assert correct landing datetime:
        landing_datetime = flight_attrs['FDR Landing Datetime'].value
        self.assertEqual(landing_datetime - start_datetime,
                         timedelta(0, 3243, 900000))

        # Assert sensible values for approaches:
        approaches = flight_attrs['FDR Approaches'].value
        self.assertEqual(len(approaches), 1)
        approach = approaches[0]
        self.assertEqual(approach['airport'], airport['id'])
        self.assertEqual(approach['type'], 'LANDING')
        self.assertEqual(approach['runway'], runway['identifier'])
        self.assertEqual(approach['datetime'] - start_datetime,
                         timedelta(0, 3492))

        # Assert correct flight type:
        self.assertEqual(flight_attrs['FDR Flight Type'].value, 'COMPLETE')

        # Assert correct airport and runway:
        self.assertEqual(api_handler.get_nearest_airport.call_args_list,
                         [((40418.0, -3339.21875), {}), ((37917.0, -450.0), {}),
                          ((37917.0, -450.0), {})])
        self.assertEqual(api_handler.get_nearest_runway.call_args_list,
                         [((100, 310.22130556082084), {}),
                          ((100, 219.42928588921563), {}),
                          ((100, 219.42928588921563), {})])
        self.assertEqual(flight_attrs['FDR Takeoff Airport'].value, airport)
        self.assertEqual(flight_attrs['FDR Takeoff Runway'].value, runway)
        self.assertEqual(flight_attrs['FDR Landing Airport'].value, airport)
        self.assertEqual(flight_attrs['FDR Landing Runway'].value, runway)

        # Check other flight attributes:
        self.assertEqual(flight_attrs['FDR Duration'].value, 2816.65)
        self.assertEqual(flight_attrs['FDR Takeoff Fuel'].value, 1000)
        self.assertEqual(flight_attrs['FDR Landing Fuel'].value, 500)
        self.assertEqual(flight_attrs['FDR Version'].value, ___version___)
        self.assertEqual(\
            flight_attrs['FDR Off Blocks Datetime'].value - start_datetime,
            timedelta(0, 172))
        self.assertEqual(\
            flight_attrs['FDR On Blocks Datetime'].value - start_datetime,
            timedelta(0, 3490))

        # NOTE: 'FDR Takeoff Gross Weight' and 'FDR Landing Gross Weight' cannot be tested as 'Gross Weight' is not recorded or derived.
        # NOTE: 'FDR Takeoff Runway' cannot be tested as 'Takeoff Peak Acceleration' does not exist for 'Heading At Takeoff'.
        #
        # FIXME: 'TakeoffDatetime' requires missing 'Liftoff' KTI.
        # FIXME: 'Duration' requires missing 'Takeoff Datetime' and 'Landing Datetime' FlightAttributes.
        #
        # TODO: Test cases for attributes which are currently not implemented.
        #  - Flight number   (May not be recorded.)
        #  - All datetimes
        #  - Pilots          (Hercules: Might not be available.)
        #  - V2, Vapp, Vref  (Hercules: Will be AFR based.)

    def test_time_taken_4_3377853_146_301(self):
        '''
        '''
        timer = Timer(self.test_4_3377853_146_301)
        time_taken = min(timer.repeat(2, 1))
        print 'Time taken %s secs' % time_taken
        self.assertLess(time_taken, 10.0, msg='Took too long!')

    ############################################################################
    # Test 6: 737-1 Frame

    @unittest.skipIf(not os.path.isfile('test_data/6_737_1_RD0001851371.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_6_737_1_RD0001851371(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/6_737_1_RD0001851371.hdf5'
        hdf_path = 'test_data/6_737_1_RD0001851371_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-1',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (58.2000732421875, 8.0804443359375):   self.airports['KRS'],
            (60.18585205078125, 11.1126708984375): self.airports['OSL'],
        }

        runways = {
            2456: self.runways['KRS']['22'],
            2461: self.runways['OSL']['01R'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 7: 737-i Frame

    @unittest.skipIf(not os.path.isfile('test_data/7_737_i_RD0001839773.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_7_737_i_RD0001839773(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/7_737_i_RD0001839773.hdf5'
        hdf_path = 'test_data/7_737_i_RD0001839773_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-i',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (60.18907904624939, 11.098754405975342):  self.airports['OSL'],
            (60.296865999698639, 5.2152204513549805): self.airports['BGO'],
        }

        runways = {
            2455: self.runways['BGO']['35'],  # FIXME: Should be 17?
            2461: self.runways['OSL']['19R'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 9: 737-5 Frame

    @unittest.skipIf(not os.path.isfile('test_data/9_737_5_RD0001860694.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_9_737_5_RD0001860694(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/9_737_5_RD0001860694.hdf5'
        hdf_path = 'test_data/9_737_5_RD0001860694_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-i',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (58.20556640625, 8.0880683118646797):   self.airports['KRS'],
            (60.19134521484375, 11.07696533203125): self.airports['OSL'],
        }

        runways = {
            2461: self.runways['OSL']['01L'],
            2456: self.runways['KRS']['04'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 10: 737-3C Frame

    @unittest.skipIf(not os.path.isfile('test_data/10_737_3C_RD0001861142.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_10_737_3C_RD0001861142_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/10_737_3C_RD0001861142.001.hdf5'
        hdf_path = 'test_data/10_737_3C_RD0001861142.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-3C',
            'Precise Positioning': True,
            'Tail Number': 'G-ABCD',
        }

        def airports(lat, lon):
            if int(lat) == 63 and int(lon) == 10:
                return self.airports['TRD']
            if int(lat) == 60 and int(lon) == 11:
                return self.airports['OSL']
            raise ValueError

        runways = {
            2461: self.runways['OSL']['01L'],
            2472: self.runways['TRD']['09'],  # FIXME: Should be 27?
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 11: 737-3C Frame

    # This sample has a single 232kt sample which the validation traps, leaving
    # no data to examine. Falls over in analysis engine but probably shouldn't
    # reach this far.

    @unittest.skipIf(not os.path.isfile('test_data/11_737_3C_RD0001861129.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_11_737_3C_no_fast_data_RD0001861129_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/11_737_3C_RD0001861129.001.hdf5'
        hdf_path = 'test_data/11_737_3C_RD0001861129.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-3C',
            'Precise Positioning': True,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (58.20556640625, 8.0878186225891113):   self.airports['KRS'],
            (60.19134521484375, 11.07696533203125): self.airports['OSL'],
        }

        runways = {
            2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
            2456: self.runways['KRS']['04'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 12: 737-3C Frame

    @unittest.skipIf(not os.path.isfile('test_data/12_737_3C_RD000183818.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_12_737_3C_RD000183818_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/12_737_3C_RD000183818.001.hdf5'
        hdf_path = 'test_data/12_737_3C_RD000183818.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-3C',
            'Precise Positioning': True,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (50.108969665785381, 14.250219723680361): self.airports['KRS'],
            (60.18798957977976, 11.114856132439204):  self.airports['OSL'],
        }

        runways = {
            2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
            2456: self.runways['KRS']['04'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    @unittest.skipIf(not os.path.isfile('test_data/12_737_3C_RD0001830229.001.hdf5'),
                         'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_12_737_3C_RD0001830229_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/12_737_3C_RD0001830229.001.hdf5'
        hdf_path = 'test_data/12_737_3C_RD0001830229.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-3C',
            'Precise Positioning': True,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (60.181233670030323, 11.111000648566653): self.airports['OSL'],
            (63.457546234130859, 10.928155781080219): self.airports['TRD'],
        }

        runways = {
            2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
            2472: self.runways['TRD']['09'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 13: 737-3C Frame

    @unittest.skipIf(not os.path.isfile('test_data/13_737_3C_RD000183818.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_13_737_3C_RD000183818_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/13_737_3C_RD000183818.001.hdf5'
        hdf_path = 'test_data/13_737_3C_RD000183818.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-3C',
            'Precise Positioning': True,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (50.108961165864415, 14.250177224075527): self.airports['KRS'],  # FIXME: Incorrect?
            (60.18798957977976, 11.114856132439204):  self.airports['OSL'],
        }

        runways = {
            2456: self.runways['KRS']['04'],  # FIXME: Incorrect?
            2461: self.runways['OSL']['19L'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 14: 737-i Frame

    @unittest.skipIf(not os.path.isfile('test_data/14_737_i_RD0001834649.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_14_737_i_RD0001834649_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/14_737_i_RD0001834649.001.hdf5'
        hdf_path = 'test_data/14_737_i_RD0001834649.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-i',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (53.414026200771332, -6.3064670562744141): self.airports['DUB'],
            (60.215677917003632, 11.046042442321777):  self.airports['OSL'],
        }

        runways = {
            2429: self.runways['DUB']['28'],
            2461: self.runways['OSL']['19R'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 14a: 737-3C Frame

    @unittest.skipIf(not os.path.isfile('test_data/14a_737_3C_RD0001802061.001.hdf5'),
                     'Test file not present')
    #@mock.patch('analysis_engine.flight_attribute.get_api_handler')
    #def test_14a_RD0001802061(self, get_api_handler):
    def test_14a_RD0001802061(self):
        '''
        '''
        hdf_orig = 'test_data/14a_737_3C_RD0001802061.001.hdf5'
        hdf_path = 'test_data/14a_737_3C_RD0001802061.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Family': u'B737 NG',
            'Frame': u'737-3C',
            'Identifier': u'15',
            'Main Gear To Lowest Point Of Tail': None,
            'Main Gear To Radio Altimeter Antenna': None,
            'Manufacturer Serial Number': u'39009',
            'Manufacturer': u'Boeing',
            'Model': u'B737-8JP',
            'Precise Positioning': True,
            'Series': u'B737-800',
            'Tail Number': 'G-ABCD',
        }

        #airports = {
        #    (60.181234387708784, 11.111000827986269): self.airports['OSL'],
        #    (63.457546234130859, 10.928016315005772): self.airports['TRD'],
        #}
        #
        #runways = {
        #    2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
        #    2472: self.runways['TRD']['09'],
        #}
        #
        #get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 15: 737-4 Frame (Digital Engine Parameters)

    @unittest.skipIf(not os.path.isfile('test_data/15_737_4_RD0001833760.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_15_737_4_RD0001833760(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/15_737_4_RD0001833760.hdf5'
        hdf_path = 'test_data/15_737_4_RD0001833760_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-4',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (54.630581988738136, 25.283337957583939): self.airports['VNO'],
            (60.209477341638205, 11.08785446913123):  self.airports['OSL'],
            # Second definition for go-around:
            (60.285488069057465, 11.131654903292656): self.airports['OSL'],
        }

        runways = {
            2461: self.runways['OSL']['19R'],
            2518: self.runways['VNO']['20'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 16: 737-4a Frame (Analogue Engine Parameters)

    @unittest.skipIf(not os.path.isfile('test_data/16_737_4a_RD0001821019.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_16_737_4a_RD0001821019(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/16_737_4a_RD0001821019.hdf5'
        hdf_path = 'test_data/16_737_4a_RD0001821019_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Flap Selections': [0, 1, 2, 5, 10, 15, 25, 30, 40],
            'Frame': '737-4a',
            'Precise Positioning': False,
            'Tail Number': 'G-ABCD',
        }

        airports = {
            (48.728805184364319, 2.3640793561935425): self.airports['ORY'],
            (60.220885479215063, 11.114044189453125): self.airports['OSL'],
        }

        runways = {
            2461: self.runways['OSL']['19R'],
            3031: self.runways['ORY']['02'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 17: 737-3C Frame (ILS Approach - Not Tuned?)

    @unittest.skipIf(not os.path.isfile('test_data/17_737_3C_RD0001830259.001.hdf5'),
                     'Test file not present')
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_17_737_3C_RD0001861142_001(self, get_api_handler):
        '''
        '''
        hdf_orig = 'test_data/17_737_3C_RD0001830259.001.hdf5'
        hdf_path = 'test_data/17_737_3C_RD0001830259.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Family': u'Boeing 737',
            'Frame': u'737-3C',
            'Identifier': u'10',
            'Main Gear To Lowest Point Of Tail': None,
            'Main Gear To Radio Altimeter Antenna': None,
            'Manufacturer Serial Number': u'',
            'Manufacturer': u'Boeing',
            'Model': u'737-300',
            'Precise Positioning': True,
            'Series': u'Boeing 737',
            'Tail Number': u'AB-CDE',
        }

        def airports(lat, lon):
            if int(lat) == 63 and int(lon) == 10:
                return self.airports['TRD']
            if int(lat) == 60 and int(lon) == 11:
                return self.airports['OSL']
            raise ValueError

        runways = {
            2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
            2472: self.runways['TRD']['09'],
        }

        get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False, required_params=PROCESS_PARAMETERS)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Test 18: 737-4 Frame (ILS Approach - Not Tuned?)

    @unittest.skipIf(not os.path.isfile('test_data/18_747_4_ILS_not_tuned_RD0001864580.001.hdf5'),
                     'Test file not present')
    #@mock.patch('analysis_engine.flight_attribute.get_api_handler')
    #def test_18_747_4_ILS_not_tuned_RD0001864580_001(self, get_api_handler):
    def test_18_747_4_ILS_not_tuned_RD0001864580_001(self):
        '''
        '''
        hdf_orig = 'test_data/18_747_4_ILS_not_tuned_RD0001864580.001.hdf5'
        hdf_path = 'test_data/18_747_4_ILS_not_tuned_RD0001864580.001_copy.hdf5'

        self._copy_hdf_file(hdf_orig, hdf_path)

        aircraft_info = {
            'Family': u'B737 Classic',
            'Frame': u'737-4',
            'Identifier': u'30',
            'Main Gear To Lowest Point Of Tail': None,
            'Main Gear To Radio Altimeter Antenna': None,
            'Manufacturer Serial Number': u'123456',
            'Manufacturer': u'Boeing',
            'Model': u'B737-31S',
            'Precise Positioning': False,
            'Series': u'B737-300',
            'Tail Number': u'G-ABCD',
        }

        #def airports(lat, lon):
        #    if int(lat) == 63 and int(lon) == 10:
        #        return self.airports['TRD']
        #    if int(lat) == 60 and int(lon) == 11:
        #        return self.airports['OSL']
        #    raise ValueError
        #
        #runways = {
        #    2461: self.runways['OSL']['01L'],  # FIXME: Should be 19R?
        #    2472: self.runways['TRD']['09'],
        #}
        #
        #get_api_handler.return_value = self._mock_api_handler(airports, runways)

        results = process_flight(hdf_path, aircraft_info, draw=False) ##, required_params=PROCESS_PARAMETERS)

        # TODO: Further assertions on the results!
        self.assertEqual(len(results), 4)

        self._render_output_files(hdf_path, results)

    ############################################################################
    # Other Tests

    def test_get_derived_nodes(self):
        '''
        '''
        nodes = get_derived_nodes(['sample_derived_parameters'])
        self.assertEqual(len(nodes), 13)
        self.assertEqual(sorted(nodes.keys())[0], 'Heading Rate')
        self.assertEqual(sorted(nodes.keys())[-1], 'Vertical g')

    @unittest.skip('Not Implemented')
    def test_get_required_params(self):
        '''
        '''
        self.assertTrue(False)

    @unittest.skip('Not Implemented')
    def test_process_flight(self):
        '''
        '''
        self.assertTrue(False)


################################################################################
# Program


if __name__ == '__main__':

    suite = unittest.TestSuite()
    suite.addTest(TestProcessFlight('test_17_737_3C_RD0001861142_001'))
    ####suite = unittest.TestLoader().loadTestsFromName('test_l382_herc_2')
    unittest.TextTestRunner(verbosity=2).run(suite)
    ####unittest.main()


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
