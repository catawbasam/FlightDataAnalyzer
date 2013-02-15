# -*- coding: utf-8 -*-
################################################################################

'''
Test processing the specimen flight through the FlightDataAnalyzer.
'''

################################################################################
# Imports


import os
import unittest

from datetime import datetime

# Ensure that we use the local API handler for data validation:
import data_validation.settings as dv_settings
dv_settings.VALIDATION_API_HANDLER = 'data_validation.api_handler.ValidationAPIHandlerLocal'

from analysis_engine.node import ApproachItem, Attribute
from analysis_engine.process_flight import process_flight

from utilities.filesystem_tools import copy_file


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

################################################################################
# Test Cases


class TestProcessSpecimenFlight(unittest.TestCase):
    '''
    '''

    def setUp(self):
        self.data_path = copy_file(os.path.join(test_data_path,
                                                'Specimen_Flight.hdf5'),
                                   dest_dir=os.path.join(test_data_path,
                                                         'temp'))

    def tearDown(self):
        try:
            os.remove(self.data_path)
        except:
            pass

    def test_process_specimen_flight(self):
        '''
        '''
        aircraft_info = {
            'Tail Number': 'G-ABCD',
            'Model': 'B737-301',
            'Series': 'B737-300',
            'Family': 'B737',
            'Manufacturer': 'Boeing',
            'Precise Positioning': False,
            'Frame': '737-5',
            'Frame Qualifier': 'Altitude_Radio_EFIS',
        }

        afr = {}

        results = process_flight(
            self.data_path, aircraft_info,
            start_datetime=datetime(2012, 12, 30, 19, 9, 6),
            achieved_flight_record=afr)

        # TODO: Further assertions on the results!
        self.assertEqual(
            results['approach'],
            [ApproachItem(type='LANDING',
                          slice=slice(2036.5, 2322.5, None),
                          airport={'distance': 1582.789862070702,
                                   'magnetic_variation': 'E001226 0106',
                                   'code': {'icao': 'ENGM', 'iata': 'OSL'},
                                   'elevation': 689,
                                   'name': 'Oslo Gardermoen',
                                   'longitude': 11.1004,
                                   'location': {'city': 'Oslo',
                                                'country': 'Norway'},
                                   'latitude': 60.1939,
                                   'id': 2461},
                          runway={'start': {'latitude': 60.185019,
                                            'elevation': 650,
                                            'longitude': 11.073491},
                                  'end': {'latitude': 60.216092,
                                          'elevation': 682,
                                          'longitude': 11.091397},
                                  'magnetic_heading': 13.7,
                                  'strip': {'width': 147,
                                            'length': 11811,
                                            'id': 4076,
                                            'surface': 'ASP'},
                                  'localizer': {'elevation': 686,
                                                'longitude': 11.093536,
                                                'frequency': 110300.0,
                                                'latitude': 60.219775,
                                                'heading': 16,
                                                'beam_width': 4.5},
                                  'identifier': '01L',
                                  'id': 8151,
                                  'glideslope': {'latitude': 60.186858,
                                                 'elevation': 669,
                                                 'angle': 3.0,
                                                 'longitude': 11.072234,
                                                 'threshold_distance': 943}},
                          gs_est=slice(2039, 2235, None),
                          loc_est=slice(2038, 2318, None),
                          ils_freq=None,
                          turnoff=2297.671875,
                          lowest_lat=60.18765449523926,
                          lowest_lon=11.074669361114502,
                          lowest_hdg=16.875)])
        attributes = [
            Attribute('FDR Takeoff Airport', {
                'code': {'iata': 'KRS', 'icao': 'ENCN'},
                'distance': 44.95628388633336,
                'elevation': 43,
                'id': 2456,
                'latitude': 58.2042,
                'location': {'city': 'Kjevik',
                             'country': 'Norway'},
                'longitude': 8.08537,
                'magnetic_variation': 'E000091 0106',
                'name': 'Kristiansand Lufthavn Kjevik'}),
             Attribute('FDR Takeoff Runway', {
                 'distance': 1061.645105402618,
                 'end': {'elevation': 43,
                         'latitude': 58.211678,
                         'longitude': 8.095269},
                 'glideslope': {'angle': 3.4,
                                'elevation': 39,
                                'latitude': 58.198664,
                                'longitude': 8.080164,
                                'threshold_distance': 720},
                 'id': 8127,
                 'identifier': '04',
                 'localizer': {'beam_width': 4.5,
                               'elevation': 43,
                               'frequency': 110300.0,
                               'heading': 36,
                               'latitude': 58.212397,
                               'longitude': 8.096228},
                 'magnetic_heading': 33.9,
                 'start': {'elevation': 26,
                           'latitude': 58.196703,
                           'longitude': 8.075406},
                 'strip': {'id': 4064,
                           'length': 6660,
                           'surface': 'ASP',
                           'width': 147}}),
             Attribute('FDR Landing Airport',
                       {'code': {'iata': 'OSL',
                                 'icao': 'ENGM'},
                        'distance': 1582.789862070702,
                        'elevation': 689,
                        'id': 2461,
                        'latitude': 60.1939,
                        'location': {'city': 'Oslo', 'country': 'Norway'},
                        'longitude': 11.1004,
                        'magnetic_variation': 'E001226 0106',
                        'name': 'Oslo Gardermoen'}),
             Attribute('FDR Version', '0.0.1'),
             Attribute('FDR Takeoff Gross Weight',
                       47671.209672019446),
             Attribute('FDR Landing Datetime',
                       datetime(2012, 12, 30, 19, 46, 39, 500000)),
             Attribute('FDR Takeoff Datetime',
                       datetime(2012, 12, 30, 19, 15, 2, 921919)),
             Attribute('FDR Duration', 1896.578081),
             Attribute('FDR Off Blocks Datetime',
                       datetime(2012, 12, 30, 19, 12, 12)),
             Attribute('FDR On Blocks Datetime',
                       datetime(2012, 12, 30, 19, 51, 1)),
             Attribute('FDR Takeoff Fuel', 5980.402727122727),
             Attribute('FDR Landing Fuel', 4655.053711706836),
             Attribute('FDR Landing Gross Weight',
                       46374.56716198159),
             Attribute('FDR Flight Type', 'COMPLETE'),
        ]
        self.assertEqual(results['flight'][:-3] + results['flight'][-2:],
                         attributes)


################################################################################
# Program


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.TextTestRunner(verbosity=2).run(suite)


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
