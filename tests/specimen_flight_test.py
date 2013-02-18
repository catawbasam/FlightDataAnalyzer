# -*- coding: utf-8 -*-
##############################################################################

'''
Test processing the specimen flight through the FlightDataAnalyzer.
'''

##############################################################################
# Imports


import logging
import os
import unittest

from datetime import datetime

from analysis_engine import hooks, settings
from analysis_engine.node import ApproachItem, ApproachNode, Attribute
from analysis_engine.process_flight import process_flight

from utilities.filesystem_tools import copy_file


##############################################################################
# Constants


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')


##############################################################################
# Setup


def setUpModule():
    settings.API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal'
    hooks.PRE_FILE_ANALYSIS = None
    hooks.PRE_FLIGHT_ANALYSIS = None
    logging.disable(logging.DEBUG)


##############################################################################
# Test Cases


class TestProcessSpecimenFlight(unittest.TestCase):
    '''
    '''

    def setUp(self):
        hdf_path = os.path.join(DATA_PATH, 'Specimen_Flight.hdf5')
        tmp_path = os.path.join(DATA_PATH, 'temp')
        self.data_path = copy_file(hdf_path, dest_dir=tmp_path)

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

        # Prepare results for testing (removing awkward values):
        approaches = results['approach']
        attributes = [attribute for attribute in results['flight']
                if not attribute.name == 'FDR Analysis Datetime']

        self.assertItemsEqual(approaches, ApproachNode(
            name='Approach Information',
            items=[
                ApproachItem(
                    type='LANDING',
                    slice=slice(2036.5, 2322.5, None),
                    airport={
                        'code': {'icao': 'ENGM', 'iata': 'OSL'},
                        'distance': 1582.789862070702,
                        'elevation': 689,
                        'id': 2461,
                        'latitude': 60.1939,
                        'location': {'city': 'Oslo', 'country': 'Norway'},
                        'longitude': 11.1004,
                        'magnetic_variation': 'E001226 0106',
                        'name': 'Oslo Gardermoen',
                    },
                    runway={
                        'start': {
                            'elevation': 650,
                            'latitude': 60.185019,
                            'longitude': 11.073491,
                        },
                        'end': {
                            'elevation': 682,
                            'latitude': 60.216092,
                            'longitude': 11.091397,
                        },
                        'magnetic_heading': 13.7,
                        'strip': {
                            'id': 4076,
                            'length': 11811,
                            'surface': 'ASP',
                            'width': 147,
                        },
                        'localizer': {
                            'beam_width': 4.5,
                            'elevation': 686,
                            'frequency': 110300.0,
                            'heading': 16,
                            'latitude': 60.219775,
                            'longitude': 11.093536,
                        },
                        'identifier': '01L',
                        'id': 8151,
                        'glideslope': {
                            'angle': 3.0,
                            'elevation': 669,
                            'latitude': 60.186858,
                            'longitude': 11.072234,
                            'threshold_distance': 943,
                        },
                    },
                    gs_est=slice(2039, 2235, None),
                    loc_est=slice(2038, 2318, None),
                    ils_freq=None,
                    turnoff=2297.671875,
                    lowest_lat=60.18765449523926,
                    lowest_lon=11.074669361114502,
                    lowest_hdg=16.875,
                ),
            ]
        ))

        self.assertItemsEqual(attributes, [
            Attribute('FDR Takeoff Airport', {
                'code': {'iata': 'KRS', 'icao': 'ENCN'},
                'distance': 44.95628388633336,
                'elevation': 43,
                'id': 2456,
                'latitude': 58.2042,
                'location': {'city': 'Kjevik', 'country': 'Norway'},
                'longitude': 8.08537,
                'magnetic_variation': 'E000091 0106',
                'name': 'Kristiansand Lufthavn Kjevik',
            }),
            Attribute('FDR Takeoff Runway', {
                'distance': 1061.645105402618,
                'end': {
                    'elevation': 43,
                    'latitude': 58.211678,
                    'longitude': 8.095269,
                },
                'glideslope': {
                    'angle': 3.4,
                    'elevation': 39,
                    'latitude': 58.198664,
                    'longitude': 8.080164,
                    'threshold_distance': 720,
                },
                'id': 8127,
                'identifier': '04',
                'localizer': {
                    'beam_width': 4.5,
                    'elevation': 43,
                    'frequency': 110300.0,
                    'heading': 36,
                    'latitude': 58.212397,
                    'longitude': 8.096228,
                },
                'magnetic_heading': 33.9,
                'start': {
                    'elevation': 26,
                    'latitude': 58.196703,
                    'longitude': 8.075406,
                },
                'strip': {
                    'id': 4064,
                    'length': 6660,
                    'surface': 'ASP',
                    'width': 147,
                },
            }),
            Attribute('FDR Landing Airport', {
                'code': {'iata': 'OSL', 'icao': 'ENGM'},
                'distance': 1582.789862070702,
                'elevation': 689,
                'id': 2461,
                'latitude': 60.1939,
                'location': {'city': 'Oslo', 'country': 'Norway'},
                'longitude': 11.1004,
                'magnetic_variation': 'E001226 0106',
                'name': 'Oslo Gardermoen',
            }),
            Attribute('FDR Version', '0.0.1'),
            Attribute('FDR Takeoff Gross Weight', 47671.209672019446),
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
            Attribute('FDR Landing Gross Weight', 46374.56716198159),
            Attribute('FDR Flight Type', 'COMPLETE'),
        ])


##############################################################################
# Program


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.TextTestRunner(verbosity=2).run(suite)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
