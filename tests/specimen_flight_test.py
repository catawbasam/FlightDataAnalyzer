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
from analysis_engine.node import (
    ApproachItem,
    ApproachNode,
    Attribute,
    KeyPointValue,
)
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
        
        # TODO: Change asserts for floating points to 1 decimal place
        #       (many asserts?).
        self.assertItemsEqual(approaches, ApproachNode(
            'Approach Information',
            items=[
                ApproachItem(
                    'LANDING',
                    slice(2036.0, 2323.0, None),
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
                    gs_est=slice(2038, 2235, None),
                    loc_est=slice(2037, 2318, None),
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
    
        result_kpvs = results['kpv']
        
        expected_kpvs = [
            KeyPointValue(2256.15625, 0.08139081678700373, 'Acceleration Lateral At Touchdown'),
            KeyPointValue(347.90625, 0.03662081678700353, 'Acceleration Lateral During Takeoff Max'),
            KeyPointValue(189.40625, -0.06105918321299646, 'Acceleration Lateral Max'),
            KeyPointValue(2354.40625, 0.10581081678700362, 'Acceleration Lateral Max'),
            KeyPointValue(123.90625, -1.0826291832129964, 'Acceleration Lateral While Taxiing Straight Max'),
            KeyPointValue(2323.65625, 0.0752858167870037, 'Acceleration Lateral While Taxiing Straight Max'),
            KeyPointValue(189.40625, -0.06105918321299646, 'Acceleration Lateral While Taxiing Turn Max'),
            KeyPointValue(2354.40625, 0.10581081678700362, 'Acceleration Lateral While Taxiing Turn Max'),
            KeyPointValue(2249.174033736791, 0.019714999999999927, 'Acceleration Longitudinal During Landing Max'),
            KeyPointValue(360.08854166666669, 0.33106999999999975, 'Acceleration Longitudinal During Takeoff Max'),
            KeyPointValue(2252.640625, 1.3484865941299795, 'Acceleration Normal 20 Ft To Flare Max'),
            KeyPointValue(359.890625, 1.2111585941299794, 'Acceleration Normal At Liftoff'),
            KeyPointValue(2252.765625, 1.4423273941299792, 'Acceleration Normal At Touchdown'),
            KeyPointValue(360.57771664540326, 1.2455653654900614, 'Acceleration Normal Liftoff To 35 Ft Max'),
            KeyPointValue(2252.796875, 1.4423273941299792, 'Acceleration Normal Max'),
            KeyPointValue(2252.875, 1.4423273941299792, 'Acceleration Normal With Flap Down While Airborne Max'),
            KeyPointValue(2254.875, 0.7511097941299794, 'Acceleration Normal With Flap Down While Airborne Min'),
            KeyPointValue(410.75, 1.1951369941299792, 'Acceleration Normal With Flap Up While Airborne Max'),
            KeyPointValue(1548.625, 0.8426617941299794, 'Acceleration Normal With Flap Up While Airborne Min'),
            KeyPointValue(2173.109375, 141.0, 'Airspeed 1000 To 500 Ft Max'),
            KeyPointValue(2206.109375, 133.0, 'Airspeed 1000 To 500 Ft Min'),
            KeyPointValue(467.109375, 253.0, 'Airspeed 1000 To 8000 Ft Max'),
            KeyPointValue(1645.109375, 289.0, 'Airspeed 10000 To 8000 Ft Max'),
            KeyPointValue(2037.109375, 200.0, 'Airspeed 3000 To 1000 Ft Max'),
            KeyPointValue(372.109375, 167.0, 'Airspeed 35 To 1000 Ft Max'),
            KeyPointValue(361.109375, 161.0, 'Airspeed 35 To 1000 Ft Min'),
            KeyPointValue(2225.109375, 136.0, 'Airspeed 500 To 20 Ft Max'),
            KeyPointValue(2251.109375, 131.0, 'Airspeed 500 To 20 Ft Min'),
            KeyPointValue(1835.109375, 254.0, 'Airspeed 5000 To 3000 Ft Max'),
            KeyPointValue(555.109375, 253.0, 'Airspeed 8000 To 10000 Ft Max'),
            KeyPointValue(1701.109375, 261.0, 'Airspeed 8000 To 5000 Ft Max'),
            KeyPointValue(360.57771664540326, 160.46834164540326, 'Airspeed At 35 Ft During Takeoff'),
            KeyPointValue(362.53125, 163.578125, 'Airspeed At Gear Down Selection'),
            KeyPointValue(2106.5, 179.390625, 'Airspeed At Gear Up Selection'),
            KeyPointValue(356.93754400765022, 149.31267603060087, 'Airspeed At Liftoff'),
            KeyPointValue(2254.046875, 127.0625, 'Airspeed At Touchdown'),
            KeyPointValue(1655.109375, 289.0, 'Airspeed Below 10000 Ft During Descent Max'),
            KeyPointValue(1645.109375, 289.0, 'Airspeed Below 10000 Ft Max'),
            KeyPointValue(428.109375, 213.0, 'Airspeed Below 3000 Ft Max'),
            KeyPointValue(1835.109375, 254.0, 'Airspeed Below 5000 Ft Max'),
            KeyPointValue(1701.109375, 261.0, 'Airspeed Below 8000 Ft Max'),
            KeyPointValue(1012.109375, 288.0, 'Airspeed During Cruise Max'),
            KeyPointValue(888.109375, 280.0, 'Airspeed During Cruise Min'),
            KeyPointValue(2254.109375, 127.0, 'Airspeed During Level Flight Max'),
            KeyPointValue(2250.5309913583251, 1.8106134371956841, 'Airspeed Gusts During Final Approach'),
            KeyPointValue(1578.109375, 299.0, 'Airspeed Max'),
            KeyPointValue(1578.109375, 299.0, 'Airspeed Top Of Descent To 10000 Ft Max'),
            KeyPointValue(2254.046875, 125.35712298459734, 'Airspeed True At Touchdown'),
            KeyPointValue(2297.671875, 33.57419431780523, 'Airspeed Vacating Runway'),
            KeyPointValue(362.109375, 164.0, 'Airspeed While Gear Extending Max'),
            KeyPointValue(2105.109375, 180.0, 'Airspeed While Gear Retracting Max'),
            KeyPointValue(2110.234375, 178.0, 'Airspeed With Flap 10 During Descent Max'),
            KeyPointValue(2112.234375, 176.875, 'Airspeed With Flap 10 During Descent Min'),
            KeyPointValue(2110.234375, 178.0, 'Airspeed With Flap 10 Max'),
            KeyPointValue(2112.234375, 176.875, 'Airspeed With Flap 10 Min'),
            KeyPointValue(2113.234375, 175.875, 'Airspeed With Flap 15 During Descent Max'),
            KeyPointValue(2147.234375, 155.875, 'Airspeed With Flap 15 During Descent Min'),
            KeyPointValue(2113.234375, 175.875, 'Airspeed With Flap 15 Max'),
            KeyPointValue(2147.234375, 155.875, 'Airspeed With Flap 15 Min'),
            KeyPointValue(2148.234375, 155.0, 'Airspeed With Flap 20 During Descent Max'),
            KeyPointValue(2148.234375, 155.0, 'Airspeed With Flap 20 During Descent Min'),
            KeyPointValue(2148.234375, 155.0, 'Airspeed With Flap 20 Max'),
            KeyPointValue(2148.234375, 155.0, 'Airspeed With Flap 20 Min'),
            KeyPointValue(2149.234375, 154.75, 'Airspeed With Flap 25 During Descent Max'),
            KeyPointValue(2149.234375, 154.75, 'Airspeed With Flap 25 During Descent Min'),
            KeyPointValue(2149.234375, 154.75, 'Airspeed With Flap 25 Max'),
            KeyPointValue(2149.234375, 154.75, 'Airspeed With Flap 25 Min'),
            KeyPointValue(2150.234375, 152.875, 'Airspeed With Flap 30 During Descent Max'),
            KeyPointValue(2208.234375, 133.0, 'Airspeed With Flap 30 During Descent Min'),
            KeyPointValue(2150.234375, 152.875, 'Airspeed With Flap 30 Max'),
            KeyPointValue(2256.234375, 118.625, 'Airspeed With Flap 30 Min'),
            KeyPointValue(394.234375, 175.125, 'Airspeed With Flap 5 During Climb Max'),
            KeyPointValue(381.234375, 161.0, 'Airspeed With Flap 5 During Climb Min'),
            KeyPointValue(2016.234375, 203.0, 'Airspeed With Flap 5 During Descent Max'),
            KeyPointValue(2091.234375, 178.0, 'Airspeed With Flap 5 During Descent Min'),
            KeyPointValue(2016.234375, 203.0, 'Airspeed With Flap 5 Max'),
            KeyPointValue(357.234375, 150.5, 'Airspeed With Flap 5 Min'),
            KeyPointValue(1578.109375, 299.0, 'Airspeed With Gear Down Max'),
            KeyPointValue(394.234375, 1770.421875, 'Altitude AAL At First Flap Change After Liftoff'),
            KeyPointValue(362.53125, 88.253173828125, 'Altitude AAL At Gear Down Selection'),
            KeyPointValue(2106.5, 1923.00390625, 'Altitude AAL At Gear Up Selection'),
            KeyPointValue(2150.234375, 1262.19140625, 'Altitude AAL At Last Flap Change Before Landing'),
            KeyPointValue(2008.734375, 3589.37890625, 'Altitude AAL Flap Extension Max'),
            KeyPointValue(2220.71875, 375.50390625, 'Altitude At AP Disengaged Selection'),
            KeyPointValue(401.71875, 1932.296875, 'Altitude At AP Engaged Selection'),
            KeyPointValue(2221.90625, 358.12890625, 'Altitude At AT Disengaged Selection'),
            KeyPointValue(356.93754400765022, 241.75017603060087, 'Altitude At Liftoff'),
            KeyPointValue(1179.109375, 23004.0, 'Altitude At Mach Max'),
            KeyPointValue(835.5, 23024.0, 'Altitude Max'),
            KeyPointValue(2009.234375, 4470.375, 'Altitude With Flaps Max'),
            KeyPointValue(354.06458739441194, 0.20594911640897687, 'Deceleration To Abort Takeoff At Rotation'),
            KeyPointValue(2264.8613424564592, 15.023344092606294, 'Delayed Braking After Touchdown'),
            KeyPointValue(356.93754400765022, 898.1373473731496, 'Distance From Liftoff To Runway End'),
            KeyPointValue(354.06458739441194, 1123.7222565911654, 'Distance From Rotation To Runway End'),
            KeyPointValue(116.078125, 610.5, 'Eng Gas Temp During Eng Start Max'),
            KeyPointValue(1556.578125, 384.0, 'Eng Gas Temp During Flight Min'),
            KeyPointValue(736.578125, 808.0, 'Eng Gas Temp During Maximum Continuous Power Max'),
            KeyPointValue(380.578125, 817.5, 'Eng Gas Temp During Takeoff 5 Min Rating Max'),
            KeyPointValue(2211.296875, 54.0, 'Eng N1 500 To 20 Ft Max'),
            KeyPointValue(2251.296875, 39.25, 'Eng N1 500 To 20 Ft Min'),
            KeyPointValue(2198.296875, 1.0, 'Eng N1 Cycles During Final Approach'),
            KeyPointValue(762.296875, 94.0, 'Eng N1 During Maximum Continuous Power Max'),
            KeyPointValue(612.296875, 89.75, 'Eng N1 During Takeoff 5 Min Rating Max'),
            KeyPointValue(2364.0, 44.5, 'Eng N1 During Taxi Max'),
            KeyPointValue(716.5625, 96.0, 'Eng N2 During Maximum Continuous Power Max'),
            KeyPointValue(380.5625, 95.25, 'Eng N2 During Takeoff 5 Min Rating Max'),
            KeyPointValue(2362.0, 79.5, 'Eng N2 During Taxi Max'),
            KeyPointValue(362.53125, 5.0, 'Flap At Gear Down Selection'),
            KeyPointValue(356.93754400765022, 5.0, 'Flap At Liftoff'),
            KeyPointValue(2254.046875, 30.0, 'Flap At Touchdown'),
            KeyPointValue(57.234375, 0.0, 'Flap With Gear Up Max'),
            KeyPointValue(143.234375, 5.0, 'Flap With Gear Up Max'),
            KeyPointValue(2150.234375, 30.0, 'Flap With Gear Up Max'),
            KeyPointValue(2254.046875, 380.0078125, 'Flare Distance 20 Ft To Touchdown'),
            KeyPointValue(2254.046875, 3.678793435289208, 'Flare Duration 20 Ft To Touchdown'),
            KeyPointValue(356.93754400765022, 5980.402727122727, 'Fuel Qty At Liftoff'),
            KeyPointValue(2254.046875, 4655.053711706836, 'Fuel Qty At Touchdown'),
            KeyPointValue(356.93754400765022, 47671.209672019446, 'Gross Weight At Liftoff'),
            KeyPointValue(2254.046875, 46374.56716198159, 'Gross Weight At Touchdown'),
            KeyPointValue(2254.046875, 123.25, 'Groundspeed At Touchdown'),
            KeyPointValue(356.0, 140.140625, 'Groundspeed Max'),
            KeyPointValue(2254.171875, 123.0, 'Groundspeed Max'),
            KeyPointValue(2297.671875, 31.5, 'Groundspeed Vacating Runway'),
            KeyPointValue(269.171875, 14.0, 'Groundspeed While Taxiing Straight Max'),
            KeyPointValue(2389.171875, 25.0, 'Groundspeed While Taxiing Straight Max'),
            KeyPointValue(305.171875, 7.0, 'Groundspeed While Taxiing Turn Max'),
            KeyPointValue(2435.171875, 14.0, 'Groundspeed While Taxiing Turn Max'),
            KeyPointValue(2264.921875, 16.875, 'Heading At Landing'),
            KeyPointValue(2307.0, 32.6568603515625, 'Heading At Lowest Point On Approach'),
            KeyPointValue(342.921875, 36.2109375, 'Heading At Takeoff'),
            KeyPointValue(348.421875, 2.659918639476075, 'Heading Deviation From Runway Above 80 Kts Airspeed During Takeoff'),
            KeyPointValue(2302.671875, 25.3125, 'Heading Vacating Runway'),
            KeyPointValue(2248.421875, 1.7578125, 'Heading Variation 500 To 50 Ft'),
            KeyPointValue(2257.4365255816419, 1.40625, 'Heading Variation Above 100 Kts Airspeed During Landing'),
            KeyPointValue(2273.9403935185187, 3.8671875, 'Heading Variation Touchdown Plus 4 Sec To 60 Kts Airspeed'),
            KeyPointValue(2194.046875, 707.00390625, 'Height 1 Mins To Touchdown'),
            KeyPointValue(2134.046875, 1503.00390625, 'Height 2 Mins To Touchdown'),
            KeyPointValue(2074.046875, 2463.00390625, 'Height 3 Mins To Touchdown'),
            KeyPointValue(2014.046875, 3459.00390625, 'Height 4 Mins To Touchdown'),
            KeyPointValue(1954.046875, 4335.00390625, 'Height 5 Mins To Touchdown'),
            KeyPointValue(2170.46875, -0.08928571428571429, 'ILS Glideslope Deviation 1000 To 500 Ft Max'),
            KeyPointValue(2170.46875, -0.08928571428571429, 'ILS Glideslope Deviation 1500 To 1000 Ft Max'),
            KeyPointValue(2230.46875, 0.17857142857142858, 'ILS Glideslope Deviation 500 To 200 Ft Max'),
            KeyPointValue(2170.484375, -0.0907258064516129, 'ILS Localizer Deviation 1000 To 500 Ft Max'),
            KeyPointValue(2169.484375, -0.0907258064516129, 'ILS Localizer Deviation 1500 To 1000 Ft Max'),
            KeyPointValue(2218.484375, -0.06048387096774195, 'ILS Localizer Deviation 500 To 200 Ft Max'),
            KeyPointValue(2254.046875, 0.056073588709677435, 'ILS Localizer Deviation At Touchdown'),
            KeyPointValue(358.3076388888889, 1.3700948812386855, 'Liftoff To Climb Pitch Duration'),
            KeyPointValue(1179.109375, 0.6637046228346292, 'Mach Max'),
            KeyPointValue(362.109375, 0.24952616558796145, 'Mach While Gear Extending Max'),
            KeyPointValue(2105.109375, 0.286361389180504, 'Mach While Gear Retracting Max'),
            KeyPointValue(1179.109375, 0.6637046228346292, 'Mach With Gear Down Max'),
            KeyPointValue(2188.21875, 2.8125, 'Pitch 1000 To 500 Ft Max'),
            KeyPointValue(2184.21875, 1.40625, 'Pitch 1000 To 500 Ft Min'),
            KeyPointValue(2254.046875, 2.8839111328125, 'Pitch 20 Ft To Touchdown Min'),
            KeyPointValue(364.21875, 19.6875, 'Pitch 35 To 400 Ft Max'),
            KeyPointValue(361.21875, 15.46875, 'Pitch 35 To 400 Ft Min'),
            KeyPointValue(375.21875, 21.4453125, 'Pitch 400 To 1000 Ft Max'),
            KeyPointValue(368.21875, 19.6875, 'Pitch 400 To 1000 Ft Min'),
            KeyPointValue(2252.21875, 5.625, 'Pitch 50 Ft To Touchdown Max'),
            KeyPointValue(2230.21875, 1.7578125, 'Pitch 500 To 20 Ft Min'),
            KeyPointValue(2243.21875, 3.515625, 'Pitch 500 To 50 Ft Max'),
            KeyPointValue(2254.046875, 2.8839111328125, 'Pitch 7 Ft To Touchdown Min'),
            KeyPointValue(422.234375, 13.359375, 'Pitch After Flap Retraction Max'),
            KeyPointValue(360.53669695551184, 14.269828632735662, 'Pitch At 35 Ft During Climb'),
            KeyPointValue(356.93754400765022, 7.7893530549435885, 'Pitch At Liftoff'),
            KeyPointValue(2254.046875, 2.8839111328125, 'Pitch At Touchdown'),
            KeyPointValue(361.21875, 15.46875, 'Pitch Liftoff To 35 Ft Max'),
            KeyPointValue(355.21875, 2.4609375, 'Pitch Rate 2 Deg Pitch To 35 Ft Max'),
            KeyPointValue(353.21875, 0.52734375, 'Pitch Rate 2 Deg Pitch To 35 Ft Min'),
            KeyPointValue(2251.21875, 0.87890625, 'Pitch Rate 20 Ft To Touchdown Max'),
            KeyPointValue(2254.046875, -1.43646240234375, 'Pitch Rate 20 Ft To Touchdown Min'),
            KeyPointValue(361.21875, 1.7578125, 'Pitch Rate 35 To 1000 Ft Max'),
            KeyPointValue(361.5, 2280.0, 'Rate Of Climb 35 To 1000 Ft Min'),
            KeyPointValue(471.5, 4740.0, 'Rate Of Climb Below 10000 Ft Max'),
            KeyPointValue(716.5, 5100.0, 'Rate Of Climb Max'),
            KeyPointValue(2192.5, -840.0, 'Rate Of Descent 1000 To 500 Ft Max'),
            KeyPointValue(1632.5, -2640.0, 'Rate Of Descent 10000 To 5000 Ft Max'),
            KeyPointValue(2102.5, -960.0, 'Rate Of Descent 2000 To 1000 Ft Max'),
            KeyPointValue(2081.5, -1080.0, 'Rate Of Descent 3000 To 2000 Ft Max'),
            KeyPointValue(2247.515625, -758.6519286881273, 'Rate Of Descent 50 Ft To Touchdown Max'),
            KeyPointValue(2232.5, -840.0, 'Rate Of Descent 500 To 50 Ft Max'),
            KeyPointValue(1997.5, -1560.0, 'Rate Of Descent 5000 To 3000 Ft Max'),
            KeyPointValue(2254.4036085948655, -214.89463186777564, 'Rate Of Descent At Touchdown'),
            KeyPointValue(1632.5, -2640.0, 'Rate Of Descent Below 10000 Ft Max'),
            KeyPointValue(1558.5, -3540.0, 'Rate Of Descent Max'),
            KeyPointValue(1997.5, -1560.0, 'Rate Of Descent Max'),
            KeyPointValue(1558.5, -3540.0, 'Rate Of Descent Top Of Descent To 10000 Ft Max'),
            KeyPointValue(2170.34375, -2.4609375, 'Roll 1000 To 300 Ft Max'),
            KeyPointValue(2252.34375, -1.7578125, 'Roll 20 Ft To Touchdown Max'),
            KeyPointValue(368.34375, -1.0546875, 'Roll 20 To 400 Ft Max'),
            KeyPointValue(2235.34375, -1.40625, 'Roll 300 To 20 Ft Max'),
            KeyPointValue(375.34375, -4.921875, 'Roll 400 To 1000 Ft Max'),
            KeyPointValue(1920.34375, -22.5, 'Roll Above 1000 Ft Max'),
            KeyPointValue(358.34375, -1.40625, 'Roll Liftoff To 20 Ft Max'),
            KeyPointValue(431.609375, 2628.5, 'Terrain Clearance Above 3000 Ft Min'),
            KeyPointValue(2125.296875, 3.25, 'Thrust Asymmetry During Approach Max'),
            KeyPointValue(1313.296875, 11.0, 'Thrust Asymmetry During Flight Max'),
            KeyPointValue(331.296875, 4.75, 'Thrust Asymmetry During Takeoff Max'),
            KeyPointValue(2273.5052083333335, 19.458333333333485, 'Touchdown To 60 Kts Duration'),
            KeyPointValue(2255.53125, 1.484375, 'Touchdown To Spoilers Deployed Duration'),
            KeyPointValue(2150.390625, 0.055905395082465816, 'Turbulence During Approach Max'),
            KeyPointValue(1241.5, 0.027310323713430208, 'Turbulence During Cruise Max'),
            KeyPointValue(2253.640625, 0.15200050075950688, 'Turbulence During Flight Max'),
            KeyPointValue(2242.3707882534777, 8.391644390878792, 'Wind Direction At 100 Ft During Descent'),
            KeyPointValue(2169.7503255208335, 345.2998352050779, 'Wind Direction At 1000 Ft During Descent'),
            KeyPointValue(2133.687744140625, 349.33489322662354, 'Wind Direction At 1500 Ft During Descent'),
            KeyPointValue(2101.687744140625, 357.577428817749, 'Wind Direction At 2000 Ft During Descent'),
            KeyPointValue(2247.4242928452577, 18.08817122422875, 'Wind Direction At 50 Ft During Descent'),
            KeyPointValue(2210.37548828125, 337.0931625366211, 'Wind Direction At 500 Ft During Descent'),
            KeyPointValue(2242.3707882534777, 4.606521686630572, 'Wind Speed At 100 Ft During Descent'),
            KeyPointValue(2169.7503255208335, 6.0, 'Wind Speed At 1000 Ft During Descent'),
            KeyPointValue(2133.687744140625, 14.0, 'Wind Speed At 1500 Ft During Descent'),
            KeyPointValue(2101.687744140625, 15.77728271484375, 'Wind Speed At 2000 Ft During Descent'),
            KeyPointValue(2247.4242928452577, 4.0, 'Wind Speed At 50 Ft During Descent'),
            KeyPointValue(2210.37548828125, 4.6053466796875, 'Wind Speed At 500 Ft During Descent'),
            KeyPointValue(62.75, 41746.37377295, 'Zero Fuel Weight'),
        ]
        
        for expected_kpv in expected_kpvs:
            matching_kpvs = [k for k in result_kpvs
                             if k.name == expected_kpv.name]
            for matching_kpv in matching_kpvs:
                try:
                    self.assertAlmostEqual(matching_kpv.index,
                                           expected_kpv.index, places=1)
                    self.assertAlmostEqual(matching_kpv.value,
                                           expected_kpv.value, places=1)
                    break
                except AssertionError:
                    pass
            else:
                raise AssertionError(
                    "Expected KPV '%s' does not match any result KPVs '%s'." %
                    (expected_kpv, matching_kpvs)
                )
            


##############################################################################
# Program


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.TextTestRunner(verbosity=2).run(suite)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
