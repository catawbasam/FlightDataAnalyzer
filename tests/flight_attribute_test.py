import numpy as np
import unittest

from datetime import datetime
from mock import Mock, patch

from analysis.api_handler import NotFoundError
from analysis.node import (A, KeyPointValue, KeyTimeInstance, KPV, KTI, P, S,
                           Section)
from analysis.flight_attribute import (
    Approaches, 
    Duration, 
    FlightID, 
    FlightNumber,
    FlightType, 
    LandingAirport, 
    LandingDatetime, 
    LandingFuel, 
    LandingGrossWeight,
    LandingPilot,
    LandingRunway,
    OffBlocksDatetime,
    OnBlocksDatetime,
    TakeoffAirport,
    TakeoffDatetime, 
    TakeoffFuel,
    TakeoffGrossWeight,
    TakeoffRunway,
)


class TestApproaches(unittest.TestCase):
    def test_can_operate(self):
        # Can operate with approach lat lng.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Latitude At Low Point On Approach',
             'Longitude At Low Point On Approach']))
        # Can operate with landing lat lng.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Latitude At Landing',
             'Longitude At Landing']))
        # Can operate with everything.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Latitude At Low Point On Approach',
             'Longitude At Low Point On Approach',
             'Heading At Low Point On Approach',
             'Latitude At Landing',
             'Longitude At Landing',
             'Touch And Go',
             'Go Around']))
        # Cannot operate missing latitude.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Longitude At Low Point On Approach']))
        # Cannot operate missing Approach and Landing.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Heading At Landing',
             'Latitude At Low Point On Approach',
             'Longitude At Low Point On Approach']))
        # Cannot operate with differing sources of lat lng.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Latitude At Low Point On Approach',
             'Longitude At Landing']))
    
    def test__get_lat_lon(self):
        # Landing KPVs.
        approaches = Approaches()
        approach_slice = slice(3,10)
        landing_lat_kpvs = KPV('Latitude At Landing',
                               items=[KeyPointValue(1, 13, 'b'),
                                      KeyPointValue(5, 10, 'b'),
                                      KeyPointValue(17, 14, 'b')])
        landing_lon_kpvs = KPV('Longitude At Landing',
                               items=[KeyPointValue(1, -1, 'b'),
                                      KeyPointValue(5, -2, 'b'),
                                      KeyPointValue(17, 2, 'b')])
        lat, lon = approaches._get_lat_lon(approach_slice, landing_lat_kpvs,
                                           landing_lon_kpvs, None, None)
        self.assertEqual(lat, 10)
        self.assertEqual(lon, -2)
        # Approach KPVs.
        approach_slice = slice(10,15)
        approach_lat_kpvs = KPV('Latitude At Low Point On Approach',
                                items=[KeyPointValue(12, 4, 'b')])
        approach_lon_kpvs = KPV('Longitude At Low Point On Approach',
                                items=[KeyPointValue(12, 3, 'b')])
        lat, lon = approaches._get_lat_lon(approach_slice, None, None,
                                           approach_lat_kpvs, approach_lon_kpvs)
        self.assertEqual(lat, 4)
        self.assertEqual(lon, 3)
        # Landing and Approach KPVs, Landing KPVs preferred.
        approach_slice = slice(4, 15)
        lat, lon = approaches._get_lat_lon(approach_slice, landing_lat_kpvs,
                                           landing_lon_kpvs, approach_lat_kpvs,
                                           approach_lon_kpvs)
        self.assertEqual(lat, 10)
        self.assertEqual(lon, -2)
        approach_slice = slice(20,40)
        lat, lon = approaches._get_lat_lon(approach_slice, landing_lat_kpvs,
                                           landing_lon_kpvs, approach_lat_kpvs,
                                           approach_lon_kpvs)
        self.assertEqual(lat, None)
        self.assertEqual(lon, None)
    
    def test__get_hdg(self):
        approaches = Approaches()
        # Landing KPV
        approach_slice = slice(1,10)
        landing_hdg_kpvs = KPV('Heading At Landing',
                               items=[KeyPointValue(2, 30, 'a'),
                                      KeyPointValue(14, 40, 'b')])
        hdg = approaches._get_hdg(approach_slice, landing_hdg_kpvs, None)
        self.assertEqual(hdg, 30)
        # Approach KPV
        approach_hdg_kpvs = KPV('Heading At Landing',
                                items=[KeyPointValue(4, 15, 'a'),
                                       KeyPointValue(23, 30, 'b')])
        hdg = approaches._get_hdg(approach_slice, None, approach_hdg_kpvs)
        self.assertEqual(hdg, 15)
        # Landing and Approach KPV, Landing preferred
        hdg = approaches._get_hdg(approach_slice, landing_hdg_kpvs,
                                  approach_hdg_kpvs)
        self.assertEqual(hdg, 30)
        # No KPVs in slice.
        approach_slice = slice(30,60)
        hdg = approaches._get_hdg(approach_slice, landing_hdg_kpvs,
                                  approach_hdg_kpvs)
        self.assertEqual(hdg, None)
    
    def test__get_approach_type(self):
        approaches = Approaches()
        # Heading At Landing KPVs.
        landing_hdg_kpvs = KPV('Heading At Landing',
                               items=[KeyPointValue(9, 21, 'a')])
        approach_slice = slice(7, 10)
        approach_type = approaches._get_approach_type(approach_slice,
                                                      landing_hdg_kpvs, None,
                                                      None)
        self.assertEqual(approach_type, 'LANDING')
        # Touch and Go KTIs.
        touch_and_gos = KTI('Touch And Go', items=[KeyTimeInstance(12, 'a'),
                                                   KeyTimeInstance(16, 'a')])
        approach_slice = slice(8,14)
        approach_type = approaches._get_approach_type(approach_slice, None,
                                                      touch_and_gos, None)
        self.assertEqual(approach_type, 'TOUCH_AND_GO')
        # Go Around KTIs.
        go_arounds = KTI('Go Arounds', items=[KeyTimeInstance(12, 'a'),
                                              KeyTimeInstance(16, 'a')])
        approach_type = approaches._get_approach_type(approach_slice, None,
                                                      None, go_arounds)
        self.assertEqual(approach_type, 'GO_AROUND')
        # Heading At Landing and Touch And Gos, Heading preferred.
        approach_type = approaches._get_approach_type(approach_slice,
                                                      landing_hdg_kpvs,
                                                      touch_and_gos, None)
        self.assertEqual(approach_type, 'LANDING')
        # Heading At Landing and Go Arounds. Heading preferred.
        approach_type = approaches._get_approach_type(approach_slice,
                                                      landing_hdg_kpvs,
                                                      None, go_arounds)
        self.assertEqual(approach_type, 'LANDING')
        # Touch And Gos and Go Arounds. Touch And Gos preferred.
        approach_type = approaches._get_approach_type(approach_slice,
                                                      None, touch_and_gos,
                                                      go_arounds)
        self.assertEqual(approach_type, 'TOUCH_AND_GO')
        # All 3, Heading preferred.
        approach_type = approaches._get_approach_type(approach_slice,
                                                      landing_hdg_kpvs,
                                                      touch_and_gos, go_arounds)
        self.assertEqual(approach_type, 'LANDING')
        # No KPVs/KTIs within slice.
        approach_slice = slice(100, 200)
        approach_type = approaches._get_approach_type(approach_slice,
                                                      landing_hdg_kpvs,
                                                      touch_and_gos, go_arounds)
        self.assertEqual(approach_type, None)
        
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway, get_nearest_airport):
        approaches = Approaches()
        approaches.set_flight_attr = Mock()
        # No approach type due to missing 'Touch And Go', 'Go Around' and
        # 'Heading At Landing' KTI/KPVs.
        start_datetime = A('Start Datetime', value=datetime(1970, 1,1))
        approach_and_landing = S('Approach and Landing',
                                 items=[Section('a', slice(0,10))])
        landing_lat_kpvs = KPV('Latitude At Landing',
                               items=[KeyPointValue(5, 10, 'b')])
        landing_lon_kpvs = KPV('Longitude At Landing',
                               items=[KeyPointValue(5, -2, 'b')])
        landing_hdg_kpvs = KPV('Heading At Landing',
                               items=[KeyPointValue(15, 60, 'a')])
        go_arounds = KTI('Go Around', items=[KeyTimeInstance(25, 'Go Around')])
        touch_and_gos = KTI('Touch And Go',
                            items=[KeyTimeInstance(35, 'Touch And Go')])
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, None)
        approaches.set_flight_attr.assert_called_once_with([])
        # Go Around KTI exists within slice.
        get_nearest_airport.return_value = {'id': 1}
        go_arounds = KTI('Go Around', items=[KeyTimeInstance(5, 'Go Around')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, None)
        expected_datetime = datetime(1970, 1, 1, 0, 0,
                                     approach_and_landing[0].slice.stop) # 10 seconds offset.
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'GO_AROUND', 'runway': None,
              'datetime': expected_datetime}])
        get_nearest_airport.assert_called_once_with(10, -2)
        # Touch And Go KTI exists within the slice.
        touch_and_gos = KTI('Touch And Go',
                            items=[KeyTimeInstance(5, 'Touch And Go')])
        go_arounds = KTI('Go Around', items=[KeyTimeInstance(25, 'Go Around')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, None)
        expected_datetime = datetime(1970, 1, 1, 0, 0,
                                     approach_and_landing[0].slice.stop) # 10 seconds offset.
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'TOUCH_AND_GO', 'runway': None, 'datetime':
              expected_datetime}])
        get_nearest_airport.assert_called_with(10, -2)
        # Use 'Heading At Low Point Of Approach' to query for runway.
        get_nearest_runway.return_value = {'identifier': '06L'}
        approach_hdg_kpvs = KPV('Heading At Low Point Of Approach',
                                items=[KeyPointValue(5, 25, 'a'),
                                       KeyPointValue(12, 35, 'b')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None,
                          approach_hdg_kpvs, None, None)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'TOUCH_AND_GO', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_airport.assert_called_with(10, -2)
        get_nearest_runway.assert_called_once_with(1, 25)
        # Landing Heading KPV exists within slice.
        touch_and_gos = KTI('Touch And Go',
                            items=[KeyTimeInstance(35, 'Touch And Go')])
        landing_hdg_kpvs = KPV('Heading At Landing',
                               items=[KeyPointValue(5, 60, 'a')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, None)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_airport.assert_called_with(10, -2)
        get_nearest_runway.assert_called_with(1, 60)
        # Do not use Latitude and Longitude when requesting runway if Precise
        # precisioning is False.
        precision = A('Precise Positioning', value=False)
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, precision)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_airport.assert_called_with(10, -2)
        get_nearest_runway.assert_called_with(1, 60)
        # Pass Latitude and Longitude into get_nearest_runway if 'Precise
        # Positioning' is True.
        precision.value = True
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, None, None, None,
                          None, precision)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_airport.assert_called_with(10, -2)
        get_nearest_runway.assert_called_with(1, 60, latitude=10, longitude=-2)
        # Use Approach Lat Lon KPVs if available.
        approach_lat_kpvs = KPV('Latitude At Low Point On Approach',
                                items=[KeyPointValue(5, 8, 'b')])
        approach_lon_kpvs = KPV('Longitude At Low Point On Approach',
                                items=[KeyPointValue(5, 4, 'b')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds, None,
                          None, approach_lat_kpvs, approach_lon_kpvs, None,
                          None, precision)
        get_nearest_runway.assert_called_with(1, 60, latitude=8, longitude=4)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        # Prefer Landing Lat Lon KPVs if both Landing and Approach are provided.
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, approach_lat_kpvs,
                          approach_lon_kpvs, None, None, precision)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_runway.assert_called_with(1, 60, latitude=10, longitude=-2)
        # Use 'ILS Frequency On Approach' to query for runway if available.
        precision.value = False
        approach_ilsfreq_kpvs = KPV('ILS Frequency on Approach',
                                    items=[KeyPointValue(5, 330150, 'b')])
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, approach_lat_kpvs,
                          approach_lon_kpvs, None, approach_ilsfreq_kpvs,
                          precision)
        approaches.set_flight_attr.assert_called_once_with(\
            [{'airport': 1, 'type': 'LANDING', 'runway': '06L',
              'datetime': expected_datetime}])
        get_nearest_runway.assert_called_with(1, 60, ilsfreq=330150)
        # Airport cannot be found.
        get_nearest_airport.side_effect = NotFoundError('', '')
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, approach_lat_kpvs,
                          approach_lon_kpvs, None, approach_ilsfreq_kpvs,
                          precision)
        approaches.set_flight_attr.assert_called_once_with([])
        # Runway cannot be found.
        get_nearest_runway.side_effect = NotFoundError('', '')
        approaches.set_flight_attr = Mock()
        approaches.derive(start_datetime, approach_and_landing,
                          landing_hdg_kpvs, touch_and_gos, go_arounds,
                          landing_lat_kpvs, landing_lon_kpvs, approach_lat_kpvs,
                          approach_lon_kpvs, None, approach_ilsfreq_kpvs,
                          precision)
        approaches.set_flight_attr.assert_called_once_with([])


class TestDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Duration.get_operational_combinations(),
                         [('FDR Takeoff Datetime', 'FDR Landing Datetime')])
    
    def test_derive(self):
        duration = Duration()
        duration.set_flight_attr = Mock()
        takeoff_dt = A('FDR Takeoff Datetime', value=datetime(1970, 1, 1, 0, 1, 0))
        landing_dt = A('FDR Landing Datetime', value=datetime(1970, 1, 1, 0, 2, 30))
        duration.derive(takeoff_dt, landing_dt)
        duration.set_flight_attr.assert_called_once_with(90)


class TestFlightID(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FlightID.get_operational_combinations(),
                         [('AFR Flight ID',)])
    
    def test_derive(self):
        afr_flight_id = A('AFR Flight ID', value=10245)
        flight_id = FlightID()
        flight_id.set_flight_attr = Mock()
        flight_id.derive(afr_flight_id)
        flight_id.set_flight_attr.assert_called_once_with(10245)


class TestFlightNumber(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FlightNumber.get_operational_combinations(),
                         [('Flight Number',)])
    
    def test_derive(self):
        flight_number_param = P('Flight Number',
                                array=np.ma.masked_array(['10 2H', '102H',
                                                          '102H']))
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('102H')
        

class TestLandingAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingAirport.get_operational_combinations(),
                         [('Latitude At Landing', 'Longitude At Landing')])
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = NotFoundError('Not Found.')
        latitude = KPV('Latitude At Landing',
                       items=[KeyPointValue(12, 0.5, 'a'),
                              KeyPointValue(32, 0.9, 'a'),])
        longitude = KPV('Longitude At Landing',
                        items=[KeyPointValue(12, 7.1, 'a'),
                               KeyPointValue(32, 8.4, 'a')])
        landing_airport = LandingAirport()
        landing_airport.set_flight_attr = Mock()
        landing_airport.derive(latitude, longitude)
        get_nearest_airport.assert_called_once_with(0.9, 8.4)
        landing_airport.set_flight_attr.assert_called_once_with(None)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        airport_info = {'id': 123}
        latitude = KPV('Latitude At Landing',
                       items=[KeyPointValue(12, 0.5, 'a'),
                              KeyPointValue(32, 0.9, 'a'),])
        longitude = KPV('Longitude At Landing',
                        items=[KeyPointValue(12, 7.1, 'a'),
                               KeyPointValue(32, 8.4, 'a')])
        landing_airport = LandingAirport()
        landing_airport.set_flight_attr = Mock()
        get_nearest_airport.return_value = airport_info
        landing_airport.set_flight_attr = Mock()
        landing_airport.derive(latitude, longitude)
        get_nearest_airport.assert_called_once_with(0.9, 8.4)
        landing_airport.set_flight_attr.assert_called_once_with(airport_info)


class TestLandingDatetime(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingDatetime.get_operational_combinations(),
                         [('Start Datetime', 'Touchdown')])
    
    def test_derive(self):
        landing_datetime = LandingDatetime()
        landing_datetime.set_flight_attr = Mock()
        start_datetime = A('Start Datetime', datetime(1970, 1, 1))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(12, 'a'),
                                            KeyTimeInstance(30, 'b')])
        touchdown.frequency = 0.5
        landing_datetime.derive(start_datetime, touchdown)
        expected_datetime = datetime(1970, 1, 1, 0, 0, 15)
        landing_datetime.set_flight_attr.assert_called_once_with(\
            expected_datetime)
        touchdown = KTI('Touchdown')
        landing_datetime.set_flight_attr = Mock()
        landing_datetime.derive(start_datetime, touchdown)
        landing_datetime.set_flight_attr.assert_called_once_with(None)


class TestLandingFuel(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingFuel.get_operational_combinations(),
                         [('AFR Landing Fuel',), ('Fuel Qty At Touchdown',),
                          ('AFR Landing Fuel', 'Fuel Qty At Touchdown')])
    
    def test_derive(self):
        landing_fuel = LandingFuel()
        landing_fuel.set_flight_attr = Mock()
        # Only 'AFR Takeoff Fuel' dependency.
        afr_landing_fuel = A('AFR Landing Fuel', value=100)
        landing_fuel.derive(afr_landing_fuel, None)
        landing_fuel.set_flight_attr.assert_called_once_with(100)
        # Only 'Fuel Qty At Liftoff' dependency.
        fuel_qty_at_touchdown = KPV('Fuel Qty At Touchdown',
                                    items=[KeyPointValue(87, 160),
                                           KeyPointValue(132, 200)])
        landing_fuel.set_flight_attr = Mock()
        landing_fuel.derive(None, fuel_qty_at_touchdown)
        landing_fuel.set_flight_attr.assert_called_once_with(200)
        # Both, 'AFR Takeoff Fuel' used.
        landing_fuel.set_flight_attr = Mock()
        landing_fuel.derive(afr_landing_fuel, fuel_qty_at_touchdown)
        landing_fuel.set_flight_attr.assert_called_once_with(100)


class TestLandingGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingGrossWeight.get_operational_combinations(),
                         [('Gross Weight At Touchdown',)])
    
    def test_derive(self):
        landing_gross_weight = LandingGrossWeight()
        landing_gross_weight.set_flight_attr = Mock()
        touchdown_gross_weight = KPV('Gross Weight At Touchdown',
                                     items=[KeyPointValue(5, 15, 'a'),
                                            KeyPointValue(12, 120, 'b')])
        landing_gross_weight.derive(touchdown_gross_weight)
        landing_gross_weight.set_flight_attr.assert_called_once_with(120)


class TestLandingPilot(unittest.TestCase):
    def test_can_operate(self):
        opts = LandingPilot.get_operational_combinations()
        self.assertTrue(('Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown') in opts)
        
            [('Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'), 
             ('Roll (Capt)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Roll (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Landing', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Roll (Capt)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Pitch (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Roll (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Landing', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Roll (Capt)', 'Pitch (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Roll (Capt)', 'Roll (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Roll (Capt)', 'Landing', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (FO)', 'Roll (FO)', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Roll (FO)',
              'Landing'),
             ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
              'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Roll (Capt)', 'Roll (FO)',
              'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Roll (Capt)', 'Landing',
              'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'),
             ('Pitch (Capt)', 'Pitch (FO)', 'Roll (FO)',
              'Autopilot Engaged 1 At Touchdown',
              'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Pitch (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Roll (Capt)', 'Pitch (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Roll (Capt)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown'), ('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)', 'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown', 'Autopilot Engaged 2 At Touchdown')])
    
    def test_derive(self):
        node = LandingPilot()
        node.set_flight_attr = Mock()
        # Only Autopilot available, neither engaged.
        # Q: What should discrete values be?
        autopilot1 = KPV('Autopilot Engaged 1 At Touchdown', value=0)
        autopilot2 = KPV('Autopilot Engaged 2 At Touchdown', value=0)
        node.derive(None, None, None, None, None, autopilot1, autopilot2)
        node.set_flight_attr.assert_called_once_with(None)
        # Autopilot 1 Engaged.
        autopilot1 = KPV('Autopilot Engaged 1 At Touchdown', value=1)
        autopilot2 = KPV('Autopilot Engaged 2 At Touchdown', value=0)
        node.set_flight_attr = Mock()
        node.derive(None, None, None, None, None, autopilot1, autopilot2)
        node.set_flight_attr.assert_called_once_with('Captain')
        # Autopilot 2 Engaged.
        autopilot1 = KPV('Autopilot Engaged 1 At Touchdown', value=0)
        autopilot2 = KPV('Autopilot Engaged 2 At Touchdown', value=1)
        node.set_flight_attr = Mock()
        node.derive(None, None, None, None, None, autopilot1, autopilot2)
        node.set_flight_attr.assert_called_once_with('First Officer')
        # Both Autopilots Engaged.
        autopilot1 = KPV('Autopilot Engaged 1 At Touchdown', value=1)
        autopilot2 = KPV('Autopilot Engaged 2 At Touchdown', value=1)
        node.set_flight_attr = Mock()
        node.derive(None, None, None, None, None, autopilot1, autopilot2)
        node.set_flight_attr.assert_called_once_with(None)




class TestLandingRunway(unittest.TestCase):
    def test_can_operate(self):
        '''
        There may be a neater way to test this, but at least it's verbose.
        '''
        combinations = LandingRunway.get_operational_combinations()
        self.assertEqual(len(combinations), 16)
        self.assertEqual(combinations[0], ('Approach And Landing',
                                           'Heading At Landing',
                                           'FDR Landing Airport'))
        self.assertEqual(combinations[-1], ('Approach And Landing',
                                            'Heading At Landing',
                                            'FDR Landing Airport',
                                            'Latitude At Landing',
                                            'Longitude At Landing',
                                            'ILS Frequency On Approach',
                                            'Precise Positioning'))
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        runway_info = {'ident': '27L', 'runways': [{'length': 20}]}
        get_nearest_runway.return_value = runway_info
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr = Mock()
        # Airport and Heading At Landing arguments.
        airport = A('Takeoff Airport')
        airport.value = {'id':25}
        landing_hdg = KPV('Heading At Landing',
                          items=[KeyPointValue(15, 20.0, 'a')])
        approach_and_landing = S('Approach and Landing',
                                 items=[Section('b', slice(14, 20))])
        
        landing_runway.derive(approach_and_landing, landing_hdg, airport)
        get_nearest_runway.assert_called_once_with(25, 20.0)
        landing_runway.set_flight_attr.assert_called_once_with(runway_info)
        approach_ilsfreq = KPV('ILS Frequency On Approach',
                               items=[KeyPointValue(15, 330150, 'a')])
        landing_runway.set_flight_attr = Mock()
        landing_runway.derive(approach_and_landing, landing_hdg, airport,
                              None, None, approach_ilsfreq, None)
        get_nearest_runway.assert_called_with(25, 20.0, ilsfreq=330150)
        landing_runway.set_flight_attr.assert_called_once_with(runway_info)
        
        # Airport, Landing Heading, Latitude, Longitude and Precision
        # arguments. Latitude and Longitude are only passed with all these
        # parameters available and Precise Positioning is True.
        latitude = KPV('Latitude At Landing',
                       items=[KeyPointValue(15, 1.2, 'DATA')])
        longitude = KPV('Latitude At Landing',
                        items=[KeyPointValue(15, 3.2, 'DATA')])
        precision = A('Precision')
        precision.value = False
        landing_runway.set_flight_attr = Mock()
        landing_runway.derive(approach_and_landing, landing_hdg, airport, latitude,
                              longitude, approach_ilsfreq, precision)
        get_nearest_runway.assert_called_with(25, 20.0, ilsfreq=330150)
        landing_runway.set_flight_attr.assert_called_once_with(runway_info)
        precision.value = True
        landing_runway.set_flight_attr = Mock()
        landing_runway.derive(approach_and_landing, landing_hdg, airport,
                              latitude, longitude, approach_ilsfreq, precision)
        get_nearest_runway.assert_called_with(25, 20.0, ilsfreq=330150,
                                              latitude=1.2, longitude=3.2)


class TestOffBlocksDatetime(unittest.TestCase):
    def test_derive_without_turning(self):
        # Empty 'Turning'.
        turning = S('Turning')
        off_blocks_datetime = OffBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # Only 'Turning In Air'.
        turning = S('Turning', items=[KeyPointValue(name='Turning In Air',
                                                    slice=slice(0, 100))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(20, 60))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(20)
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(10, 20)),
                                      KeyPointValue(name='Turning In Air',
                                                    slice=slice(20, 60)),
                                      KeyPointValue(name='Turning On Ground',
                                                    slice=slice(70, 90))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(10)


class TestOnBlocksDatetime(unittest.TestCase):
    def test_derive_without_turning(self):
        # Empty 'Turning'.
        turning = S('Turning')
        off_blocks_datetime = OnBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # Only 'Turning In Air'.
        turning = S('Turning', items=[KeyPointValue(name='Turning In Air',
                                                    slice=slice(0, 100))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(20, 60))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(60)
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(10, 20)),
                                      KeyPointValue(name='Turning In Air',
                                                    slice=slice(20, 60)),
                                      KeyPointValue(name='Turning On Ground',
                                                    slice=slice(70, 90))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(90)


class TestTakeoffAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual([('Liftoff', 'Latitude', 'Longitude')],
                         TakeoffAirport.get_operational_combinations())
        
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = NotFoundError('Not Found.')
        liftoff = KTI('Liftoff')
        liftoff.create_kti(1)
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(liftoff, latitude, longitude)
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        self.assertFalse(takeoff_airport.set_flight_attr.called)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        airport_info = {'id': 123}
        get_nearest_airport.return_value = airport_info
        liftoff = KTI('Liftoff')
        liftoff.create_kti(1)
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(liftoff, latitude, longitude)
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        takeoff_airport.set_flight_attr.assert_called_once_with(airport_info)


class TestTakeoffDatetime(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffDatetime.get_operational_combinations(),
                         [('Liftoff', 'Start Datetime')])
    
    def test_derive(self):
        takeoff_dt = TakeoffDatetime()
        takeoff_dt.set_flight_attr = Mock()
        start_dt = A('Start Datetime', value=datetime(1970, 1, 1))
        liftoff = KTI('Liftoff', frequency=0.25,
                      items=[KeyTimeInstance(100, 'a')])
        takeoff_dt.derive(liftoff, start_dt)
        takeoff_dt.set_flight_attr.assert_called_once_with(\
            datetime(1970, 1, 1, 0, 0, 25))
        liftoff = KTI('Liftoff', frequency=0.25, items=[])
        takeoff_dt.set_flight_attr = Mock()
        takeoff_dt.derive(liftoff, start_dt)
        takeoff_dt.set_flight_attr.assert_called_once_with(None)


class TestTakeoffFuel(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffFuel.get_operational_combinations(),
                         [('AFR Takeoff Fuel',), ('Fuel Qty At Liftoff',),
                          ('AFR Takeoff Fuel', 'Fuel Qty At Liftoff')])
    
    def test_derive(self):
        takeoff_fuel = TakeoffFuel()
        takeoff_fuel.set_flight_attr = Mock()
        # Only 'AFR Takeoff Fuel' dependency.
        afr_takeoff_fuel = A('AFR Takeoff Fuel', value=100)
        takeoff_fuel.derive(afr_takeoff_fuel, None)
        takeoff_fuel.set_flight_attr.assert_called_once_with(100)
        # Only 'Fuel Qty At Liftoff' dependency.
        fuel_qty_at_liftoff = KPV('Fuel Qty At Liftoff',
                                  items=[KeyPointValue(132, 200)])
        takeoff_fuel.set_flight_attr = Mock()
        takeoff_fuel.derive(None, fuel_qty_at_liftoff)
        takeoff_fuel.set_flight_attr.assert_called_once_with(200)
        # Both, 'AFR Takeoff Fuel' used.
        takeoff_fuel.set_flight_attr = Mock()
        takeoff_fuel.derive(afr_takeoff_fuel, fuel_qty_at_liftoff)
        takeoff_fuel.set_flight_attr.assert_called_once_with(100)


class TestTakeoffGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffGrossWeight.get_operational_combinations(),
                         [('Gross Weight At Liftoff',)])
    
    def test_derive(self):
        takeoff_gross_weight = TakeoffGrossWeight()
        takeoff_gross_weight.set_flight_attr = Mock()
        liftoff_gross_weight = KPV('Gross Weight At Liftoff',
                                   items=[KeyPointValue(5, 135, 'a'),
                                          KeyPointValue(12, 120, 'b')])
        takeoff_gross_weight.derive(liftoff_gross_weight)
        takeoff_gross_weight.set_flight_attr.assert_called_once_with(135)


class TestTakeoffRunway(unittest.TestCase):
    def test_can_operate(self):
        '''
        There may be a neater way to test this, but at least it's verbose.
        '''
        expected = \
        [('FDR Takeoff Airport', 'Heading At Takeoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Latitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Longitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude', 'Longitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude', 
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude', 
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Latitude', 
          'Longitude'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Latitude',
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Longitude',
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude', 'Longitude',
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Liftoff', 'Latitude',
          'Longitude', 'Precise Positioning')]
        self.assertEqual(TakeoffRunway.get_operational_combinations(),
                         expected)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        runway_info = {'ident': '27L', 'runways': [{'length': 20}]}
        get_nearest_runway.return_value = runway_info
        takeoff_runway = TakeoffRunway()
        takeoff_runway.set_flight_attr = Mock()
        # Airport and Heading At Takeoff arguments.
        airport = A('Takeoff Airport')
        airport.value = {'id':25}
        takeoff_heading = KPV('Heading At Takeoff')
        takeoff_heading.create_kpv(1, 20.0)
        takeoff_runway.derive(airport, takeoff_heading)
        get_nearest_runway.assert_called_with(25, 20.0)
        takeoff_runway.set_flight_attr.assert_called_once_with(runway_info)
        # Airport, Heading At Takeoff, Liftoff, Latitude, Longitude and Precision
        # arguments. Latitude and Longitude are only passed with all these
        # parameters available and Precise Positioning is True.
        liftoff = KTI('Liftoff')
        liftoff.create_kti(1)
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        precision = A('Precision')
        precision.value = True
        takeoff_runway.derive(airport, takeoff_heading, liftoff, latitude,
                              longitude, precision)
        get_nearest_runway.assert_called_with(25, 20.0, latitude=4.0,
                                              longitude=3.0)
        takeoff_runway.set_flight_attr.assert_called_with(runway_info)
        # When Precise Positioning's value is False, Latitude and Longitude
        # are not used.
        precision.value = False
        takeoff_runway.derive(airport, takeoff_heading, liftoff, latitude,
                              longitude, precision)
        get_nearest_runway.assert_called_with(25, 20.0)
        takeoff_runway.set_flight_attr.assert_called_with(runway_info)


class TestFlightType(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FlightType.get_operational_combinations(),
          [('Fast', 'Liftoff', 'Touchdown'),
           ('AFR Type', 'Fast', 'Liftoff', 'Touchdown'),
           ('Fast', 'Liftoff', 'Touchdown', 'Touch And Go'),
           ('Fast', 'Liftoff', 'Touchdown', 'Groundspeed'),
           ('AFR Type', 'Fast', 'Liftoff', 'Touchdown', 'Touch And Go'),
           ('AFR Type', 'Fast', 'Liftoff', 'Touchdown', 'Groundspeed'),
           ('Fast', 'Liftoff', 'Touchdown', 'Touch And Go', 'Groundspeed'),
           ('AFR Type', 'Fast', 'Liftoff', 'Touchdown', 'Touch And Go', 
            'Groundspeed')])
    
    def test_derive(self):
        '''
        Tests every flow, but does not test every conceivable set of arguments.
        '''
        type_node = FlightType()
        type_node.set_flight_attr = Mock()
        # Liftoff and Touchdown.
        fast = S('Fast', items=[slice(5,10)])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'a')])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10, 'x')])
        type_node.derive(None, fast, liftoffs, touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with('COMPLETE')
        # Would be 'COMPLETE', but 'AFR Type' overrides it.
        afr_type = A('AFR Type', value='FERRY')
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, liftoffs, touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with('FERRY')
        # Liftoff missing.
        empty_liftoffs = KTI('Liftoff')
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, empty_liftoffs, touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with('TOUCHDOWN_ONLY')
        # Touchdown missing.
        empty_touchdowns = KTI('Touchdown')
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, liftoffs, empty_touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with('LIFTOFF_ONLY')
        # Liftoff and Touchdown missing, only Fast.
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, empty_liftoffs, empty_touchdowns, None,
                         None)
        type_node.set_flight_attr.assert_called_once_with('REJECTED_TAKEOFF')
        # Liftoff, Touchdown and Fast missing.
        empty_fast = fast = S('Fast')
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, None)
        type_node.set_flight_attr.assert_called_once_with('ENGINE_RUN_UP')
        # Liftoff, Touchdown and Fast missing, Groundspeed changes.
        groundspeed = P('Groundspeed', np.ma.arange(20))
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, groundspeed)
        type_node.set_flight_attr.assert_called_once_with('GROUND_RUN')
        # Liftoff, Touchdown and Fast missing, Groundspeed stays the same.
        groundspeed = P('Groundspeed', np.ma.masked_array([0] * 20))
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, groundspeed)
        type_node.set_flight_attr.assert_called_once_with('ENGINE_RUN_UP',)
        # Liftoff after Touchdown.
        late_liftoffs = KTI('Liftoff', items=[KeyTimeInstance(20, 'a')])
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, late_liftoffs, touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with(\
            'TOUCHDOWN_BEFORE_LIFTOFF')
        # Touch and Go before Touchdown.
        afr_type = A('AFR Type', value='TRAINING')
        touch_and_gos = KTI('Touch and Gos', items=[KeyTimeInstance(7, 'a')])
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, liftoffs, touchdowns, touch_and_gos,
                         None)
        type_node.set_flight_attr.assert_called_once_with('TRAINING')
        # Touch and Go after Touchdown.
        afr_type = A('AFR Type', value='TRAINING')
        touch_and_gos = KTI('Touch and Gos', items=[KeyTimeInstance(15, 'a')])
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, liftoffs, touchdowns, touch_and_gos,
                         None)
        type_node.set_flight_attr.assert_called_once_with('LIFTOFF_ONLY')
        
        
