import numpy as np
import unittest

from datetime import datetime, timedelta
from mock import Mock, patch

from analysis_engine import __version__
from analysis_engine.api_handler import NotFoundError
from analysis_engine.node import (
    A, KPV, KTI, P, S,
    KeyPointValue,
    KeyTimeInstance,
    Section,
)
from analysis_engine.settings import CONTROLS_IN_USE_TOLERANCE
from analysis_engine.flight_attribute import (
    DeterminePilot,
    Duration,
    FlightID,
    FlightNumber,
    FlightType,
    InvalidFlightType,
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
    Version,
)


def setUpModule():
    from analysis_engine import settings
    settings.API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal'


class NodeTest(object):
    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            self.assertEqual(
                self.node_class.get_operational_combinations(),
                self.operational_combinations,
            )


class TestDeterminePilot(unittest.TestCase):
    def test__autopilot_engaged(self):
        determine_pilot = DeterminePilot()
        # Q: What should discrete values be?
        autopilot1 = KeyPointValue('AP Engaged 1 At Touchdown', value=0)
        autopilot2 = KeyPointValue('AP Engaged 2 At Touchdown', value=0)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, None)
        # Autopilot 1 Engaged.
        autopilot1 = KeyPointValue('AP Engaged 1 At Touchdown', value=1)
        autopilot2 = KeyPointValue('AP Engaged 2 At Touchdown', value=0)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, 'Captain')
        # Autopilot 2 Engaged.
        autopilot1 = KeyPointValue('AP Engaged 1 At Touchdown', value=0)
        autopilot2 = KeyPointValue('AP Engaged 2 At Touchdown', value=1)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, 'First Officer')
        # Both Autopilots Engaged.
        autopilot1 = KeyPointValue('AP Engaged 1 At Touchdown', value=1)
        autopilot2 = KeyPointValue('AP Engaged 2 At Touchdown', value=1)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, None)

    def test__pitch_roll_changed(self):
        determine_pilot = DeterminePilot()
        slice_ = slice(0,3)
        below_tolerance = np.ma.array([CONTROLS_IN_USE_TOLERANCE/4.0, 0, 0,
                                       CONTROLS_IN_USE_TOLERANCE/2.0, 0, 0])
        above_tolerance = np.ma.array([CONTROLS_IN_USE_TOLERANCE*4, 0, 0,
                                       CONTROLS_IN_USE_TOLERANCE*2, 0, 0])
        # Both pitch and roll below tolerance.
        pitch = below_tolerance
        roll = below_tolerance
        pilot = determine_pilot._pitch_roll_changed(slice_, pitch, roll)
        self.assertFalse(pilot)
        # Pitch above tolerance.
        pitch = above_tolerance
        roll = below_tolerance
        pilot = determine_pilot._pitch_roll_changed(slice_, pitch, roll)
        self.assertTrue(pilot)
        # Roll above tolerance.
        pitch = below_tolerance
        roll = above_tolerance
        pilot = determine_pilot._pitch_roll_changed(slice_, pitch, roll)
        self.assertTrue(pilot)
        # Pitch and Roll above tolerance.
        pitch = above_tolerance
        roll = above_tolerance
        pilot = determine_pilot._pitch_roll_changed(slice_, pitch, roll)
        self.assertTrue(pilot)
        # Pitch and Roll above tolerance outside of slice.
        slice_ = slice(1,3)
        pitch = above_tolerance
        roll = above_tolerance
        pilot = determine_pilot._pitch_roll_changed(slice_, pitch, roll)
        self.assertFalse(pilot)

    def test__controls_in_use(self):
        # Instantiating subclass of DeterminePilot since it will gain a logging
        # methods through multiple inheritance.
        determine_pilot = LandingPilot()
        pitch_captain = Mock()
        roll_captain = Mock()
        pitch_fo = Mock()
        roll_fo = Mock()
        section = Section('Takeoff', slice(0,3), 0, 3)
        determine_pilot._pitch_roll_changed = Mock()
        # Neither pilot's controls changed.
        in_use = [False, False]
        def side_effect(*args, **kwargs):
            return in_use.pop()
        determine_pilot._pitch_roll_changed.side_effect = side_effect
        pilot = determine_pilot._controls_in_use(pitch_captain, roll_captain,
                                                 pitch_fo, roll_fo, section)
        self.assertEqual(determine_pilot._pitch_roll_changed.call_args_list,
                         [((section.slice, pitch_captain, roll_captain), {}),
                          ((section.slice, pitch_fo, roll_fo), {})])
        self.assertEqual(pilot, None)
         # Captain's controls changed.
        in_use = [False, True]
        pilot = determine_pilot._controls_in_use(pitch_captain, roll_captain,
                                                 pitch_fo, roll_fo, section)
        self.assertEqual(pilot, 'Captain')
        # First Officer's controls changed.
        in_use = [True, False]
        pilot = determine_pilot._controls_in_use(pitch_captain, roll_captain,
                                                 pitch_fo, roll_fo, section)
        self.assertEqual(pilot, 'First Officer')
        # Both controls changed.
        in_use = [True, True]
        pilot = determine_pilot._controls_in_use(pitch_captain, roll_captain,
                                                 pitch_fo, roll_fo, section)
        self.assertEqual(pilot, None)

    def test__determine_pilot(self):
        determine_pilot = DeterminePilot()
        # Controls in use, no takeoff_or_landing.
        takeoff_or_landing = None
        pitch_captain = Mock()
        pitch_captain.array = Mock()
        roll_captain = Mock()
        roll_captain.array = Mock()
        pitch_fo = Mock()
        pitch_fo.array = Mock()
        roll_fo = Mock()
        roll_fo.array = Mock()
        determine_pilot.set_flight_attr = Mock()
        pilot_returned = determine_pilot._determine_pilot(pitch_captain,
                                                          roll_captain, pitch_fo,
                                                          roll_fo, takeoff_or_landing, None,
                                                          None)
        self.assertEqual(pilot_returned, None)
        # Controls in use with takeoff_or_landing. Pilot cannot be discerned.
        determine_pilot._controls_in_use = Mock()
        pilot = None
        determine_pilot._controls_in_use.return_value = pilot
        takeoff_or_landing = Mock()
        determine_pilot.set_flight_attr = Mock()
        pilot_returned = determine_pilot._determine_pilot(pitch_captain, roll_captain, pitch_fo,
                                                          roll_fo, takeoff_or_landing, None,
                                                          None)
        determine_pilot._controls_in_use.assert_called_once_with(
            pitch_captain.array, roll_captain.array, pitch_fo.array,
            roll_fo.array, takeoff_or_landing)
        self.assertEqual(pilot, pilot_returned)
        # Controls in use with takeoff_or_landing. Pilot returned
        determine_pilot._controls_in_use = Mock()
        pilot = 'Captain'
        determine_pilot._controls_in_use.return_value = pilot
        determine_pilot.set_flight_attr = Mock()
        pilot_returned = determine_pilot._determine_pilot(pitch_captain, roll_captain, pitch_fo,
                                                          roll_fo, takeoff_or_landing, None,
                                                          None)
        self.assertEqual(pilot_returned, pilot)
        # Only Autopilot.
        autopilot1 = Mock()
        autopilot2 = Mock()
        determine_pilot._autopilot_engaged = Mock()
        determine_pilot._controls_in_use = Mock()
        pilot = 'Captain'
        determine_pilot._autopilot_engaged.return_value = pilot
        determine_pilot.set_flight_attr = Mock()
        pilot_returned = determine_pilot._determine_pilot(None, None, None, None, None,
                                                          autopilot1, autopilot2)
        determine_pilot._autopilot_engaged.assert_called_once_with(autopilot1,
                                                                 autopilot2)
        self.assertEqual(determine_pilot._controls_in_use.called, False)
        self.assertEqual(pilot_returned, pilot)
        # Controls in Use overrides Autopilot.
        controls_pilot = 'Captain'
        autopilot_pilot = 'First Officer'
        determine_pilot._controls_in_use.return_value = controls_pilot
        determine_pilot._autopilot_engaged.return_value = autopilot_pilot
        pilot_returned = determine_pilot._determine_pilot(pitch_captain, roll_captain, pitch_fo,
                                                          roll_fo, takeoff_or_landing,
                                                          autopilot1, autopilot2)
        self.assertEqual(pilot_returned, controls_pilot)
        # Autopilot is used when Controls in Use does not provide an answer.
        determine_pilot._controls_in_use.return_value = None
        pilot = 'First Officer'
        determine_pilot._autopilot_engaged.return_value = pilot
        pilot_returned = determine_pilot._determine_pilot(pitch_captain, roll_captain, pitch_fo,
                                                         roll_fo, takeoff_or_landing, autopilot1,
                                                         autopilot2)
        self.assertEqual(pilot_returned, pilot)


class TestDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Duration.get_operational_combinations(),
                         [('FDR Takeoff Datetime', 'FDR Landing Datetime')])

    def test_derive(self):
        duration = Duration()
        duration.set_flight_attr = Mock()
        takeoff_dt = A('FDR Takeoff Datetime',
                       value=datetime(1970, 1, 1, 0, 1, 0))
        landing_dt = A('FDR Landing Datetime',
                       value=datetime(1970, 1, 1, 0, 2, 30))
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
                                array=np.ma.masked_array([103, 102,102]))
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('102')

    def test_derive_ascii(self):
        flight_number_param = P('Flight Number',
                                array=np.ma.masked_array(['ABC', 'DEF', 'DEF']))
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('DEF')
        flight_number.set_flight_attr.reset_mock()
        # Entirely masked.
        flight_number_param.array[:] = np.ma.masked
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.called = False

    def test_derive_most_common_positive_float(self):
        flight_number = FlightNumber()

        neg_number_param = P(
            'Flight Number',
            array=np.ma.array([-1,2,-4,10]))
        flight_number.derive(neg_number_param)
        self.assertEqual(flight_number.value, None)

        # TODO: Implement variance checks as below
        ##high_variance_number_param = P(
            ##'Flight Number',
            ##array=np.ma.array([2,2,4,4,4,7,7,7,4,5,4,7,910]))
        ##self.assertRaises(ValueError, flight_number.derive, high_variance_number_param)

        flight_number_param= P(
            'Flight Number',
            array=np.ma.array([2,555.6,444,444,444,444,444,444,888,444,444,444,
                               444,444,444,444,444,7777,9100]))
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('444')


class TestLandingAirport(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = LandingAirport
        self.operational_combinations = [
            ('AFR Landing Airport',),
            ('Latitude At Landing', 'Longitude At Landing'),
            ('Latitude At Landing', 'AFR Landing Airport'),
            ('Longitude At Landing', 'AFR Landing Airport'),
            ('Latitude At Landing', 'Longitude At Landing', 'AFR Landing Airport'),
        ]

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = NotFoundError('Not Found.')
        lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=12, value=0.5),
            KeyPointValue(index=32, value=0.9),
        ])
        lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=12, value=7.1),
            KeyPointValue(index=32, value=8.4),
        ])
        afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that no attribute is created if not found via API:
        apt.derive(lat, lon, None)
        apt.set_flight_attr.assert_called_once_with(None)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(0.9, 8.4)
        get_nearest_airport.reset_mock()
        # Check that the AFR airport was used if not found via API:
        apt.derive(lat, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(0.9, 8.4)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        info = {'id': 123}
        get_nearest_airport.return_value = info
        lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=12, value=0.5),
            KeyPointValue(index=32, value=0.9),
        ])
        lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=12, value=7.1),
            KeyPointValue(index=32, value=8.4),
        ])
        afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the airport returned via API is used for the attribute:
        apt.derive(lat, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(info)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(0.9, 8.4)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_afr_fallback(self, get_nearest_airport):
        info = {'id': '50'}
        get_nearest_airport.return_value = info
        lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=12, value=0.5),
            KeyPointValue(index=32, value=0.9),
        ])
        lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=12, value=7.1),
            KeyPointValue(index=32, value=8.4),
        ])
        afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        apt.derive(None, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(lat, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(None, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'


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
        expected_datetime = datetime(1970, 1, 1, 0, 1)
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
        # Only controls in use parameters.
        self.assertTrue(('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)',
                         'Sidestick Pitch (FO)', 'Sidestick Roll (FO)',
                         'Landing') in opts)
        # Only Autopilot.
        self.assertTrue(('Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown') in opts)
        # Combinations.
        self.assertTrue(('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)',
                         'Sidestick Pitch (FO)', 'Sidestick Roll (FO)',
                         'Landing', 'Autopilot Engaged 1 At Touchdown') in opts)
        self.assertTrue(('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)',
                         'Landing', 'Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown' in opts))
        # All.
        self.assertTrue(('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)',
                         'Sidestick Pitch (FO)', 'Sidestick Roll (FO)',
                         'Landing', 'Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown') in opts)

    def test_derive(self):
        landing_pilot = LandingPilot()
        landings = Mock()
        landings.get_last = Mock()
        last_landing = Mock()
        landings.get_last.return_value = last_landing
        pitch_captain = Mock()
        roll_captain = Mock()
        pitch_fo = Mock()
        roll_fo = Mock()
        autopilot1 = Mock()
        autopilot1.get_last = Mock()
        last_autopilot1 = Mock()
        autopilot1.get_last.return_value = last_autopilot1
        autopilot2 = Mock()
        autopilot2.get_last = Mock()
        last_autopilot2 = Mock()
        autopilot2.get_last.return_value = last_autopilot2
        landing_pilot._determine_pilot = Mock()
        landing_pilot._determine_pilot.return_value = Mock()
        landing_pilot.set_flight_attr = Mock()
        landing_pilot.derive(pitch_captain, roll_captain, pitch_fo, roll_fo,
                             landings, autopilot1, autopilot2)
        self.assertTrue(landings.get_last.called)
        self.assertTrue(autopilot1.get_last.called)
        self.assertTrue(autopilot2.get_last.called)
        landing_pilot._determine_pilot.assert_called_once_with(pitch_captain,
                                                               roll_captain,
                                                               pitch_fo,
                                                               roll_fo,
                                                               last_landing,
                                                               last_autopilot1,
                                                               last_autopilot2)
        landing_pilot.set_flight_attr.assert_called_once_with(\
            landing_pilot._determine_pilot.return_value)


class TestLandingRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = LandingRunway
        self.operational_combination_length = 144
        self.check_operational_combination_length_only = True

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        info = {
            'end': {'latitude': 58.211678, 'longitude': 8.095269},
            'glideslope': {'latitude': 58.198664, 'frequency': '335000M', 'angle': 3.4, 'longitude': 8.080164, 'threshold_distance': 720},
            'id': 8127,
            'identifier': '27L',
            'localizer': {'latitude': 58.212397, 'beam_width': 4.5, 'frequency': '110300M', 'heading': 36, 'longitude': 8.096228},
            'start': {'latitude': 58.196703, 'longitude': 8.075406},
            'strip': {'width': 147, 'length': 6660, 'id': 4064, 'surface': 'ASP'},
        }
        get_nearest_runway.return_value = info
        fdr_apt = A(name='FDR Landing Airport', value={'id': 25})
        afr_apt = None
        lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=16, value=4.0),
            KeyPointValue(index=18, value=6.0),
        ])
        lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=16, value=3.0),
            KeyPointValue(index=18, value=9.0),
        ])
        hdg = KPV(name='Heading At Landing', items=[
            KeyPointValue(index=16, value=60.0),
            KeyPointValue(index=18, value=20.0),
        ])
        precise = A(name='Precise Positioning')
        approaches = S(name='Approach', items=[
            Section(name='Approach', slice=slice(14, 20), start_edge=14, stop_edge=20),
        ])
        ils_freq_on_app = KPV(name='ILS Frequency On Approach', items=[
            KeyPointValue(index=18, value=330150),
        ])
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Test with bare minimum information:
        rwy.derive(fdr_apt, afr_apt, hdg)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, hint='landing')
        get_nearest_runway.reset_mock()
        # Test with ILS frequency:
        rwy.derive(fdr_apt, afr_apt, hdg, None, None, None, approaches,
                   ils_freq_on_app)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, ils_freq=330150, hint='landing')
        get_nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        precise.value = True
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, precise, approaches, ils_freq_on_app)
        rwy.set_flight_attr.assert_called_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, latitude=6.0, longitude=9.0, ils_freq=330150)
        get_nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        precise.value = False
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, precise, approaches, ils_freq_on_app)
        rwy.set_flight_attr.assert_called_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, ils_freq=330150, hint='landing')
        get_nearest_runway.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    def test_derive_afr_fallback(self, get_nearest_runway):
        info = {'ident': '09L'}
        def runway_side_effect(apt, hdg, *args, **kwargs):
            if hdg == 90.0:
                return info
            raise NotFoundError('No runway found.')
        get_nearest_runway.side_effect = runway_side_effect
        fdr_apt = A(name='FDR Landing Airport', value={'id': 50})
        afr_rwy = A(name='AFR Landing Runway', value={'ident': '27R'})
        hdg_a = KPV(name='Heading At Landing', items=[
            KeyPointValue(index=1, value=360.0),
            KeyPointValue(index=2, value=270.0),
        ])
        hdg_b = KPV(name='Heading At Landing', items=[
            KeyPointValue(index=1, value=180.0),
            KeyPointValue(index=2, value=90.0),
        ])
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        rwy.derive(None, None, None)
        rwy.set_flight_attr.assert_called_once_with(None)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(fdr_apt, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, hdg_a)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        # Check wrong heading triggers AFR:
        rwy.derive(fdr_apt, afr_rwy, hdg_a)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(fdr_apt.value['id'], hdg_a.get_last().value, hint='landing')
        get_nearest_runway.reset_mock()
        rwy.derive(fdr_apt, afr_rwy, hdg_b)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(fdr_apt.value['id'], hdg_b.get_last().value, hint='landing')
        get_nearest_runway.reset_mock()


class TestOffBlocksDatetime(unittest.TestCase):
    def test_derive(self):
        # Empty 'Turning'.
        turning = S('Turning On Ground')
        start_datetime = A(name='Start Datetime', value=datetime.now())
        off_blocks_datetime = OffBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning On Ground', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(20, 60))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=20))

        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(10, 20)),
                                      KeyPointValue(name='Turning On Ground',
                                                    slice=slice(70, 90))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=10))


class TestOnBlocksDatetime(unittest.TestCase):
    def test_derive_without_turning(self):
        # Empty 'Turning'.
        turning = S('Turning On Ground')
        start_datetime = A(name='Start Datetime', value=datetime.now())
        off_blocks_datetime = OnBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning On Ground',
                    items=[KeyPointValue(name='Turning On Ground',
                                         slice=slice(20, 60))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=60))
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(10, 20)),
                                      KeyPointValue(name='Turning In Air',
                                                    slice=slice(20, 60)),
                                      KeyPointValue(name='Turning On Ground',
                                                    slice=slice(70, 90))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=90))


class TestTakeoffAirport(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffAirport
        self.operational_combinations = [
            ('AFR Takeoff Airport',),
            ('Latitude At Takeoff', 'Longitude At Takeoff'),
            ('Latitude At Takeoff', 'AFR Takeoff Airport'),
            ('Longitude At Takeoff', 'AFR Takeoff Airport'),
            ('Latitude At Takeoff', 'Longitude At Takeoff', 'AFR Takeoff Airport'),
        ]

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = NotFoundError('Not Found.')
        lat = KPV(name='Latitude At Takeoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Takeoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that no attribute is created if not found via API:
        apt.derive(lat, lon, None)
        apt.set_flight_attr.assert_called_once_with(None)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()
        # Check that the AFR airport was used if not found via API:
        apt.derive(lat, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        info = {'id': 123}
        get_nearest_airport.return_value = info
        lat = KPV(name='Latitude At Takeoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Takeoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the airport returned via API is used for the attribute:
        apt.derive(lat, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(info)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive_afr_fallback(self, get_nearest_airport):
        info = {'id': '50'}
        get_nearest_airport.return_value = info
        lat = KPV(name='Latitude At Takeoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Takeoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        apt.derive(None, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(lat, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(None, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'


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
            datetime(1970, 1, 1, 0, 6, 40))
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


'''
class TestTakeoffPilot(unittest.TestCase):
    def test_can_operate(self):
        opts = TakeoffPilot.get_operational_combinations()
        # Only controls in use parameters.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff') in opts)
        # Only Autopilot.
        self.assertTrue(('AP Engaged 1 At Liftoff',
                         'AP Engaged 2 At Liftoff') in opts)
        # Combinations.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff',
                         'AP Engaged 1 At Liftoff') in opts)
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Takeoff',
                         'AP Engaged 1 At Liftoff',
                         'AP Engaged 2 At Liftoff' in opts))
        # All.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff', 'AP Engaged 1 At Liftoff',
                         'AP Engaged 2 At Liftoff') in opts)

    def test_derive(self):
        takeoff_pilot = TakeoffPilot()
        takeoffs = Mock()
        takeoffs.get_first = Mock()
        first_takeoff = Mock()
        takeoffs.get_first.return_value = first_takeoff
        pitch_captain = Mock()
        roll_captain = Mock()
        pitch_fo = Mock()
        roll_fo = Mock()
        autopilot1 = Mock()
        autopilot1.get_first = Mock()
        first_autopilot1 = Mock()
        autopilot1.get_first.return_value = first_autopilot1
        autopilot2 = Mock()
        autopilot2.get_first = Mock()
        first_autopilot2 = Mock()
        autopilot2.get_first.return_value = first_autopilot2
        takeoff_pilot._determine_pilot = Mock()
        takeoff_pilot._determine_pilot.return_value = Mock()
        takeoff_pilot.set_flight_attr = Mock()
        takeoff_pilot.derive(pitch_captain, roll_captain, pitch_fo, roll_fo,
                             takeoffs, autopilot1, autopilot2)
        self.assertTrue(takeoffs.get_first.called)
        self.assertTrue(autopilot1.get_first.called)
        self.assertTrue(autopilot2.get_first.called)
        takeoff_pilot._determine_pilot.assert_called_once_with(pitch_captain,
                                                               roll_captain,
                                                               pitch_fo,
                                                               roll_fo,
                                                               first_takeoff,
                                                               first_autopilot1,
                                                               first_autopilot2)
        takeoff_pilot.set_flight_attr.assert_called_once_with(\
            takeoff_pilot._determine_pilot.return_value)
'''


class TestTakeoffRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffRunway
        self.operational_combinations = [
            ('AFR Takeoff Runway',),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway'),
            ('FDR Takeoff Airport', 'Heading At Takeoff'),
            ('AFR Takeoff Runway', 'Heading At Takeoff'),
            ('AFR Takeoff Runway', 'Latitude At Takeoff'),
            ('AFR Takeoff Runway', 'Longitude At Takeoff'),
            ('AFR Takeoff Runway', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Longitude At Takeoff'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Latitude At Takeoff', 'Longitude At Takeoff'),
            ('AFR Takeoff Runway', 'Latitude At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude At Takeoff', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Latitude At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Latitude At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
            ('FDR Takeoff Airport', 'AFR Takeoff Runway', 'Heading At Takeoff', 'Latitude At Takeoff', 'Longitude At Takeoff', 'Precise Positioning'),
        ]

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        info = {'ident': '27L', 'length': 20}
        get_nearest_runway.return_value = info
        fdr_apt = A(name='FDR Takeoff Airport', value={'id': 25})
        afr_apt = None
        lat = KPV(name='Latitude At Takeoff', items=[
            KeyPointValue(index=1, value=4.0),
            KeyPointValue(index=2, value=6.0),
        ])
        lon = KPV(name='Longitude At Takeoff', items=[
            KeyPointValue(index=1, value=3.0),
            KeyPointValue(index=2, value=9.0),
        ])
        hdg = KPV(name='Heading At Takeoff', items=[
            KeyPointValue(index=1, value=20.0),
            KeyPointValue(index=2, value=60.0),
        ])
        precise = A(name='Precise Positioning')
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Test with bare minimum information:
        rwy.derive(fdr_apt, afr_apt, hdg)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, hint='takeoff')
        get_nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        precise.value = True
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, precise)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, latitude=4.0, longitude=3.0)
        get_nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        # NOTE: Latitude and longitude are still used for determining the
        #       takeoff runway, even when positioning is not precise!
        precise.value = False
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, precise)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.assert_called_once_with(25, 20.0, latitude=4.0, longitude=3.0,  hint='takeoff')
        get_nearest_runway.reset_mock()

    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    def test_derive_afr_fallback(self, get_nearest_runway):
        info = {'ident': '09L'}
        def runway_side_effect(apt, hdg, *args, **kwargs):
            if hdg == 90.0:
                return info
            raise NotFoundError('No runway found.')
        get_nearest_runway.side_effect = runway_side_effect
        fdr_apt = A(name='FDR Takeoff Airport', value={'id': 50})
        afr_rwy = A(name='AFR Takeoff Runway', value={'ident': '27R'})
        hdg_a = KPV(name='Heading At Takeoff', items=[
            KeyPointValue(index=1, value=270.0),
            KeyPointValue(index=2, value=360.0),
        ])
        hdg_b = KPV(name='Heading At Takeoff', items=[
            KeyPointValue(index=1, value=90.0),
            KeyPointValue(index=2, value=180.0),
        ])
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        rwy.derive(None, None, None)
        rwy.set_flight_attr.assert_called_once_with(None)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(fdr_apt, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, hdg_a)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not get_nearest_runway.called, 'method should not have been called'
        # Check wrong heading triggers AFR:
        rwy.derive(fdr_apt, afr_rwy, hdg_a)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        get_nearest_runway.assert_called_once_with(fdr_apt.value['id'], hdg_a.get_first().value, hint='takeoff')
        rwy.set_flight_attr.reset_mock()
        get_nearest_runway.reset_mock()
        rwy.derive(fdr_apt, afr_rwy, hdg_b)
        rwy.set_flight_attr.assert_called_once_with(info)
        get_nearest_runway.assert_called_once_with(fdr_apt.value['id'], hdg_b.get_first().value, hint='takeoff')


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
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.COMPLETE)
        # Would be 'COMPLETE', but 'AFR Type' overrides it.
        afr_type = A('AFR Type', value=FlightType.Type.FERRY)
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, liftoffs, touchdowns, None, None)
        type_node.set_flight_attr.assert_called_once_with(FlightType.Type.FERRY)
        # Liftoff missing.
        empty_liftoffs = KTI('Liftoff')
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, empty_liftoffs, touchdowns, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'TOUCHDOWN_ONLY')
        # Touchdown missing.
        empty_touchdowns = KTI('Touchdown')
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, liftoffs, empty_touchdowns, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'LIFTOFF_ONLY')

        # Liftoff and Touchdown missing, only Fast.
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, empty_liftoffs, empty_touchdowns, None,
                         None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.REJECTED_TAKEOFF)
        # Liftoff, Touchdown and Fast missing.
        empty_fast = fast = S('Fast')
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.ENGINE_RUN_UP)
        # Liftoff, Touchdown and Fast missing, Groundspeed changes.
        groundspeed = P('Groundspeed', np.ma.arange(20))
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, groundspeed)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.GROUND_RUN)
        # Liftoff, Touchdown and Fast missing, Groundspeed stays the same.
        groundspeed = P('Groundspeed', np.ma.masked_array([0] * 20))
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, empty_liftoffs, empty_touchdowns,
                         None, groundspeed)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.ENGINE_RUN_UP)
        # Liftoff after Touchdown.
        late_liftoffs = KTI('Liftoff', items=[KeyTimeInstance(20, 'a')])
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, late_liftoffs, touchdowns, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'TOUCHDOWN_BEFORE_LIFTOFF')
        # Touch and Go before Touchdown.
        afr_type = A('AFR Type', value=FlightType.Type.TRAINING)
        touch_and_gos = KTI('Touch And Go', items=[KeyTimeInstance(7, 'a')])
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, liftoffs, touchdowns, touch_and_gos,
                         None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.TRAINING)
        # Touch and Go after Touchdown.
        afr_type = A('AFR Type', value=FlightType.Type.TRAINING)
        touch_and_gos = KTI('Touch And Go', items=[KeyTimeInstance(15, 'a')])
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(afr_type, fast, liftoffs, touchdowns,
                             touch_and_gos, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'LIFTOFF_ONLY')


class TestAnalysisDatetime(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVersion(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Version.get_operational_combinations(),
                         [('Start Datetime',)])

    def test_derive(self):
        version = Version()
        version.set_flight_attr = Mock()
        version.derive()
        version.set_flight_attr.assert_called_once_wth(__version__)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
