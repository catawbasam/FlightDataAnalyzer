import numpy as np
import unittest

from datetime import datetime, timedelta
from mock import Mock, patch

from analysis_engine.api_handler import NotFoundError
from analysis_engine.node import (A, KeyPointValue, KeyTimeInstance, KPV, KTI,
                                  P, S, Section)
from analysis_engine.settings import CONTROLS_IN_USE_TOLERANCE
from analysis_engine.flight_attribute import (
    Approaches, 
    DeterminePilot,
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
    TakeoffPilot,
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
             'Latitude At Lowest Point On Approach',
             'Longitude At Lowest Point On Approach']))
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
             'Latitude At Lowest Point On Approach',
             'Longitude At Lowest Point On Approach',
             'Heading At Lowest Point On Approach',
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
             'Longitude At Lowest Point On Approach']))
        # Cannot operate missing Approach and Landing.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Heading At Landing',
             'Latitude At Lowest Point On Approach',
             'Longitude At Lowest Point On Approach']))
        # Cannot operate with differing sources of lat lng.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Approach And Landing',
             'Heading At Landing',
             'Touch And Go',
             'Go Around',
             'Latitude At Lowest Point On Approach',
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
        approach_lat_kpvs = KPV('Latitude At Lowest Point On Approach',
                                items=[KeyPointValue(12, 4, 'b')])
        approach_lon_kpvs = KPV('Longitude At Lowest Point On Approach',
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
        
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_airport')
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_runway')
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
        approach_lat_kpvs = KPV('Latitude At Lowest Point On Approach',
                                items=[KeyPointValue(5, 8, 'b')])
        approach_lon_kpvs = KPV('Longitude At Lowest Point On Approach',
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


class TestDeterminePilot(unittest.TestCase):
    def test__autopilot_engaged(self):
        determine_pilot = DeterminePilot()
        # Q: What should discrete values be?
        autopilot1 = KeyPointValue('Autopilot Engaged 1 At Touchdown', value=0)
        autopilot2 = KeyPointValue('Autopilot Engaged 2 At Touchdown', value=0)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, None)
        # Autopilot 1 Engaged.
        autopilot1 = KeyPointValue('Autopilot Engaged 1 At Touchdown', value=1)
        autopilot2 = KeyPointValue('Autopilot Engaged 2 At Touchdown', value=0)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, 'Captain')
        # Autopilot 2 Engaged.
        autopilot1 = KeyPointValue('Autopilot Engaged 1 At Touchdown', value=0)
        autopilot2 = KeyPointValue('Autopilot Engaged 2 At Touchdown', value=1)
        pilot = determine_pilot._autopilot_engaged(autopilot1, autopilot2)
        self.assertEqual(pilot, 'First Officer')
        # Both Autopilots Engaged.
        autopilot1 = KeyPointValue('Autopilot Engaged 1 At Touchdown', value=1)
        autopilot2 = KeyPointValue('Autopilot Engaged 2 At Touchdown', value=1)
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
        determine_pilot = DeterminePilot()
        pitch_captain = Mock()
        roll_captain = Mock()
        pitch_fo = Mock()
        roll_fo = Mock()
        section = Section(name='Takeoff', slice=slice(0,3))
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
                                array=np.ma.masked_array([103, 102,102]))
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('102')
        
    def test_derive_most_common_positive_float(self):
        flight_number = FlightNumber()
        
        neg_number_param = P(
            'Flight Number',
            array=np.ma.array([-1,2,-4,10]))
        self.assertRaises(ValueError, flight_number.derive, neg_number_param)
        
        # TODO: Implement variance checks as below
        ##high_variance_number_param = P(
            ##'Flight Number',
            ##array=np.ma.array([2,2,4,4,4,7,7,7,4,5,4,7,910]))
        ##self.assertRaises(ValueError, flight_number.derive, high_variance_number_param)
        
        flight_number_param= P(
            'Flight Number',
            array=np.ma.array([2,555.6,444,444,444,444,444,444,888,444,444,444,444,444,444,444,444,7777,9100]))
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param)
        flight_number.set_flight_attr.assert_called_with('444')
        
        
class TestLandingAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingAirport.get_operational_combinations(),
                         [('Latitude At Landing', 'Longitude At Landing')])
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_airport')
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
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_airport')
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
        # Only controls in use parameters.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Landing') in opts)
        # Only Autopilot.
        self.assertTrue(('Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown') in opts)
        # Combinations.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Landing',
                         'Autopilot Engaged 1 At Touchdown') in opts)
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Landing',
                         'Autopilot Engaged 1 At Touchdown',
                         'Autopilot Engaged 2 At Touchdown' in opts))
        # All.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Landing', 'Autopilot Engaged 1 At Touchdown',
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
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        ##runway_info = {'ident': '27L', 'runways': [{'length': 20}]}
        runway_info = {
            'ident': '27L', 
            'items':[ 
                {
                    "end":{"latitude":58.211678,"longitude":8.095269},
                    "localizer":{"latitude":58.212397,"beam_width":4.5,"frequency":"110300M","heading":36,"longitude":8.096228},
                    "glideslope":{"latitude":58.198664,"frequency":"335000M","angle":3.4,"longitude":8.080164,"threshold_distance":720},
                    "start":{"latitude":58.196703,"longitude":8.075406},
                    "strip":{"width":147,"length":6660,"id":4064,"surface":"ASP"},
                    "identifier":"27L",
                    "id":8127,
                }
            ]}
        
        
        get_nearest_runway.return_value = runway_info
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr = Mock()
        # Airport and Heading At Landing arguments.
        airport = A('Landing Airport')
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
    def test_derive(self):
        # Empty 'Turning'.
        turning = S('Turning')
        start_datetime = A(name='Start Datetime', value=datetime.now())
        off_blocks_datetime = OffBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # Only 'Turning In Air'.
        turning = S('Turning', items=[KeyPointValue(name='Turning In Air',
                                                    slice=slice(0, 100))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(20, 60))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(\
            start_datetime.value + timedelta(seconds=20))
        
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
                                                    slice=slice(10, 20)),
                                      KeyPointValue(name='Turning In Air',
                                                    slice=slice(20, 60)),
                                      KeyPointValue(name='Turning On Ground',
                                                    slice=slice(70, 90))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=10))


class TestOnBlocksDatetime(unittest.TestCase):
    def test_derive_without_turning(self):
        # Empty 'Turning'.
        turning = S('Turning')
        start_datetime = A(name='Start Datetime', value=datetime.now())
        off_blocks_datetime = OnBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # Only 'Turning In Air'.
        turning = S('Turning', items=[KeyPointValue(name='Turning In Air',
                                                    slice=slice(0, 100))])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(turning, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)
        # 'Turning On Ground'.
        turning = S('Turning', items=[KeyPointValue(name='Turning On Ground',
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


class TestTakeoffAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual([('Latitude At Takeoff', 'Longitude At Takeoff')],
                         TakeoffAirport.get_operational_combinations())
        
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = NotFoundError('Not Found.')
        latitude = KPV(name='Latitude At Takeoff',
                       items=[KeyPointValue(index=1, value=4.0),
                              KeyPointValue(index=2, value=6.0)])
        longitude = KPV(name='Longitude At Takeoff',
                        items=[KeyPointValue(index=1, value=3.0),
                               KeyPointValue(index=2, value=9.0)])
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(latitude, longitude)
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        self.assertFalse(takeoff_airport.set_flight_attr.called)
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        airport_info = {'id': 123}
        get_nearest_airport.return_value = airport_info
        latitude = KPV(name='Latitude At Takeoff',
                       items=[KeyPointValue(index=1, value=4.0),
                              KeyPointValue(index=2, value=6.0)])
        longitude = KPV(name='Longitude At Takeoff',
                        items=[KeyPointValue(index=1, value=3.0),
                               KeyPointValue(index=2, value=9.0)])
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(latitude, longitude)
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


class TestTakeoffPilot(unittest.TestCase):
    def test_can_operate(self):
        opts = TakeoffPilot.get_operational_combinations()
        # Only controls in use parameters.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff') in opts)
        # Only Autopilot.
        self.assertTrue(('Autopilot Engaged 1 At Liftoff',
                         'Autopilot Engaged 2 At Liftoff') in opts)
        # Combinations.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff',
                         'Autopilot Engaged 1 At Liftoff') in opts)
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Takeoff',
                         'Autopilot Engaged 1 At Liftoff',
                         'Autopilot Engaged 2 At Liftoff' in opts))
        # All.
        self.assertTrue(('Pitch (Capt)', 'Roll (Capt)', 'Pitch (FO)',
                         'Roll (FO)', 'Takeoff', 'Autopilot Engaged 1 At Liftoff',
                         'Autopilot Engaged 2 At Liftoff') in opts)
    
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
    

class TestTakeoffRunway(unittest.TestCase):
    def test_can_operate(self):
        '''
        There may be a neater way to test this, but at least it's verbose.
        '''
        expected = \
        [('FDR Takeoff Airport', 'Heading At Takeoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude At Takeoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff',
          'Longitude At Takeoff'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff', 
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Longitude At Takeoff', 
          'Precise Positioning'),
         ('FDR Takeoff Airport', 'Heading At Takeoff', 'Latitude At Takeoff',
          'Longitude At Takeoff', 'Precise Positioning')]
        self.assertEqual(TakeoffRunway.get_operational_combinations(),
                         expected)
    
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP.get_nearest_runway')
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
        latitude = KPV(name='Latitude At Takeoff',
                       items=[KeyPointValue(index=1, value=4.0),
                              KeyPointValue(index=2, value=6.0)])
        longitude = KPV(name='Longitude At Takeoff',
                        items=[KeyPointValue(index=1, value=3.0),
                               KeyPointValue(index=2, value=9.0)])
        precision = A('Precision')
        precision.value = True
        takeoff_runway.derive(airport, takeoff_heading, latitude, longitude,
                              precision)
        get_nearest_runway.assert_called_with(25, 20.0, latitude=4.0,
                                              longitude=3.0)
        takeoff_runway.set_flight_attr.assert_called_with(runway_info)
        # When Precise Positioning's value is False, Latitude and Longitude
        # are not used.
        precision.value = False
        takeoff_runway.derive(airport, takeoff_heading, latitude, longitude,
                              precision)
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

