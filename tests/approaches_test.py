import numpy as np
import unittest

from datetime import datetime
from mock import call, Mock, patch

from flight_phase_test import buildsection

from analysis_engine.approaches import Approaches
from analysis_engine.node import (
    A, Approach, KPV, KeyPointValue, P, Parameter, S, Section)


class TestApproaches(unittest.TestCase):
    def test_can_operate(self):
        combinations = Approaches.get_operational_combinations()
        self.assertTrue(('Altitude AAL For Flight Phases', 'Landing',
                         'Go Around And Climbout', 'Altitude AAL',
                         'Fast') in combinations)

    def test__approach_slices_basic(self):
        alt = np.ma.array(range(5000, 500, -500) + [0] * 10)
        land = buildsection('Landing', 11, 20)
        # Go-around above 3000ft will be ignored.
        ga = buildsection('Go Around And Climbout', 8, 13)
        app = Approaches()
        result = app._approach_slices(
            Parameter('Altitude AAL For Flight Phases', alt), land, ga)
        self.assertEqual(result, [slice(4.0, 20)])

    def test__approach_slices_landing_and_go_around_overlap(self):
        alt = np.ma.array([3500, 2500, 2000, 2500, 3500, 3500])
        land=buildsection('Landing', 5, 6)
        ga=buildsection('Go Around And Climbout', 2.5, 3.5)
        app = Approaches()
        result = app._approach_slices(
            Parameter('Altitude AAL For Flight Phases', alt), land, ga)
        self.assertEqual(result, [slice(0, 6)])

    def test__approach_slices_separate_landing_phase_go_around(self):
        alt = np.ma.array([3500, 2500, 2000, 2500, 3500, 3500])
        land = buildsection('Landing', 5, 6)
        ga = buildsection('Go Around And Climbout', 1.5, 2.0)
        app = Approaches()
        result = app._approach_slices(
            Parameter('Altitude AAL For Flight Phases', alt), land, ga)
        self.assertEqual(result, [slice(0, 2), slice(3, 6)])
        
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal')
    def test_derive(self, api, get_nearest_airport, get_nearest_runway):

        def _fake_approach(t, a, b):
            return {
                'airport': None,
                'runway': None,
                'type': t,
                'datetime': datetime(1970, 1, 1, 0, 0, a),
                'slice_start': a,
                'slice_stop': b,
            }

        approaches = Approaches()
        approaches._lookup_airport_and_runway = Mock()
        approaches._lookup_airport_and_runway.return_value = [None, None]
        approaches._approach_slices = Mock()
        approaches._approach_slices.return_value = []        
        
        landing = S('Landing')
        go_arounds = S('Go Arounds')

        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            10, 5, 0, 0, 5, 10, 20, 30, 40, 50,      # Touch & Go
            50, 45, 30, 35, 30, 30, 35, 40, 40, 40,  # Go Around
            30, 20, 10, 0, 0, 0, 0, 0, 0, 0,         # Landing
        ]))

        #land_afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        #land_afr_rwy = A(name='AFR Landing Runway', value={'ident': '09L'})

        #precise = A(name='Precise Positioning')
        fast = S(name='Fast', items=[
            Section(name='Fast', slice=slice(0, 22), start_edge=0, stop_edge=22.5),
        ])

        land_hdg = KPV(name='Heading At Landing', items=[
            KeyPointValue(index=15, value=60),
        ])
        land_lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=5, value=10),
        ])
        land_lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=5, value=-2),
        ])
        appr_hdg = KPV(name='Heading At Low Point On Approach', items=[
            KeyPointValue(index=5, value=25),
            KeyPointValue(index=12, value=35),
        ])
        appr_lat = KPV(name='Latitude At Lowest Point On Approach', items=[
            KeyPointValue(index=5, value=8),
        ])
        appr_lon = KPV(name='Longitude At Lowest Point On Approach', items=[
            KeyPointValue(index=5, value=4),
        ])
        appr_ils_freq = KPV(name='ILS Frequency on Approach', items=[
            KeyPointValue(name=5, value=330150),
        ])

        # No approaches if no approach sections in the flight:
        approaches.derive(alt_aal, landing, go_arounds, alt_aal, fast)
        self.assertEqual(approaches, [])
        # Test the different approach types:
        approach_slices = [slice(0, 5), slice(10, 15), slice(20, 25)]
        approaches._approach_slices.return_value = approach_slices
        land_afr_apt_none = A(name='AFR Landing Airport', value=None)
        land_afr_rwy_none = A(name='AFR Landing Runway', value=None)
        approaches.derive(
            alt_aal, landing, go_arounds, alt_aal, fast,
            land_afr_apt=land_afr_apt_none, land_afr_rwy=land_afr_rwy_none)
        self.assertEqual(approaches,
                         [Approach(0, 'TOUCH_AND_GO', slice(0, 5)),
                          Approach(10, 'GO_AROUND', slice(10, 15)),
                          Approach(20, 'LANDING', slice(20, 25))])
        #approaches.set_flight_attr.assert_called_once_with()
        #approaches.set_flight_attr.reset_mock()
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=approach_slices[0], appr_hdg=[], appr_lat=[], appr_lon=[], appr_ils_freq=[], precise=False),
            call(_slice=approach_slices[1], appr_hdg=[], appr_lat=[], appr_lon=[], appr_ils_freq=[], precise=False),
            call(_slice=approach_slices[2], appr_hdg=[], appr_lat=[], appr_lon=[], appr_ils_freq=[], precise=False, land_afr_apt=land_afr_apt_none, land_afr_rwy=land_afr_rwy_none, hint='landing'),
        ])
        del approaches[:]
        approaches._lookup_airport_and_runway.reset_mock()
        # Test that landing lat/lon/hdg used for landing only, else use approach lat/lon/hdg:
        approaches.derive(alt_aal, landing, go_arounds, alt_aal, fast, land_hdg, land_lat, land_lon, appr_hdg, appr_lat, appr_lon, land_afr_apt=land_afr_apt_none, land_afr_rwy=land_afr_rwy_none)
        self.assertEqual(approaches,
                         [Approach(0, 'TOUCH_AND_GO', slice(0, 5)),
                          Approach(10, 'GO_AROUND', slice(10, 15)),
                          Approach(20, 'LANDING', slice(20, 25))])
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=approach_slices[0], appr_hdg=appr_hdg, appr_lat=appr_lat, appr_lon=appr_lon, appr_ils_freq=[], precise=False),
            call(_slice=approach_slices[1], appr_hdg=appr_hdg, appr_lat=appr_lat, appr_lon=appr_lon, appr_ils_freq=[], precise=False),
            call(_slice=approach_slices[2], appr_hdg=land_hdg, appr_lat=land_lat, appr_lon=land_lon, appr_ils_freq=[], precise=False, land_afr_apt=land_afr_apt_none, land_afr_rwy=land_afr_rwy_none, hint='landing'),
        ])
        approaches._lookup_airport_and_runway.reset_mock()

        # FIXME: Finish implementing these tests to check that using the API
        #        works correctly and any fall back values are used as
        #        appropriate.
    
    @unittest.skip('Test Not Implemented')
    def test_derive_afr_fallback(self):
        self.assertTrue(False, msg='Test not implemented.')    