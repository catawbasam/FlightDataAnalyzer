import numpy as np
import unittest

from datetime import datetime
from mock import call, Mock, patch

from flight_phase_test import buildsection

from analysis_engine.approaches import ApproachInformation
from analysis_engine.flight_phase import ApproachAndLanding
from analysis_engine.node import (
    A, ApproachItem, KPV, KeyPointValue, P, Parameter, S, Section)


class TestApproachInformation(unittest.TestCase):
    def test_can_operate(self):
        combinations = ApproachInformation.get_operational_combinations()
        self.assertTrue(('Approach And Landing', 'Altitude AAL', 'Fast')
                        in combinations)
        
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_runway')
    @patch('analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerLocal.get_nearest_airport')
    def test_derive(self, get_nearest_airport, get_nearest_runway):
        approaches = ApproachInformation()
        approaches._lookup_airport_and_runway = Mock()
        approaches._lookup_airport_and_runway.return_value = [None, None]
        
        app = ApproachAndLanding()

        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            10, 5, 0, 0, 5, 10, 20, 30, 40, 50,      # Touch & Go
            50, 45, 30, 35, 30, 30, 35, 40, 40, 40,  # Go Around
            30, 20, 10, 0, 0, 0, 0, 0, 0, 0,         # Landing
        ]))

        #land_afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        #land_afr_rwy = A(name='AFR Landing Runway', value={'ident': '09L'})

        #precise = A(name='Precise Positioning')
        fast = S(name='Fast', items=[
            Section(name='Fast', slice=slice(0, 22), start_edge=0,
                    stop_edge=22.5),
        ])

        land_hdg = KPV(name='Heading At Landing', items=[
            KeyPointValue(index=22, value=60),
        ])
        land_lat = KPV(name='Latitude At Landing', items=[
            KeyPointValue(index=22, value=10),
        ])
        land_lon = KPV(name='Longitude At Landing', items=[
            KeyPointValue(index=22, value=-2),
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
        #appr_ils_freq = KPV(name='ILS Frequency on Approach', items=[
            #KeyPointValue(name=5, value=330150),
        #])

        # No approaches if no approach sections in the flight:
        approaches.derive(app, alt_aal, fast)
        self.assertEqual(approaches, [])
        # Test the different approach types:
        slices = [slice(0, 5), slice(10, 15), slice(20, 25)]
        app.create_phases(slices)
        land_afr_apt_none = A(name='AFR Landing Airport', value=None)
        land_afr_rwy_none = A(name='AFR Landing Runway', value=None)
        approaches.derive(app, alt_aal, fast, land_afr_apt=land_afr_apt_none,
                          land_afr_rwy=land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15)),
                          ApproachItem('LANDING', slice(20, 25))])
        #approaches.set_flight_attr.assert_called_once_with()
        #approaches.set_flight_attr.reset_mock()
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], appr_ils_freq=[], precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[1], appr_ils_freq=[], precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[2], appr_ils_freq=[], precise=False, lowest_lat=None,
                 lowest_lon=None, lowest_hdg=None,
                 land_afr_apt=land_afr_apt_none,
                 land_afr_rwy=land_afr_rwy_none, hint='landing'),
        ])
        del approaches[:]
        approaches._lookup_airport_and_runway.reset_mock()
        # Test that landing lat/lon/hdg used for landing only, else use approach
        # lat/lon/hdg:
        approaches.derive(app, alt_aal, fast, land_hdg, land_lat, land_lon,
                          appr_hdg, appr_lat, appr_lon,
                          land_afr_apt=land_afr_apt_none,
                          land_afr_rwy=land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15),
                                       lowest_hdg=appr_hdg[1]),
                          ApproachItem('LANDING', slice(20, 25),
                                       lowest_lat=land_lat[0],
                                       lowest_lon=land_lon[0],
                                       lowest_hdg=land_hdg[0])])
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], lowest_hdg=None, lowest_lat=None,
                 lowest_lon=None, appr_ils_freq=[], precise=False),
            call(_slice=slices[1], lowest_hdg=appr_hdg[1], lowest_lat=None,
                 lowest_lon=None, appr_ils_freq=[], precise=False),
            call(_slice=slices[2], lowest_hdg=land_hdg[0],
                 lowest_lat=land_lat[0], lowest_lon=land_lon[0], appr_ils_freq=[],
                 precise=False, land_afr_apt=land_afr_apt_none,
                 land_afr_rwy=land_afr_rwy_none, hint='landing'),
        ])
        approaches._lookup_airport_and_runway.reset_mock()

        # FIXME: Finish implementing these tests to check that using the API
        #        works correctly and any fall back values are used as
        #        appropriate.
    
    @unittest.skip('Test Not Implemented')
    def test_derive_afr_fallback(self):
        self.assertTrue(False, msg='Test not implemented.')    