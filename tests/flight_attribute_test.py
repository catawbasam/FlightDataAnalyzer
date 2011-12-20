import unittest

from datetime import datetime
from mock import Mock, patch

import numpy as np

from analysis.api_handler import NotFoundError
from analysis.node import (A, KeyPointValue, KeyTimeInstance, KPV, KTI, P, S,
                           Section)
from analysis.flight_attribute import (
    TakeoffAirport, TakeoffRunway, Approaches
    )

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
        liftoff.create_kti(1, 'STATE')
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(liftoff, latitude, longitude)
        self.assertEqual(get_nearest_airport.call_args, ((4.0, 3.0), {}))
        self.assertFalse(takeoff_airport.set_flight_attr.called)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        airport_info = {'id': 123}
        get_nearest_airport.return_value = airport_info
        liftoff = KTI('Liftoff')
        liftoff.create_kti(1, 'STATE')
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        takeoff_airport = TakeoffAirport()
        takeoff_airport.set_flight_attr = Mock()
        takeoff_airport.derive(liftoff, latitude, longitude)
        self.assertEqual(get_nearest_airport.call_args,
                         ((4.0, 3.0), {}))
        self.assertEqual(takeoff_airport.set_flight_attr.call_args,
                         ((airport_info,), {}))


class TestTakeoffRunway(unittest.TestCase):
    def test_can_operate(self):
        '''
        There may be a neater way to test this, but at least it's verbose.
        '''
        expected = \
        [('Takeoff Airport', 'Takeoff Heading'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff'),
         ('Takeoff Airport', 'Takeoff Heading', 'Latitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Longitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Latitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Longitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Latitude', 'Longitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Latitude', 
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Longitude', 
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Latitude', 
          'Longitude'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Latitude',
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Longitude',
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Latitude', 'Longitude',
          'Precise Positioning'),
         ('Takeoff Airport', 'Takeoff Heading', 'Liftoff', 'Latitude',
          'Longitude', 'Precise Positioning')]
        self.assertEqual(TakeoffRunway.get_operational_combinations(),
                         expected)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway):
        runway_info = {'ident': '27L', 'runways': [{'length': 20}]}
        get_nearest_runway.return_value = runway_info
        takeoff_runway = TakeoffRunway()
        takeoff_runway.set_flight_attr = Mock()
        # Airport and Takeoff Heading arguments.
        airport = A('Takeoff Airport')
        airport.value = {'id':25}
        takeoff_heading = KPV('Takeoff Heading')
        takeoff_heading.create_kpv(1, 20.0)
        takeoff_runway.derive(airport, takeoff_heading)
        self.assertEqual(get_nearest_runway.call_args, ((25, 20.0), {}))
        self.assertEqual(takeoff_runway.set_flight_attr.call_args,
                         ((runway_info,), {}))
        # Airport, Takeoff Heading, Liftoff, Latitude, Longitude and Precision
        # arguments. Latitude and Longitude are only passed with all these
        # parameters available and Precise Positioning is True.
        liftoff = KTI('Liftoff')
        liftoff.create_kti(1, 'STATE')
        latitude = P('Latitude', array=np.ma.masked_array([2.0,4.0,6.0]))
        longitude = P('Longitude', array=np.ma.masked_array([1.0,3.0,5.0]))
        precision = A('Precision')
        precision.value = True
        takeoff_runway.derive(airport, takeoff_heading, liftoff, latitude,
                              longitude, precision)
        self.assertEqual(get_nearest_runway.call_args, ((25, 20.0),
                                                        {'latitude': 4.0,
                                                         'longitude': 3.0}))
        self.assertEqual(takeoff_runway.set_flight_attr.call_args,
                         ((runway_info,), {}))
        # When Precise Positioning's value is False, Latitude and Longitude
        # are not used.
        precision.value = False
        takeoff_runway.derive(airport, takeoff_heading, liftoff, latitude,
                              longitude, precision)
        self.assertEqual(get_nearest_runway.call_args, ((25, 20.0), {}))
        self.assertEqual(takeoff_runway.set_flight_attr.call_args,
                         ((runway_info,), {}))
        

class TestApproaches(unittest.TestCase):
    def test_can_operate(self):
        # Can operate with approach lat lng.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach and Landing',
             'Latitude At Low Point On Approach',
             'Longitude At Low Point On Approach']))
        # Can operate with landing lat lng.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach and Landing',
             'Latitude At Landing',
             'Longitude At Landing']))
        # Can operate with everything.
        self.assertTrue(Approaches.can_operate(\
            ['Start Datetime',
             'Approach and Landing',
             'Heading At Landing',
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
             'Approach and Landing',
             'Longitude At Low Point On Approach']))
        # Cannot operate missing Approach and Landing.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Latitude At Low Point On Approach',
             'Longitude At Low Point On Approach']))
        # Cannot operate with differing sources of lat lng.
        self.assertFalse(Approaches.can_operate(\
            ['Start Datetime',
             'Approach and Landing',
             'Latitude At Low Point On Approach',
             'Longitude At Landing']))
    
    def test__get_lat_lon(self):
        approaches = Approaches()
        landing_lat_kpvs = KPV('Latitude At Landing',
                               items=[KeyPointValue(5, 10, 'b')])
        landing_lng_kpvs = KPV('Longitude At Landing',
                               items=[KeyPointValue(5, -2, 'b')])
        lat, lon = appraoches._get_lat_lon(landing_lat_kpvs, landing_lng_kpvs,
                                           None, None)
        self.assertEqual(lat, 10)
        self.assertEqual(lon, -2)
        
        
    
    def test__get_hdg(self):
        self.assertTrue(False)
    
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_airport')
    @patch('analysis.api_handler_http.APIHandlerHTTP.get_nearest_runway')
    def test_derive(self, get_nearest_runway, get_nearest_airport):
        approaches = Approaches()
        approaches.set_flight_attr = Mock()
        get_nearest_airport.return_value = {'id': 1}
        start_datetime = A('Start Datetime', value=datetime.now())
        approach_and_landing = S('Approach and Landing',
                                 items=[Section('a', slice(0,10))])
        landing_lat_kpvs = KPV('Latitude At Landing',
                               items=[KeyPointValue(5, 10, 'b')])
        landing_lng_kpvs = KPV('Longitude At Landing',
                               items=[KeyPointValue(5, -2, 'b')])
        go_arounds = KTI('Go Around', items=[KeyTimeInstance(5, 'Go Around')])
        approaches.derive(start_datetime, approach_and_landing, None,
                          go_arounds, landing_lat_kpvs, landing_lng_kpvs, None,
                          None, None, None)
        self.assertEqual(get_nearest_airport.call_args, ((10, -2), {}))
        self.assertEqual(get_nearest_runway.call_args_list, [])
        self.assertEqual(approaches.set_flight_attr.call_args,
                         (([{'airport':1,},]), {}))
        
        


#class TestLandingAirport(unittest.TestCase):
    #def test_can_operate(self):
        #self.assertTrue(False)
    
    #def test_derive(self):
        #self.assertTrue(False)


#class TestLandingRunway(unittest.TestCase):
    #def test_can_operate(self):
        #self.assertTrue(False)
    
    #def test_derive(self):
        #self.assertTrue(False)

#class TestLandingAirport(unittest.TestCase):
    #def test_can_operate(self):
        #self.assertTrue(False)
    
    #def test_derive(self):
        #self.assertTrue(False)


#class TestLandingRunway(unittest.TestCase):
    #def test_can_operate(self):
        #self.assertTrue(False)
    
    #def test_derive(self):
        #self.assertTrue(False)


