import httplib2
import simplejson
import socket
import unittest

from mock import Mock, patch

from analysis_engine.api_handler import (
    APIConnectionError,
    APIError,
    APIHandlerHTTP,
    InvalidAPIInputError,
    NotFoundError,
    UnknownAPIError
)
from analysis_engine.api_handler_analysis_engine import (
    AnalysisEngineAPIHandlerHTTP,
    AnalysisEngineAPIHandlerLocal,
)


class APIHandlerHTTPTest(unittest.TestCase):
    
    @patch('analysis_engine.api_handler.httplib2.Http.request')
    def test__request(self, http_request_patched):
        '''
        Test error handling.
        '''
        url = 'www.testcase.com'
        handler = APIHandlerHTTP()
        resp = {'status': 200}
        content = '{}'
        http_request_patched.return_value = resp, content
        result = handler._request(url)
        self.assertEqual(result, {})
        # Responded with invalid JSON.
        content = 'invalid JSON'
        http_request_patched.return_value = resp, content
        self.assertRaises(simplejson.JSONDecodeError, handler._request, url)
        content = "{error: 'Server error.'}"
        resp['status'] = 400 # Bad Request
        self.assertRaises(InvalidAPIInputError, handler._request, url)
        resp['status'] = 401 # Unauthorised
        self.assertRaises(APIConnectionError, handler._request, url)
        resp['status'] = 404 # Not Found
        self.assertRaises(NotFoundError, handler._request, url)
        resp['status'] = 500 # Internal Server Error
        self.assertRaises(UnknownAPIError, handler._request, url)
        resp['status'] = 200
        http_request_patched.side_effect = socket.error()
        self.assertRaises(APIConnectionError, handler._request, url)
        http_request_patched.side_effect = httplib2.ServerNotFoundError()
        self.assertRaises(APIConnectionError, handler._request, url)
    
    def test__attempt_request(self):
        handler = APIHandlerHTTP(attempts=3)
        handler._request = Mock()
        handler._request.return_value = {}
        self.assertEqual(handler._attempt_request(1, x=2), {})
        handler._request.assert_called_once_with(1, x=2)
        # Raises out Exception if it is raised in every attempt.
        handler._request.side_effect = UnknownAPIError('')
        self.assertRaises(UnknownAPIError, handler._attempt_request, 3)
        self.assertEqual(handler._request.call_args_list,
                         [((1,), {'x': 2})] + [((3,), {})] * 3)
        # Returns value after Exception being raised in earlier attempts.
        return_values = [{}, UnknownAPIError(''), APIConnectionError('')]
        def side_effect(*args, **kwargs):
            elem = return_values.pop()
            if isinstance(elem, APIError):
                raise elem
            else:
                return elem
        handler._request.side_effect = side_effect
        self.assertEqual(handler._attempt_request(4, y=5), {})
        self.assertEqual(handler._request.call_args_list,
                [((1,), {'x': 2})] + [((3,), {})] * 3 + [((4,), {'y': 5})] * 3)
    
    def test_get_nearest_airport(self):
        handler = AnalysisEngineAPIHandlerHTTP(attempts=3)
        handler._request = Mock()
        request_return_value = {
            "status": 200,
            "airport": {
                "distance": 1.5125406009017226,
                "magnetic_variation": "W002241 0106",
                "code": {
                    "icao":"EGLL",
                    "iata":"LHR"
                    },
                "name":"London Heathrow",
                "longitude":-0.461389,
                "location": {
                    "city":"London",
                    "country":"United Kingdom"
                    },
                "latitude":51.4775,
                "id":2383
            }
        }
        handler._request.return_value = request_return_value
        self.assertEqual(handler.get_nearest_airport(14.1, 0.52),
                         request_return_value['airport'])
    
    @unittest.skip("Remove skip if a server is online to test against.")
    def test_get_nearest_airport_integration(self):
        '''
        Make an HTTP request rather than mocking the response. Requires the
        BASE_URL server being online.
        '''
        handler = AnalysisEngineAPIHandlerHTTP(attempts=3)
        self.assertEqual(handler.get_nearest_airport(51.4775, -0.461389),
                         {'code': {'iata': 'LHR', 'icao': 'EGLL'},
                          'id': 2383,
                          'latitude': 51.4775,
                          'location': {
                              'city': 'London',
                              'country': 'United Kingdom'
                              },
                          'longitude': -0.461389,
                          'magnetic_variation': 'W002241 0106',
                          'name': 'London Heathrow'})
    
    def test_get_nearest_runway(self):
        handler = AnalysisEngineAPIHandlerHTTP(attempts=3)
        handler._request = Mock()
        handler._request.return_value = {'status': 200, 'runway': {'x': 1}}
        self.assertEqual(handler.get_nearest_runway('ICAO', 120),
                         {'x': 1})
        # TODO: Test GET parameters.


class AnalysisEngineAPIHandlerLocalTest(unittest.TestCase):
    def setUp(self):
        self.handler = AnalysisEngineAPIHandlerLocal()
    
    def test_get_airport(self):
        self.assertEqual(self.handler.get_airport(2456),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport('KRS'),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport('ENCN'),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport(2461),
                         self.handler.airports[1])
        self.assertEqual(self.handler.get_airport('OSL'),
                         self.handler.airports[1])
        self.assertEqual(self.handler.get_airport('ENGM'),
                         self.handler.airports[1])
    
    def test_get_nearest_airport(self):
        airport = self.handler.get_nearest_airport(58, 8)
        self.assertEqual(airport['distance'], 23253.447237062534)
        del airport['distance']
        self.assertEqual(airport, self.handler.airports[0])
        airport = self.handler.get_nearest_airport(60, 11)
        self.assertEqual(airport['distance'], 22267.45203750386)
        del airport['distance']
        self.assertEqual(airport, self.handler.airports[1])
    
    def test_get_nearest_runway(self):
        runway = self.handler.get_nearest_runway(None, None, latitude=58,
                                                 longitude=8)
        self.assertEqual(runway['distance'], 22316.691624918927)
        del runway['distance']
        self.assertEqual(runway, self.handler.runways[0])
        runway = self.handler.get_nearest_runway(None, None, latitude=60,
                                                 longitude=11)
        self.assertEqual(runway['distance'], 20972.761983734454)
        del runway['distance']
        self.assertEqual(runway, self.handler.runways[1])