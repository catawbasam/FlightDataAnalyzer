try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

import socket
import httplib2
from mock import Mock, patch

from analysis.api_handler import (APIConnectionError, InvalidAPIInputError,
                                  NotFoundError, UnknownAPIError)
from analysis.api_handler_http import APIHandlerHTTP


class APIHandlerHTTPTest(unittest.TestCase):
    
    @patch('analysis.api_handler_http.httplib2.Http.request')
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
        self.assertRaises(UnknownAPIError, handler._request, url)
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
        self.assertEqual(handler._request.call_args, ((1,), {'x': 2}))
        # Raises out Exception if it is raised in every attempt.
        handler._request.side_effect = Exception()
        self.assertRaises(Exception, handler._attempt_request, 3)
        self.assertEqual(handler._request.call_args_list,
                         [((1,), {'x': 2})] + [((3,), {})] * 3)
        # Returns value after Exception being raised in earlier attempts.
        return_values = [{}, Exception(), Exception()]
        def side_effect(*args, **kwargs):
            elem = return_values.pop()
            if isinstance(elem, Exception):
                raise elem
            else:
                return elem
        handler._request.side_effect = side_effect
        self.assertEqual(handler._attempt_request(4, y=5), {})
        self.assertEqual(handler._request.call_args_list,
                [((1,), {'x': 2})] + [((3,), {})] * 3 + [((4,), {'y': 5})] * 3)
    
    def test_get_nearest_airport(self):
        handler = APIHandlerHTTP(attempts=3)
        handler._request = Mock()
        request_return_value = {
            "status":200,
            "airport": {"distance":1.5125406009017226,
                        "magnetic_variation":"W002241 0106",
                        "code":{"icao":"EGLL",
                                "iata":"LHR"},
                        "name":"London Heathrow",
                        "longitude":-0.461389,
                        "location":{"city":"London",
                                    "country":"United Kingdom"},
                        "latitude":51.4775,
                        "id":2383}}
        handler._request.return_value = request_return_value
        self.assertEqual(handler.get_nearest_airport(14.1, 0.52),
                         request_return_value['airport'])
    
    @unittest.skip("Remove skip if a server is online to test against.")
    def test_get_nearest_airport_integration(self):
        '''
        Make an HTTP request rather than mocking the response. Requires the
        BASE_URL server being online.
        '''
        handler = APIHandlerHTTP(attempts=3)
        self.assertEqual(handler.get_nearest_airport(51.4775, -0.461389),
                         {"distance":1.5125406009017226,
                          "magnetic_variation":"W002241 0106",
                          "code":{"icao":"EGLL",
                                  "iata":"LHR"},
                          "name":"London Heathrow",
                          "longitude":-0.461389,
                          "location":{"city":"London",
                                      "country":"United Kingdom"},
                          "latitude":51.4775,
                          "id":2383})
    
    def test_get_nearest_runway(self):
        handler = APIHandlerHTTP(attempts=3)
        handler._request = Mock()
        handler._request.return_value = {'status': 200, 'runway': {'x': 1}}
        self.assertEqual(handler.get_nearest_runway('ICAO', 120),
                         {'x': 1})
        # TODO: Test GET parameters.