import urllib
import socket
import httplib
import time
import httplib2
try:
    import simplejson as json
except ImportError:
    import json
    
from analysis_engine.api_handler import (APIConnectionError, APIHandler,
                                         InvalidAPIInputError, NotFoundError,
                                         UnknownAPIError)
from analysis_engine import settings

TIMEOUT = 60

socket.setdefaulttimeout(TIMEOUT)


class APIHandlerHTTP(APIHandler):
    '''
    Restful HTTP API Handler.
    '''
    def __init__(self, attempts=3, delay=2):
        '''
        :param attempts: Attempts to retry the same request before raising an exception.
        :type attempts: int
        :param delay: Time to sleep between API requests.
        :type delay: int or float
        '''
        if attempts >= 1:
            self.attempts = attempts
        else:
            raise ValueError('APIHandlerHTTP must attempt requests at least once.')
        self.delay = delay
    
    def _request(self, uri, method='GET', body='', timeout=TIMEOUT):
        '''
        Makes a request to a URL and attempts to return the decoded content.
        
        :param uri: URI to request.
        :type uri: str
        :param method: Method of request.
        :type method: str
        :param timeout: Request timeout in seconds.
        :type timeout: int
        :param body: Body to be encoded.
        :type body: str, dict or tuple
        :raises InvalidAPIInputError: If server returns 400.
        :raises NotFoundError: If server returns 404.
        :raises APIConnectionError: If the server does not respond or returns 401.
        :raises UnknownAPIError: If the server returns 500 or an unexpected status code.
        '''
        # Encode body as GET parameters.
        body = urllib.urlencode(body)
        http = httplib2.Http(timeout=timeout)
        try:
            resp, content = http.request(uri, method, body)
        except (httplib2.ServerNotFoundError, socket.error, AttributeError): # DNS..
            raise APIConnectionError(uri, method, body)
        status = int(resp['status'])
        try:
            decoded_content = json.loads(content)
        except ValueError:
            decoded_content = None
        # Test HTTP Status.
        if status != 200:
            if decoded_content:
                # Try to get 'error' message from JSON which may not be
                # available.
                error_msg = decoded_content['error']
            else:
                error_msg = ''
            if status == httplib.BAD_REQUEST: # 400
                raise InvalidAPIInputError(error_msg, uri, method, body)
            elif status == httplib.UNAUTHORIZED: # 401
                raise APIConnectionError(error_msg, uri, method, body)
            elif status == httplib.NOT_FOUND: # 404
                raise NotFoundError(error_msg, uri, method, body)
            elif status == httplib.INTERNAL_SERVER_ERROR: # 500
                raise UnknownAPIError(error_msg, uri, method, body)
            else:
                raise UnknownAPIError(error_msg, uri, method, body)
        
        if decoded_content is None:
            raise UnknownAPIError('JSON response could not be decoded.',
                                  uri, method, body)
        return decoded_content
    
    def _attempt_request(self, *args, **kwargs):
        '''
        Attempt the request the number of times specified by self.attempts.
        If the specified number of attempts have failed, raise the exception
        last raised.
        
        :param args: Arguments passed into self._request.
        :type args: list
        :param kwargs: Keyword arguments passed into self._request.
        :type kwargs: dict
        :raises Exception: If self._request() does so in all attempts.
        :returns: Decoded JSON object if successful.
        :rtype: dict
        '''
        for attempt in range(self.attempts):
            try:
                return self._request(*args, **kwargs)
            except Exception as error:
                time.sleep(self.delay)
        raise error
    
    def get_nearest_airport(self, latitude, longitude):
        '''
        Either returns the nearest airport to the specified latitude and
        longitude, or raises an exception if one cannot be found.
        
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :raises NotFoundError: If airport cannot be found.
        :raises InvalidAPIInputError: If latitude or longitude are out of bounds.
        :returns: Airport information.
        :rtype: dict
        '''
        url = '%(base_url)s/api/airport/nearest.json?ll=%(ll)s' % \
            {'base_url': settings.BASE_URL, 'll': '%f,%f' % (latitude, longitude)}
        return self._attempt_request(url)['airport']
    
    def get_nearest_runway(self, airport, heading, latitude=None,
                           longitude=None, ilsfreq=None):
        '''
        Returns the nearest runway from the specified airport using latitude,
        longitude, precision and ilsfreq.
        
        :param airport: Either ICAO code, IATA code or database ID of airport.
        :type airport: int or str
        :param heading: Magnetic heading.
        :type heading: int # Q: could it be float?
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :param ilsfreq: ILS Localizer frequency of the runway in KHz.
        :type ilsfreq: float # Q: could/should it be int?
        :raises NotFoundError: If runway cannot be found.
        :raises InvalidAPIInputError: If latitude, longitude or heading are out of bounds.
        :returns: Runway info in the format {'ident': '27*', 'items': [{# ...}, {# ...},]}, 'ident' is either specific ('09L') or generalised ('09*'). 'items' is a list of matching runways.
        :rtype: dict
        '''
        url = '%(base_url)s/api/airport/%(airport)s/runway/nearest.json' % \
            {'base_url': settings.BASE_URL, 'airport': airport}
        
        params = {'heading': heading}
        if latitude and longitude:
            params['ll'] = '%f,%f' % (latitude, longitude)
        if ilsfreq:
            # While ILS frequency is recorded in MHz, the API expects KHz.
            params['ilsfreq'] = int(ilsfreq * 1000)
        get_params = urllib.urlencode(params)
        url += '?' + get_params
        return self._attempt_request(url)['runway']
        
    def get_vspeed_limit(self, *args, **kwargs):
        '''
        
        '''
        pass
