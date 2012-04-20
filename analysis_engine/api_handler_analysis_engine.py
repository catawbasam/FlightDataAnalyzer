import urllib
    
from abc import ABCMeta, abstractmethod

from analysis_engine.api_handler import (APIHandlerHTTP, NotFoundError)
from analysis_engine.settings import BASE_URL


class AnalysisEngineAPI(object):
    '''
    Abstract base class for API handler classes.
    '''
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_nearest_airport(self, latitude, longitude):
        '''
        Will either return the nearest airport to the specified latitude and
        longitude, or raise an exception if one cannot be found.
        
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :raises NotFoundError: If airport cannot be found.
        :raises InvalidAPIInputError: If latitude or longitude are out of bounds.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_airport(self, code):
        '''
        Will either return an airport matching the code or raise an exception
        if one cannot be found.
        
        :param code: Either the id, ICAO or IATA of the airport.
        :type code: int or str
        :raises NotFoundError: If airport cannot be found.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_nearest_runway(self, airport, heading, latitude=None, 
                           longitude=None, ilsfreq=None):
        '''
        Will return the nearest runway from the specified airport using
        latitude, longitude, precision and ilsfreq.
        
        :param airport: Value identifying the airport.
        :type airport: undefined
        :param heading: Magnetic heading.
        :type heading: int # Q: could it be float?
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :param ilsfreq: ILS frequency of runway # Q: Glideslope or Localizer frequency?
        :type ilsfreq: float # Q: could/should it be int?
        :raises NotFoundError: If runway cannot be found.
        :raises InvalidAPIInputError: If latitude, longitude or heading are out of bounds.
        :returns: Runway info dictionary.
        :rtype: dict
        '''
        raise NotImplementedError
    
    # TODO: Determine method signature...
    @abstractmethod
    def get_vspeed_limit(self, *args, **kwargs):
        '''
        TODO: Define what this does..
        '''
        raise NotImplementedError
    
    
class AnalysisEngineAPIHandlerDUMMY(AnalysisEngineAPI):
    '''
    DummyAPIHandler will always raise NotFoundError.
    '''
    def get_nearest_airport(self, *args, **kwargs):
        raise NotFoundError('DummyAPIHandler will always raise NotFoundError.')       
    
    def get_nearest_runway(self, *args, **kwargs):
        raise NotFoundError('DummyAPIHandler will always raise NotFoundError.')
    
    def get_vspeed_limit(self, *args, **kwargs):
        raise NotFoundError('DummyAPIHandler will always raise NotFoundError.')


class AnalysisEngineAPIHandlerHTTP(AnalysisEngineAPI, APIHandlerHTTP):
    
    def get_airport(self, code):
        '''
        Will either return an airport matching the code or raise an exception
        if one cannot be found.
        
        :param code: Either the id, ICAO or IATA of the airport.
        :type code: int or str
        :raises NotFoundError: If airport cannot be found.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        url = '%(base_url)s/api/airport/%(code)s/' % \
            {'base_url': BASE_URL.rstrip('/'), 'code': code}
        return self._attempt_request(url)['airport']
    
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
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        url = '%(base_url)s/api/airport/nearest.json?ll=%(ll)s' % \
            {'base_url': BASE_URL.rstrip('/'), 'll': '%f,%f' % (latitude, longitude)}
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
            {'base_url': BASE_URL.rstrip('/'), 'airport': airport}
        
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
    
    