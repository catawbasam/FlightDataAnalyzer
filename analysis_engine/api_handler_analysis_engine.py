# -*- coding: utf-8 -*-
##############################################################################

'''
'''

##############################################################################
# Imports

import os
import simplejson
import urllib
import yaml

from abc import ABCMeta, abstractmethod
from copy import copy
from operator import itemgetter

from analysis_engine.api_handler import (APIHandlerHTTP,
                                         IncompleteEntryError,
                                         NotFoundError)
from analysis_engine.library import bearing_and_distance


##############################################################################
# Analysis Engine API Handlers


########################################
# API Handler Interface


class AnalysisEngineAPI(object):
    '''
    Abstract base class for API handler classes.
    '''

    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_aircraft(self, tail_number):
        '''
        Will either return an aircraft matching the tail number or raise an
        exception if one cannot be found.
        
        :param tail_number: Aircraft tail number.
        :type tail_number: str
        :raises NotFoundError: If the aircraft cannot be found.
        :returns: Aircraft info dictionary
        :rtype: dict
        '''
        try:
            return self.aircraft[tail_number]
        except KeyError:
            raise NotFoundError("Local API Handler: Aircraft with tail number "
                                "'%s' could not be found." % tail_number)

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
    def get_nearest_airport(self, latitude, longitude):
        '''
        Will either return the nearest airport to the specified latitude and
        longitude, or raise an exception if one cannot be found.

        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :raises NotFoundError: If airport cannot be found.
        :raises InvalidAPIInputError: If latitude or longitude are out of
                bounds.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        raise NotImplementedError

    @abstractmethod
    def get_nearest_runway(self, airport, heading, latitude=None,
            longitude=None, ils_freq=None, hint=None):
        '''
        Will return the nearest runway from the specified airport using
        latitude, longitude, precision and ils freq.

        :param airport: Value identifying the airport.
        :type airport: undefined
        :param heading: Magnetic heading.
        :type heading: int # Q: could it be float?
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :param ils_freq: ILS localizer frequency of runway
        :type ils_freq: float # Q: could/should it be int?
        :raises NotFoundError: If runway cannot be found.
        :raises InvalidAPIInputError: If latitude, longitude or heading are out
                of bounds.
        :returns: Runway info dictionary.
        :rtype: dict
        '''
        raise NotImplementedError


########################################
# Dummy API Handler


class AnalysisEngineAPIHandlerDummy(AnalysisEngineAPI):
    '''
    DummyAPIHandler will always raise NotFoundError.
    '''

    def get_airport(self, *args, **kwargs):
        '''
        '''
        raise NotFoundError('Dummy API handler always returns nothing...')

    def get_nearest_airport(self, *args, **kwargs):
        '''
        '''
        raise NotFoundError('Dummy API handler always returns nothing...')

    def get_nearest_runway(self, *args, **kwargs):
        '''
        '''
        raise NotFoundError('Dummy API handler always returns nothing...')


########################################
# HTTP API Handler


class AnalysisEngineAPIHandlerHTTP(AnalysisEngineAPI, APIHandlerHTTP):
    '''
    '''
    
    def get_aircraft(self, tail_number):
        '''
        Will either return an aircraft matching the tail number or raise an
        exception if one cannot be found.
        
        :param tail_number: Aircraft tail number.
        :type tail_number: str
        :raises NotFoundError: If the aircraft cannot be found.
        :returns: Aircraft info dictionary
        :rtype: dict
        '''
        from analysis_engine.settings import BASE_URL
        url = '%(base_url)s/api/aircraft/%(tail_number)s/' % {
            'base_url': BASE_URL.rstrip('/'),
            'tail_number': tail_number,
        }
        return self._attempt_request(url)['aircraft']
    
    def get_airport(self, code):
        '''
        Will either return an airport matching the code or raise an exception
        if one cannot be found.

        :param code: Either the id, ICAO or IATA of the airport.
        :type code: int or str
        :raises NotFoundError: If the airport cannot be found.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        from analysis_engine.settings import BASE_URL
        url = '%(base_url)s/api/airport/%(code)s/' % {
            'base_url': BASE_URL.rstrip('/'),
            'code': code,
        }
        return self._attempt_request(url)['airport']

    def get_nearest_airport(self, latitude, longitude):
        '''
        Either returns the nearest airport to the specified latitude and
        longitude, or raises an exception if one cannot be found.

        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :raises InvalidAPIInputError: If latitude or longitude are out of
                bounds.
        :returns: Airport info dictionary.
        :rtype: dict
        '''
        from analysis_engine.settings import BASE_URL
        url = '%(base_url)s/api/airport/nearest.json?ll=%(ll)s' % {
            'base_url': BASE_URL.rstrip('/'),
            'll': '%f,%f' % (latitude, longitude),
        }
        return self._attempt_request(url)['airport']

    def get_nearest_runway(self, airport, heading, latitude=None,
                           longitude=None, ils_freq=None, hint=None):
        '''
        Returns the nearest runway from the specified airport using latitude,
        longitude, precision and ils frequency.

        :param airport: Either ICAO code, IATA code or database ID of airport.
        :type airport: int or str
        :param heading: Magnetic heading.
        :type heading: int # Q: could it be float?
        :param latitude: Latitude in decimal degrees.
        :type latitude: float
        :param longitude: Longitude in decimal degrees.
        :type longitude: float
        :param ils_freq: ILS Localizer frequency of the runway in KHz.
        :type ils_freq: float # Q: could/should it be int?
        :param hint: Whether we are looking up a runway for 'takeoff',
                'landing', or 'approach'.
        :type hint: str
        :raises NotFoundError: If the runway cannot be found.
        :raises InvalidAPIInputError: If latitude, longitude or heading are out
                of bounds.
        :returns: Runway info in the format {'ident': '27*', 'items': [{# ...},
                {# ...},]}, 'ident' is either specific ('09L') or generalised
                ('09*').  'items' is a list of matching runways.
        :rtype: dict
        '''
        from analysis_engine.settings import BASE_URL
        url = '%(base_url)s/api/airport/%(airport)s/runway/nearest.json' % {
            'airport': airport,
            'base_url': BASE_URL.rstrip('/'),
        }

        params = {'heading': heading}
        if latitude and longitude:
            params['ll'] = '%f,%f' % (latitude, longitude)
        if ils_freq:
            # While ILS frequency is recorded in MHz, the API expects KHz.
            params['ilsfreq'] = int(ils_freq * 1000)
        if hint in ['takeoff', 'landing', 'approach']:
            params['hint'] = hint
        url += '?' + urllib.urlencode(params)
        runway = self._attempt_request(url)['runway']
        if not runway.get('end'):
            raise IncompleteEntryError(
                "Runway ident '%s' at '%s' has no end" %
                (runway.get('identifier', 'unknown'), airport))
        return runway


# Local API Handler
###################

class AnalysisEngineAPIHandlerLocal(AnalysisEngineAPI):
    
    
    @staticmethod
    def _load_data(path):
        '''
        Support loading both yaml and json. yaml is too slow for large files.
        '''
        if os.path.splitext(path)[1] == '.json':
            return simplejson.load(open(path, 'rb'))
        else:
            return yaml.load(open(path, 'rb'))
    
    def __init__(self):
        '''
        Load aircraft, airports and runways from yaml config files.
        '''
        from analysis_engine.settings import (
            LOCAL_API_AIRCRAFT_PATH,
            LOCAL_API_AIRPORT_PATH,
            LOCAL_API_RUNWAY_PATH,
            LOCAL_API_EXPORTS_PATH,
        )
        self.aircraft = self._load_data(LOCAL_API_AIRCRAFT_PATH)
        self.airports = self._load_data(LOCAL_API_AIRPORT_PATH)
        self.runways = self._load_data(LOCAL_API_RUNWAY_PATH)
        self.exports = self._load_data(LOCAL_API_EXPORTS_PATH)
    
    def get_aircraft(self, tail_number):
        '''
        Will either return an aircraft matching the tail number or raise an
        exception if one cannot be found.
        
        :param tail_number: Aircraft tail number.
        :type tail_number: str
        :raises NotFoundError: If the aircraft cannot be found.
        :returns: Aircraft info dictionary
        :rtype: dict
        '''
        try:
            return self.aircraft[tail_number]
        except KeyError:
            raise NotFoundError("Local API Handler: Aircraft with tail number "
                                "'%s' could not be found." % tail_number)
    
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
        for airport in self.airports:
            if code in (airport.get('id'), airport['code'].get('iata'),
                        airport['code'].get('icao')):
                break
        else:
            raise NotFoundError("Local API Handler: Airport with code '%s' "
                                "could not be found." % code)
        return airport
        
    
    def get_nearest_airport(self, latitude, longitude):
        '''
        Get the nearest airport from a pre-defined list.
        
        :param latitude: Latitude value for looking up a runway.
        :type latitude: float
        :param longitude: Longitude value for looking up a runway.
        :type longitude: float
        :returns: Airport dictionary.
        :rtype: dict
        '''
        airports = []
        for airport in self.airports:
            if 'latitude' not in airport or 'longitude' not in airport:
                continue
            airport = copy(airport)
            airport['distance'] = bearing_and_distance(latitude, longitude,
                                                       airport['latitude'],
                                                       airport['longitude'])[1]
            airports.append(airport)
        
        return min(airports, key=itemgetter('distance'))
    
    def get_nearest_runway(self, airport, heading, latitude=None,
                           longitude=None, ils_freq=None, hint=None):
        '''
        Get the nearest runway from a pre-defined list.
        
        :param airport: Not used.
        :param heading: Not used.
        :param latitude: Latitude value for looking up a runway.
        :type latitude: float
        :param longitude: Longitude value for looking up a runway.
        :type longitude: float
        :param ils_freq: Not used.
        :param hint: Not used.
        :raises NotFoundError:  If longitude or latitude is not defined.
        :returns: Runway dictionary.
        :rtype: dict
        '''
        if not latitude or not longitude:
            # Not precise
            if hint == 'landing':
                return self.runways[1]
        
            raise NotFoundError('Local API Handler: Runway could not be found')
        runways = []
        for runway in self.runways:
            runway = copy(runway)
            runway['distance'] = bearing_and_distance(
                latitude, longitude, runway['start']['latitude'],
                runway['start']['longitude'])[1]
            runways.append(runway)
        return min(runways, key=itemgetter('distance'))

    def get_data_exports(self, tail_number):
        '''
        Will either return data exports configuration for an aircraft matching
        the tail number or raise an exception if one cannot be found.

        :param tail_number: Aircraft tail number.
        :type tail_number: str
        :raises NotFoundError: If the aircraft cannot be found.
        :returns: Aircraft info dictionary
        :rtype: dict
        '''
        try:
            return self.exports[tail_number]
        except (KeyError, TypeError):
            raise NotFoundError("Local API Handler: Aircraft with tail number "
                                "'%s' could not be found." % tail_number)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
