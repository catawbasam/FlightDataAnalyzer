# -*- coding: utf-8 -*-
##############################################################################

'''
'''

##############################################################################
# Imports


import urllib

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
        :raises NotFoundError: If airport cannot be found.
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
        :raises NotFoundError: If runway cannot be found.
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
            raise IncompleteEntryError("Runway ident '%s' at '%s' has no end" % (runway.get('identifier', 'unknown'), airport))
        return runway


# Local API Handler
###################

class AnalysisEngineAPIHandlerLocal(AnalysisEngineAPI):
    
    airports = [
        {'code': {
            'iata': 'KRS',
            'icao': 'ENCN',
            },
         'elevation': 43,
         'id': 2456,
         'latitude': 58.2042,
         'location': {
             'city': 'Kjevik',
             'country': 'Norway',
             },         
         'longitude': 8.08537,
         'magnetic_variation': 'E000091 0106',
         'name': 'Kristiansand Lufthavn Kjevik',
         },
        {'code': {
             'iata': 'OSL',
             'icao': 'ENGM',
             },
         'elevation': 689,
         'id': 2461,
         'latitude': 60.1939,
         'location': {
             'city': 'Oslo',
             'country': 'Norway'
             },
         'longitude': 11.1004,
         'magnetic_variation': 'E001226 0106',
         'name': 'Oslo Gardermoen',
         },
    ]
    runways = [
        {'end': {
            'elevation': 43,
            'latitude': 58.211678,
            'longitude': 8.095269,
            },
         'glideslope': {
             'angle': 3.4, 
             'elevation': 39, 
             'latitude': 58.198664, 
             'longitude': 8.080164, 
             'threshold_distance': 720
             },
         'id': 8127,
         'identifier': '04',
         'localizer': {
             'beam_width': 4.5,
             'elevation': 43,
             'frequency': 110300.0,
             'heading': 36,
             'latitude': 58.212397,
             'longitude': 8.096228,
             },         
         'magnetic_heading': 33.9,
         'start': {
             'elevation': 26,
             'latitude': 58.196703,
             'longitude': 8.075406,
             },         
         'strip': {
             'id': 4064,
             'length': 6660,
             'surface': 'ASP',
             'width': 147,
             },         
         },
        {'end': {
            'elevation': 682,
            'latitude': 60.216092,
            'longitude': 11.091397,
            },
         'glideslope': {
             'angle': 3.0,
             'elevation': 669,
             'latitude':  60.186858,
             'longitude':  11.072234,
             'threshold_distance': 943
             },
         'id': 8151,
         'identifier': '01L',
         'localizer': {
             'beam_width': 4.5,
             'elevation': 686,
             'frequency': 110300.0,
             'heading': 16,
             'latitude': 60.219775,
             'longitude': 11.093536,
             },
         'magnetic_heading': 13.7,
         'start': {
             'latitude': 60.185019,
             'elevation': 650,
             'longitude': 11.073491,
             },
         
         'strip': {
             'width': 147,
             'length': 11811,
             'id': 4076,
             'surface': 'ASP',
             }, 
         },
    ]
    
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
            raise NotFoundError("Local API Handler: Airport with code '%s' could not be found." % 
                                code)
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
        
            raise NotFoundError('Local API Handler: Runway not found')
        runways = []
        for runway in self.runways:
            runway = copy(runway)
            runway['distance'] = bearing_and_distance(
                latitude, longitude, runway['start']['latitude'],
                runway['start']['longitude'])[1]
            runways.append(runway)
        return min(runways, key=itemgetter('distance'))


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
