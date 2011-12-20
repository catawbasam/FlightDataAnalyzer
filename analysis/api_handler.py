from abc import ABCMeta, abstractmethod

from analysis.settings import HANDLER


class APIError(Exception):
    def __init__(self, message, uri=None, method=None, body=None):
        super(APIError, self).__init__(message)
        self.uri = uri
        self.method = method
        self.body = body


class APIConnectionError(APIError):
    pass


class InvalidAPIInputError(APIError):
    pass


class NotFoundError(APIError):
    pass


class UnknownAPIError(APIError): # Q: Name?
    pass


class APIHandler(object):
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
        :returns: Airport info dictionary or None if the airport cannot be found.
        :rtype: dict or None
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
        :returns: Runway info dictionary or None if the runway cannot be found.
        :rtype: dict or None
        '''
        raise NotImplementedError
    
    # TODO: Determine method signature...
    @abstractmethod
    def get_vspeed_limit(self, *args, **kwargs):
        '''
        TODO: Define what this does..
        '''
        raise NotImplementedError


def get_api_handler(*args, **kwargs):
    '''
    Returns an instance of the class specified by the settings.HANDLER import
    path.
    
    :param args: Handler class instantiation args.
    :type args: list
    :param kwargs: Handler class instantiation kwargs.
    :type kwargs: dict
    '''
    import_path_split = HANDLER.split('.')
    class_name = import_path_split[-1]
    module_path = '.'.join(import_path_split[:-1])
    handler_module = __import__(module_path, globals(), locals(),
                               fromlist=[class_name])
    handler_class = getattr(handler_module, class_name)
    return handler_class(*args, **kwargs)