import httplib
import httplib2
import logging
import os
import socket
import time
import urllib
try:
    import simplejson as json
except ImportError:
    import json

from analysis_engine import settings

TIMEOUT = 15
socket.setdefaulttimeout(TIMEOUT)

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



class APIHandlerHTTP(object):
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
        disable_validation = not os.path.exists(settings.CA_CERTIFICATE_FILE)
        http = httplib2.Http(
            ca_certs=settings.CA_CERTIFICATE_FILE,
            disable_ssl_certificate_validation=disable_validation,
            timeout=timeout,
        )
        try:
            resp, content = http.request(uri, method, body)
        except (httplib2.ServerNotFoundError, socket.error, AttributeError): # DNS..
            raise APIConnectionError(uri, method, body)
        status = int(resp['status'])
        try:
            decoded_content = json.loads(content)
        except ValueError:
            # Only JSON return types supported, any other return means server
            # is not configured correctly
            logging.exception("JSON decode error for '%s' - only JSON supported"
                             " by this API. Server configuration error? %s\nBody: %s",
                             method, uri, body)
            raise
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
            except (APIConnectionError, UnknownAPIError) as error:
                logging.exception("'%s' error in request, retrying in %.2f", 
                                  error, self.delay)
                time.sleep(self.delay)
        raise error
    



def get_api_handler(handler_path, *args, **kwargs):
    '''
    Returns an instance of the class specified by the hanlder_path.
    
    :param handler_path: Path to handler module, e.g. project.module.APIHandler
    :type handler_path: string
    :param args: Handler class instantiation args.
    :type args: list
    :param kwargs: Handler class instantiation kwargs.
    :type kwargs: dict
    '''
    import_path_split = handler_path.split('.')
    class_name = import_path_split[-1]
    module_path = '.'.join(import_path_split[:-1])
    handler_module = __import__(module_path, globals(), locals(),
                               fromlist=[class_name])
    handler_class = getattr(handler_module, class_name)
    return handler_class(*args, **kwargs)


