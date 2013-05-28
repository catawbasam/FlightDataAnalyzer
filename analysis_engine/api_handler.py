# -*- coding: utf-8 -*-
##############################################################################

'''
'''

##############################################################################
# Imports


import httplib
import httplib2
import logging
import os
import simplejson as json
import socket
import time
import urllib

from analysis_engine.settings import API_PROXY_INFO, CA_CERTIFICATE_FILE


##############################################################################
# Globals

logger = logging.getLogger(name=__name__)

TIMEOUT = 15


##############################################################################
# Exceptions


class APIError(Exception):
    '''
    A generic exception class for an error when calling an API.
    '''

    def __init__(self, message, uri=None, method=None, body=None):
        '''
        '''
        super(APIError, self).__init__(message)
        self.uri = uri
        self.method = method
        self.body = body


class APIConnectionError(APIError):
    '''
    An exception to be raised when unable to connect to an API.
    '''
    pass


class InvalidAPIInputError(APIError):
    '''
    An exception to be raised when input to an API in not valid.
    '''
    pass


class IncompleteEntryError(APIError):
    '''
    An exception to be raised when and entry does not contain all required data.
    '''
    pass


class NotFoundError(APIError):
    '''
    An exception to be raised when something could not be found via the API.
    '''
    pass


class UnknownAPIError(APIError):
    '''
    An exception to be raised when some unexpected API error occurred.
    '''
    pass


##############################################################################
# HTTP API Handler


class APIHandlerHTTP(object):
    '''
    Restful HTTP API Handler.
    '''

    def __init__(self, attempts=3, delay=2):
        '''
        Initialises an HTTP API handler.

        :param attempts: Number of retry attempts before raising an exception.
        :type attempts: int
        :param delay: Time to wait between retrying requests.
        :type delay: int or float
        '''
        self.attempts = max(attempts, 1)
        self.delay = abs(delay)

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
        :raises APIConnectionError: If the server does not respond or returns
                401.
        :raises UnknownAPIError: If the server returns 500 or an unexpected
                status code.
        :raises JSONDecodeError: If status code is 200, but content is not
                JSON.
        '''
        # Prepare the request object:
        body = urllib.urlencode(body)
        disable_validation = not os.path.exists(CA_CERTIFICATE_FILE)
        http = httplib2.Http(
            ca_certs=CA_CERTIFICATE_FILE,
            disable_ssl_certificate_validation=disable_validation,
            timeout=timeout,
            proxy_info=API_PROXY_INFO,
        )
        
        # Attempt to make the API request:
        socket_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            response, content = http.request(uri, method, body)
        except (httplib2.ServerNotFoundError, socket.error, AttributeError):
            # Usually a result of errors with DNS...
            raise APIConnectionError(uri, method, body)
        finally:
            socket.setdefaulttimeout(socket_timeout)

        # Check the status code of the response:
        status = int(response['status'])
        if not status == httplib.OK:
            # Try to get an 'error' message from JSON if available:
            try:
                message = json.loads(content)['error']
            except (json.JSONDecodeError, KeyError):
                message = ''
            # Try to get an 'error' message from JSON if available:
            raise {
                httplib.BAD_REQUEST: InvalidAPIInputError,
                httplib.UNAUTHORIZED: APIConnectionError,
                httplib.NOT_FOUND: NotFoundError,
                httplib.INTERNAL_SERVER_ERROR: UnknownAPIError,
            }.get(status, UnknownAPIError)(message, uri, method, body)

        # Attempt to decode the response:
        try:
            # TODO: Set use_decimal to improve accuracy?
            return json.loads(content)
        except json.JSONDecodeError:
            # Only JSON return types supported, any other return means server
            # is not configured correctly
            logger.exception("JSON decode error for '%s' - only JSON "
                             "supported by this API. Server configuration "
                             "error? %s\nBody: %s", method, uri, body)
            raise

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
        error = None
        for attempt in range(self.attempts):
            try:
                logger.info('API Request args: %s | kwargs: %s', args, kwargs)
                return self._request(*args, **kwargs)
            except (APIConnectionError, UnknownAPIError) as error:
                msg = "'%s' error in request, retrying in %.2f seconds..."
                logger.exception(msg, error, self.delay)
                time.sleep(self.delay)
        if isinstance(error, Exception):
            raise error


##############################################################################
# API Handler Lookup Function


def get_api_handler(handler_path, *args, **kwargs):
    '''
    Returns an instance of the class specified by the handler_path.

    :param handler_path: Path to handler module, e.g. project.module.APIHandler
    :type handler_path: string
    :param args: Handler class instantiation args.
    :type args: list
    :param kwargs: Handler class instantiation kwargs.
    :type kwargs: dict
    '''
    import_path_split = handler_path.split('.')
    class_name = import_path_split.pop()
    module_path = '.'.join(import_path_split)
    handler_module = __import__(module_path, globals(), locals(),
                                fromlist=[class_name])
    handler_class = getattr(handler_module, class_name)
    return handler_class(*args, **kwargs)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
