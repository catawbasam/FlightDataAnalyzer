# -*- coding: utf-8 -*-
################################################################################

'''
Custom exceptions raised by the flight data analyser.
'''

################################################################################
# Exceptions


# NOTE: Take care when creating new exceptions. They most likely will need to
#       be pickleable to prevent errors when running the analysis engine
#       through celery.
#
# http://docs.celeryproject.org/en/latest/userguide/tasks.html#creating-pickleable-exceptions


class DataFrameError(Exception):
    '''
    Error handling for cases where new LFLs require a derived parameter
    algorithm which has not yet been programmed.
    '''

    def __init__(self, param_name, frame_name):
        '''
        :param param_name: Recorded parameter where frame specific processing
                           is required.
        :type param_name: string
        :param frame_name: Data frame identifier. For this parameter, no frame
                           condition evaluates true.
        :type frame_name: string
        '''
        self.param_name = param_name
        self.frame_name = frame_name
        super(DataFrameError, self).__init__(param_name, frame_name)

    def __str__(self):
        '''
        '''
        return "Parameter '%s' not in LFL and no procedure for frame '%s'." \
            % (self.param_name, self.frame_name)



class FlightPhaseError(Exception):
    '''
    Error raised when an unexpected issues is encountered within a Flight Phase
    '''
    pass

################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
