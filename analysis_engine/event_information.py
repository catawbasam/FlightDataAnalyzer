# -*- coding: utf-8 -*-
################################################################################

'''
'''

################################################################################
# Imports


import logging


################################################################################
# Exports


__all__ = ['populate_events']


################################################################################
# Globals


logger = logging.getLogger(__name__)


################################################################################
# Functions


def populate_events(hdf_file_path, events_info):
    '''
    '''
    for id, info in events_info:

        # 1. Determine duration of the event:
        info['duration'] = None  # TODO

        # 2. Determine the flight phase in which the event occurred:
        info['flight_phase'] = None  # TODO

        # 3. Determine the coordinates at which the event occurred:
        info['latitude'] = None  # TODO
        info['longitude'] = None  # TODO

        # 4. Record additional information about the event for comments:
        info['variables'] = {}  # TODO

    return events_info


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
