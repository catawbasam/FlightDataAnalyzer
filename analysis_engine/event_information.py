# -*- coding: utf-8 -*-
################################################################################

'''
'''

################################################################################
# Imports


import logging

from hdfaccess.file import hdf_file


################################################################################
# Exports


__all__ = ['populate_events']


################################################################################
# Globals


logger = logging.getLogger(__name__)


################################################################################
# Functions


def populate_events(hdf_path, events_info):
    '''
    '''
    logger.info('Populating events %r from HDF: %s',
        sorted(events_info.keys()), hdf_path)

    with hdf_file(hdf_path) as hdf:

        def lookup(parameter, offset):
            '''
            '''
            if not parameter in hdf:
                return None
            return hdf[parameter].get(offset)

        for id, info in events_info.iteritems():

            logger.debug('Populating event #%d with extra data...', id)

            offset = info['segment_offset']

            # 1. Determine duration of the event:
            info['duration'] = None  # TODO

            # 2. Determine the flight phase in which the event occurred:
            info['flight_phase'] = None  # TODO: Determine how this works!

            # 3. Determine the coordinates at which the event occurred:
            latitude = lookup('Latitude Smoothed', offset)
            longitude = lookup('Longitude Smoothed', offset)
            if latitude and longitude:
                info['coordinates'] = (latitude, longitude)
            else:
                info['coordinates'] = None

            # 4. Record additional information about the event for comments:
            info['variables'] = {}  # TODO

        return events_info


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
