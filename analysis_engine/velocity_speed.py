# -*- coding: utf-8 -*-
##############################################################################

'''
Utilities for looking up velocity speeds for various aircraft.

Note that we do not take every factor into account when selecting a value from
the lookup tables as we cannot always determine suitable values for some of the
parameters (temperature, altitude, runway slope, tailwind, etc.) nor presume to
guess what value was chosen for a particular flight.

It is necessary to ensure that flap/conf values reflect those held for the
aircraft family or series in the model information module.

When deciding on the table to select, we use the following order of precedence:

- aircraft series, engine series.
- aircraft family, engine series.
- aircraft series.
- aircraft family.

'''

##############################################################################
# Imports


import logging
import numpy as np

from abc import ABCMeta
from bisect import bisect_left

import scipy.interpolate as interp

from flightdatautilities import units


##############################################################################
# Globals


logger = logging.getLogger(__name__)


##############################################################################
# Abstract Classes


class VelocitySpeed(object):
    '''
    '''

    __meta__ = ABCMeta

    interpolate = False
    minimum_speed = None
    source = None
    weight_unit = 'kg'  # Can be one of 'lb', 'kg', 't'.

    tables = {
        'v2': {'weight': ()},
        'vref': {'weight': ()},
    }

    def v2(self, weight, setting):
        '''
        Look up a value for V2.

        Will use interpolation if configured and convert units if necessary.

        None will be returned if weight is outside of the table range or no
        entries are available in the table for the provided flap/conf value.

        :param weight: Weight of the aircraft.
        :type weight: float
        :param setting: Flap or conf setting to use in lookup.
        :type setting: string
        :returns: V2 value or None.
        :rtype: float
        :raises: KeyError -- when table or flap/conf settings is not found.
        :raises: ValueError -- when weight units cannot be converted.
        '''
        return self._get_velocity_speed(self.tables['v2'], weight, setting)

    def vref(self, weight, setting):
        '''
        Look up a value for Vref.

        Will use interpolation if configured and convert units if necessary.

        None will be returned if weight is outside of the table range or no
        entries are available in the table for the provided flap/conf value.

        :param weight: Weight of the aircraft.
        :type weight: float
        :param setting: Flap or conf setting to use in lookup.
        :type setting: string
        :returns: Vref value or None.
        :rtype: float
        :raises: KeyError -- when table or flap/conf settings is not found.
        :raises: ValueError -- when weight units cannot be converted.
        '''
        return self._get_velocity_speed(self.tables['vref'], weight, setting)

    def _get_velocity_speed(self, lookup, weight, setting):
        '''
        Looks up the velocity speed in the provided lookup table.

        Will use interpolation if configured and convert units if necessary.

        None will be returned if weight is outside of the table range or no
        entries are available in the table for the provided flap/conf value.

        :param lookup: The velocity speed lookup table.
        :type lookup: dict
        :param weight: Weight of the aircraft.
        :type weight: float
        :param setting: Flap or conf setting to use in lookup.
        :type setting: string
        :returns: A velocity speed value or None.
        :rtype: float
        :raises: KeyError -- when flap/conf settings is not found.
        :raises: ValueError -- when weight units cannot be converted.
        '''
        # Convert the aircraft weight to match the lookup table:
        weight = units.convert(weight, 'kg', self.weight_unit)

        if setting not in lookup:
            msg = "Velocity speed table '%s' has no entry for flap/conf '%s'."
            arg = (self.__class__.__name__, setting)
            logger.error(msg, *arg)
            raise KeyError(msg % arg)

        wt = lookup['weight']
        if not min(wt) <= weight <= max(wt) or weight is np.ma.masked:
            msg = "Weight '%s' outside of range for velocity speed table '%s'."
            arg = (weight, self.__class__.__name__)
            logger.warning(msg, *arg)
            return None

        # Determine the value for the velocity speed:
        if self.interpolate:
            f = interp.interp1d(lookup['weight'], lookup[setting])
            value = f(weight)
        else:
            index = bisect_left(lookup['weight'], weight)
            value = lookup[setting][index]

        # Return a minimum speed if we have one and the value is below it:
        if self.minimum_speed is not None and value < self.minimum_speed:
            return self.minimum_speed

        return value


##############################################################################
# Velocity Speed Table Classes


class B737_300(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B737-300.
    '''
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                   1: (124, 131, 138, 145, 153, 160, 168),
                   5: (119, 126, 132, 139, 146, 153, 160),
                  15: (113, 120, 126, 132, 139, 145, 152),
        },
        'vref': {
            'weight': ( 32,  36,  40,  44,  48,  52,  56,  60,  64),
                  15: (111, 118, 125, 132, 138, 143, 149, 154, 159),
                  30: (105, 111, 117, 123, 129, 135, 140, 144, 149),
                  40: (101, 108, 114, 120, 125, 130, 135, 140, 145),
        },
    }


class B737_400(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B737-400.
    '''
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 40,  45,  50,  55,  60,  65,  70),
                   5: (130, 136, 143, 149, 155, 162, 168),
        },
        'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65,  70,  70),
                  15: (123, 132, 141, 149, 156, 164, 171, 177, 177),
                  30: (111, 119, 127, 134, 141, 147, 154, 159, 159),
                  40: (109, 116, 124, 130, 137, 143, 149, 155, 155),
        },
    }


class B737_500(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B737-500.
    '''
    interpolate = True
    minimum_speed = 109
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 30,  35,  40,  45,  50,  55,  60,  65),
                   5: (112, 119, 126, 133, 139, 146, 152, 157),
                  15: (107, 113, 120, 126, 132, 138, 143, 146),
        },
        'vref': {
            'weight': ( 36,  40,  44,  48,  52,  56,  60),
                  15: (118, 125, 132, 137, 143, 149, 153),
                  30: (111, 117, 125, 130, 134, 140, 144),
                  40: (108, 114, 121, 126, 130, 135, 140),
        },
    }


class B737_700(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B737-700.

    Note: This table is available but should never be needed as the Boeing B737
    NG family of aircraft record the V2 and VREF parameters in the data frame.
    '''
    interpolate = True
    minimum_speed = 110
    source = 'B737-5_925017_07'
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 40,  45,  49,  54,  58,  63,  68,  72),
                   1: (114, 120, 126, 132, 137, 142, 147, 152),
                   5: (111, 117, 123, 129, 134, 139, 144, 148),
                  15: (107, 113, 117, 122, 127, 131, 135, 138),
        },
        'vref': {
            'weight': ( 40,  45,  49,  54,  58,  63,  68,  72,  77),
                  15: (115, 121, 127, 133, 139, 145, 150, 155, 159),
                  30: (111, 117, 123, 129, 134, 140, 144, 149, 153),
                  40: (108, 114, 120, 126, 132, 137, 142, 147, 151),
        },
    }


class B767(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767.
    '''
    interpolate = False
    source = '767 Flight Crew Operations Manual'
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                   5: (127, 134, 139, 145, 151, 156, 161, 166, 171, 176),
                  15: (122, 128, 134, 139, 144, 149, 154, 159, 164, 168),
                  20: (118, 124, 129, 134, 140, 144, 149, 154, 159, 164),
        },
        'vref': {
            # FIXME: Flap detents look wrong here!
            'weight': (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                   5: (114, 121, 128, 134, 141, 147, 153, 158, 164, 169),
                  15: (109, 116, 122, 129, 141, 135, 146, 151, 157, 162),
                  20: (105, 111, 118, 124, 130, 135, 141, 147, 152, 158),
        },
    }


class B767_200_CF6_80A(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767-200 w/ GE CF6-80A.
    '''
    interpolate = False
    source = ''  # FIXME: Populate this attribute.
    weight_unit = 'lb'
    tables = {
        'v2': {
            'weight': (220, 240, 260, 280, 300, 320, 340, 360),
                   1: (135, 140, 145, 150, 155, 160, 164, 169),
                   5: (130, 135, 140, 144, 149, 154, 158, 162),
                  15: (123, 128, 133, 138, 142, 146, 151, None),
                  20: (120, 125, 129, 134, 138, 143, 148, None),
        },
        'vref': {
            'weight': (220, 240, 260, 280, 300, 320, 340, 360),
                  20: (129, 135, 141, 146, 151, 156, 161, 165),
                  25: (126, 132, 137, 142, 147, 152, 157, 161),
                  30: (122, 127, 133, 138, 143, 147, 152, 156),
        },
    }


class B767_300_CF6_80C2(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767-300 w/ GE CF6-80C2.
    '''
    interpolate = False
    source = ''  # FIXME: Populate this attribute.
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                   5: (128, 134, 140, 145, 151, 156, 161, 166, 171, 175),
                  15: (122, 128, 134, 139, 144, 150, 154, 159, 164, 168),
                  20: (118, 124, 129, 134, 139, 145, 149, 154, 159, 165),
        },
        'vref': {
            'weight': (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                  20: (128, 135, 141, 146, 151, 157, 162, 168, 173, 179),
                  25: (123, 129, 135, 141, 146, 151, 156, 161, 166, 170),
                  30: (119, 125, 131, 137, 142, 148, 156, 164, 171, 179),
        },
    }


class B767_300_PW4000_94(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767-300 w/ P&W 4000-94.
    '''
    interpolate = False
    source = ''  # FIXME: Populate this attribute.
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                   5: (121, 127, 134, 140, 145, 151, 156, 161, 166, 171, 175),
                  15: (116, 122, 128, 134, 139, 144, 149, 154, 159, 164, 168),
                  20: (112, 118, 124, 129, 135, 140, 144, 149, 154, 160, 165),
        },
        'vref': {
            'weight': (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                  20: (128, 134, 140, 146, 152, 157, 162, 168, 174, 179),
                  25: (123, 129, 135, 140, 146, 151, 156, 161, 166, 170),
                  30: (119, 125, 131, 137, 142, 148, 156, 164, 171, 179),
        },
    }


##############################################################################
# Constants


AIRCRAFT_FAMILY_VELOCITY_SPEED_MAP = {
    'B767': B767,
}


AIRCRAFT_SERIES_VELOCITY_SPEED_MAP = {
    'B737-300': B737_300,
    'B737-300(QC)': B737_300,
    'B737-400': B737_400,
    'B737-500': B737_500,
}


AIRCRAFT_FAMILY_ENGINE_SERIES_VELOCITY_SPEED_MAP = {
}


AIRCRAFT_SERIES_ENGINE_SERIES_VELOCITY_SPEED_MAP = {
    # All combinations listed
    # TODO: better lookup solution needed
    ('B767-200', 'CF6-80A'): B767_200_CF6_80A,
    ('B767-200(F)', 'CF6-80A'): B767_200_CF6_80A,
    ('B767-200(ER)', 'CF6-80A'): B767_200_CF6_80A,
    ('B767-200(ER/F)', 'CF6-80A'): B767_200_CF6_80A,
    ('B767-300', 'CF6-80C2'): B767_300_CF6_80C2,
    ('B767-300(ER)', 'CF6-80C2'): B767_300_CF6_80C2,
    ('B767-300F(ER)', 'CF6-80C2'): B767_300_CF6_80C2,
    ('B767-300(ER/F)', 'CF6-80C2'): B767_300_CF6_80C2,
    ('B767-300', 'PW4000-94'): B767_300_PW4000_94,
    ('B767-300(ER)', 'PW4000-94'): B767_300_PW4000_94,
    ('B767-300F(ER)', 'PW4000-94'): B767_300_PW4000_94,
    ('B767-300(ER/F)', 'PW4000-94'): B767_300_PW4000_94,
}


##############################################################################
# Functions


def get_vspeed_map(series=None, family=None, engine=None):
    '''
    Accessor for fetching velocity speed table classes.

    :param series: An aircraft series e.g. B737-300
    :type series: string
    :param family: An aircraft family e.g. B737
    :type family: string
    :param engine: An engine series e.g. CF6-80C2
    :type engine: string
    :returns: associated VelocitySpeed class
    :rtype: VelocitySpeed
    :raises: KeyError -- if no velocity speed mapping found.
    '''
    if (series, engine) in AIRCRAFT_SERIES_ENGINE_SERIES_VELOCITY_SPEED_MAP:
        return AIRCRAFT_SERIES_ENGINE_SERIES_VELOCITY_SPEED_MAP[(series, engine)]

    if (family, engine) in AIRCRAFT_FAMILY_ENGINE_SERIES_VELOCITY_SPEED_MAP:
        return AIRCRAFT_FAMILY_ENGINE_SERIES_VELOCITY_SPEED_MAP[(family, engine)]

    if series in AIRCRAFT_SERIES_VELOCITY_SPEED_MAP:
        return AIRCRAFT_SERIES_VELOCITY_SPEED_MAP[series]

    if family in AIRCRAFT_FAMILY_VELOCITY_SPEED_MAP:
        return AIRCRAFT_FAMILY_VELOCITY_SPEED_MAP[family]

    msg = "No velocity speed table mapping for series '%s' or family '%s'."
    raise KeyError(msg % (series, family))


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
