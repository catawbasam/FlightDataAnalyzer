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

    @property
    def reference_settings(self):
        ref_settings = self.tables['vref'].keys()
        ref_settings.remove('weight')
        return ref_settings

    @property
    def v2_settings(self):
        v2_settings = self.tables['v2'].keys()
        v2_settings.remove('weight')
        return v2_settings

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
    source = ''  # FIXME: Populate this attribute.
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


class B737_800(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B737-800.

    Note: This table is available but should never be needed as the Boeing B737
    NG family of aircraft record the V2 and VREF parameters in the data frame.
    '''
    interpolate = True
    minimum_speed = 110
    source = 'B737-5_925017_07'
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 41,  45,  50,  54,  59,  63,  68,  73,  77,  82),
                   1: (167, 164, 160, 156, 151, 147, 142, 137, 131, 126),
                   5: (161, 158, 154, 150, 146, 141, 137, 132, 127, 121),
                  15: (156, 153, 149, 145, 141, 137, 133, 128, 123, 118),
        },
        'vref': {
            'weight': ( 41,  45,  50,  54,  59,  63,  68,  73,  77,  82),
                  15: (174, 169, 164, 159, 154, 148, 142, 135, 129, 122),
                  30: (165, 160, 156, 151, 146, 141, 135, 129, 123, 116),
                  40: (157, 153, 148, 144, 139, 133, 128, 122, 116, 109),
        }
    }


# FIXME: This is only applicable to RR RB211-535E4 engines!
class B757_200_RB211_535(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B757-200 with Rolls Royce 
    RB211-535E4 engines.
    '''
    interpolate = True
    source = 'Boeing Manual'  # Original name of source unknown.
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': (  62,   64,   66,   68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116),
                   5: (None, None, None, None, 130, 132, 134, 135, 137, 139, 141, 142, 144, 146, 147, 149, 150, 152, 154, 155, 157, 158, 159, 161, 163, 165, 166, 168),
                  15: ( 124,  124,  124,  124, 123, 125, 126, 128, 130, 131, 133, 135, 136, 138, 140, 141, 143, 144, 146, 147, 149, 151, 152, 153, 154, 156, 157, 159),
        },
        'vref': {
            'weight': (  62,   64,   66,   68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116),
                   5: (None, None, None, None, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 141, 143, 145, 147, 149, 151, 153, 155, 157, 158, 160, 162, 164),
                  15: ( 111,  117,  117,  117, 114, 116, 118, 120, 122, 123, 125, 127, 129, 131, 133, 135, 137, 139, 140, 141, 143, 145, 147, 149, 150, 152, 154, 156),
                  30: ( 107,  109,  111,  113, 115, 116, 118, 120, 122, 123, 125, 127, 129, 131, 132, 134, 136, 137, 139, 141, 142, 144, 146, 147, 149, 150, 152, 154),
        },
    }

class B757_200_RB211_535C_37(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B757-200 with Rolls Royce 
    RB211-535C-37 engines.
    '''
    interpolate = True
    source = 'Customer 20 ticket #243'
    weight_unit = 't'
    tables = {
        'v2': { # TODO: this is copy of 575 v2 table above, update once v2 table received.
            'weight': (  62,   64,   66,   68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116),
                   5: (None, None, None, None, 130, 132, 134, 135, 137, 139, 141, 142, 144, 146, 147, 149, 150, 152, 154, 155, 157, 158, 159, 161, 163, 165, 166, 168),
                  15: ( 124,  124,  124,  124, 123, 125, 126, 128, 130, 131, 133, 135, 136, 138, 140, 141, 143, 144, 146, 147, 149, 151, 152, 153, 154, 156, 157, 159),
        },
        'vref': {
            'weight': ( 60,  70,  80,  90, 100, 110, 120),
                  20: (115, 121, 127, 133, 139, 145, 150),
                  25: (111, 117, 123, 129, 134, 140, 144),
                  30: (108, 114, 120, 126, 132, 137, 142),
        },
    }


class B767(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767.
    '''
    interpolate = True
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
    interpolate = True
    source = 'FDS Customer #78'
    weight_unit = 'lb'
    tables = {
        'v2': {
            'weight': (220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000),
                   1: (   135,    140,    145,    150,    155,    160,    164,    169),
                   5: (   130,    135,    140,    144,    149,    154,    158,    162),
                  15: (   123,    128,    133,    138,    142,    146,    151,   None),
                  20: (   120,    125,    129,    134,    138,    143,    148,   None),
        },
        'vref': {
            'weight': (220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000),
                  20: (   129,    135,    141,    146,    151,    156,    161,    165),
                  25: (   126,    132,    137,    142,    147,    152,    157,    161),
                  30: (   122,    127,    133,    138,    143,    147,    152,    156),
        },
    }


class B767_300_CF6_80C2(VelocitySpeed):
    '''
    Velocity speed tables for Boeing B767-300 w/ GE CF6-80C2.
    '''
    interpolate = True
    source = 'FDS Customer #78'
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
    interpolate = True
    source = 'FDS Customer #78'
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

class F28_0070(VelocitySpeed):
    '''
    Velocity speed tables for Fokker F28-0070 (Fokker 70).
    '''
    interpolate = True
    source = ''
    weight_unit = 't'
    tables = {
        'v2': {
            'weight': ( 26,  28,  30,  32,  34,  36,  38,  40,  42),
                   0: (117, 119, 123, 127, 131, 135, 139, 143, 146),
                   8: (114, 115, 118, 122, 126, 129, 133, 135, 139),
                  15: (112, 113, 115, 117, 120, 124, 127, 131, 134),
        },
        'vref': {
            'weight': ( 26,  28,  30,  32,  34,  36,  38,  40,  42),
                   0: (124, 129, 133, 137, 142, 146, 150, 154, 157),
                  25: (113, 117, 121, 125, 129, 133, 137, 140, 143),
                  42: (104, 108, 112, 115, 119, 122, 126, 129, 132),
        },
    }

##############################################################################
# Constants

VELOCITY_SPEED_MAP = {
    # All combinations listed
    # Boeing
    ('B737-300', None): B737_300,
    ('B737-300(QC)', None): B737_300,
    ('B737-400', None): B737_400,
    ('B737-500', None): B737_500,
    ('B737-700', None): B737_700,
    ('B737-800', None): B737_800,

    ('B757-200', 'RB211-535'): B757_200_RB211_535,
    ('B757-200(PCF)', 'RB211-535'): B757_200_RB211_535,
    ('B757-200', 'RB211-535C-37'): B757_200_RB211_535C_37,
    ('B757-200(F)', 'RB211-535C-37'): B757_200_RB211_535C_37,

    ('B767', None): B767,
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
    
    # Fokker
    ('F28-0070', None): F28_0070,
}


##############################################################################
# Functions


def get_vspeed_map(series=None, family=None, engine_series=None, engine_type=None):
    '''
    Accessor for fetching velocity speed table classes.

    :param series: An aircraft series e.g. B737-300
    :type series: string
    :param family: An aircraft family e.g. B737
    :type family: string
    :param engine_series: An engine series e.g. CF6-80C2
    :type engine_series: string
    :returns: associated VelocitySpeed class
    :rtype: VelocitySpeed
    :raises: KeyError -- if no velocity speed mapping found.
    '''
    lookup_combinations = ((series, engine_type),
                           (family, engine_type),
                           (series, engine_series),
                           (family, engine_series),
                           (series, None),
                           (family, None))

    for combination in lookup_combinations:
        if combination in VELOCITY_SPEED_MAP:
            return VELOCITY_SPEED_MAP[combination]

    msg = "No velocity speed table mapping for series '%s' or family '%s'."
    raise KeyError(msg % (series, family))


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
