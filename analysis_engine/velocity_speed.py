#!/usr/bin/env python
#coding:utf-8
'''
we do not take into account temperature, altitude 

Need to ensure flap/conf settings reflect those held for family/series in
model_information.
'''
import logging
import numpy as np

from bisect import bisect_left
import scipy.interpolate as interp

from settings import (
    KG_TO_LB
)

logger = logging.getLogger(__name__)

class VelocitySpeed(object):
    '''
    unit = 1, 10's 1000's etc
    weight_unit = t, kg, lb
    '''
    weight_unit = 'kg'
    interpolate = False
    source = ''

    v2_table = {'weight': ()}
    airspeed_reference_table = {'weight': ()}

    def __init__(self):
        assert 'weight' in self.v2_table, 'Weight not in V2 lookup table'
        assert 'weight' in self.airspeed_reference_table, \
               'Weight not in Vref lookup table'

    def v2(self, weight, setting):
        '''
        lookup v2 value using interpolation if set
        converts value to kg if weight_unit in lb

        :param weight: Weight of aircraft
        :type weight: float
        :param setting: Flap/Conf setting to use in lookup
        :type setting: String
        :raises: ValueError
        :returns: v2 value
        :rtype: float
        '''
        return self._get_vspeed(self.v2_table, weight, setting)

    def airspeed_reference(self, weight, setting):
        '''
        lookup v2 value using interpolation if set
        converts value to kg if weight_unit in lb
        returns None if weight is outside of table range or no entries in table
        for Flap/Conf setting

        :param weight: Weight of aircraft
        :type weight: float
        :param setting: Flap/Conf setting to use in lookup
        :type setting: String
        :returns: v2 value
        :rtype: float
        '''
        return self._get_vspeed(self.airspeed_reference_table,
                                weight,
                                setting)

    def _get_vspeed(self, lookup, aircraft_weight, setting):
        '''
        
        '''
        if self.weight_unit == 'lb':
            # Convert to tonnes
            weight = aircraft_weight / KG_TO_LB / 1000.0
        elif self.weight_unit == 'kg':
            # Convert to tonnes
            weight = aircraft_weight / 1000.0
        elif self.weight_unit == 't':
            weight = aircraft_weight
        else:
            raise ValueError("Unrecognised weight units '%s'" %
                             self.weight_unit)

        if setting not in lookup:
            msg = "Vspeed table '%s' does not have v_speed entry for '%s' flap." % (
                           self.__class__.__name__,
                           setting)
            logger.error(msg)
            raise KeyError(msg)

        if weight < lookup['weight'][0] or \
           weight > lookup['weight'][-1] or \
           weight is np.ma.masked:
            logger.warning("Weight of '%s' is outside of table range for '%s'",
                           weight,
                           self.__class__.__name__)
            return None

        if self.interpolate:
            # numpy interpolate
            f = interp.interp1d(lookup['weight'], lookup[setting])
            value = f(weight)
        else:
            # bisect lookup
            value_index = bisect_left(lookup['weight'], weight)
            value = lookup[setting][value_index]
        return value


class B737_300(VelocitySpeed):
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 'kg'
    v2_table = {
             'weight': ( 35,  40,  45,  50,  55,  60,  65),
                    1: (124, 131, 138, 145, 153, 160, 168),
                    5: (119, 126, 132, 139, 146, 153, 160),
                   15: (113, 120, 126, 132, 139, 145, 152),
    }
    airspeed_reference_table = {
             'weight':  ( 32,  36,  40,  44,  48,  52,  56,  60,  64),
                   15:  (111, 118, 125, 132, 138, 143, 149, 154, 159),
                   30:  (105, 111, 117, 123, 129, 135, 140, 144, 149),
                   40:  (101, 108, 114, 120, 125, 130, 135, 140, 145),
    }

class B737_400(VelocitySpeed):
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 'kg'
    v2_table = {
            'weight': ( 40,  45,  50,  55,  60,  65,  70),
                   5: (130, 136, 143, 149, 155, 162, 168),
    }
    airspeed_reference_table = {
             'weight':  ( 35,  40,  45,  50,  55,  60,  65,  70,  70),  # duplicate column?
                    15: (123, 132, 141, 149, 156, 164, 171, 177, 177),
                    30: (111, 119, 127, 134, 141, 147, 154, 159, 159),
                    40: (109, 116, 124, 130, 137, 143, 149, 155, 155),
    }

class B737_500(VelocitySpeed):
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 'kg'
    airspeed_reference_table = {
             'weight':  ( 32,  36,  40,  44,  48,  52,  56,  60,  64),
                    15: (111, 118, 125, 132, 138, 143, 149, 154, 159),
                    30: (105, 111, 117, 123, 129, 135, 140, 144, 149),
                    40: (101, 108, 114, 120, 125, 130, 135, 140, 145),
    }


"""

This table is available but should never be needed as the 737 NG family of
aircraft record V2 and VREF in the data frame.

class B737_700(VelocitySpeed):
    interpolate = True
    source = 'B737-5_925017_07'
    weight_unit = 'kg'
    minimum_speed = 110
    v2_table = {
        'weight': (40, 45, 49, 54, 58, 63, 68, 72),
        1: (114, 120, 126, 132, 137, 142, 147, 152),
        5: (111, 117, 123, 129, 134, 139, 144, 148),
        15: (107, 113, 117, 122, 127, 131, 135, 138),
    }
    airspeed_reference_table = {
        'weight':  (40, 45, 49, 54, 58, 63, 68, 72, 77),
        15: (115, 121, 127, 133, 139, 145, 150, 155, 159),
        30: (111, 117, 123, 129, 134, 140, 144, 149, 153),
        40: (108, 114, 120, 126, 132, 137, 142, 147, 151),
    }
"""

class B767(VelocitySpeed):
    '''
    '''
    interpolate = False
    source = '767 Flight Crew Operations Manual'
    weight_unit = 'kg'
    v2_table = {
             'weight':  (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                    5:  (127, 134, 139, 145, 151, 156, 161, 166, 171, 176),
                   15:  (122, 128, 134, 139, 144, 149, 154, 159, 164, 168),
                   20:  (118, 124, 129, 134, 140, 144, 149, 154, 159, 164),
        }
    airspeed_reference_table = {
             'weight':  (100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                    5:  (114, 121, 128, 134, 141, 147, 153, 158, 164, 169),
                   15:  (109, 116, 122, 129, 141, 135, 146, 151, 157, 162),
                   20:  (105, 111, 118, 124, 130, 135, 141, 147, 152, 158),
        }


def get_vspeed_map(series=None, family=None):
    """
    Accessor for fetching vspeed mapping classes.
    
    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: associated VelocitySpeed class
    :rtype: VelocitySpeed
    """
    if series in series_vspeed_map:
        return series_vspeed_map[series]
    elif family in family_vspeed_map:
        return family_vspeed_map[family]
    else:
        raise KeyError("No vspeed mapping for Series '%s' Family '%s'" % (
            series, family))

#############################################################################

# Notes:
# - Series vspeeds will be used over Family vspeeds settings
# - Familys/series which do not require vspeeds should be entered as None

series_vspeed_map = {
    # this will take precidence over family_vspeed_map
    'B737-300' : B737_300,
    'B737-300(QC)' : B737_300,
    'B737-400' : B737_400,
    'B737-500' : B737_500,
}

family_vspeed_map = {
    'B767' : B767,
}
