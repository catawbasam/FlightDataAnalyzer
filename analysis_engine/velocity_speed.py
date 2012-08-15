from bisect import bisect_left
import scipy.interpolate as interp

from settings import (
    KG_TO_LB
)


class VelocitySpeed(object):
    weight_unit = 'kg'
    interpolate = False
    source = ''

    weight_boundaries = []
    v2_table = {}
    airspeed_reference_table = {}

    def v2(self, weight, setting):
        '''
        lookup v2 value using interpolation if set
        converts value to kg if weight_unit in lb

        :param weight: Weight of aircraft
        :type weight: float
        :param setting: Flap/Conf setting to use in lookup
        :type setting: String
        :returns: v2 value
        :rtype: float
        '''
        return self._get_vspeed(self.v2_table, weight, setting)

    def airspeed_reference(self, weight, setting):
        '''
        lookup v2 value using interpolation if set
        converts value to kg if weight_unit in lb

        :param weight: Weight of aircraft
        :type weight: float
        :param setting: Flap/Conf setting to use in lookup
        :type setting: String
        :returns: v2 value
        :rtype: float
        '''
        return self._get_vspeed(self.airspeed_reference_table, weight, setting)

    def _get_vspeed(self, lookup, weight, setting):
        if self.weight_unit == 'lb':
            weight = weight * KG_TO_LB

        if self.interpolate:
            # numpy interpolate
            f = interp.interp1d(self.weight_boundaries, lookup[setting])
            value = f(weight)
        else:
            # bisect lookup
            value_index = bisect_left(self.weight_boundaries, weight)
            value = lookup[setting][value_index]
        return value


class boeing_767(VelocitySpeed):
    '''
    '''

    interpolate = False
    source = 'DHL 767 Flight Crew Operations Manual'
    weight_unit = 'kg'
    weight_boundaries = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    v2_table = {
                   5:   [127, 134, 139, 145, 151, 156, 161, 166, 171, 176],
                   15:  [122, 128, 134, 139, 144, 149, 154, 159, 164, 168],
                   20:  [118, 124, 129, 134, 140, 144, 149, 154, 159, 164],
        }
    airspeed_reference_table = {
                   5:   [114, 121, 128, 134, 141, 147, 153, 158, 164, 169],
                   15:  [109, 116, 122, 129, 141, 135, 146, 151, 157, 162],
                   20:  [105, 111, 118, 124, 130, 135, 141, 147, 152, 158],
        }


'''

bert.v2(125, 'flap 20') * np.ma.ones(20, np.double)

a = np.ma.zeros(20, np.double)
a[2:10] = 12.3
'''
