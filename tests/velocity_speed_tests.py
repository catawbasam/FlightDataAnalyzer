import mock
import unittest

from analysis_engine.velocity_speed import VelocitySpeed

class TestVelocitySpeed(unittest.TestCase):
    def setUp(self):
        self.velocity_speed = VelocitySpeed()
        self.velocity_speed.v2_table = {
            'weight': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
                   5: [127, 134, 139, 145, 151, 156, 161, 166, 171, 176],
                  15: [122, 128, 134, 139, 144, 149, 154, 159, 164, 168],
                  20: [118, 124, 129, 134, 140, 144, 149, 154, 159, 164],
        }
        self.velocity_speed.airspeed_reference_table = {
            'weight': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
                   5: [114, 121, 128, 134, 141, 147, 153, 158, 164, 169],
                  15: [109, 116, 122, 129, 141, 135, 146, 151, 157, 162],
                  20: [105, 111, 118, 124, 130, 135, 141, 147, 152, 158],
        }

    def test_v2(self):
        self.velocity_speed.interpolate = False
        self.velocity_speed.unit = 1000
        self.assertEquals(self.velocity_speed.v2(119000, 20), 129)
        self.assertEquals(self.velocity_speed.v2(120000, 20), 129)
        self.assertEquals(self.velocity_speed.v2(121000, 20), 134)

    def test_v2_interpolated(self):
        self.velocity_speed.interpolate = True
        self.velocity_speed.unit = 1000
        self.assertEquals(self.velocity_speed.v2(145000, 20), 142)
        self.assertEquals(self.velocity_speed.v2(120000, 20), 129)
        self.assertEquals(self.velocity_speed.v2(165000, 5), 163.5)

    def test_airspeed_reference(self):
        self.velocity_speed.interpolate = False
        self.velocity_speed.unit = 1000
        self.assertEquals(self.velocity_speed.airspeed_reference(119000, 15), 122)
        self.assertEquals(self.velocity_speed.airspeed_reference(120000, 15), 122)
        self.assertEquals(self.velocity_speed.airspeed_reference(121000, 15), 129)

    def test_airspeed_reference_interpolated(self):
        self.velocity_speed.interpolate = True
        self.velocity_speed.unit = 1000
        self.assertEquals(self.velocity_speed.airspeed_reference(120000, 5), 128)
        self.assertEquals(self.velocity_speed.airspeed_reference(120000, 15), 122)
        self.assertEquals(self.velocity_speed.airspeed_reference(145000, 20), 132.5)
