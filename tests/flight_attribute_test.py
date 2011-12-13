import unittest

from analysis.flight_attribute import (
    TakeoffAirport, TakeoffRunway, Approaches
    )

class TestTakeoffRunway(unittest.TestCase):
    def test_can_operate(self):
        option1 = ('','')
        TakeoffRunway.can_operate(option1)
        self.assertTrue(False)