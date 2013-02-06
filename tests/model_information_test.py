import unittest

from analysis_engine.model_information import get_flap_detents


class TestFlapDetents(unittest.TestCase):
    def test_get_flap_detents(self):
        detents = get_flap_detents()
        # must be lots of them
        self.assertGreater(len(detents), 25)
        self.assertLess(len(detents), 100)
        self.assertIn(0, detents)
        self.assertIn(45, detents)
        self.assertIn(50, detents) # herc
        # no duplication
        self.assertEqual(len(set(detents)), len(detents))
        