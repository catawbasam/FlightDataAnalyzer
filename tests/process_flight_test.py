import unittest

import os
import shutil

from datetime import datetime
        
from analysis.process_flight import process_flight, derive_parameters, get_derived_nodes

class TestProcessFlight(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_l382_herc(self):
        hdf_orig = "test_data/2_6748957_L382-Hercules.hdf5"
        hdf_path = "test_data/2_6748957_L382-Hercules_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': u'L382-Hercules',
                   'Identifier': u'',
                   'Manufacturer': u'Lockheed',
                   'Manufacturer Serial Number': u'',
                   'Model': u'L382',
                   'Tail Number': u'A-HERC',
                   'Precise Positioning': False,
                   }
        afr = {'AFR Destination Airport': 3279,
               'AFR Flight ID': 4041843,
               'AFR Flight Number': u'ISF51VC',
               'AFR Landing Aiport': 3279,
               'AFR Landing Datetime': datetime(2011, 4, 4, 8, 7, 42),
               'AFR Landing Fuel': 0,
               'AFR Landing Gross Weight': 0,
               'AFR Landing Pilot': 'CAPTAIN',
               'AFR Landing Runway': '23*',
               'AFR Off Blocks Datetime': datetime(2011, 4, 4, 6, 48),
               'AFR On Blocks Datetime': datetime(2011, 4, 4, 8, 18),
               'AFR Takeoff Airport': 3282,
               'AFR Takeoff Datetime': datetime(2011, 4, 4, 6, 48, 59),
               'AFR Takeoff Fuel': 0,
               'AFR Takeoff Gross Weight': 0,
               'AFR Takeoff Pilot': 'FIRST_OFFICER',
               'AFR Takeoff Runway': '11*',
               'AFR Type': u'LINE_TRAINING',
               'AFR V2': 149,
               'AFR Vapp': 135,
               'AFR Vref': 120
              }
        res = process_flight(hdf_path, ac_info, achieved_flight_record=afr, 
                             draw=False)
        self.assertEqual(len(res), 3)
    
    
    def test_146_301(self):
        hdf_path = "test_data/4_3377853_146-301.005.hdf5"
        ac_info = {'Frame': '737-3C',
                   'Identifier': '5',
                   'Main Gear To Altitude Radio': 10,
                   'Manufacturer': 'Boeing',
                   'Tail Number': 'G-ABCD',
                   }
        res = process_flight(hdf_path, ac_info, draw=True)
        self.assertEqual(len(res), 3)
    
    @unittest.skip('Not Implemented')
    def test_get_required_params(self):
        self.assertTrue(False)
    
    @unittest.skip('Not Implemented')    
    def test_process_flight(self):
        self.assertTrue(False)
        
    def test_get_derived_nodes(self):
        nodes = get_derived_nodes(['sample_derived_parameters'])
        self.assertEqual(len(nodes), 13)
        self.assertEqual(sorted(nodes.keys())[0], 'Heading Rate')
        self.assertEqual(sorted(nodes.keys())[-1], 'Vertical g')
        