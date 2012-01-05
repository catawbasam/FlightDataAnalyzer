import unittest
import csv
import os
import shutil
import sys
debug = sys.gettrace() is not None

from datetime import datetime, timedelta
        
from analysis.library import value_at_time
from analysis.node import KeyPointValueNode, P, KeyTimeInstanceNode, S
from analysis.process_flight import (process_flight, derive_parameters, 
                                     get_derived_nodes)

debug = sys.gettrace() is not None
if debug:
    # only import if we're going to use this as it's slow!
    from analysis.plot_flight import plot_flight

class TestProcessFlight(unittest.TestCase):
    
    def setUp(self):
        pass
    
    @unittest.skipIf(not os.path.isfile("test_data/1_7295949_737-3C.001.hdf5"), "Test file not present")
    def test_1_7295949_737_3C(self):
        hdf_orig = "test_data/1_7295949_737-3C.001.hdf5"
        hdf_path = "test_data/1_7295949_737-3C.001_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-3C',
                   'Identifier': '5',
                   'Main Gear To Altitude Radio': 10,
                   'Manufacturer': 'Boeing',
                   'Tail Number': 'G-ABCD',
                   }
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        if debug:
            from analysis.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])

        #TODO: Further assertions on the results!

    @unittest.skipIf(not os.path.isfile("test_data/2_6748957_L382-Hercules.hdf5"), "Test file not present")
    def test_2_6748957_L382_Hercules(self):
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
        self.assertEqual(len(res), 4)

        if debug:
            from analysis.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])

        tdwn = res['kti'].get(name='Touchdown')[0]
        tdwn_minus_1 = res['kti'].get(name='1 Mins To Touchdown')[0]
        
        self.assertAlmostEqual(tdwn.index, 4967.0, places=0)
        self.assertAlmostEqual(tdwn_minus_1.index, 4907.0, places=0)
        self.assertEqual(tdwn.datetime - tdwn_minus_1.datetime, timedelta(minutes=1))
        #TODO: Further assertions on the results!
        

    @unittest.skipIf(not os.path.isfile("test_data/3_6748984_L382-Hercules.hdf5"), "Test file not present")
    def test_3_6748984_L382_Hercules(self):
        # test copied from herc_2 so AFR may not be accurate
        hdf_orig = "test_data/3_6748984_L382-Hercules.hdf5"
        hdf_path = "test_data/3_6748984_L382-Hercules_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': u'L382-Hercules',
                   'Identifier': u'',
                   'Manufacturer': u'Lockheed',
                   'Manufacturer Serial Number': u'',
                   'Model': u'L382',
                   'Tail Number': u'B-HERC',
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
        self.assertEqual(len(res), 4)
        if debug:
            from analysis.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        #TODO: Further assertions on the results!
        
    @unittest.skipIf(not os.path.isfile("test_data/4_3377853_146-301.005.hdf5"), "Test file not present")
    def test_4_3377853_146_301(self):
        hdf_orig = "test_data/4_3377853_146-301.005.hdf5"
        hdf_path = "test_data/4_3377853_146-301.005_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '146-301',
                   'Identifier': '1',
                   'Manufacturer': 'BAE',
                   'Tail Number': 'G-ABCD',
                   }
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        if debug:
            from analysis.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        #TODO: Further assertions on the results!

            
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
        
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestProcessFlight('test_l382_herc_2'))

    ##suite = unittest.TestLoader().loadTestsFromName("test_l382_herc_2")
    unittest.TextTestRunner(verbosity=2).run(suite)
    ##unittest.main()