import unittest
import csv
import os
import shutil
import mock
import sys
debug = sys.gettrace() is not None

from datetime import datetime, timedelta
        
from analysis.library import value_at_time
from analysis.node import (Attribute, FlightAttributeNode, KeyPointValueNode,
                           KeyTimeInstanceNode, P, S)
from analysis.process_flight import (process_flight, derive_parameters, 
                                     get_derived_nodes)
from analysis import settings, ___version___

debug = sys.gettrace() is not None
if debug:
    # only import if we're going to use this as it's slow!
    from analysis.plot_flight import plot_flight

class TestProcessFlight(unittest.TestCase):
    
    def setUp(self):
        pass
    
    @unittest.skipIf(not os.path.isfile("test_data/1_7295949_737-3C.001.hdf5"),
                     "Test file not present")
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
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        if debug:
            from analysis.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])

        #TODO: Further assertions on the results!

    @unittest.skipIf(not os.path.isfile("test_data/2_6748957_L382-Hercules.hdf5"),
                     "Test file not present")
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
        

    #@unittest.skipIf(not os.path.isfile("test_data/3_6748984_L382-Hercules.hdf5"), "Test file not present")
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
        afr = {'AFR Destination Airport': 3279, # TODO: Choose another airport.
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
        
    @unittest.skipIf(not os.path.isfile("test_data/4_3377853_146-301.018.hdf5"),
                     "Test file not present")
    @mock.patch('analysis.flight_attribute.get_api_handler')
    def test_4_3377853_146_301(self, get_api_handler):
        # Avoid side effects which may be caused by PRE_FLIGHT_ANALYSIS.
        settings.PRE_FLIGHT_ANALYSIS = None
        hdf_orig = "test_data/4_3377853_146-301.018.hdf5"
        hdf_path = "test_data/4_3377853_146-301.018_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        
        ac_info = {'Frame': '146-301',
                   'Identifier': '1',
                   'Manufacturer': 'BAE',
                   'Tail Number': 'G-ABCD',
                   'Flap Selections': [0,18,24,30,33],
                   }
        
        afr_flight_id = 3377853
        afr_landing_fuel = 500
        afr_takeoff_fuel = 1000
        afr = {'AFR Flight ID': afr_flight_id,
               'AFR Landing Fuel': afr_landing_fuel,
               'AFR Takeoff Fuel': afr_takeoff_fuel,
               }
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        airport = {'id': 100, 'icao': 'EGLL'}
        runway = {'identifier': '09L'}
        api_handler.get_nearest_airport = mock.Mock()
        api_handler.get_nearest_airport.return_value = airport
        api_handler.get_nearest_runway = mock.Mock()
        api_handler.get_nearest_runway.return_value = runway
        start_datetime = datetime.now()
        res = process_flight(hdf_path, ac_info, achieved_flight_record=afr,
                             start_datetime=start_datetime)
        ##if debug:
            ##from analysis.plot_flight import csv_flight_details
            ##csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            ##plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        self.assertEqual(len(res), 4)
        self.assertTrue('flight' in res)
        from pprint import pprint
        pprint(res)
        flight_attrs = {attr.name: attr for attr in res['flight']}
        # 'FDR Flight ID' is sourced from 'AFR Flight ID'.
        self.assertEqual(flight_attrs['FDR Flight ID'].value, afr_flight_id)
        # 'FDR Analysis Datetime' is created during processing from
        # datetime.now(). Ensure the value is sensible.
        fdr_analysis_dt = flight_attrs['FDR Analysis Datetime']
        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)
        self.assertTrue(now > fdr_analysis_dt.value > five_minutes_ago)
        
        takeoff_datetime = flight_attrs['FDR Takeoff Datetime'].value
        self.assertEqual(takeoff_datetime - start_datetime,
                         timedelta(0, 427, 250000))
        
        landing_datetime = flight_attrs['FDR Landing Datetime'].value
        self.assertEqual(landing_datetime - start_datetime,
                         timedelta(0, 3243, 900000))
        
        approaches = flight_attrs['FDR Approaches'].value
        self.assertEqual(len(approaches), 1)
        approach = approaches[0]
        self.assertEqual(approach['airport'], airport['id'])
        self.assertEqual(approach['type'], 'LANDING')
        self.assertEqual(approach['runway'], runway['identifier'])
        self.assertEqual(approach['datetime'] - start_datetime,
                         timedelta(0, 3492))
        
        self.assertEqual(flight_attrs['FDR Flight Type'].value, 'COMPLETE')
        
        self.assertEqual(api_handler.get_nearest_airport.call_args_list,
                         [((40418.0, -3339.21875), {}), ((37917.0, -450.0), {}),
                          ((37917.0, -450.0), {})])
        self.assertEqual(api_handler.get_nearest_runway.call_args_list,
                         [((100, 310.22130556082084), {}),
                          ((100, 219.42928588921563), {}),
                          ((100, 219.42928588921563), {})])
        self.assertEqual(flight_attrs['FDR Takeoff Airport'].value, airport)
        self.assertEqual(flight_attrs['FDR Takeoff Runway'].value, runway)
        self.assertEqual(flight_attrs['FDR Landing Airport'].value, airport)
        self.assertEqual(flight_attrs['FDR Landing Runway'].value, runway)
        
        self.assertEqual(flight_attrs['FDR Duration'].value, 2816.65)
        self.assertEqual(flight_attrs['FDR Takeoff Fuel'].value,
                         afr_takeoff_fuel)
        self.assertEqual(flight_attrs['FDR Landing Fuel'].value,
                         afr_landing_fuel)
        self.assertEqual(flight_attrs['FDR Version'].value, ___version___)
        
        
        
        # 'FDR Takeoff Gross Weight' and 'FDR Landing Gross Weight' cannot be
        # tested as 'Gross Weight' is not recorded or derived.
        # 'FDR Takeoff Runway' cannot be tested as 'Takeoff Peak Acceleration'
        # does not exist for 'Heading At Takeoff'.
        
        # 
        # ''
        # FIXME: 'TakeoffDatetime' requires missing 'Liftoff' KTI.
        # FIXME: 'Duration' requires missing 'Takeoff Datetime' and 'Landing
        #         Datetime' FlightAttributes.
        # 
        # 'Flight Number' is not recorded.
        #TODO: Further assertions on the results!
        # TODO: Test cases for attributes which should be coming out but are NotImplemented.
        # FlightNumber? May not be recorded.
        # All datetimes.
        # Pilots. (might not be for Herc)
        # V2, Vapp, Version (Herc will be AFR based).
    
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
        self.assertEqual(sorted(nodes.keys())[-1], 'Vertical Speed')
        
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestProcessFlight('test_l382_herc_2'))

    ##suite = unittest.TestLoader().loadTestsFromName("test_l382_herc_2")
    unittest.TextTestRunner(verbosity=2).run(suite)
    ##unittest.main()