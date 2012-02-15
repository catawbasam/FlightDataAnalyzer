import unittest
import csv
import os
import shutil
import mock
import sys
debug = sys.gettrace() is not None

from datetime import datetime, timedelta
from analysis_engine.plot_flight import track_to_kml
        
from analysis_engine.library import value_at_time
from analysis_engine.node import (Attribute, FlightAttributeNode,
                                  KeyPointValueNode, KeyTimeInstanceNode, P, S)
from analysis_engine.process_flight import (process_flight, derive_parameters,
                                            get_derived_nodes)
from analysis_engine import settings, ___version___

debug = sys.gettrace() is not None
if debug:
    # only import if we're going to use this as it's slow!
    from analysis_engine.plot_flight import plot_flight

class TestProcessFlight(unittest.TestCase):
    
    def setUp(self):
        pass
    
    @unittest.skipIf(not os.path.isfile("test_data/1_7295949_737-3C.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_1_7295949_737_3C(self, get_api_handler):
        hdf_orig = "test_data/1_7295949_737-3C.hdf5"
        hdf_path = "test_data/1_7295949_737-3C_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-3C',
                   'Identifier': '5',
                   'Main Gear To Altitude Radio': 10,
                   'Manufacturer': 'Boeing',
                   'Model Series': '737',
                   'Tail Number': 'G-ABCD',
                   'Precise Positioning': True,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }
        
        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_trd = {"distance":0.52169665188063608,"magnetic_variation":"E001220 0706","code":{"icao":"ENVA","iata":"TRD"},"name":"Vaernes","longitude":10.9399,"location":{"city":"Trondheim","country":"Norway"},"latitude":63.457599999999999,"id":2472}
        airport_bgo = {"distance":0.065627843313191145,"magnetic_variation":"W001185 0106","code":{"icao":"ENBR","iata":"BGO"},"name":"Bergen Lufthavn Flesland","longitude":5.21814,"location":{"city":"Bergen","country":"Norway"},"latitude":60.293399999999998,"id":2455}
        airports = \
            {(60.207918026368986, 11.087010689351679):airport_osl,
             (63.457546234130859, 10.920455589077017):airport_trd,
             (60.209332779049873, 11.08782559633255):airport_osl,
             (60.297126313897756, 5.2168199977260254):airport_bgo,
             (60.201646909117699, 11.083488464355469):airport_osl,
             (60.292314738035202, 5.2184030413627625):airport_bgo,
             (60.292314738035202, 5.2184030413627625):airport_bgo,
             (60.295075884447485, 5.2175367817352285):airport_bgo}
        
        runway_osl_19r = {"end":{"latitude":60.184763,"longitude":11.073319},"glideslope":{"latitude":60.213763999999998,"frequency":"332300M","angle":3.0,"longitude":11.088044,"threshold_distance":991},"start":{"latitude":60.216242,"longitude":11.091471},"localizer":{"latitude":60.182059,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.071759},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_trd_09 = {"end":{"latitude":63.457572,"longitude":10.941974},"glideslope":{"latitude":63.457085999999997,"frequency":"335000M","angle":3.0,"longitude":10.901011,"threshold_distance":1067},"start":{"latitude":63.457614,"longitude":10.894439},"localizer":{"latitude":63.457539,"beam_width":4.5,"frequency":"110300M","heading":89,"longitude":10.947803},"strip":{"width":147,"length":9347,"surface":"ASP"},"identifier":"09","id":8129}
        runway_bgo_17 = {"end":{"latitude":60.282283,"longitude":5.221859},"glideslope":{"latitude":60.300981,"frequency":"333800M","angle":3.1000000000000001,"longitude":5.2140919999999999,"threshold_distance":1161},"start":{"latitude":60.304365,"longitude":5.214447},"localizer":{"latitude":60.278892,"beam_width":4.5,"frequency":"109900M","heading":173,"longitude":5.223010},"strip":{"width":147,"length":9810,"surface":"ASP"},"identifier":"17","id":8193}
        
        runways = \
            {2461: runway_osl_19r, 2472: runway_trd_09, 2455: runway_bgo_17}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lon, lat, **kwargs):
            return airports[(lon, lat)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)

        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
                
        #if debug:
            #plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])

        #TODO: Further assertions on the results!
        
    #----------------------------------------------------------------------
    # Test 6 = 737-1 frame
    #----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/6_737_1_RD0001851371.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_6_737_1_RD0001851371(self, get_api_handler):
        hdf_orig = "test_data/6_737_1_RD0001851371.hdf5"
        hdf_path = "test_data/6_737_1_RD0001851371_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-1',
                   'Precise Positioning': False,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }
        

        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_trd = {"distance":0.52169665188063608,"magnetic_variation":"E001220 0706","code":{"icao":"ENVA","iata":"TRD"},"name":"Vaernes","longitude":10.9399,"location":{"city":"Trondheim","country":"Norway"},"latitude":63.457599999999999,"id":2472}
        airport_bgo = {"distance":0.065627843313191145,"magnetic_variation":"W001185 0106","code":{"icao":"ENBR","iata":"BGO"},"name":"Bergen Lufthavn Flesland","longitude":5.21814,"location":{"city":"Bergen","country":"Norway"},"latitude":60.293399999999998,"id":2455}
        airport_krs = {"distance":0.29270199259899349,"magnetic_variation":"E000091 0106","code":{"icao":"ENCN","iata":"KRS"},"name":"Kristiansand Lufthavn Kjevik","longitude":8.0853699999999993,"location":{"city":"Kjevik","country":"Norway"},"latitude":58.2042,"id":2456}
        airports = \
            {(58.2000732421875, 8.0804443359375):airport_krs,
             (60.18585205078125, 11.1126708984375):airport_osl}
        
        runway_osl_01r = {"end":{"latitude":60.201367,"longitude":11.122289},"glideslope":{"latitude":60.177936000000003,"frequency":"330950M","angle":3.0,"longitude":11.111328,"threshold_distance":945},"start":{"latitude":60.175513,"longitude":11.107355},"localizer":{"latitude":60.204934,"beam_width":4.5,"frequency":"111950M","heading":16,"longitude":11.124370},"strip":{"width":147,"length":9678,"surface":"ASP"},"identifier":"01R","id":8149}       
        runway_osl_19r = {"end":{"latitude":60.185000000000002,"longitude":11.073744},"glideslope":{"latitude":60.213763999999998,"frequency":"332300M","angle":3.0,"longitude":11.088044,"threshold_distance":991},"start":{"latitude":60.216067000000002,"longitude":11.091664},"localizer":{"latitude":60.182102999999998,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.072075},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_trd_09 = {"end":{"latitude":63.457552769999999,"longitude":10.94666812},"glideslope":{"latitude":63.457085999999997,"frequency":"335000M","angle":3.0,"longitude":10.901011,"threshold_distance":1067},"start":{"latitude":63.457656229999998,"longitude":10.88929278},"localizer":{"latitude":63.457549999999998,"beam_width":4.5,"frequency":"110300M","heading":89,"longitude":10.947803},"strip":{"width":147,"length":9347,"surface":"ASP"},"identifier":"09","id":8129}
        runway_bgo_17 = {"end":{"latitude":60.280150999999996,"longitude":5.2225789999999996},"glideslope":{"latitude":60.300981,"frequency":"333800M","angle":3.1000000000000001,"longitude":5.2140919999999999,"threshold_distance":1161},"start":{"latitude":60.306624939999999,"longitude":5.2137007400000002},"localizer":{"latitude":60.2789,"beam_width":4.5,"frequency":"109900M","heading":173,"longitude":5.2229999999999999},"strip":{"width":147,"length":9810,"surface":"ASP"},"identifier":"17","id":8193}
        runway_krs_22 = {"end":{"latitude":58.196636,"longitude":8.075328},"glideslope":{"latitude":58.208922000000001,"frequency":"330800M","angle":3.6000000000000001,"longitude":8.0932750000000002,"threshold_distance":422},"start":{"latitude":58.211733,"longitude":8.095353},"localizer":{"latitude":58.196164000000003,"beam_width":4.5,"frequency":"110900M","heading":216,"longitude":8.0746920000000006},"strip":{"width":147,"length":6660,"surface":"ASP"},"identifier":"22","id":8128}
        runway_krs_04 = {"end":{"latitude":58.211678,"longitude":8.095269},"localizer":{"latitude":58.212397,"beam_width":4.5,"frequency":"110300M","heading":36,"longitude":8.096228},"glideslope":{"latitude":58.198664,"frequency":"335000M","angle":3.4,"longitude":8.080164,"threshold_distance":720},"start":{"latitude":58.196703,"longitude":8.075406},"strip":{"width":147,"length":6660,"id":4064,"surface":"ASP"},"identifier":"04","id":8127}
        runways = \
            {2461: runway_osl_01r, 2472: runway_trd_09, 2455: runway_bgo_17, 2456: runway_krs_22}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)

        track_to_kml(hdf_path, res['kti'], res['kpv'])
                     
        from analysis_engine.plot_flight import csv_flight_details

        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])

    #----------------------------------------------------------------------
    # Test 7 = 737-i frame
    #----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/7_737_i_RD0001839773.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_7_737_i_RD0001839773(self, get_api_handler):
        hdf_orig = "test_data/7_737_i_RD0001839773.hdf5"
        hdf_path = "test_data/7_737_i_RD0001839773_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-i',
                   'Precise Positioning': False,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }
        
        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_bgo = {"distance":0.065627843313191145,"magnetic_variation":"W001185 0106","code":{"icao":"ENBR","iata":"BGO"},"name":"Bergen Lufthavn Flesland","longitude":5.21814,"location":{"city":"Bergen","country":"Norway"},"latitude":60.293399999999998,"id":2455}
        airports = \
            {(60.296829215117867, 5.2152368000575473):airport_bgo,
             (60.189057247979299, 11.098743506840297):airport_osl}
        
        runway_osl_19r = {"end":{"latitude":60.185000000000002,"longitude":11.073744},"glideslope":{"latitude":60.213678,"frequency":"332300M","angle":3.0,"longitude": 11.087713,"threshold_distance":991},"start":{"latitude":60.216067000000002,"longitude":11.091664},"localizer":{"latitude": 60.182054,"beam_width":4.5,"frequency":"111300M","heading":016,"longitude": 11.071766},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_bgo_35 = {"start":{"latitude":60.280150999999996,"longitude":5.2225789999999996},"glideslope":{"latitude":60.285492,"frequency":"333800M","angle":3.1000000000000001,"longitude":5.219389,"threshold_distance":1161},"end":{"latitude":60.306624939999999,"longitude":5.2137007400000002},"localizer":{"latitude": 60.307589,"beam_width":4.5,"frequency":"109900M","heading":353,"longitude":  5.213357},"strip":{"width":147,"length":9810,"surface":"ASP"},"identifier":"17","id":8193}
        runways = {2455: runway_bgo_35, 2461: runway_osl_19r}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])

    #----------------------------------------------------------------------
    # Test 8 = 737-i frame with short HDF file
    #----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/8_737_i_RD0001835658.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_8_737_i_RD0001835658(self, get_api_handler):
        hdf_orig = "test_data/8_737_i_RD0001835658.hdf5"
        hdf_path = "test_data/8_737_i_RD0001835658_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-i'}
        
        airports = {}
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        # We want this to fail gracefully as there is no data in the HDF file.

    #----------------------------------------------------------------------
    # Test 9 = 737-5 frame
    #----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/9_737_5_RD0001860694.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_9_737_5_RD0001860694(self, get_api_handler):
        hdf_orig = "test_data/9_737_5_RD0001860694.hdf5"
        hdf_path = "test_data/9_737_5_RD0001860694_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-i',
                   'Precise Positioning': False,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }

        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_krs = {"distance":0.29270199259899349,"magnetic_variation":"E000091 0106","code":{"icao":"ENCN","iata":"KRS"},"name":"Kristiansand Lufthavn Kjevik","longitude":8.0853699999999993,"location":{"city":"Kjevik","country":"Norway"},"latitude":58.2042,"id":2456}
        airports = \
            {(58.20556640625, 8.0878186225891113):airport_krs,
             (60.19134521484375, 11.07696533203125):airport_osl}
        
        runway_osl_01l = {"end":{"latitude":60.216113,"longitude":11.091418},"glideslope":{"latitude":60.187709,"frequency":"332300M","angle":3.0,"longitude":11.072739,"threshold_distance":991},"start":{"latitude":60.185048,"longitude":11.073522},"localizer":{"latitude":60.219793,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.093544},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_krs_04 = {"end":{"latitude":58.211678,"longitude":8.095269},"localizer":{"latitude":58.212397,"beam_width":4.5,"frequency":"110300M","heading":36,"longitude":8.096228},"glideslope":{"latitude":58.198664,"frequency":"335000M","angle":3.4,"longitude":8.080164,"threshold_distance":720},"start":{"latitude":58.196703,"longitude":8.075406},"strip":{"width":147,"length":6660,"id":4064,"surface":"ASP"},"identifier":"04","id":8127}
        runways = \
            {2461: runway_osl_01l, 2456: runway_krs_04}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])


    #----------------------------------------------------------------------
    # Test 10 = 737-3C frame 
    #
    #Traceback (most recent call last): File\
        #"/var/local/polaris/virtualenvs/celery/local/lib/python2.7/site-packages/celery/execute/trace.py",\
        #line 47, in trace return cls(states.SUCCESS, retval=fun(*args, **kwargs))\
        #File\
        #"/var/local/polaris/roles/celery/app/PolarisTaskManagement/polaris_tasks/base.py",\
        #line 105, in __call__ return super(FileLoggingTask, self).__call__(*args,
                                                                           #**kwargs) File\
        #"/var/local/polaris/virtualenvs/celery/local/lib/python2.7/site-packages/celery/app/task/__init__.py",\
        #line 247, in __call__ return self.run(*args, **kwargs) File\
        #"/var/local/polaris/roles/celery/app/PolarisTaskManagement/polaris_tasks/analysis_tasks.py",\
        #line 340, in run flight_info, required_params) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/process_flight.py",\
        #line 261, in process_flight hdf, node_mgr, process_order) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/process_flight.py",\
        #line 119, in derive_parameters result = node.get_derived(deps) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/node.py",\
        #line 193, in get_derived res = self.derive(*args) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/derived_parameters.py",\
        #line 1296, in derive runway_distances(app_info.value[num_loc]['runway'])\
        #File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/library.py",\
        #line 660, in runway_distances start_lat = runway['start']['latitude']\
        #KeyError: 'start'
    ##----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/10_737_3C_RD0001861142.001.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_10_737_3C_RD0001861142_001(self, get_api_handler):
        hdf_orig = "test_data/10_737_3C_RD0001861142.001.hdf5"
        hdf_path = "test_data/10_737_3C_RD0001861142.001_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-3C',
                   'Precise Positioning': True,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }

        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_trd = {"distance":0.52169665188063608,"magnetic_variation":"E001220 0706","code":{"icao":"ENVA","iata":"TRD"},"name":"Vaernes","longitude":10.9399,"location":{"city":"Trondheim","country":"Norway"},"latitude":63.457599999999999,"id":2472}
        runway_osl_01l = {"end":{"latitude":60.216113,"longitude":11.091418},"glideslope":{"latitude":60.187709,"frequency":"332300M","angle":3.0,"longitude":11.072739,"threshold_distance":991},"start":{"latitude":60.185048,"longitude":11.073522},"localizer":{"latitude":60.219793,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.093544},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_trd_09 = {"end":{"latitude":63.457572,"longitude":10.941974},"glideslope":{"latitude":63.457085999999997,"frequency":"335000M","angle":3.0,"longitude":10.901011,"threshold_distance":1067},"start":{"latitude":63.457614,"longitude":10.894439},"localizer":{"latitude":63.457539,"beam_width":4.5,"frequency":"110300M","heading":89,"longitude":10.947803},"strip":{"width":147,"length":9347,"surface":"ASP"},"identifier":"09","id":8129}
        runways = {2461: runway_osl_01l, 2472: runway_trd_09}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            if int(lat) == 63 and int(lon) == 10:
                # we're in TRD:
                return airport_trd
            elif int(lat) == 60 and int(lon) == 11:
                return airport_osl
            else:
                raise ValueError

        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])


    #----------------------------------------------------------------------
    # Test 11 = 737-3C frame
    #Traceback (most recent call last): File\
        #"/var/local/polaris/virtualenvs/celery/local/lib/python2.7/site-packages/celery/execute/trace.py",\
        #line 47, in trace return cls(states.SUCCESS, retval=fun(*args, **kwargs))\
        #File\
        #"/var/local/polaris/roles/celery/app/PolarisTaskManagement/polaris_tasks/base.py",\
        #line 105, in __call__ return super(FileLoggingTask, self).__call__(*args,
                                                                           #**kwargs) File\
        #"/var/local/polaris/virtualenvs/celery/local/lib/python2.7/site-packages/celery/app/task/__init__.py",\
        #line 247, in __call__ return self.run(*args, **kwargs) File\
        #"/var/local/polaris/roles/celery/app/PolarisTaskManagement/polaris_tasks/analysis_tasks.py",\
        #line 340, in run flight_info, required_params) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/process_flight.py",\
        #line 261, in process_flight hdf, node_mgr, process_order) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/process_flight.py",\
        #line 119, in derive_parameters result = node.get_derived(deps) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/node.py",\
        #line 193, in get_derived res = self.derive(*args) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/flight_phase.py",\
        #line 42, in derive slice(speedy.slice.start,midpoint)) File\
        #"/var/local/polaris/roles/celery/app/AnalysisEngine/analysis_engine/library.py",\
        #line 1832, in index_at_value raise ValueError, 'No range for seek\
        #function to scan across' ValueError: No range for seek function to scan\
                                 #across
    
    #----------------------------------------------------------------------
    
    @unittest.skipIf(not os.path.isfile("test_data/11_737_3C_RD0001861129.001.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_11_737_3C_RD0001861129_001(self, get_api_handler):
        hdf_orig = "test_data/11_737_3C_RD0001861129.001.hdf5"
        hdf_path = "test_data/11_737_3C_RD0001861129.001_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-3C',
                   'Precise Positioning': True,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }

        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_krs = {"distance":0.29270199259899349,"magnetic_variation":"E000091 0106","code":{"icao":"ENCN","iata":"KRS"},"name":"Kristiansand Lufthavn Kjevik","longitude":8.0853699999999993,"location":{"city":"Kjevik","country":"Norway"},"latitude":58.2042,"id":2456}
        airports = \
            {(58.20556640625, 8.0878186225891113):airport_krs,
             (60.19134521484375, 11.07696533203125):airport_osl}
        
        runway_osl_01l = {"end":{"latitude":60.216113,"longitude":11.091418},"glideslope":{"latitude":60.187709,"frequency":"332300M","angle":3.0,"longitude":11.072739,"threshold_distance":991},"start":{"latitude":60.185048,"longitude":11.073522},"localizer":{"latitude":60.219793,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.093544},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_krs_04 = {"end":{"latitude":58.211678,"longitude":8.095269},"localizer":{"latitude":58.212397,"beam_width":4.5,"frequency":"110300M","heading":36,"longitude":8.096228},"glideslope":{"latitude":58.198664,"frequency":"335000M","angle":3.4,"longitude":8.080164,"threshold_distance":720},"start":{"latitude":58.196703,"longitude":8.075406},"strip":{"width":147,"length":6660,"id":4064,"surface":"ASP"},"identifier":"04","id":8127}
        runways = \
            {2461: runway_osl_01l, 2456: runway_krs_04}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
        
    @unittest.skipIf(not os.path.isfile("test_data/RD0001830229.001.hdf5"),
                         "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_12_RD0001830229(self, get_api_handler):
        hdf_orig = "test_data/RD0001830229.001.hdf5"
        hdf_path = "test_data/RD0001830229.001_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Frame': '737-3C',
                   'Precise Positioning': True,
                   'Flap Selections': [0,1,2,5,10,15,25,30,40],
                   }

        airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        airport_trd = {"distance":0.52169665188063608,"magnetic_variation":"E001220 0706","code":{"icao":"ENVA","iata":"TRD"},"name":"Vaernes","longitude":10.9399,"location":{"city":"Trondheim","country":"Norway"},"latitude":63.457599999999999,"id":2472}
        airports = \
            {(63.457546234130859, 10.928016315005772):airport_trd,
             (60.181234387708784, 11.111000827986269):airport_osl}
        
        runway_osl_01l = {"end":{"latitude":60.216113,"longitude":11.091418},"glideslope":{"latitude":60.187709,"frequency":"332300M","angle":3.0,"longitude":11.072739,"threshold_distance":991},"start":{"latitude":60.185048,"longitude":11.073522},"localizer":{"latitude":60.219793,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.093544},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        runway_trd_09 = {"end":{"latitude":63.457572,"longitude":10.941974},"glideslope":{"latitude":63.457085999999997,"frequency":"335000M","angle":3.0,"longitude":10.901011,"threshold_distance":1067},"start":{"latitude":63.457614,"longitude":10.894439},"localizer":{"latitude":63.457539,"beam_width":4.5,"frequency":"110300M","heading":89,"longitude":10.947803},"strip":{"width":147,"length":9347,"surface":"ASP"},"identifier":"09","id":8129}
        runways = \
            {2461: runway_osl_01l, 2472: runway_trd_09}        
        
        # Mock API handler return values so that we do not make http requests.
        # Will return the same airport and runway for each query, can be
        # avoided with side_effect.
        api_handler = mock.Mock()
        get_api_handler.return_value = api_handler
        api_handler.get_nearest_airport = mock.Mock()
        def mocked_nearest_airport(lat, lon, **kwargs):
            return airports[(lat, lon)]
        api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        api_handler.get_nearest_runway = mock.Mock()
        def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            return runways[airport_id]
        api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])    


    @unittest.skipIf(not os.path.isfile("test_data/RD0001802061.001.hdf5"),
                             "Test file not present")
    #@mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_13_RD0001802061(self):
        hdf_orig = "test_data/RD0001802061.001.hdf5"
        hdf_path = "test_data/RD0001802061.001_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        ac_info = {'Family': u'B737 NG',
                   'Series': u'B737-800',
                   'Tail Number': u'LN-DYV',
                   'Main Gear To Lowest Point Of Tail': None,
                   'Manufacturer Serial Number': u'39009', 
                   'Main Gear To Radio Altimeter Antenna': None,
                   'Precise Positioning': True, 
                   'Model': u'B737-8JP', 
                   'Identifier': u'15', 
                   'Frame': u'737-3C',
                   'Manufacturer': u'Boeing'}

        #airport_osl = {"distance":0.93165142982548599,"magnetic_variation":"E001226 0106","code":{"icao":"ENGM","iata":"OSL"},"name":"Oslo Gardermoen","longitude":11.1004,"location":{"city":"Oslo","country":"Norway"},"latitude":60.193899999999999,"id":2461}
        #airport_trd = {"distance":0.52169665188063608,"magnetic_variation":"E001220 0706","code":{"icao":"ENVA","iata":"TRD"},"name":"Vaernes","longitude":10.9399,"location":{"city":"Trondheim","country":"Norway"},"latitude":63.457599999999999,"id":2472}
        #airports = \
            #{(63.457546234130859, 10.928016315005772):airport_trd,
             #(60.181234387708784, 11.111000827986269):airport_osl}
        
        #runway_osl_01l = {"end":{"latitude":60.216113,"longitude":11.091418},"glideslope":{"latitude":60.187709,"frequency":"332300M","angle":3.0,"longitude":11.072739,"threshold_distance":991},"start":{"latitude":60.185048,"longitude":11.073522},"localizer":{"latitude":60.219793,"beam_width":4.5,"frequency":"111300M","heading":196,"longitude":11.093544},"strip":{"width":147,"length":11811,"surface":"ASP"},"identifier":"19R","id":8152}
        #runway_trd_09 = {"end":{"latitude":63.457572,"longitude":10.941974},"glideslope":{"latitude":63.457085999999997,"frequency":"335000M","angle":3.0,"longitude":10.901011,"threshold_distance":1067},"start":{"latitude":63.457614,"longitude":10.894439},"localizer":{"latitude":63.457539,"beam_width":4.5,"frequency":"110300M","heading":89,"longitude":10.947803},"strip":{"width":147,"length":9347,"surface":"ASP"},"identifier":"09","id":8129}
        #runways = \
            #{2461: runway_osl_01l, 2472: runway_trd_09}        
        
        ## Mock API handler return values so that we do not make http requests.
        ## Will return the same airport and runway for each query, can be
        ## avoided with side_effect.
        #api_handler = mock.Mock()
        #get_api_handler.return_value = api_handler
        #api_handler.get_nearest_airport = mock.Mock()
        #def mocked_nearest_airport(lat, lon, **kwargs):
            #return airports[(lat, lon)]
        #api_handler.get_nearest_airport.side_effect = mocked_nearest_airport
        #api_handler.get_nearest_runway = mock.Mock()
        #def mocked_nearest_runway(airport_id, mag_hdg, **kwargs):
            #return runways[airport_id]
        #api_handler.get_nearest_runway.side_effect = mocked_nearest_runway
        start_datetime = datetime.now()
        
        res = process_flight(hdf_path, ac_info, draw=False)
        self.assertEqual(len(res), 4)
        track_to_kml(hdf_path, res['kti'], res['kpv'])
        from analysis_engine.plot_flight import csv_flight_details
        csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])    


        
    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_1_7295949_737_3C)
        time = min(timer.repeat(1, 1))
        print "Time taken %s secs" % time
        self.assertLess(time, 1.0, msg="Took too long")    

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
                   'Model Series': 'L382',
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
            from analysis_engine.plot_flight import csv_flight_details
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
                   'Model Series': 'L382',
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
            from analysis_engine.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        #TODO: Further assertions on the results!

   
    @unittest.skipIf(not os.path.isfile("test_data/HERCDIP.hdf5"), "Test file not present")
    def test_3A_L382_Hercules_NODIP(self):
        # test copied from herc_2 so AFR may not be accurate
        hdf_orig = "test_data/HERCNODIP.hdf5"
        hdf_path = "test_data/HERCNODIP_copy.hdf5"
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
            from analysis_engine.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        #TODO: Further assertions on the results!
    
    @unittest.skipIf(not os.path.isfile("test_data/HERCDIP.hdf5"), "Test file not present")
    def test_3B_L382_Hercules_DIP(self):
        # test copied from herc_2 so AFR may not be accurate
        hdf_orig = "test_data/HERCDIP.hdf5"
        hdf_path = "test_data/HERCDIP_copy.hdf5"
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
            from analysis_engine.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        #TODO: Further assertions on the results!
     
    @unittest.skipIf(not os.path.isfile("test_data/4_3377853_146_301.hdf5"),
                     "Test file not present")
    @mock.patch('analysis_engine.flight_attribute.get_api_handler')
    def test_4_3377853_146_301(self, get_api_handler):
        # Avoid side effects which may be caused by PRE_FLIGHT_ANALYSIS.
        settings.PRE_FLIGHT_ANALYSIS = None
        hdf_orig = "test_data/4_3377853_146_301.hdf5"
        hdf_path = "test_data/4_3377853_146_301_copy.hdf5"
        if os.path.isfile(hdf_path):
            os.remove(hdf_path)
        shutil.copy(hdf_orig, hdf_path)
        
        ac_info = {'Frame': '146-301',
                   'Identifier': '1',
                   'Manufacturer': 'BAE',
                   'Model Series': '146',
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
        if debug:
            from analysis_engine.plot_flight import csv_flight_details
            csv_flight_details(hdf_path, res['kti'], res['kpv'], res['phases'])
            plot_flight(hdf_path, res['kti'], res['kpv'], res['phases'])
        
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
        self.assertEqual(\
            flight_attrs['FDR Off Blocks Datetime'].value - start_datetime, 
            timedelta(0, 172))
        self.assertEqual(\
            flight_attrs['FDR On Blocks Datetime'].value - start_datetime, 
            timedelta(0, 3490))
        
        
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
        
        
    def test_time_taken_4_3377853_146_301(self):
        from timeit import Timer
        timer = Timer(self.test_4_3377853_146_301)
        time_taken = min(timer.repeat(2, 1))
        print "Time taken %s secs" % time_taken
        self.assertLess(time_taken, 10.0, msg="Took too long")
    
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
