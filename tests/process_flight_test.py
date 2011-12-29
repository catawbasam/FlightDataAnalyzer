import unittest
import csv
import os
import shutil

from datetime import datetime
        
from analysis.process_flight import process_flight, derive_parameters, get_derived_nodes

from analysis.node import KeyPointValueNode, P, KeyTimeInstanceNode, S
from analysis.library import value_at_time
from analysis.key_time_instances import TriggerPassiveNodes

import itertools
def sort_by_index_or_slice(x):
    try:
        return float(x.index)
    except:
        try:
            return x.slice.start
        except:
            pass
      
    #except (TypeError, AttributeError):
        #return x.slice.start

def extend_output(output):
    get = GetParamsForDevelopmentOutput()
    index = output[-1][2]
    output.extend([get.airspeed(index), get.alt_aal(index)])
    return output

def output_phase_kti_kpv_for_development(result):
    output=[]
    file_for_indexed_output = open('C:/temp/try.csv', 'wb')
    to_csv = csv.writer(file_for_indexed_output)
    for phase in result['phases']:
        output.append(['Phase Start', phase.name, phase.slice.start])
        #output = extend_output(output)
        output.append(['Phase Stop' , None, phase.slice.stop,  phase.name])
    for kti in result['kti']:
        output.append(['KTI', None, kti.index, None, kti.name])
    for kpv in result['kpv']:
        output.append(['KPV', None, kpv.index, None, kpv.name, kpv.value])
    for row in sorted(output,key=lambda index:index[2]):
        to_csv.writerow(row)
    return
           
class GetParamsForDevelopmentOutput(KeyPointValueNode):
    def derive(self, speed=P('Airspeed'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        return

    def airspeed(self,index, speed=P('Airspeed')):
        return value_at_time (speed.array, speed.hz, speed.offset, index)
        
    def alt_aal(self,index, alt_aal=P('Altitude AAL For Flight Phases')):
        return value_at_time (alt_aal.array, alt_aal.hz, alt_aal.offset, index)
                                         

    
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
        #res = process_flight(args, kwargs)
        clouseau = TriggerPassiveNodes()
        clouseau.derive()
        res = process_flight(hdf_path, ac_info, achieved_flight_record=afr, draw=False)
        output_phase_kti_kpv_for_development(res)
        
        #index_sorted_res = sorted(itertools.chain(*res.values()), key=sort_by_index_or_slice)
        self.assertEqual(len(res), 4)
    
    
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
        