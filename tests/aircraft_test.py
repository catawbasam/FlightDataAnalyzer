try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
    
from utilities.struct import Struct


from analysis.aircraft import populate_aircraft_params

class TestAircraft(unittest.TestCase):
    def test_populate_aircraft_params(self):
        aircraft = Struct({
            'engines': {
                'manufacturer': 'CFM International',
                'model': 'CFM56-7B24',
                'type': 'jet'},
            'frame': {
                'doubled': False,
                'name': '737-3C'},
            'id': 1001,
            'identifier': '5',
            'manufacturer': 'Boeing',
            'manufacturer_serial_number': '',
            'model': {
                'name':'737-800',
                'geometry': {
                    'wing_span': 120,
                    'something': None},
                },
            'recorder': {
                'name': 'L3UQAR', 
                'serial': '123456'},
            'tail_number': 'G-ABCD'}
             )
        # add geometry by hand (coz it wasn't in API at this time!)
        aircraft.model.geometry.main_gear_to_alt_rad = 10
        res = populate_aircraft_params(aircraft)
        
        self.assertEqual(res['Tail Number'], 'G-ABCD')
        self.assertEqual(res['Frame'], '737-3C')
        self.assertEqual(res['Wing Span'], 120)
        self.assertFalse('Manufacturer Serial Number' in res)