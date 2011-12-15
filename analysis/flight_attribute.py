from datetime import datetime

from analysis import ___version___
from analysis.node import A, KTI, KPV, FlightAttributeNode, P
from api_handler import get_api_handler

class FlightID(FlightAttributeNode):
    "Flight ID if provided via a known input attribute"
    name = 'Flight ID'
    def derive(self, flight_id=A('Flight ID')):
        return flight_id
    
        
class FlightNumber(FlightAttributeNode):
    "Airline route flight number"
    def derive(self):
        # e.g. 'DEM23'
        return NotImplemented
    
    
class Type(FlightAttributeNode):
    "Type of flight flown"
    def derive(self):
        'TEST'
        return NotImplemented
         
         
class TakeoffAirport(FlightAttributeNode):
    "Takeoff Airport including ID and Name"
    def derive(self, lat=KPV('Latitude At Liftoff'), lon=KPV('Longitude At Liftoff')):
        """
        Airport information is in the following format:
        {'code': {'iata': 'LHR', 'icao': 'EGLL'},
        'distance': 1.512545797147365,
        'id': 2383,
        'latitude': 51.4775,
        'location': {'city': 'London', 'country': 'United Kingdom'},
        'longitude': -0.461389,
        'magnetic_variation': 'W002241 0106', # Format subject to change.
        'name': 'London Heathrow'}"""
        # get_airport API
        api_handler = get_api_handler()
        airport = api_handler.get_nearest_airport(lat, lon)
        self.set_flight_attr(airport)
        #self.set_flight_attr({'id':1234, 'name': 'Int. Airport'})
        return NotImplemented
                
                
class TakeoffRunway(FlightAttributeNode):
    "Runway identifier name"
    @classmethod
    def can_operate(self, available):
        # does not require all parameters to be available
        return NotImplemented
        
    def derive(self, lat=KPV('Latitude At Liftoff'), lon=KPV('Longitude At Liftoff'), 
               hdg=P('Heading Magnetic At Liftoff'), airport=A('Takeoff Airport')):
        """
        Expects Airport to store a dictionary of airport attributes.
        
        Runway information is in the following format:
        {'id': 1234,
         'identifier': '29L',
         'magnetic_heading': 290,
         'start': {
             'latitude': 14.1,
             'longitude': 7.1,
         },
         'end': {
             'latitude': 14.2,
             'longitude': 7.2,
         },
             'glideslope': {
                  'angle': 120, # Q: Sensible example value?
                  'frequency': 330, # Q: Sensible example value?
                  'latitude': 14.3,
                  'longitude': 7.3,
                  'threshold_distance': 20,
              },
              'localiser': {
                  'beam_width': 14, # Q: Sensible example value?
                  'frequency': 335, # Q: Sensible example value?
                  'heading': 291,
                  'latitude': 14.4,
                  'longitude': 7.4,
              },
         'strip': {
             'length': 150,
             'surface': 'ASPHALT',
             'width': 30,
        }}
        """
        ils_freq = airport.value.get('Localizer Frequency')
        airport_id = airport.value['id'], 
        precision = True #TODO: precision == using GPS
        api_handler = get_api_handler()
        runway = api_handler.get_runway(airport_id, hdg, latitude=lat, longitude=lon,
                                        precision=precision, ils_freq=ils_freq)
        self.set_flight_attr(runway)
        return NotImplemented
    
    
class OffBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    def derive(self):
        return NotImplemented
    
                
class TakeoffDatetime(FlightAttributeNode):
    """
    Datetime at takeoff (first liftoff) or as close to this as possible.
    If no takeoff (incomplete flight / ground run) the start of data will is
    to be used.
    """
    def derive(self):
        return NotImplemented
    
                
class TakeoffPilot(FlightAttributeNode):
    "Pilot flying at takeoff, Captain, First Officer or None"
    def derive(self):
        pilot = None
        assert pilot in ("FIRST_OFFICER", "CAPTAIN", None)
        return NotImplemented
        
                
class TakeoffGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Takeoff"
    def derive(self):
        return NotImplemented
         
         
class TakeoffFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Takeoff"
    def derive(self):
        return NotImplemented


#Q: Not sure if we can identify Destination from the data?
##class DestinationAirport(FlightAttributeNode):
    ##""
    ##def derive(self):
        ##return NotImplemented
                    ##{'id':9456, 'name':'City. Airport'}
                    
                    
class Approaches(FlightAttributeNode):
    "All airports which were approached, including the final landing airport"
    def derive(self):
        # approaches need runway, 'TOUCH_AND_GO' or 'APPROACH', and datetime
        approach = {
            'runway' : '15R',  # or None
            'type' : 'TOUCH_AND_GO',  # or 'APPROACH'
            'datetime' : datetime.now(), # TODO!
        }
        ##self.set_flight_attr(list(approaches))
        return NotImplemented
        
    
class LandingAirport(FlightAttributeNode):
    "Takeoff Airport including ID and Name"
    def derive(self, lat=KPV('Latitude At Touchdown'), lon=KPV('Longitude At Touchdown')):
        # get_airport API
        airport = get_airport(lat, lon)
        self.set_flight_attr(airport)
        #self.set_flight_attr({'id':1234, 'name': 'Int. Airport'})
        return NotImplemented
                
                
class LandingRunway(FlightAttributeNode):
    "Runway identifier name"
    @classmethod
    def can_operate(self, available):
        # does not require all parameters to be available
        return NotImplemented
        
    def derive(self, lat=KPV('Latitude At Touchdown'), lon=KPV('Longitude At Touchdown'), 
               hdg=P('Heading Magnetic At Touchdown'), airport=A('Landing Airport')):
        """
        Expects Airport to store a dictionary of airport attributes
        """
        ils_freq = airport.value.get('Localiser Frequency')
        airport_id = airport.value['id'], 
        precision = True #TODO: precision == using GPS
        runway = get_runway(airport=airport_id, latitude=lat, longitude=lon,
                            heading=hdg, ils_freq=ils_freq, precision=precision)
        self.set_flight_attr(runway)
        return NotImplemented
    
    
class OnBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    def derive(self):
        return NotImplemented
    
                
class LandingDatetime(FlightAttributeNode):
    """ Datetime at landing (final touchdown) or as close to this as possible.
    If no landing (incomplete flight / ground run) store None.
    """
    def derive(self):
        return NotImplemented
    
                
class LandingPilot(FlightAttributeNode):
    "Pilot flying at landing, Captain, First Officer or None"
    def derive(self):
        pilot = None
        assert pilot in ("FIRST_OFFICER", "CAPTAIN", None)
        return NotImplemented
        
                
class LandingGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Landing"
    def derive(self):
        return NotImplemented
         
         
class LandingFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Landing"
    def derive(self):
        return NotImplemented
    
        
class V2(FlightAttributeNode):
    def derive(self):
        return NotImplemented
         
         
class Vapp(FlightAttributeNode):
    def derive(self):
        return NotImplemented
         
         
class Vref(FlightAttributeNode):
    def derive(self):
        return NotImplemented
         
            
class Version(FlightAttributeNode):
    "Version of code used for analysis"
    def derive(self):
        self.set_flight_attr(___version___)
        return NotImplemented


class Duration(FlightAttributeNode):
    "Duration of the flight (between takeoff and landing) in seconds"
    def derive(self, takeoff_dt=A('Takeoff Datetime'), landing_dt=A('Landing Datetime')):
        duration = landing_dt - takeoff_dt
        self.set_flight_attr(duration.total_seconds()) # py2.7

                
class AnalysisDatetime(FlightAttributeNode):
    "Datetime flight was analysed (local datetime)"
    def derive(self):
        self.set_flight_attr(datetime.now())
        
    