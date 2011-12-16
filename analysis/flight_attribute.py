from datetime import datetime

from analysis import ___version___
from analysis.api_handler import get_api_handler, NotFoundError
from analysis.node import A, KTI, KPV, FlightAttributeNode, P, S


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
    def derive(self, liftoff=KTI('Liftoff'), latitude=P('Latitude'),
               longitude=P('Longitude')):
        '''
        Gets the nearest airport to the latitude and longitude at liftoff and
        sets it as an attribute.
        
        Airport information is in the following format:
        {'code': {'iata': 'LHR', 'icao': 'EGLL'},
         'distance': 1.512545797147365,
         'id': 2383,
         'latitude': 51.4775,
         'longitude': -0.461389,
         'location': {'city': 'London', 'country': 'United Kingdom'},
         'magnetic_variation': 'W002241 0106', # Format subject to change.
         'name': 'London Heathrow'}
        '''
        api_handler = get_api_handler()
        liftoff_index = liftoff[0].index
        latitude_at_liftoff = latitude.array[liftoff_index]
        longitude_at_liftoff = longitude.array[liftoff_index]
        try:
            airport = api_handler.get_nearest_airport(latitude_at_liftoff,
                                                      longitude_at_liftoff)
        except NotFoundError:
            pass
        else:
            self.set_flight_attr(airport)
                
                
class TakeoffRunway(FlightAttributeNode):
    "Runway identifier name"
    @classmethod
    def can_operate(self, available):
        return 'Takeoff Airport' in available and 'Takeoff Heading' in available

    def derive(self, airport=A('Takeoff Airport'), hdg=KPV('Takeoff Heading'),
               liftoff=KTI('Liftoff'), latitude=P('Latitude'),
               longitude=P('Longitude'), precision=A('Precise Positioning')):
        '''
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
        '''
        kwargs = {}
        if precision and precision.value and liftoff and latitude and longitude:
            liftoff_index = liftoff[0].index
            latitude_at_liftoff = latitude.array[liftoff_index]
            longitude_at_liftoff = longitude.array[liftoff_index]
            kwargs.update(latitude=latitude_at_liftoff,
                          longitude=longitude_at_liftoff)
        airport_id = airport.value['id']
        hdg_value = hdg[0].value
        api_handler = get_api_handler()
        try:
            runway = api_handler.get_nearest_runway(airport_id, hdg_value,
                                                    **kwargs)
        except NotFoundError:
            pass
        else:
            self.set_flight_attr(runway)
    
    
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
    "Landing Airport including ID and Name"
    def derive(self, approaches=S('Approach'), latitude=P('Latitude'),
               longitude=P('Longitude')):
        '''
        See TakeoffAirport for airport dictionary format.
        '''
        for approach in approaches:
            end_of_approach = approach.slice.stop
            approach_latitude = latitude.array[end_of_approach]
            approach_longitude = longitude.array[end_of_approach]
            airport = get_airport(approach_latitude, approach_longitude)
            # TODO: Multiple LandingAirport attributes?
            self.set_flight_attr(airport)
                
                
class LandingRunway(FlightAttributeNode):
    "Runway identifier name"
    @classmethod
    def can_operate(self, available):
        '''
        'Landing Heading' is the only required parameter.
        '''
        return 'Landing Heading' in available
        
    def derive(self, final_approaches=S('Final Approach'),
               hdg=P('Landing Heading'), touchdown=KTI('Touchdown'),
               latitude=P('Latitude'), longitude=P('Longitude'),
               ilsfreq=KPV('Landing ILS Freq'), airport=A('Landing Airport'),
               precision=A('Precise Positioning')):
        '''
        See TakeoffRunway for runway information.
        '''
        kwargs = {}
        if ilsfreq:
            kwargs['ilsfreq'] = ilsfreq[0].value
        if precision and precision.value and touchdown and latitude and \
           longitude:
            touchdown_index = touchdown[0].index
            latitude_at_touchdown = latitude.array[touchdown_index]
            longitude_at_touchdown = longitude.array[touchdown_index]
            kwargs.update(latitude=latitude_at_touchdown,
                          longitude=longitude_at_touchdown)
        hdg_value = hdg[0].value
        airport_id = airport.value['id']
        runway = get_runway(airport_id, hdg, **kwargs)
        self.set_flight_attr(runway)
    
    
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
        
    