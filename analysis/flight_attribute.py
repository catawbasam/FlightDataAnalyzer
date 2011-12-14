
from analysis.node import A, KTI, KPV, FlightAttributeNode, P



# EXAMPLES - they expect some liftoff KTIs!


class TakeoffAirport(FlightAttributeNode):
    def derive(self, lat=KPV('Latitude At Liftoff'), lon=KPV('Longitude At Liftoff')):
        # get_airport API
        airport = get_airport(lat, lon)
        self.set_flight_attr(airport)
        
        
class TakeoffRunway(FlightAttributeNode):
    @classmethod
    def can_operate(self, available):
        pass
        
    def derive(self, lat=KPV('Latitude At Liftoff'), lon=KPV('Longitude At Liftoff'), 
               hdg=P('Heading Magnetic At Liftoff'), airport=A('Takeoff Airport')):
        """
        Expects Airport to store a dictionary of airport attributes
        """
        ils_freq = airport.value.get('Localizer Frequency')
        airport_id = airport.value['id'], 
        precision = True #TODO: precision == using GPS
        runway = get_runway(airport=airport_id, latitude=lat, longitude=lon,
                            heading=hdg, ils_freq=ils_freq, precision=precision)
        self.set_flight_attr(runway)
        
                      
##class Approaches(FlightAttributeNode):
    ##def derive(self):
        ####self.set_flight_attr(list(approaches))
        ##return NotImplemented
        
            