
from analysis.node import FlightAttributeNode

# EXAMPLE:
class TakeoffAirport(FlightAttributeNode):
    def derive(self, lat_long=P('Smoothed Track'), lift_off=P('Liftoff')):
        # get_airport API
        airport = get_airport(lat_long[lift_off[0]])
        self.set_flight_attr('Takeoff Airport', airport.id)


