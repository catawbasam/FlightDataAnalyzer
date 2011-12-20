from datetime import datetime, timedelta
import logging

from analysis import ___version___
from analysis.api_handler import get_api_handler, NotFoundError
from analysis.library import datetime_of_index, is_slice_within_slice
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
        Requests the nearest airport to the latitude and longitude at liftoff
        from the API and sets it as an attribute.
        
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
        liftoff_index = liftoff[0].index
        latitude_at_liftoff = latitude.array[liftoff_index]
        longitude_at_liftoff = longitude.array[liftoff_index]
        api_handler = get_api_handler()
        try:
            airport = api_handler.get_nearest_airport(latitude_at_liftoff,
                                                      longitude_at_liftoff)
        except NotFoundError:
            logging.warning("Takeoff Airport could not be found with latitude "
                            "'%f' and longitude '%f'.", latitude_at_liftoff,
                            longitude_at_liftoff)
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
            logging.warning("Runway not found for airport id '%d', heading "
                            "'%f' and kwargs '%s'.", airport_id, hdg_value,
                            kwargs)
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
    '''
    All airports which were approached, including the final landing airport.
    '''
    
    def _get_lat_lng_within_approach(self, approach, lat_kpvs, lng_kpvs):
        lat = self._get_single_kpv(lat_kpvs, within_slice=approach.slice)
        lng = self._get_single_kpv(lng_kpvs, within_slice=approach.slice)
        if lat and lng:
            return lat, lng
        else:
            return None
    
    def _get_single_kpv(self, kpvs, **kwargs):
        matching = kpvs.get(**kwargs)
        if len(matching) == 1:
            return matching[0]
        else:
            return None
    
    @classmethod
    def can_operate(self, available):
        required = all([n in available for n in ['Start Datetime',
                                                 'Approach And Landing',
                                                 'Touch And Go',
                                                 'Go Around']])
        
        approach_lat_lon = 'Latitude At Low Point On Approach' in available and\
                           'Longitude At Low Point On Approach' in available
        landing_lat_lon = 'Latitude At Landing' in available and \
                          'Longitude At Landing' in available
        return required and (approach_lat_lon or landing_lat_lon)
    
    def _get_approach_type(self, landing_hdg_kpvs, touch_and_gos, go_arounds):
        if landing_hdg_kpvs:
            hdg_kpvs = landing_hdg_kpvs.get(within_slice=approach.slice)
            if len(hdg_kpvs) == 1:
                return 'LANDING'
        if touch_and_gos:
            approach_touch_and_gos = touch_and_gos.get(within_slice=
                                                       approach.slice)
            if len(approach_touch_and_gos) == 1:
                return 'TOUCH_AND_GO'
        if go_arounds:
            approach_go_arounds = go_arounds.get(within_slice=
                                                     approach.slice)
            if len(approach_go_arounds) == 1:
                return 'GO_AROUND'
        return None
    
    def _get_lat_lon(self, landing_lat_kpvs, landing_lon_kpvs,
                     approach_lat_kpvs, approach_lon_kpvs):
        if landing_lat_kpvs and landing_lon_kpvs:
            lat_kpvs = landing_lat_kpvs.get(within_slice=approach.slice)
            lon_kpvs = landing_lon_kpvs.get(within_slice=approach.slice)
            if len(lat_kpvs) == 1 and len(lon_kpvs) == 1:
                return (lat_kpvs[0].value, lon_kpvs[0].value)
        if approach_lat_kpvs and approach_lon_kpvs:
            # Try approach KPVs.
            lat_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            lon_kpvs = approach_lon_kpvs.get(within_slice=approach.slice)
            if len(lat_kpvs) == 1 and len(lon_kpvs) == 1:
                return (lat_kpvs[0].value, lon_kpvs[0].value)
        return (None, None)
    
    def _get_hdg(self, landing_hdg_kpvs, appraoch_hdg_kpvs):
        if landing_hdg_kpvs:
            hdg_kpvs = landing_hdg_kpvs.get(within_slice=approach.slice)
            if len(hdg_kpvs) == 1:
                return hdg_kpvs[0].value
        if not hdg and approach_hdg_kpvs:
            # Try approach KPV.
            hdg_kpvs = approach_hdg_kpvs.get(within_slice=approach.slice)
            if len(hdg_kpvs) == 1:
                return hdg_kpvs[0].value
        return None
    
    def derive(self, start_datetime=A('Start Datetime'),
               approach_and_landing=S('Approach And Landing'),
               touch_and_gos=KTI('Touch And Go'), go_arounds=KTI('Go Around'),
               landing_lat_kpvs=KPV('Latitude At Landing'),
               landing_lon_kpvs=KPV('Longitude At Landing'),
               landing_hdg_kpvs=KPV('Heading At Landing'),
               approach_lat_kpvs=KPV('Latitude At Low Point On Approach'),
               approach_lon_kpvs=KPV('Longitude At Low Point On Approach'),
               approach_hdg_kpvs=KPV('Heading At Low Point On Approach'),
               approach_ilsfreq_kpvs=KPV('ILS Frequency On Approach')):
        api_handler = get_api_handler()
        approaches = []
        for approach in approach_and_landing:
            approach_datetime = datetime_of_index(start_datetime.value,
                                                  approach.slice.stop, # Q: Should it be start of approach?
                                                  frequency=approach_and_landing.frequency)
            # Type.
            approach_type = self._get_approach_type(landing_hdg_kpvs,
                                                    touch_and_gos, go_arounds)
            if not approach_type:
                logging.warning("No instance of 'Touch And Go', 'Go Around' or "
                                "'Heading At Landing' within 'Approach And "
                                "Landing' slice indices '%d' and '%d'.",
                                approach.slice.start, approach.slice.stop)
                continue
            # Latitude and Longitude (required for airport query).
            # Try landing KPVs if aircraft landed.
            lat, lon = self._get_lat_lon(landing_lat_kpvs, landing_lon_kpvs,
                                         appraoch_lat_kpvs, approach_lon_kpvs)
            if not lat or not lon:
                logging.warning("Latitude and/or Longitude KPVs not found "
                                "within 'Approach and Landing' phase between "
                                "indices '%d' and '%d'.", approach.slice.start,
                                approach.slice.stop)
                continue
            # Get nearest airport.
            try:
                airport = api_handler.get_nearest_airport(lat, lon)
            except NotFoundError:
                logging.warning("Airport could not be found with latitude '%f' "
                                "and longitude '%f'.", lat, lon)
                continue
            airport_id = airport['id']
            # Heading. Try landing KPV if aircraft landed.
            hdg = self._get_hdg(landing_hdg_kpvs, appraoch_hdg_kpvs)
            if not hdg:
                logging.info("Heading not available for appraoch between "
                             "indices '%d' and '%d'.", approach.slice.start,
                             approach.slice.stop)
                approaches.append({'airport': airport_id,
                                   'runway': None,
                                   'type': appraoch_type,
                                   'datetime': approach_datetime})
                continue
            # ILS Frequency.
            kwargs = {}
            if approach_ilsfreq_kpvs:
                ilsfreq_kpvs = approach_ilsfreq_kpvs.get(within_slice=
                                                         approach.slice)
                if len(ilsfreq_kpvs) == 1:
                    kwargs['ilsfreq'] = ilsfreq[0].value
            if precision and precision.value:
                kwargs.update(latitude=lat, longitude=lon)
            try:
                runway = api_handler.get_nearest_runway(airport_id, hdg,
                                                        **kwargs)
                runway_ident = runway['identifier']
            except NotFoundError:
                logging.warning("Runway could not be found with airport id '%d'"
                                "heading '%s' and kwargs '%s'.", airport_id,
                                hdg, kwargs)
                runway_ident = None
            
            approaches.append({'airport': airport_id,
                               'runway': runway_ident,
                               'type': approach_type, # TODO: Type of approach.
                               'datetime': approach_datetime})
            
            
            
            
            ##approach_ilsfreq_kpvs.get(within_slice=approach.slice)
            ##if len(approach_ilsfreq_kpvs
            
            
            
            
            ##lat_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            ##lon_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            
            ##lat_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            ##lon_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            ##if len(lat_kpvs) != 1 or len(lon_kpvs) != 1:
                ##logging.error("Found '%d' '%s' KPVs and '%d' '%s' KPVs within "
                              ##"approach between indices '%d' and '%d'. '%s' "
                              ##"requires one of each.", len(lat_kpvs),
                              ##lat_kpvs.name, len(lon_kpvs),
                              ##lon_kpvs.name, approach.slice.start,
                              ##approach.slice.stop, self.__class__.__name__)
                ##continue
            ##lat = lat_kpvs[0].value
            ##lon_kpvs = approach_lat_kpvs.get(within_slice=approach.slice)
            ##if len(lat_kpvs) != 1:
                ##logging.error("Found '%d' '%s' KPVs within approach between "
                              ##"indices '%d' and '%d'. '%s' requires 1.",
                              ##len(lat_kpvs), lat_kpvs.name,
                              ##approach.slice.start, approach.slice.stop,
                              ##self.__class__.__name__)
            
                
            
                ##lat = next((kpv.value for kpv in approach_lat_kpvs if \
                            ##is_index_within_slice(kpv.index, approach.slice)))
                ##lng = next((kpv.value for kpv in approach_lng_kpvs if \
                            ##is_index_within_slice(kpv.index, approach.slice)))
            ##except StopIteration:
                ### Latitude and/or Longitude KPVs could not be found within this
                ### approach phase.
                ##logging.warning("'Approach Minimum Latitude' and/or 'Approach "
                                ##"Minimum Longitude' KPVs not found within "
                                ##"'Approach and Landing' phase between indices "
                                ##"'%d' and '%d'.", approach.slice.start,
                                ##approach.slice.stop) 
                ##continue
            
            
            
                
            
            
            ### Try to find a Final Approach phase within the Approach.
            ##for final_approach in final_approaches:
                ##if is_slice_within_slice(final_approach, approach):
                    ##lowest_index = final_approach.slice.stop
                    ##heading.array[lowest_index]
            ##else:
                ##if precision:
                    ##lowest_index = approach.slice.stop
                    ##lowest_latitude = latitude.array[approach.stop]
                    ##lowest_longitude = longitude.array[approach.stop]
            ##try:
                ##airport = api_handler.get_nearest_airport(lowest_latitude,
                                                          ##lowest_longitude)
            
            
            ##ilsfreq.array[]
            
        
        self.set_flight_attr(approaches)


class LandingAirport(FlightAttributeNode):
    "Landing Airport including ID and Name"
    def derive(self, final_approaches=S('Final Approach'),
               latitude=P('Latitude'), longitude=P('Longitude')):
        '''
        See TakeoffAirport for airport dictionary format.
        
        Latitude and longitude are sourced from the end of the last final
        approach in the data.
        Q: What if the data is not complete? last_final
        '''
        last_final_approach = final_approaches.get_last()
        if not last_final_approach:
            return # TODO: Log incomplete data.
        end_of_approach = last_final_approach.slice.stop
        approach_latitude = latitude.array[end_of_approach]
        approach_longitude = longitude.array[end_of_approach]
        api_handler = get_api_handler()
        try:
            airport = api_handler.get_nearest_airport(approach_latitude,
                                                      approach_longitude)
        except NotFoundError:
            logging.warning("Airport could not be found with latitude '%f' "
                            "and longitude '%f'.", lowest_latitude,
                            lowest_longitude)
        else:
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
        api_handler = get_api_handler()
        try:
            runway = api_handler.get_nearest_runway(airport_id, hdg, **kwargs)
        except NotFoundError:
            logging.warning("Runway not found for airport id '%d', heading "
                            "'%f' and kwargs '%s'.", airport_id, hdg_value,
                            kwargs)
        else:
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
        
    