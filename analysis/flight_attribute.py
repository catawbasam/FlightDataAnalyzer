from datetime import datetime, timedelta
import logging

from analysis import ___version___
from analysis.api_handler import get_api_handler, NotFoundError
from analysis.library import datetime_of_index, is_slice_within_slice
from analysis.node import A, KTI, KPV, FlightAttributeNode, P, S


class AnalysisDatetime(FlightAttributeNode):
    "Datetime flight was analysed (local datetime)"
    name = 'FDR Analysis Duration'
    def derive(self, unknown_dep=P('UNKNOWN')): # TODO: Remove dependency if possible?
        self.set_flight_attr(datetime.now())


class Approaches(FlightAttributeNode):
    '''
    All airports which were approached, including the final landing airport.
    '''
    name = 'FDR Approaches'
    @classmethod
    def can_operate(self, available):
        required = all([n in available for n in ['Start Datetime',
                                                 'Approach And Landing',
                                                 'Heading At Landing',
                                                 'Touch And Go',
                                                 'Go Around']])
        
        approach_lat_lon = 'Latitude At Low Point On Approach' in available and\
                           'Longitude At Low Point On Approach' in available
        landing_lat_lon = 'Latitude At Landing' in available and \
                          'Longitude At Landing' in available
        return required and (approach_lat_lon or landing_lat_lon)
    
    def _get_approach_type(self, approach_slice, landing_hdg_kpvs,
                           touch_and_gos, go_arounds):
        '''
        Decides the approach type depending on whether or not a KPV or KTI
        exists or approach.
        
        * Landing At Low Point On Approach KPV exists - LANDING
        * Touch And Go - TOUCH_AND_GO
        * Go Around - GO_AROUND
        
        :param approach_slice: Slice of approach section to get KPVs or KTIs within.
        :type approach_slice: slice
        :param landing_hdg_kpvs: 'Landing At Low Point On Approach' KeyPointValueNode.
        :type landing_hdg_kpvs: KeyPointValueNode
        :param touch_and_gos: 'Touch And Go' KeyTimeInstanceNode.
        :type touch_and_gos: KeyTimeInstanceNode
        :param go_arounds: 'Go Arounds' KeyTimeInstanceNode.
        :type go_arounds: KeyTimeInstanceNode
        '''
        if landing_hdg_kpvs:
            hdg_kpvs = landing_hdg_kpvs.get(within_slice=approach_slice)
            if len(hdg_kpvs) == 1:
                return 'LANDING'
        if touch_and_gos:
            approach_touch_and_gos = touch_and_gos.get(within_slice=
                                                       approach_slice)
            if len(approach_touch_and_gos) == 1:
                return 'TOUCH_AND_GO'
        if go_arounds:
            approach_go_arounds = go_arounds.get(within_slice=approach_slice)
            if len(approach_go_arounds) == 1:
                return 'GO_AROUND'
        return None
    
    def _get_lat_lon(self, approach_slice, landing_lat_kpvs, landing_lon_kpvs,
                     approach_lat_kpvs, approach_lon_kpvs):
        '''
        Returns the latitude and longitude KPV values from landing_lat_kpvs and
        landing_lon_kpvs if they are available (not None) and there is exactly
        one of each within the slice, otherwise will return the latitude and
        longitude KPV values from approach_lat_kpvs and approach_lon_kpvs if
        there is exactly one of each within the slice, otherwise returns None.
        
        :param approach_slice: Slice of approach section to get latitude and longitude within.
        :type approach_slice: slice
        :param landing_lat_kpvs: 'Latitude At Landing' KeyPointValueNode.
        :type landing_lat_kpvs: KeyPointValueNode
        :param landing_lon_kpvs: 'Longitude At Landing' KeyPointValueNode.
        :type landing_lon_kpvs: KeyPointValueNode
        :param approach_lat_kpvs: 'Latitude At Low Point Of Approach' KeyPointValueNode.
        :type approach_lat_kpvs: KeyPointValueNode
        :param approach_lon_kpvs: 'Longitude At Low Point Of Approach' KeyPointValueNode.
        :type approach_lon_kpvs: KeyPointValueNode
        :returns: Latitude and longitude within slice (landing preferred) or pair of Nones.
        :rtype: (int, int) or (None, None)
        '''
        if landing_lat_kpvs and landing_lon_kpvs:
            lat_kpvs = landing_lat_kpvs.get(within_slice=approach_slice)
            lon_kpvs = landing_lon_kpvs.get(within_slice=approach_slice)
            if len(lat_kpvs) == 1 and len(lon_kpvs) == 1:
                return (lat_kpvs[0].value, lon_kpvs[0].value)
        if approach_lat_kpvs and approach_lon_kpvs:
            # Try approach KPVs.
            lat_kpvs = approach_lat_kpvs.get(within_slice=approach_slice)
            lon_kpvs = approach_lon_kpvs.get(within_slice=approach_slice)
            if len(lat_kpvs) == 1 and len(lon_kpvs) == 1:
                return (lat_kpvs[0].value, lon_kpvs[0].value)
        return (None, None)
    
    def _get_hdg(self, approach_slice, landing_hdg_kpvs, approach_hdg_kpvs):
        '''
        Returns the value of a KPV from landing_hdg_kpvs if it is available
        (not None) and there is exactly one within the slice, otherwise will
        return the value of a KPV from approach_hdg_kpvs if there is
        exactly one within the slice, otherwise returns None.
        
        :param approach_slice: Slice of approach section to get a heading within.
        :type approach_slice: slice
        :param landing_hdg_kpvs: 'Heading At Landing' KeyPointValueNode.
        :type landing_hdg_kpvs: KeyPointValueNode
        :param approach_hdg_kpvs: 'Heading At Low Point On Approach' KeyPointValueNode.
        :type approach_hdg_kpvs: KeyPointValueNode
        :returns: Heading within slice (landing preferred) or None.
        :rtype: int or None
        '''
        if landing_hdg_kpvs:
            hdg_kpvs = landing_hdg_kpvs.get(within_slice=approach_slice)
            if len(hdg_kpvs) == 1:
                return hdg_kpvs[0].value
        if approach_hdg_kpvs:
            # Try approach KPV.
            hdg_kpvs = approach_hdg_kpvs.get(within_slice=approach_slice)
            if len(hdg_kpvs) == 1:
                return hdg_kpvs[0].value
        return None
    
    def derive(self, start_datetime=A('Start Datetime'),
               approach_and_landing=S('Approach And Landing'),
               landing_hdg_kpvs=KPV('Heading At Landing'),
               touch_and_gos=KTI('Touch And Go'), go_arounds=KTI('Go Around'),
               landing_lat_kpvs=KPV('Latitude At Landing'),
               landing_lon_kpvs=KPV('Longitude At Landing'),
               approach_lat_kpvs=KPV('Latitude At Low Point On Approach'),
               approach_lon_kpvs=KPV('Longitude At Low Point On Approach'),
               approach_hdg_kpvs=KPV('Heading At Low Point On Approach'),
               approach_ilsfreq_kpvs=KPV('ILS Frequency On Approach'),
               precision=A('Precise Positioning')):
        api_handler = get_api_handler()
        approaches = []
        for approach in approach_and_landing:
            approach_datetime = datetime_of_index(start_datetime.value,
                                                  approach.slice.stop, # Q: Should it be start of approach?
                                                  frequency=approach_and_landing.frequency)
            # Type.
            approach_type = self._get_approach_type(approach.slice,
                                                    landing_hdg_kpvs,
                                                    touch_and_gos, go_arounds)
            if not approach_type:
                logging.warning("No instance of 'Touch And Go', 'Go Around' or "
                                "'Heading At Landing' within 'Approach And "
                                "Landing' slice indices '%d' and '%d'.",
                                approach.slice.start, approach.slice.stop)
                continue
            # Latitude and Longitude (required for airport query).
            # Try landing KPVs if aircraft landed.
            lat, lon = self._get_lat_lon(approach.slice, landing_lat_kpvs,
                                         landing_lon_kpvs, approach_lat_kpvs,
                                         approach_lon_kpvs)
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
            hdg = self._get_hdg(approach.slice, landing_hdg_kpvs,
                                approach_hdg_kpvs)
            if not hdg:
                logging.info("Heading not available for approach between "
                             "indices '%d' and '%d'.", approach.slice.start,
                             approach.slice.stop)
                approaches.append({'airport': airport_id,
                                   'runway': None,
                                   'type': approach_type,
                                   'datetime': approach_datetime})
                continue
            # ILS Frequency.
            kwargs = {}
            if approach_ilsfreq_kpvs:
                ilsfreq_kpvs = approach_ilsfreq_kpvs.get(within_slice=
                                                         approach.slice)
                if len(ilsfreq_kpvs) == 1:
                    kwargs['ilsfreq'] = ilsfreq_kpvs[0].value
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
                               'type': approach_type,
                               'datetime': approach_datetime})
        
        self.set_flight_attr(approaches)


class Duration(FlightAttributeNode):
    "Duration of the flight (between takeoff and landing) in seconds"
    name = 'FDR Duration'
    def derive(self, takeoff_dt=A('Takeoff Datetime'), landing_dt=A('Landing Datetime')):
        duration = landing_dt.value - takeoff_dt.value
        self.set_flight_attr(duration.total_seconds()) # py2.7


class FlightID(FlightAttributeNode):
    "Flight ID if provided via a known input attribute"
    name = 'FDR Flight ID'
    def derive(self, flight_id=A('AFR Flight ID')):
        self.set_flight_attr(flight_id.value)
    
        
class FlightNumber(FlightAttributeNode):
    "Airline route flight number"
    name = 'FDR Flight Number'
    def derive(self, num=P('Flight Number')):
        # Q: Should we validate the flight number or source from a different
        # index?
        self.set_flight_attr(num.array[len(num.array) / 2])


class LandingAirport(FlightAttributeNode):
    "Landing Airport including ID and Name"
    name = 'FDR Landing Airport'
    def derive(self, landing_latitude=KPV('Latitude At Landing'),
               landing_longitude=KPV('Longitude At Landing')):
        '''
        See TakeoffAirport for airport dictionary format.
        
        Latitude and longitude are sourced from the end of the last final
        approach in the data.
        Q: What if the data is not complete? last_final
        '''
        last_latitude = landing_latitude.get_last()
        last_longitude = landing_longitude.get_last()
        if not last_latitude or not last_longitude:
            logging.warning("'%s' and/or '%s' KPVs did not exist, therefore "
                            "'%s' cannot query for landing airport.",
                            last_latitude.name, last_longitude.name,
                            self.__class__.__name__)
            return
        api_handler = get_api_handler()
        try:
            airport = api_handler.get_nearest_airport(last_latitude.value,
                                                      last_longitude.value)
        except NotFoundError:
            logging.warning("Airport could not be found with latitude '%f' "
                            "and longitude '%f'.", last_latitude.value,
                            last_longitude.value)
        else:
            self.set_flight_attr(airport)


class LandingRunway(FlightAttributeNode):
    "Runway identifier name"
    name = 'FDR Landing Runway'
    @classmethod
    def can_operate(self, available):
        '''
        'Landing Heading' is the only required parameter.
        '''
        return all([n in available for n in ['Approach And Landing',
                                             'Landing Airport',
                                             'Heading At Landing']])
        
    def derive(self, approach_and_landing=S('Approach And Landing'),
               landing_hdg=P('Heading At Landing'),
               airport=A('Landing Airport'),
               landing_latitude=P('Latitude At Landing'),
               landing_longitude=P('Longitude At Landing'),
               approach_ilsfreq=KPV('ILS Frequency On Approach'),
               precision=A('Precise Positioning')):
        '''
        See TakeoffRunway for runway information.
        '''
        airport_id = airport.value['id']
        landing = approach_and_landing.get_last()
        if not landing:
            return
        heading_kpv = landing_hdg.get_last(within_slice=landing.slice)
        if not heading_kpv:
            logging.warning("'Heading At Landing' not available in '%s', "
                            "therefore runway cannot be queried for.",
                            self.__class__.__name__)
            return
        heading = heading_kpv.value
        # 'Last Approach And Landing' assumed to be Landing. Q: May not be true
        # for partial data.
        kwargs = {}
        ilsfreq_kpv = approach_ilsfreq.get_last(within_slice=landing.slice)
        kwargs['ilsfreq'] = ilsfreq_kpv.value if ilsfreq_kpv else None
        if precision and precision.value and landing_latitude and \
           landing_longitude:
            last_latitude = landing_latitude.get_last(within_slice=
                                                      landing.slice)
            last_longitude = landing_longitude.get_last(within_slice=
                                                        landing.slice)
            if last_latitude and last_longitude:
                kwargs.update(latitude=last_latitude.value,
                              longitude=last_longitude.value)
        
        api_handler = get_api_handler()
        try:
            runway = api_handler.get_nearest_runway(airport_id, heading,
                                                    **kwargs)
        except NotFoundError:
            logging.warning("Runway not found for airport id '%d', heading "
                            "'%f' and kwargs '%s'.", airport_id, hdg_value,
                            kwargs)
        else:
            self.set_flight_attr(runway)


class OffBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR Off Blocks Datetime'
    def derive(self, unknown_dep=P('UNKNOWN')):
        return NotImplemented


class OnBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR On Blocks Datetime'
    def derive(self, unknown_dep=P('UNKNOWN')):
        return NotImplemented

   
class TakeoffAirport(FlightAttributeNode):
    "Takeoff Airport including ID and Name"
    name = 'FDR Takeoff Airport'
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


class TakeoffDatetime(FlightAttributeNode):
    '''
    Datetime at takeoff (first liftoff) or as close to this as possible.
    If no takeoff (incomplete flight / ground run) the start of data will is
    to be used.
    '''
    name = 'FDR Takeoff Datetime'
    def derive(self, liftoff=KTI('Liftoff'), start_dt=A('Start Datetime')):
        first_liftoff = liftoff.get_first()
        if not first_liftoff:
            return
        liftoff_index = first_liftoff.index
        takeoff_dt = datetime_of_index(start_dt.value, liftoff_index,
                                       frequency=liftoff.frequency)
        self.set_flight_attr(takeoff_dt)


class TakeoffFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Takeoff"
    name = 'FDR Takeoff Fuel'
    @classmethod
    def can_operate(self, available):
        return 'AFR Takeoff Fuel' in available or \
               'Fuel Qty At Liftoff' in available
    
    def derive(self, afr_takeoff_fuel=A('AFR Takeoff Fuel'),
               liftoff_fuel_qty=KPV('Fuel Qty At Liftoff')):
        if afr_takeoff_fuel:
            #TODO: Validate that the AFR record is more accurate than the
            #flight data if available.
            self.set_flight_attr(afr_takeoff_fuel.value)
        else:
            fuel_qty_kpv = liftoff_fuel_qty.get_first()
            if fuel_qty_kpv:
                self.set_flight_attr(fuel_qty_kpv.value)


class TakeoffGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Takeoff"
    name = 'FDR Takeoff Gross Weight'
    def derive(self, liftoff_gross_weight=P('Gross Weight At Liftoff')):
        first_gross_weight = liftoff_gross_weight.get_first()
        if not first_gross_weight:
            return
        self.set_flight_attr(first_gross_weight.value)

    
class TakeoffPilot(FlightAttributeNode):
    "Pilot flying at takeoff, Captain, First Officer or None"
    name = 'FDR Takeoff Pilot'
    # TODO: Dependency name mappings.
    def derive(self, liftoff_autopilot1=KPV('Autopilot Engaged 1 At Liftoff'),
               liftoff_autopilot2=KPV('Autopilot Engaged 2 At Liftoff'),
               pitch_captain=P('Pitch (Capt)'), roll_captain=P('Roll (Capt)'),
               pitch_fo=P('Pitch (FO)'), roll_fo=P('Roll (FO)')):
        
        # TODO: Use Flight Director parameters if possible.
        #pilot = None
        #assert pilot in ("FIRST_OFFICER", "CAPTAIN", None)
        if liftoff_autopilot1 and liftoff_autopilot2:
            if liftoff_autopilot1.value and not liftoff_autopilot2.value:
                self.set_flight_attr('Captain')
                return
            elif not liftoff_autopilot1.value and liftoff_autopilot2.value:
                self.set_flight_attr('First Officer')
                return
        
        #control_input (1) Control Input (2) / contro wheeel / control column
        # Cannot determine Takeoff Pilot.
        self.set_flight_attr(None)


class TakeoffRunway(FlightAttributeNode):
    "Runway identifier name"
    name = 'FDR Takeoff Runway'
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


class FlightType(FlightAttributeNode):
    "Type of flight flown"
    name = 'Flight Type'
    
    @classmethod
    def can_operate(self, available):
        return all([n in available for n in ['Fast', 'Liftoff', 'Touchdown']])
    
    def derive(self, afr_type=A('AFR Type'), fast=S('Fast'),
               liftoffs=KTI('Liftoff'), touchdowns=KTI('Touchdown'),
               touch_and_gos=S('Touch And Go'), groundspeed=P('Groundspeed')):
        afr_type = afr_type.value if afr_type else None
        
        if liftoffs and not touchdowns:
            # In the air without having touched down.
            logging.warning("'Liftoff' KTI exists without 'Touchdown'. '%s' "
                            "will be 'INCOMPLETE'.", self.name)
            self.set_flight_attr('LIFTOFF_ONLY')
            return
        elif not liftoffs and touchdowns:
            # In the air without having lifted off.
            logging.warning("'Touchdown' KTI exists without 'Liftoff'. '%s' "
                            "will be 'INCOMPLETE'.", self.name)
            self.set_flight_attr('TOUCHDOWN_ONLY')
            return
        
        if liftoffs and touchdowns:
            first_touchdown = touchdowns.get_first()
            first_liftoff = liftoffs.get_first()
            if first_touchdown.index < first_liftoff.index:
                # Touchdown before having lifted off, data must be INCOMPLETE.
                logging.warning("'Touchdown' KTI index before 'Liftoff'. '%s' "
                                "will be 'INCOMPLETE'.", self.name)
                self.set_flight_attr('TOUCHDOWN_BEFORE_LIFTOFF')
                return
            last_touchdown = touchdowns.get_last()
            if touch_and_gos:
                last_touchdown = touchdowns.get_last()
                last_touch_and_go = touch_and_gos.get_last()
                if last_touchdown.index <= last_touch_and_go.index:
                    logging.warning("A 'Touch And Go' KTI exists after the last "
                                    "'Touchdown'. '%s' will be 'INCOMPLETE'.",
                                    self.name)
                    self.set_flight_attr('LIFTOFF_ONLY')
                    return
            
            if afr_type in ['FERRY', 'LINE_TRAINING', 'POSITIONING' 'TEST',
                            'TRAINING']:
                flight_type = afr_type
            else:
                flight_type = 'COMPLETE'
            self.set_flight_attr(flight_type)
        elif fast:
            self.set_flight_attr('REJECTED_TAKEOFF')
        elif groundspeed and groundspeed.array.ptp() > 10:
            # The aircraft moved on the ground.
            self.set_flight_attr('GROUND_RUN')
        else:
            self.set_flight_attr('ENGINE_RUN_UP')


#Q: Not sure if we can identify Destination from the data?
##class DestinationAirport(FlightAttributeNode):
    ##""
    ##def derive(self):
        ##return NotImplemented
                    ##{'id':9456, 'name':'City. Airport'}


class LandingDatetime(FlightAttributeNode):
    """ Datetime at landing (final touchdown) or as close to this as possible.
    If no landing (incomplete flight / ground run) store None.
    """
    name = 'FDR Landing Datetime'
    def derive(self, start_datetime=A('Start Datetime'),
               touchdown=KTI('Touchdown')):
        last_touchdown = touchdown.get_last()
        if not last_touchdown:
            self.set_flight_attr(None)
            return
        landing_datetime = datetime_of_index(start_datetime,
                                             last_touchdown.index,
                                             frequency=touchdown.frequency) 
        self.set_flight_attr(landing_datetime)

         
class LandingFuel(FlightAttributeNode):
    "Weight of Fuel in KG at point of Touchdown"
    name = 'FDR Landing Fuel'
    @classmethod
    def can_operate(self, available):
        return 'AFR Landing Fuel' in available or \
               'Fuel Qty At Touchdown' in available
    
    def derive(self, afr_landing_fuel=A('AFR Landing Fuel'),
               touchdown_fuel_qty=KPV('Fuel Qty At Touchdown')):
        if afr_landing_fuel:
            self.set_flight_attr(afr_landing_fuel.value)
        else:
            fuel_qty_kpv = touchdown_fuel_qty.get_last()
            if fuel_qty_kpv:
                self.set_flight_attr(fuel_qty_kpv.value)


class LandingGrossWeight(FlightAttributeNode):
    "Aircraft Gross Weight in KG at point of Landing"
    name = 'FDR Landing Gross Weight'
    def derive(self, touchdown_gross_weight=KPV('Gross Weight At Touchdown')):
        last_gross_weight = touchdown_gross_weight.get_last()
        if last_gross_weight:
            self.set_flight_attr(last_gross_weight.value)


class LandingPilot(FlightAttributeNode):
    "Pilot flying at landing, Captain, First Officer or None"
    name = 'FDR Landing Pilot'
    def derive(self, unknown_dep=P('UNKNOWN')):
        pilot = None
        assert pilot in ("FIRST_OFFICER", "CAPTAIN", None)
        return NotImplemented
    
        
class V2(FlightAttributeNode):
    name = 'FDR V2'
    def derive(self, unknown_dep=P('UNKNOWN')):
        return NotImplemented
         
         
class Vapp(FlightAttributeNode):
    name = 'FDR Vapp'
    def derive(self, unknown_dep=P('UNKNOWN')):
        return NotImplemented


class Version(FlightAttributeNode):
    "Version of code used for analysis"
    name = 'FDR Version'
    def derive(self, unknown_dep=P('UNKNOWN')):
        self.set_flight_attr(___version___)
        return NotImplemented
         
         
class Vref(FlightAttributeNode):
    name = 'FDR Vref'
    def derive(self, unknown_dep=P('UNKNOWN')):
        return NotImplemented
                
        
    