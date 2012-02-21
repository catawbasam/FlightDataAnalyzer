from datetime import datetime
import numpy as np
import logging
import itertools
import operator

from analysis_engine import ___version___
from analysis_engine.api_handler import get_api_handler, NotFoundError
from analysis_engine.library import datetime_of_index, min_value, max_value
from analysis_engine.node import A, KTI, KPV, FlightAttributeNode, P, S
from analysis_engine.settings import CONTROLS_IN_USE_TOLERANCE, API_HANDLER
from scipy.interpolate import interp1d


class AnalysisDatetime(FlightAttributeNode):
    "Datetime flight was analysed (local datetime)"
    name = 'FDR Analysis Datetime'
    def derive(self, start_datetime=A('Start Datetime')):
        '''
        Every derive method requires at least one dependency. Since this class
        should always derive a flight attribute, 'Start Datetime' is its only
        dependency as it will always be present, though it is unused.
        '''
        self.set_flight_attr(datetime.now())


class Approaches(FlightAttributeNode):
    '''
    All airports which were approached, including the final landing airport.
    '''
    name = 'FDR Approaches'
    @classmethod
    def can_operate(self, available):
        return all([n in available for n in ['Start Datetime',
                                             'Approach And Landing',
                                             'Altitude AAL',
                                             'Approach And Go Around',
                                             'Latitude At Lowest Point On Approach',
                                             'Longitude At Lowest Point On Approach',
                                             'Latitude At Landing',
                                             'Longitude At Landing']])
        
    def _get_lat_lon(self, approach_slice, lat_kpv_node, lon_kpv_node):
        lat_kpvs = lat_kpv_node.get(within_slice=approach_slice)
        lon_kpvs = lon_kpv_node.get(within_slice=approach_slice)        
        if len(lat_kpvs) == 1 and len(lon_kpvs) == 1:
            return (lat_kpvs[0].value, lon_kpvs[0].value)
        else:
            return (None, None)
    
    def _create_approach(self, start_dt, api_handler, approach, approach_type,
                         frequency, lat_kpv_node, lon_kpv_node, hdg_kpv_node,
                         ilsfreq_kpv_node, precision):
        approach_datetime = datetime_of_index(start_dt, approach.slice.stop, # Q: Should it be start of approach?
                                              frequency=frequency)        
        lat, lon = self._get_lat_lon(approach.slice,
                                     lat_kpv_node, lon_kpv_node)
        if not lat or not lon:
            logging.warning("Latitude and/or Longitude KPVs not found "
                            "within 'Approach and Landing' phase between "
                            "indices '%d' and '%d'.", approach.slice.start,
                            approach.slice.stop)
            return
        # Get nearest airport.
        try:
            airport = api_handler.get_nearest_airport(lat, lon)
        except NotFoundError:
            logging.warning("Airport could not be found with latitude '%f' "
                            "and longitude '%f'.", lat, lon)
            return
        airport_id = airport['id']
        
        # Heading. Try landing KPV if aircraft landed.
        hdg = None
        if hdg_kpv_node:
            hdg_kpvs = hdg_kpv_node.get(within_slice=approach.slice)
            if len(hdg_kpvs) == 1:
                hdg = hdg_kpvs[0].value
        if not hdg:
            logging.info("Heading not available for approach between "
                         "indices '%d' and '%d'.", approach.slice.start,
                         approach.slice.stop)
            return {'airport': airport,
                    'runway': None,
                    'type': approach_type,
                    'datetime': approach_datetime,
                    'slice_start_datetime': datetime_of_index(start_dt,
                                                              approach.slice.start,
                                                              frequency), # NB: Not in API therefore not stored in DB
                    'slice_stop_datetime': datetime_of_index(start_dt,
                                                             approach.slice.stop,
                                                             frequency), # NB: Not in API therefore not stored in DB
                    }
        # ILS Frequency.
        kwargs = {}
        if ilsfreq_kpv_node:
            ilsfreq_kpvs = ilsfreq_kpv_node.get(within_slice=approach.slice)
            if len(ilsfreq_kpvs) == 1:
                kwargs['ilsfreq'] = ilsfreq_kpvs[0].value
        if precision and precision.value:
            # Only use lat and lon if 'Precise Positioning' is True.
            kwargs.update(latitude=lat, longitude=lon)
        try:
            runway_info = api_handler.get_nearest_runway(airport_id, hdg, **kwargs)
            if len(runway_info['items']) > 1:
                # TODO: What to store in approach dictionary.
                runway = {'identifier': runway_info['ident']}
                logging.warning("Identified %d Runways, ident %s. Picking the first!", 
                             len(runway_info['items']), runway_info['ident'])
            else:
                runway = runway_info['items'][0]
        except NotFoundError:
            logging.warning("Runway could not be found with airport id '%d'"
                            "heading '%s' and kwargs '%s'.", airport_id,
                            hdg, kwargs)
            runway = None
        
        return {'airport': airport,
                'runway': runway,
                'type': approach_type,
                'datetime': approach_datetime,
                'slice_start_datetime': datetime_of_index(start_dt,
                                                          approach.slice.start,
                                                          frequency), # NB: Not in API therefore not stored in DB
                'slice_stop_datetime': datetime_of_index(start_dt,
                                                         approach.slice.stop,
                                                         frequency), # NB: Not in API therefore not stored in DB                
                }    
    
    def derive(self, start_datetime=A('Start Datetime'),
               approach_landing=S('Approach And Landing'),
               landing_hdg_kpvs=KPV('Heading At Landing'), # touch_and_gos=KTI('Touch And Go'),
               approach_go_around=KTI('Approach And Go Around'),
               alt_aal = P('Altitude AAL'),
               landing_lat_kpvs=KPV('Latitude At Landing'),
               landing_lon_kpvs=KPV('Longitude At Landing'),
               approach_lat_kpvs=KPV('Latitude At Lowest Point On Approach'),
               approach_lon_kpvs=KPV('Longitude At Lowest Point On Approach'),
               approach_hdg_kpvs=KPV('Heading At Lowest Point On Approach'),
               approach_ilsfreq_kpvs=KPV('ILS Frequency On Approach'),
               precision=A('Precise Positioning')):
        '''
        TODO: Document approaches format.
        TODO: Test!
        '''
        api_handler = get_api_handler(API_HANDLER)
        approaches = []
        
        for approach_section in approach_landing:
            approach = self._create_approach(start_datetime.value, api_handler,
                                             approach_section, 'LANDING',
                                             approach_landing.frequency,
                                             landing_lat_kpvs, landing_lon_kpvs,
                                             landing_hdg_kpvs,
                                             approach_ilsfreq_kpvs, precision)
            if approach:
                approaches.append(approach)
        
        for approach_section in approach_go_around:
            # If Altitude AAL reached 0, the approach type is 'TOUCH_AND_GO'.
            if np.ma.any(alt_aal.array[approach_section.slice] <= 0):
                approach_type = 'TOUCH_AND_GO'
            else:
                approach_type = 'GO_AROUND'
            approach = self._create_approach(start_datetime.value, api_handler,
                                             approach_section, approach_type,
                                             approach_go_around.frequency,
                                             approach_lat_kpvs, approach_lon_kpvs,
                                             approach_hdg_kpvs,
                                             approach_ilsfreq_kpvs, precision)
            if approach:
                approaches.append(approach)
            
        self.set_flight_attr(approaches)


class DeterminePilot(object):
    def _autopilot_engaged(self, autopilot1, autopilot2):
        if not autopilot1 or not autopilot2:
            return None
        elif autopilot1.value and not autopilot2.value:
            return 'Captain'
        elif not autopilot1.value and autopilot2.value:
            return 'First Officer'
    
    def _pitch_roll_changed(self, slice_, pitch, roll):
        '''
        Check if either pitch or roll changed during slice_.
        '''
        return pitch[slice_].ptp() > CONTROLS_IN_USE_TOLERANCE or \
               roll[slice_].ptp() > CONTROLS_IN_USE_TOLERANCE
    
    def _controls_in_use(self, pitch_captain, roll_captain, pitch_fo, roll_fo,
                         section):
        captain_flying = self._pitch_roll_changed(section.slice, pitch_captain,
                                                  roll_captain)
        fo_flying = self._pitch_roll_changed(section.slice, pitch_fo, roll_fo)
        if captain_flying and fo_flying:
            logging.warning("Cannot determine whether Captain or First "
                            "Officer was at the controls because both "
                            "controls change during '%s' slice.",
                            section.name)
            return None
        elif captain_flying:
            return 'Captain'
        elif fo_flying:
            return 'First Officer'
        else:
            logging.warning("Both captain and first officer controls "
                            "do not change during '%s' slice.",
                            section.name)
            return None
    
    def _determine_pilot(self, pitch_captain, roll_captain, pitch_fo, roll_fo,
                         takeoff_or_landing, autopilot1, autopilot2):
        if not takeoff_or_landing and (not autopilot1 or not autopilot2):
            return None
        # 1) Find out whether the captain or first officer's controls changed
        # during takeoff_or_landing.
        if pitch_captain and roll_captain and pitch_fo and roll_fo and \
           takeoff_or_landing:
            # Detect which controls were in use during takeoff_or_landing.
            pilot_flying = self._controls_in_use(pitch_captain.array,
                                                 roll_captain.array,
                                                 pitch_fo.array,
                                                 roll_fo.array,
                                                 takeoff_or_landing)
            if pilot_flying:
                return pilot_flying
        
        # 2) Find out which autopilot is engaged at liftoff.
        if autopilot1 and autopilot2:
            pilot_flying = self._autopilot_engaged(autopilot1, autopilot2)
            return pilot_flying


class Duration(FlightAttributeNode):
    "Duration of the flight (between takeoff and landing) in seconds"
    name = 'FDR Duration'
    def derive(self, takeoff_dt=A('FDR Takeoff Datetime'),
               landing_dt=A('FDR Landing Datetime')):
        if landing_dt.value and takeoff_dt.value:
            duration = landing_dt.value - takeoff_dt.value
            self.set_flight_attr(duration.total_seconds()) # py2.7
        else:
            self.set_flight_attr(None)
            return


class FlightID(FlightAttributeNode):
    "Flight ID if provided via a known input attribute"
    name = 'FDR Flight ID'
    def derive(self, flight_id=A('AFR Flight ID')):
        self.set_flight_attr(flight_id.value)


class FlightNumber(FlightAttributeNode):
    """
    Returns String representation of the integer Flight Number value.
    
    Raises ValueError if negative value in array or too great a variance in
    array values.
    """
    "Airline route flight number"
    name = 'FDR Flight Number'
    def derive(self, num=P('Flight Number')):
        # Q: Should we validate the flight number?
        _, minvalue = min_value(num.array)
        if minvalue < 0:
            raise ValueError("Only supports unsigned (positive) values")
        
        # TODO: Fill num.array masked values (as there is no np.ma.bincount) - perhaps with 0.0 and then remove all 0 values?
        # note reverse of value, index from max_value due to bincount usage.
        value, count = max_value(np.bincount(num.array.astype(np.integer)))
        if count > len(num.array) * 0.6:
            # this value accounts for at least 60% of the values in the array
            self.set_flight_attr(str(value))
        else:
            raise ValueError("Low variance")


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
            logging.warning("'Latitude At Landing' and/or 'Longitude At "
                            "Landing' KPVs did not exist, therefore '%s' "
                            "cannot query for landing airport.",
                            self.__class__.__name__)
            self.set_flight_attr(None)
            return
        api_handler = get_api_handler(API_HANDLER)
        try:
            airport = api_handler.get_nearest_airport(last_latitude.value,
                                                      last_longitude.value)
        except NotFoundError:
            logging.warning("Airport could not be found with latitude '%f' "
                            "and longitude '%f'.", last_latitude.value,
                            last_longitude.value)
            self.set_flight_attr(None)
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
                                             'FDR Landing Airport',
                                             'Heading At Landing']])
        
    def derive(self, approach_and_landing=S('Approach And Landing'),
               landing_hdg=KPV('Heading At Landing'),
               airport=A('FDR Landing Airport'),
               landing_latitude=P('Latitude At Landing'),
               landing_longitude=P('Longitude At Landing'),
               approach_ilsfreq=KPV('ILS Frequency On Approach'),
               precision=A('Precise Positioning')
               ):
        '''
        See TakeoffRunway for runway information.
        '''
        if not airport.value:
            logging.warning("'%s' requires '%s' to be set.", self.name,
                            airport.name)
            self.set_flight_attr(None)
            return
        airport_id = airport.value['id']
        landing = approach_and_landing.get_last()
        if not landing:
            logging.warning("No landing")
            self.set_flight_attr(None)
            return
        heading = landing_hdg[-1].value
            
        # 'Last Approach And Landing' assumed to be Landing. Q: May not be true
        # for partial data?
        kwargs = {}
        if approach_ilsfreq:
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
        
        api_handler = get_api_handler(API_HANDLER)
        try:
            runway_info = api_handler.get_nearest_runway(airport_id, heading,
                                                    **kwargs)
            if len(runway_info['items']) > 1:
                # TODO: Having a string here will cause problems..
                runway = {'identifier': runway_info['ident']}
                ##raise NotImplementedError('Multiple runways returned')
            else:
                runway = runway_info['items'][0]            
        except NotFoundError:
            logging.warning("Runway not found for airport id '%d', heading "
                            "'%f' and kwargs '%s'.", airport_id, heading,
                            kwargs)
        else:
            self.set_flight_attr(runway)


class OffBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR Off Blocks Datetime'
    def derive(self, turning=P('Turning'), start_datetime=A('Start Datetime')):
        first_turning = turning.get_first(name='Turning On Ground')
        if first_turning:
            off_blocks_datetime = datetime_of_index(start_datetime.value,
                                                    first_turning.slice.start,
                                                    turning.hz)
            self.set_flight_attr(off_blocks_datetime)
        else:
            self.set_flight_attr(None)


class OnBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR On Blocks Datetime'
    def derive(self, turning=P('Turning'), start_datetime=A('Start Datetime')):
        last_turning = turning.get_last(name='Turning On Ground')
        if last_turning:
            on_blocks_datetime = datetime_of_index(start_datetime.value,
                                                   last_turning.slice.stop,
                                                   turning.hz)
            self.set_flight_attr(on_blocks_datetime)
        else:
            self.set_flight_attr(None)


class TakeoffAirport(FlightAttributeNode):
    "Takeoff Airport including ID and Name"
    name = 'FDR Takeoff Airport'
    def derive(self, latitude_at_takeoff=KPV('Latitude At Takeoff'),
               longitude_at_takeoff=KPV('Longitude At Takeoff')):
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
         'magnetic_variation': 2.5,
         'name': 'London Heathrow'}
        '''
        first_latitude = latitude_at_takeoff.get_first()
        first_longitude = longitude_at_takeoff.get_first()
        if not first_latitude or not first_longitude:
            logging.warning("Cannot create '%s' attribute without '%s' or "
                            "'%s'.", self.name, latitude_at_takeoff.name,
                            longitude_at_takeoff.name)
            self.set_flight_attr(None)
            return
        api_handler = get_api_handler(API_HANDLER)
        try:
            airport = api_handler.get_nearest_airport(first_latitude.value,
                                                      first_longitude.value)
        except NotFoundError:
            logging.warning("Takeoff Airport could not be found with '%s' "
                            "'%f' and '%s' '%f'.", latitude_at_takeoff.name,
                            first_latitude.value, longitude_at_takeoff.name,
                            first_longitude.value)
            self.set_flight_attr(None)
        else:
            self.set_flight_attr(airport)


class TakeoffDatetime(FlightAttributeNode):
    '''
    Datetime at takeoff (first liftoff) or as close to this as possible.
    If no takeoff (incomplete flight / ground run) the start of data will is
    to be used.
    '''
    name = 'FDR Takeoff Datetime'
    def derive(self, liftoff=A('Liftoff'), start_dt=A('Start Datetime')):
        first_liftoff = liftoff.get_first()
        if not first_liftoff:
            self.set_flight_attr(None)
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
            
    
class TakeoffPilot(FlightAttributeNode, DeterminePilot):
    "Pilot flying at takeoff, Captain, First Officer or None"
    name = 'FDR Takeoff Pilot'
    @classmethod
    def can_operate(cls, available):
        controls_available = all([n in available for n in ('Pitch (Capt)',
                                                           'Pitch (FO)',
                                                           'Roll (Capt)',
                                                           'Roll (FO)',
                                                           'Takeoff')])
        autopilot_available = 'Autopilot Engaged 1 At Liftoff' in available and\
                              'Autopilot Engaged 2 At Liftoff' in available
        return controls_available or autopilot_available
    
    def derive(self, pitch_captain=P('Pitch (Capt)'),
               roll_captain=P('Roll (Capt)'), pitch_fo=P('Pitch (FO)'),
               roll_fo=P('Roll (FO)'), takeoffs=S('Takeoff'),
               autopilot1=KPV('Autopilot Engaged 1 At Liftoff'),
               autopilot2=KPV('Autopilot Engaged 2 At Liftoff')):
        first_takeoff = takeoffs.get_first() if takeoffs else None
        first_autopilot1 = autopilot1.get_first() if autopilot1 else None
        first_autopilot2 = autopilot2.get_first() if autopilot2 else None
        pilot_flying = self._determine_pilot(pitch_captain, roll_captain,
                                             pitch_fo, roll_fo, first_takeoff,
                                             first_autopilot1, first_autopilot2)
        self.set_flight_attr(pilot_flying)


class TakeoffRunway(FlightAttributeNode):
    "Runway identifier name"
    name = 'FDR Takeoff Runway'
    @classmethod
    def can_operate(self, available):
        return 'FDR Takeoff Airport' in available and \
               'Heading At Takeoff' in available

    def derive(self, airport=A('FDR Takeoff Airport'),
               hdg=KPV('Heading At Takeoff'),
               latitude_at_takeoff=KPV('Latitude At Takeoff'),
               longitude_at_takeoff=KPV('Longitude At Takeoff'),
               precision=A('Precise Positioning')):
        '''
        Runway information is in the following format:
        {'id': 1234,
         'identifier': '29L',
         'magnetic_heading': 290,
         ['start': {
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
        if not airport.value:
            logging.warning("'%s' requires '%s' to be set.", self.name,
                            airport.name)
            self.set_flight_attr(None)
            return
        airport_id = airport.value['id']
        kwargs = {}
        # Even if we do not have precise latitude and longitude information,
        # we still use this for the takeoff runway detection as it is often
        # accurate at the start of a flight, and in the absence of an ILS
        # tuned frequency we have no better option. (We did consider using
        # the last direction of turn onto the runway, but this would require
        # an airport database with terminal and taxiway details that was not
        # felt justified).
        if latitude_at_takeoff and longitude_at_takeoff:
        ##if precision and precision.value and latitude_at_takeoff and \
           ##longitude_at_takeoff:
            first_latitude = latitude_at_takeoff.get_first()
            first_longitude = longitude_at_takeoff.get_first()
            if first_latitude and first_longitude:
                kwargs.update(latitude=first_latitude.value,
                              longitude=first_longitude.value)
        
        hdg_value = hdg[0].value
        api_handler = get_api_handler(API_HANDLER)
        try:
            runway_info = api_handler.get_nearest_runway(airport_id, hdg_value,
                                                    **kwargs)
            if len(runway_info['items']) > 1:
                # TODO: This will probably break nodes which are dependent.
                runway = {'identifier': runway_info['ident']}
                raise NotImplementedError('Multiple runways returned')
            else:
                runway = runway_info['items'][0]
        except NotFoundError:
            logging.warning("Runway not found for airport id '%d', heading "
                            "'%f' and kwargs '%s'.", airport_id, hdg_value,
                            kwargs)
        else:
            self.set_flight_attr(runway)


class FlightType(FlightAttributeNode):
    "Type of flight flown"
    name = 'FDR Flight Type'
    
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
        landing_datetime = datetime_of_index(start_datetime.value,
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
        # TODO: Support flight attributes not calling set_flight_attr where appropriate.
        #if last_gross_weight:
        self.set_flight_attr(last_gross_weight.value)


class LandingPilot(FlightAttributeNode, DeterminePilot):
    "Pilot flying at takeoff, Captain, First Officer or None"
    name = 'FDR Takeoff Pilot'
    @classmethod
    def can_operate(cls, available):
        controls_available = all([n in available for n in ('Pitch (Capt)',
                                                           'Pitch (FO)',
                                                           'Roll (Capt)',
                                                           'Roll (FO)',
                                                           'Landing')])
        autopilot_available = 'Autopilot Engaged 1 At Touchdown' in available \
                          and 'Autopilot Engaged 2 At Touchdown' in available
        return controls_available or autopilot_available
    
    def derive(self, pitch_captain=P('Pitch (Capt)'),
               roll_captain=P('Roll (Capt)'), pitch_fo=P('Pitch (FO)'),
               roll_fo=P('Roll (FO)'), landings=S('Landing'),
               autopilot1=KPV('Autopilot Engaged 1 At Touchdown'),
               autopilot2=KPV('Autopilot Engaged 2 At Touchdown')):
        last_landing = landings.get_last()
        last_autopilot1 = autopilot1.get_last()
        last_autopilot2 = autopilot2.get_last()
        pilot_flying = self._determine_pilot(pitch_captain, roll_captain,
                                             pitch_fo, roll_fo, last_landing,
                                             last_autopilot1, last_autopilot2)
        self.set_flight_attr(pilot_flying)

    
class V2(FlightAttributeNode):
    '''
    Based on weight and flap at time of landing.
    '''
    name = 'FDR V2'
    def derive(self, weight_touchdown=KPV('Gross Weight At Touchdown'),
               flap_touchdown=KPV('Flap At Touchdown')):
        '''
        Do not source from AFR, only set attribute if V2 is recorded/derived.
        '''
        weight = weight_touchdown.get_last()
        flap = flap_touchdown.get_last()
        if not weight or not flap:
            # TODO: Log.
            return
        return NotImplemented
         
         
class Vapp(FlightAttributeNode):
    '''
    Based on weight and flap at time of landing.
    '''
    name = 'FDR Vapp'
    def derive(self, weight_touchdown=KPV('Gross Weight At Touchdown'),
               flap_touchdown=KPV('Flap At Touchdown')):
        '''
        Do not source from AFR, only set attribute if Vapp is recorded/derived.
        '''
        weight = weight_touchdown.get_last()
        flap = flap_touchdown.get_last()
        if not weight or not flap:
            # TODO: Log.
            return
        return NotImplemented


class Version(FlightAttributeNode):
    "Version of code used for analysis"
    name = 'FDR Version'
    def derive(self, start_datetime=P('Start Datetime')):
        '''
        Every derive method requires at least one dependency. Since this class
        should always derive a flight attribute, 'Start Datetime' is its only
        dependency as it will always be present, though it is unused.
        '''
        self.set_flight_attr(___version___)


class Vref(FlightAttributeNode):
    '''
    Based on weight and flap at time of landing.
    '''
    name = 'FDR Vref'
    def derive(self, aircraft_model=A('AFR Aircraft Model'),
               weight_touchdown=KPV('Gross Weight At Touchdown'),
               flap_touchdown=KPV('Flap At Touchdown')):
        '''
        Do not source from AFR, only set attribute if V2 is recorded/derived.
        '''
        ##weight = weight_touchdown.get_last()
        ##flap = flap_touchdown.get_last()
        ##if not weight or not flap:
            ### TODO: Log.
            ##return
        ##try:
            ##mapping = VREF_MAP[aircraft_model.value]
            ##index = index_at_value(np.array(mapping['Gross Weights']),
                                   ##weight.value)
            ##interp = interp1d(enumerate(mapping['Flaps']))
            ##interp(index)
            
        return NotImplemented
                
##VREF_MAP = 
##{'B737-2_800010_00.add':
 ##{'Gross Weights': range(32, 65, 4),
  ##'Flaps': {15: (111, 118, 125, 132, 138, 143, 149, 154, 159),
            ##30: (105, 111, 117, 123, 129, 135, 140, 144, 149),
            ##40: (101, 108, 114, 120, 125, 130, 135, 140, 145)}}}
    
