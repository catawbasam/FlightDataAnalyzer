# -*- coding: utf-8 -*-
##############################################################################


import numpy as np

from collections import Counter
from datetime import datetime
from operator import itemgetter

from analysis_engine import __version__, library, settings
from analysis_engine.api_handler import get_api_handler, NotFoundError
from analysis_engine.library import (datetime_of_index,
                                     min_value,
                                     max_value)
from analysis_engine.node import A, KTI, KPV, FlightAttributeNode, M, P, S


##############################################################################
# Superclasses


class DeterminePilot(object):
    '''
    '''

    # FIXME: Handle third autopilot!?
    def _autopilot_engaged(self, ap1, ap2, ap3):
        if ap1 and not ap2:
            return 'Captain'
        if not ap1 and ap2:
            return 'First Officer'
        return None

    def _controls_changed(self, slice_, pitch, roll):
        # Check if either pitch or roll changed during provided slice:
        return pitch[slice_].ptp() > settings.CONTROLS_IN_USE_TOLERANCE or \
                roll[slice_].ptp() > settings.CONTROLS_IN_USE_TOLERANCE

    def _controls_in_use(self, pitch_capt, pitch_fo, roll_capt, roll_fo, phase):
        capt_flying = self._controls_changed(phase.slice, pitch_capt, roll_capt)
        fo_flying = self._controls_changed(phase.slice, pitch_fo, roll_fo)

        # 1. Cannot determine who is flying - both sets of controls have input:
        if capt_flying and fo_flying:
            self.warning("Cannot determine whether captain or first officer "
                "was at the controls because both controls change during '%s' "
                "slice.", phase.name)
            return None

        # 2. The captain was flying the aircraft:
        if capt_flying:
            return 'Captain'

        # 3. The first officer was flying the aircraft:
        if fo_flying:
            return 'First Officer'

        # 4. No change in captain or first officer controls:
        self.warning("Both captain and first officer controls do not change "
            "during '%s' slice.", phase.name)
        return None

    def _determine_pilot(self, pitch_capt, pitch_fo, roll_capt, roll_fo, phase,
                               ap1, ap2, ap3):

        # 1. Check for change in controls during the phase:
        if all((pitch_capt, pitch_fo, roll_capt, roll_fo, phase)):
            pilot = self._controls_in_use(pitch_capt.array, pitch_fo.array,
                    roll_capt.array, roll_fo.array, phase)
            if pilot:
                return pilot

        # 2. Check which autopilot is engaged:
        if all((ap1, ap2)):
            pilot = self._autopilot_engaged(ap1, ap2, ap3)
            if pilot:
                return pilot

        return None


##############################################################################


class InvalidFlightType(Exception):
    def __init__(self, flight_type):
        self.flight_type = flight_type
        super(InvalidFlightType, self).__init__(flight_type)


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
        if num.array.dtype.type is np.string_:
            # XXX: Slow, but Flight Number should be sampled infrequently.
            value, count = next(reversed(sorted(Counter(num.array).items(),
                                                key=itemgetter(1))))
            if value is not np.ma.masked and count > len(num.array) * 0.45:
                self.set_flight_attr(value)
            return
        _, minvalue = min_value(num.array)
        if minvalue < 0:
            self.warning("'%s' only supports unsigned (positive) values",
                            self.name)
            self.set_flight_attr(None)
            return

        # TODO: Fill num.array masked values (as there is no np.ma.bincount) - perhaps with 0.0 and then remove all 0 values?
        # note reverse of value, index from max_value due to bincount usage.
        value, count = max_value(np.bincount(num.array.astype(np.integer)))
        if count > len(num.array) * 0.45:
            # this value accounts for at least 45% of the values in the array
            self.set_flight_attr(str(int(value)))
        else:
            self.warning("Only %d out of %d flight numbers were the same."\
                         " Flight Number attribute will be set as None.",
                         count, len(num.array))
            self.set_flight_attr(None)
            return


class LandingAirport(FlightAttributeNode):
    '''
    The airport that the aircraft landed at determined from the flight data if
    possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Landing Airport'

    @classmethod
    def can_operate(self, available):
        '''
        We can determine a landing airport in one of two ways:

        1. Find the nearest airport to the coordinates at landing.
        2. Use the airport data provided in the achieved flight record.
        '''
        return 'AFR Landing Airport' in available or all((
            'Latitude At Landing' in available,
            'Longitude At Landing' in available,
        ))

    def derive(self,
               land_lat=KPV('Latitude At Landing'),
               land_lon=KPV('Longitude At Landing'),
               land_afr_apt=A('AFR Landing Airport')):
        '''
        '''
        # 1. If we have latitude and longitude, look for the nearest airport:
        if land_lat and land_lon:
            lat = land_lat.get_last()
            lon = land_lon.get_last()
            if lat and lon:
                api = get_api_handler(settings.API_HANDLER)
                try:
                    airport = api.get_nearest_airport(lat.value, lon.value)
                except NotFoundError:
                    msg = 'No landing airport found near coordinates (%f, %f).'
                    self.warning(msg, lat.value, lon.value)
                    # No airport was found, so fall through and try AFR.
                else:
                    self.debug('Detected landing airport: %s', airport)
                    self.set_flight_attr(airport)
                    return  # We found an airport, so finish here.
            else:
                self.warning('No coordinates for looking up landing airport.')
                # No suitable coordinates, so fall through and try AFR.

        # 2. If we have an airport provided in achieved flight record, use it:
        if land_afr_apt:
            airport = land_afr_apt.value
            self.debug('Using landing airport from AFR: %s', airport)
            self.set_flight_attr(airport)
            return  # We found an airport in the AFR, so finish here.

        # 3. After all that, we still couldn't determine an airport...
        self.error('Unable to determine airport at landing!')
        self.set_flight_attr(None)


class LandingRunway(FlightAttributeNode):
    '''
    The runway that the aircraft landed at determined from the flight data if
    possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Landing Runway'

    @classmethod
    def can_operate(self, available):
        '''
        We can determine a landing runway in a number of ways:

        1. Imprecisely using airport and heading at landing.
        2. Precisely using airport, heading and coordinates at landing.
        3. Use the runway data provided in the achieved flight record.
        '''
        minimum = all((
            'FDR Landing Airport' in available,
            'Heading At Landing' in available,
            'ILS Frequency On Approach' in available
        ))

        fallback = 'AFR Landing Runway' in available

        return minimum or fallback

    def derive(self,
            land_fdr_apt=A('FDR Landing Airport'),
            land_afr_rwy=A('AFR Landing Runway'),
            land_hdg=KPV('Heading At Landing'),
            land_lat=KPV('Latitude At Landing'),
            land_lon=KPV('Longitude At Landing'),
            precision=A('Precise Positioning'),
            approaches=S('Approach And Landing'),
            ils_freq_on_app=KPV('ILS Frequency On Approach')):
        '''
        '''
        fallback = False
        precise = bool(getattr(precision, 'value', False))

        try:
            airport = int(land_fdr_apt.value['id'])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.warning('Invalid airport... Fallback to AFR Landing Runway.')
            fallback = True

        try:
            heading = land_hdg.get_last().value
            if heading is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('Invalid heading... Fallback to AFR Landing Runway.')
            fallback = True

        try:
            landing = approaches.get_last()
            if landing is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('No approaches... Fallback to AFR Landing Runway.')
            # Don't set fallback - can still attempt to use heading only...

        # 1. If we have airport and heading, look for the nearest runway:
        if not fallback:
            kwargs = {}

            # The last approach is assumed to be the landing.
            # XXX: Last approach may not be landing for partial data?!
            if ils_freq_on_app:
                ils_freq = ils_freq_on_app.get_last(within_slice=landing.slice)
                if ils_freq:
                    kwargs.update(ils_freq=ils_freq.value)

            # We only provide coordinates when looking up a landing runway if
            # the recording of latitude and longitude on the aircraft is
            # precise. Inertial recordings are too inaccurate to pinpoint the
            # correct runway and we use ILS frequencies if possible to get a
            # more exact match.
            if precise and landing and land_lat and land_lon:
                lat = land_lat.get_last(within_slice=landing.slice)
                lon = land_lon.get_last(within_slice=landing.slice)
                if lat and lon:
                    kwargs.update(
                        latitude=lat.value,
                        longitude=lon.value,
                    )
                else:
                    self.warning('No coordinates for landing runway lookup.')
            else:
                kwargs.update(hint='landing')

            api = get_api_handler(settings.API_HANDLER)
            try:
                runway = api.get_nearest_runway(airport, heading, **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport, heading, kwargs)
                # No runway was found, so fall through and try AFR.
                if 'ils_freq' in kwargs:
                    # This is a trap for airports where the ILS data is not
                    # available, but the aircraft approached with the ILS
                    # tuned. A good prompt for an omission in the database.
                    self.warning('Fix database? No runway but ILS was tuned.')
            else:
                self.debug('Detected landing runway: %s', runway)
                self.set_flight_attr(runway)
                return  # We found a runway, so finish here.

        # 2. If we have a runway provided in achieved flight record, use it:
        if land_afr_rwy:
            runway = land_afr_rwy.value
            self.debug('Using landing runway from AFR: %s', runway)
            self.set_flight_attr(runway)
            return  # We found a runway in the AFR, so finish here.

        # 3. After all that, we still couldn't determine a runway...
        self.error('Unable to determine runway at landing!')
        self.set_flight_attr(None)


class OffBlocksDatetime(FlightAttributeNode):
    "Datetime when moving away from Gate/Blocks"
    name = 'FDR Off Blocks Datetime'
    def derive(self, turning=S('Turning On Ground'),
               start_datetime=A('Start Datetime')):
        first_turning = turning.get_first()
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
    def derive(self, turning=S('Turning On Ground'),
               start_datetime=A('Start Datetime')):
        last_turning = turning.get_last()
        if last_turning:
            on_blocks_datetime = datetime_of_index(start_datetime.value,
                                                   last_turning.slice.stop,
                                                   turning.hz)
            self.set_flight_attr(on_blocks_datetime)
        else:
            self.set_flight_attr(None)


class TakeoffAirport(FlightAttributeNode):
    '''
    The airport that the aircraft took off from determined from the flight data
    if possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Takeoff Airport'

    @classmethod
    def can_operate(self, available):
        '''
        We can determine a takeoff airport in one of two ways:

        1. Find the nearest airport to the coordinates at takeoff.
        2. Use the airport data provided in the achieved flight record.
        '''
        return 'AFR Takeoff Airport' in available or all((
            'Latitude At Takeoff' in available,
            'Longitude At Takeoff' in available,
        ))

    def derive(self,
            toff_lat=KPV('Latitude At Takeoff'),
            toff_lon=KPV('Longitude At Takeoff'),
            toff_afr_apt=A('AFR Takeoff Airport')):
        '''
        '''
        # 1. If we have latitude and longitude, look for the nearest airport:
        if toff_lat and toff_lon:
            lat = toff_lat.get_first()
            lon = toff_lon.get_first()
            if lat and lon:
                api = get_api_handler(settings.API_HANDLER)
                try:
                    airport = api.get_nearest_airport(lat.value, lon.value)
                except NotFoundError:
                    msg = 'No takeoff airport found near coordinates (%f, %f).'
                    self.warning(msg, lat.value, lon.value)
                    # No airport was found, so fall through and try AFR.
                else:
                    self.debug('Detected takeoff airport: %s', airport)
                    self.set_flight_attr(airport)
                    return  # We found an airport, so finish here.
            else:
                self.warning('No coordinates for looking up takeoff airport.')
                # No suitable coordinates, so fall through and try AFR.

        # 2. If we have an airport provided in achieved flight record, use it:
        if toff_afr_apt:
            airport = toff_afr_apt.value
            self.debug('Using takeoff airport from AFR: %s', airport)
            self.set_flight_attr(airport)
            return  # We found an airport in the AFR, so finish here.

        # 3. After all that, we still couldn't determine an airport...
        self.error('Unable to determine airport at takeoff!')
        self.set_flight_attr(None)


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
    def derive(self, liftoff_gross_weight=KPV('Gross Weight At Liftoff')):
        first_gross_weight = liftoff_gross_weight.get_first()
        if first_gross_weight:
            self.set_flight_attr(first_gross_weight.value)
        else:
            # There is not a 'Gross Weight At Liftoff' KPV. Since it is sourced
            # from 'Gross Weight Smoothed', gross weight at liftoff should not
            # be masked.
            self.warning("No '%s' KPVs, '%s' attribute will be None.",
                            liftoff_gross_weight.name, self.name)
            self.set_flight_attr(None)


# FIXME: Check parameters for pitch and roll for captain and first officer!
#        What about 'Pitch Command (*)' and 'Sidestick [Pitch|Roll] (*)'?
# FIXME: This code does not identify the pilot correctly. Roll (FO) is the roll
#        attitude from the right side instrument, not the Airbus first officer
#        sidestick roll input. Needs a rewrite.
class TakeoffPilot(FlightAttributeNode, DeterminePilot):
    '''
    Pilot flying at takeoff - may be the captain, first officer or none.
    '''

    name = 'FDR Takeoff Pilot'

    @classmethod
    def can_operate(cls, available):

        controls = library.all_of((
            'Pitch (Capt)',
            'Pitch (FO)',
            'Roll (Capt)',
            'Roll (FO)',
            'Takeoff',
        ), available)
        autopilot = library.all_of((
            'AP (1) Engaged',
            'AP (2) Engaged',
            'Liftoff',
            # Optional: 'AP (3) Engaged'
        ), available)
        return controls or autopilot

    def derive(self,
               pitch_capt=P('Pitch (Capt)'),
               pitch_fo=P('Pitch (FO)'),
               roll_capt=P('Roll (Capt)'),
               roll_fo=P('Roll (FO)'),
               ap1_eng=M('AP (1) Engaged'),
               ap2_eng=M('AP (2) Engaged'),
               ap3_eng=M('AP (3) Engaged'),
               takeoffs=S('Takeoff'),
               liftoffs=KTI('Liftoff')):

        phase = takeoffs.get_first() if takeoffs else None
        index = liftoffs.get_first() if liftoffs else None
        if index is not None:
            ap_at_index = lambda ap: library.value_at_index(ap.array, index)
            ap1 = ap_at_index(ap1_eng) if ap1_eng else None
            ap2 = ap_at_index(ap2_eng) if ap2_eng else None
            ap3 = ap_at_index(ap3_eng) if ap3_eng else None
        else:
            ap1 = ap2 = ap3 = None
        args = (pitch_capt, pitch_fo, roll_capt, roll_fo, phase, ap1, ap2, ap3)
        self.set_flight_attr(self._determine_pilot(*args))


class TakeoffRunway(FlightAttributeNode):
    '''
    The runway that the aircraft took off from determined from the flight data
    if possible, otherwise falling back to information provided in the achieved
    flight record.
    '''

    name = 'FDR Takeoff Runway'

    @classmethod
    def can_operate(self, available):
        '''
        We can determine a takeoff runway in a number of ways:

        1. Imprecisely using airport and heading at takeoff.
        2. Precisely using airport, heading and coordinates at takeoff.
        3. Use the runway data provided in the achieved flight record.
        '''
        minimum = all((
            'FDR Takeoff Airport' in available,
            'Heading At Takeoff' in available,
        ))

        fallback = 'AFR Takeoff Runway' in available

        return minimum or fallback

    def derive(self,
            toff_fdr_apt=A('FDR Takeoff Airport'),
            toff_afr_rwy=A('AFR Takeoff Runway'),
            toff_hdg=KPV('Heading At Takeoff'),
            toff_lat=KPV('Latitude At Takeoff'),
            toff_lon=KPV('Longitude At Takeoff'),
            precision=A('Precise Positioning')):
        '''
        '''
        fallback = False
        precise = bool(getattr(precision, 'value', False))

        try:
            airport = int(toff_fdr_apt.value['id'])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.warning('Invalid airport... Fallback to AFR Takeoff Runway.')
            fallback = True

        try:
            heading = toff_hdg.get_first().value
            if heading is None:
                raise ValueError
        except (AttributeError, ValueError):
            self.warning('Invalid heading... Fallback to AFR Takeoff Runway.')
            fallback = True

        # 1. If we have airport and heading, look for the nearest runway:
        if not fallback:
            kwargs = {}

            # Even if we do not have precise latitude and longitude
            # information, we still use this for the takeoff runway detection
            # as it is often accurate at the start of a flight, and in the
            # absence of an ILS tuned frequency we have no better option. (We
            # did consider using the last direction of turn onto the runway,
            # but this would require an airport database with terminal and
            # taxiway details that was not felt justified).
            if toff_lat and toff_lon:
                lat = toff_lat.get_first()
                lon = toff_lon.get_first()
                if lat and lon:
                    kwargs.update(
                        latitude=lat.value,
                        longitude=lon.value,
                    )
                else:
                    self.warning('No coordinates for takeoff runway lookup.')
            if not precise:
                kwargs.update(hint='takeoff')

            api = get_api_handler(settings.API_HANDLER)
            try:
                runway = api.get_nearest_runway(airport, heading, **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport, heading, kwargs)
                # No runway was found, so fall through and try AFR.
            else:
                self.debug('Detected takeoff runway: %s', runway)
                self.set_flight_attr(runway)
                return  # We found a runway, so finish here.

        # 2. If we have a runway provided in achieved flight record, use it:
        if toff_afr_rwy:
            runway = toff_afr_rwy.value
            self.debug('Using takeoff runway from AFR: %s', runway)
            self.set_flight_attr(runway)
            return  # We found a runway in the AFR, so finish here.

        # 3. After all that, we still couldn't determine a runway...
        self.error('Unable to determine runway at takeoff!')
        self.set_flight_attr(None)


class FlightType(FlightAttributeNode):
    "Type of flight flown"
    name = 'FDR Flight Type'

    class Type(object):
        '''
        Type of flight.
        '''
        COMMERCIAL = 'COMMERCIAL'
        COMPLETE = 'COMPLETE'
        INCOMPLETE = 'INCOMPLETE'
        ENGINE_RUN_UP = 'ENGINE_RUN_UP'
        GROUND_RUN = 'GROUND_RUN'
        REJECTED_TAKEOFF = 'REJECTED_TAKEOFF'
        TEST = 'TEST'
        TRAINING = 'TRAINING'
        FERRY = 'FERRY'
        POSITIONING = 'POSITIONING'
        LINE_TRAINING = 'LINE_TRAINING'

    @classmethod
    def can_operate(self, available):
        return all(n in available for n in ['Fast', 'Liftoff', 'Touchdown'])

    def derive(self, afr_type=A('AFR Type'), fast=S('Fast'),
               liftoffs=KTI('Liftoff'), touchdowns=KTI('Touchdown'),
               touch_and_gos=S('Touch And Go'), groundspeed=P('Groundspeed')):
        '''
        TODO: Detect MID_FLIGHT.
        '''
        afr_type = afr_type.value if afr_type else None

        if liftoffs and not touchdowns:
            # In the air without having touched down.
            self.warning("'Liftoff' KTI exists without 'Touchdown'.")
            raise InvalidFlightType('LIFTOFF_ONLY')
            #self.set_flight_attr('LIFTOFF_ONLY')
            #return
        elif not liftoffs and touchdowns:
            # In the air without having lifted off.
            self.warning("'Touchdown' KTI exists without 'Liftoff'.")
            raise InvalidFlightType('TOUCHDOWN_ONLY')
            #self.set_flight_attr('TOUCHDOWN_ONLY')
            #return

        if liftoffs and touchdowns:
            first_touchdown = touchdowns.get_first()
            first_liftoff = liftoffs.get_first()
            if first_touchdown.index < first_liftoff.index:
                # Touchdown before having lifted off, data must be INCOMPLETE.
                self.warning("'Touchdown' KTI index before 'Liftoff'.")
                raise InvalidFlightType('TOUCHDOWN_BEFORE_LIFTOFF')
                #self.set_flight_attr('TOUCHDOWN_BEFORE_LIFTOFF')
                #return
            last_touchdown = touchdowns.get_last() # TODO: Delete line.
            if touch_and_gos:
                last_touchdown = touchdowns.get_last()
                last_touch_and_go = touch_and_gos.get_last()
                if last_touchdown.index <= last_touch_and_go.index:
                    self.warning("A 'Touch And Go' KTI exists after the last "
                                 "'Touchdown'.")
                    raise InvalidFlightType('LIFTOFF_ONLY')
                    #self.set_flight_attr('LIFTOFF_ONLY')
                    #return

            if afr_type in [FlightType.Type.FERRY,
                            FlightType.Type.LINE_TRAINING,
                            FlightType.Type.POSITIONING,
                            FlightType.Type.TEST,
                            FlightType.Type.TRAINING]:
                flight_type = afr_type
            else:
                flight_type = FlightType.Type.COMPLETE
        elif fast:
            flight_type = FlightType.Type.REJECTED_TAKEOFF
        elif groundspeed and groundspeed.array.ptp() > 10:
            # The aircraft moved on the ground.
            flight_type = FlightType.Type.GROUND_RUN
        else:
            flight_type = FlightType.Type.ENGINE_RUN_UP
        self.set_flight_attr(flight_type)

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
        if last_gross_weight:
            self.set_flight_attr(last_gross_weight.value)
        else:
            # There is not a 'Gross Weight At Touchdown' KPV. Since it is sourced
            # from 'Gross Weight Smoothed', gross weight at touchdown should not
            # be masked. Are there no Touchdown KTIs?
            self.warning("No '%s' KPVs, '%s' attribute will be None.",
                            touchdown_gross_weight.name, self.name)
            self.set_flight_attr(None)


# FIXME: Check parameters for pitch and roll for captain and first officer!
#        What about 'Pitch Command (*)' and 'Sidestick [Pitch|Roll] (*)'?
# FIXME: This code does not identify the pilot correctly. Roll (FO) is the roll
#        attitude from the right side instrument, not the Airbus first officer
#        sidestick roll input. Needs a rewrite.
class LandingPilot(FlightAttributeNode, DeterminePilot):
    '''
    Pilot flying at landing - may be the captain, first officer or none.
    '''

    name = 'FDR Landing Pilot'

    @classmethod
    def can_operate(cls, available):

        controls = library.all_of((
            'Pitch (Capt)',
            'Pitch (FO)',
            'Roll (Capt)',
            'Roll (FO)',
            'Landing',
        ), available)
        autopilot = library.all_of((
            'AP (1) Engaged',
            'AP (2) Engaged',
            'Touchdown',
            # Optional: 'AP (3) Engaged'
        ), available)
        return controls or autopilot

    def derive(self,
               pitch_capt=P('Pitch (Capt)'),
               pitch_fo=P('Pitch (FO)'),
               roll_capt=P('Roll (Capt)'),
               roll_fo=P('Roll (FO)'),
               ap1_eng=M('AP (1) Engaged'),
               ap2_eng=M('AP (2) Engaged'),
               ap3_eng=M('AP (3) Engaged'),
               landings=S('Landing'),
               touchdowns=KTI('Touchdown')):

        phase = landings.get_last() if landings else None
        index = touchdowns.get_last() if touchdowns else None
        if index is not None:
            ap_at_index = lambda ap: library.value_at_index(ap.array, index)
            ap1 = ap_at_index(ap1_eng) if ap1_eng else None
            ap2 = ap_at_index(ap2_eng) if ap2_eng else None
            ap3 = ap_at_index(ap3_eng) if ap3_eng else None
        else:
            ap1 = ap2 = ap3 = None
        args = (pitch_capt, pitch_fo, roll_capt, roll_fo, phase, ap1, ap2, ap3)
        self.set_flight_attr(self._determine_pilot(*args))


class Version(FlightAttributeNode):
    "Version of code used for analysis"
    name = 'FDR Version'
    def derive(self, start_datetime=P('Start Datetime')):
        '''
        Every derive method requires at least one dependency. Since this class
        should always derive a flight attribute, 'Start Datetime' is its only
        dependency as it will always be present, though it is unused.
        '''
        self.set_flight_attr(__version__)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
