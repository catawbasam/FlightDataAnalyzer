import numpy as np

from analysis_engine.api_handler import get_api_handler, NotFoundError
from analysis_engine.node import A, ApproachNode, KPV, KTI, P, S
from analysis_engine import settings


class ApproachInformation(ApproachNode):
    """
    # TODO: Update docstring for ApproachNode.
    
    Details of all approaches that were made including landing.

    If possible we attempt to determine the airport and runway associated with
    each approach.

    We also attempt to determine an approach type which may be one of the
    following:

    - Landing
    - Touch & Go
    - Go Around

    The date and time at the start and end of the approach is also determined.

    When determining the airport and runway, we use the heading, latitude and
    longitude at:
    
    a. landing for landing approaches, and
    b. the lowest point on the approach for any other approaches.

    If we are unable to determine the airport and runway for a landing
    approach, it is also possible to fall back to the achieved flight record.

    A list of approach details are returned in the following format::
    
        [
            {
                'airport': {...},  # See output provided by Airport API.
                'runway': {...},   # See output provided by Runway API.
                'type': 'LANDING',
                'datetime': datetime(1970, 1, 1, 0, 0, 0),
                'slice_start': 100,
                'slice_stop': 200,
                'ILS localizer established': slice(start, stop),
                'ILS glideslope established': slice(start, stop),
                'ILS frequency', 109.05
            },
            {
                'airport': {...},  # See output provided by Airport API.
                'runway': {...},   # See output provided by Runway API.
                'type': 'GO_AROUND',
                'datetime': datetime(1970, 1, 1, 0, 0, 0),
                'slice_start': 100,
                'slice_stop': 200,
                'ILS localizer established': slice(start, stop),
                'ILS frequency', 109.05
            },
            {
                'airport': {...},  # See output provided by Airport API.
                'runway': {...},   # See output provided by Runway API.
                'type': 'TOUCH_AND_GO',
                'datetime': datetime(1970, 1, 1, 0, 0, 0),
                'slice_start': 100,
                'slice_stop': 200,
            },
            ...
        ]
    
    
    
    This separates out the approach phase excluding the landing.
    """
    
    @classmethod
    def can_operate(self, available):
        '''
        '''
        return all(n in available for n in ['Approach And Landing',
                                            'Altitude AAL', 'Fast'])
    
    def _lookup_airport_and_runway(self, _slice, precise, lowest_lat,
                                   lowest_lon, lowest_hdg, appr_ils_freq,
                                   land_afr_apt=None, land_afr_rwy=None,
                                   hint='approach'):
        '''
        '''
        api = get_api_handler(settings.API_HANDLER)
        kwargs = {}
        airport = None
        runway = None

        # A1. If we have latitude and longitude, look for the nearest airport:
        if lowest_lat and lowest_lon:
            kwargs.update(latitude=lowest_lat.value, longitude=lowest_lon.value)
            try:
                airport = api.get_nearest_airport(**kwargs)
            except NotFoundError:
                msg = 'No approach airport found near coordinates (%f, %f).'
                self.warning(msg, lowest_lat.value, lowest_lon.value)
                # No airport was found, so fall through and try AFR.
            else:
                self.debug('Detected approach airport: %s', airport)
        else:
            # No suitable coordinates, so fall through and try AFR.
            self.warning('No coordinates for looking up approach airport.')

        # A2. If and we have an airport in achieved flight record, use it:
        # NOTE: AFR data is only provided if this approach is a landing.
        if not airport and land_afr_apt:
            airport = land_afr_apt.value
            self.debug('Using approach airport from AFR: %s', airport)

        # A3. After all that, we still couldn't determine an airport...
        if not airport:
            self.error('Unable to determine airport on approach!')
            return None, None
        
        airport_id = int(airport['id'])

        if lowest_hdg:
        #if heading is None:
            #self.warning('Invalid heading... Fallback to AFR.')
            #fallback = True
            
            # R1. If we have airport and heading, look for the nearest runway:
            if appr_ils_freq:
                ils_freq = appr_ils_freq.get_first(within_slice=_slice)
                if ils_freq:
                    kwargs['ils_freq'] = ils_freq.value

            # We already have latitude and longitude in kwargs from looking up
            # the airport. If the measurments are not precise, remove them.
            if not precise:
                kwargs['hint'] = hint
                del kwargs['latitude']
                del kwargs['longitude']

            try:
                runway = api.get_nearest_runway(airport_id, lowest_hdg.value,
                                                **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport_id, lowest_hdg.value, kwargs)
                # No runway was found, so fall through and try AFR.
                if 'ils_freq' in kwargs:
                    # This is a trap for airports where the ILS data is not
                    # available, but the aircraft approached with the ILS
                    # tuned. A good prompt for an omission in the database.
                    self.warning('Fix database? No runway but ILS was tuned.')
            else:
                self.debug('Detected approach runway: %s', runway)
                runway = runway

        # R2. If we have a runway provided in achieved flight record, use it:
        if not runway and land_afr_rwy:
            runway = land_afr_rwy.value
            self.debug('Using approach runway from AFR: %s', runway)

        # R3. After all that, we still couldn't determine a runway...
        if not runway:
            self.error('Unable to determine runway on approach!')

        return airport, runway
    
    def derive(self, app=S('Approach And Landing'),
               alt_aal=P('Altitude AAL'),
               fast=S('Fast'),
               land_hdg=KPV('Heading At Landing'),
               land_lat=KPV('Latitude At Landing'),
               land_lon=KPV('Longitude At Landing'),
               appr_hdg=KPV('Heading At Lowest Point On Approach'),
               appr_lat=KPV('Latitude At Lowest Point On Approach'),
               appr_lon=KPV('Longitude At Lowest Point On Approach'),
               loc_ests=S('ILS Localizer Established'),
               gs_ests=S('ILS Glideslope Established'),
               appr_ils_freq=KPV('ILS Frequency On Approach'),
               land_afr_apt=A('AFR Landing Airport'),
               land_afr_rwy=A('AFR Landing Runway'),
               precision=A('Precise Positioning'),
               turnoffs=KTI('Landing Turn Off Runway')):
        precise = bool(getattr(precision, 'value', False))
        
        default_kwargs = dict(
            precise=precise,
            appr_ils_freq=appr_ils_freq,
        )

        for _slice in app.get_slices():

            # a) We have a landing if approach end is outside of a fast section:
            if _slice.stop > fast.get_last().slice.stop:
                approach_type = 'LANDING'
                landing = True
            # b) We have a touch and go if Altitude AAL reached zero:
            elif np.ma.any(alt_aal.array[_slice] <= 0):
                approach_type = 'TOUCH_AND_GO'
                landing = False
            # c) In any other case we have a go-around:
            else:
                approach_type = 'GO_AROUND'
                landing = False

            # Prepare arguments for looking up the airport and runway:
            kwargs = default_kwargs.copy()
            
            # Pass latitude, longitude and heading depending whether this
            # approach is a landing or not.
            #
            # If we are not landing, we go with the lowest point on approach.
            lat = land_lat if landing else appr_lat
            lon = land_lon if landing else appr_lon
            hdg = land_hdg if landing else appr_hdg
            
            lowest_lat = lat.get_first(within_slice=_slice) if lat else None
            lowest_lon = lon.get_first(within_slice=_slice) if lon else None
            lowest_hdg = hdg.get_first(within_slice=_slice) if hdg else None
            
            kwargs.update(
                lowest_lat=lowest_lat,
                lowest_lon=lowest_lon,
                lowest_hdg=lowest_hdg,
                _slice=_slice,
            )

            # If the approach is a landing, pass through information from the
            # achieved flight record in case we cannot determine airport and
            # runway:
            if landing:
                kwargs.update(
                    land_afr_apt=land_afr_apt,
                    land_afr_rwy=land_afr_rwy,
                    hint='landing',
                )

            # Prepare approach information and populate with airport and runway
            # via API calls:
            
            gs_est = gs_ests.get_first(
                within_slice=_slice, within_use='any') if gs_ests else None
            loc_est = loc_ests.get_first(
                within_slice=_slice, within_use='any') if loc_ests else None
            
            # Add further details to save hunting when we need them later.
            ils_freq = None
            if appr_ils_freq:
                ils_freq = appr_ils_freq.get_first(within_slice=_slice)
                if ils_freq:
                    ils_freq = ils_freq.value
                     
            turnoff = None   
            if turnoffs:
                turnoff = turnoffs.get_first(within_slice=_slice)
                if turnoff:
                    turnoff = turnoff.index            
            
            airport, runway = self._lookup_airport_and_runway(**kwargs)
            self.create_approach(
                approach_type, _slice, airport=airport,
                runway=runway, gs_est=gs_est, loc_est=loc_est,
                ils_freq=ils_freq, turnoff=turnoff,
                lowest_lat=lowest_lat, lowest_lon=lowest_lon,
                lowest_hdg=lowest_hdg)        
