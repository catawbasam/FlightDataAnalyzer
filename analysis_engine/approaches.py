import numpy as np

from analysis_engine.api_handler import get_api_handler, NotFoundError
from analysis_engine.library import index_closest_value, slices_or
from analysis_engine.node import A, ApproachNode, KPV, KTI, P, S
from analysis_engine import settings


def approach_slices(self, alt_aal, lands, go_arounds):
    # Prepare to extract the slices
    app_slices = []
    ga_slices = []

    for land in lands:
        app_start = index_closest_value(
            alt_aal.array, settings.INITIAL_APPROACH_THRESHOLD,
            slice(land.slice.start, 0, -1))
        app_slices.append(slice(app_start, land.slice.stop))

    last_ga = 0
    for ga in go_arounds:
        # The go-around KTI is based on only a 500ft 'pit' but to include
        # the approach phase we stretch the start point back towards
        # 3000ft. To avoid merging multiple go-arounds, the endpoint is
        # carried across from one to the next, which is a safe thing to
        # do because the KTI algorithm works on the cycle finder results
        # which are inherently ordered.
        gapp_start = index_closest_value(
            alt_aal.array, settings.INITIAL_APPROACH_THRESHOLD,
            slice(ga.slice.start, last_ga, -1))
        ga_slices.append(slice(gapp_start, ga.slice.stop))
        last_ga = ga.slice.stop

    all_apps = slices_or(app_slices, ga_slices)
    
    if not all_apps:
        self.warning('Flight with no valid approach or go-around phase. '
                     'Probably truncated data')            

    return all_apps


class Approaches(ApproachNode):
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
        return all(n in available for n in [
            'Altitude AAL For Flight Phases',
            'Landing',
            'Go Around And Climbout',
            'Altitude AAL',
            'Fast',
        ])
    
    def _approaches(self, approach_slices, alt_aal, fast, land_hdg, land_lat,
                    land_lon, appr_hdg, appr_lat, appr_lon, loc_ests, gs_ests,
                    appr_ils_freq, land_afr_apt, land_afr_rwy, precision,
                    turnoffs):
        precise = bool(getattr(precision, 'value', False))

        default_kwargs = dict(
            precise=precise,
            appr_ils_freq=appr_ils_freq,
        )

        for _slice in approach_slices:

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
            kwargs.update(
                appr_lat=land_lat if landing else appr_lat,
                appr_lon=land_lon if landing else appr_lon,
                appr_hdg=land_hdg if landing else appr_hdg,
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
            gs_est = gs_ests.get_first(within_slice=_slice, within_use='any')
            loc_est = loc_ests.get_first(within_slice=_slice, within_use='any')
            
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
                ils_freq=ils_freq, turnoff=turnoff)
    
    def _lookup_airport_and_runway(self, _slice, precise, appr_lat, appr_lon,
                                   appr_hdg, appr_ils_freq, land_afr_apt=None,
                                   land_afr_rwy=None, hint='approach'):
        '''
        '''
        api = get_api_handler(settings.API_HANDLER)
        kwargs = {}
        airport = None
        runway = None

        # A1. If we have latitude and longitude, look for the nearest airport:
        if appr_lat and appr_lon:
            lat = appr_lat.get_first(within_slice=_slice)
            lon = appr_lon.get_first(within_slice=_slice)
            if lat and lon:
                kwargs.update(latitude=lat.value, longitude=lon.value)
                try:
                    airport = api.get_nearest_airport(**kwargs)
                except NotFoundError:
                    msg = 'No approach airport found near coordinates (%f, %f).'
                    self.warning(msg, lat.value, lon.value)
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

        heading = appr_hdg.get_first(within_slice=_slice).value
        if heading:
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
                runway = api.get_nearest_runway(airport_id, heading, **kwargs)
            except NotFoundError:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport_id, heading, kwargs)
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
    
    def derive(self, alt_aal_phases=P('Altitude AAL For Flight Phases'),
               lands=S('Landing'),
               go_arounds=S('Go Around And Climbout'),
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
        slices = approach_slices(alt_aal_phases, lands, go_arounds)
        self._approaches(
            slices, alt_aal, fast, land_hdg, land_lat, 
            land_lon, appr_hdg, appr_lat, appr_lon, loc_ests, gs_ests, 
            appr_ils_freq, land_afr_apt, land_afr_rwy, precision, turnoffs)
        