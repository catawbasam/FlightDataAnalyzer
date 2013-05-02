import logging
import numpy as np

from collections import OrderedDict, namedtuple
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import izip
from math import asin, atan2, ceil, cos, degrees, floor, radians, sin, sqrt
from scipy import interpolate as scipy_interpolate, optimize

from hdfaccess.parameter import MappedArray

from settings import (CURRENT_YEAR,
                      KTS_TO_MPS,
                      METRES_TO_FEET,
                      REPAIR_DURATION,
                      TRUCK_OR_TRAILER_INTERVAL,
                      TRUCK_OR_TRAILER_PERIOD)

# There is no numpy masked array function for radians, so we just multiply thus:
deg2rad = radians(1.0)

logger = logging.getLogger(name=__name__)

Value = namedtuple('Value', 'index value')


class InvalidDatetime(ValueError):
    pass


def actuator_mismatch(ap, ap_l, ap_r, act_l, act_r, surf, scaling, frequency):
    '''
    Computes the mismatch between a control surface and the driving actuator
    during autopilot engaged phases of flight.
    
    :param ap: autopilot engaged status, 1=engaged, 0=not engaged
    :type ap: numpy masked array
    :param ap_l: autopilot left channel engaged, 1=engaged, 0=not engaged
    :type ap_l: numpy masked array 
    :param ap_r: autopilot right channel engaged, 1=engaged, 0=not engaged
    :type ap_r: numpy masked array
    :param act_l: left channel actuator position, degrees actuator
    :type act_l: numpy masked array
    :param act_r: right channel actuator position, degrees actuator
    :type act_r: numpy masked array
    :param surf: control surface position, degrees surface movement
    :type param: numpy masked array
    :param scaling: ratio of surface movement to actuator movement
    :type scaling: float
    :param frequency: Frequency of parameters.
    :type frequency: float
    
    :returns mismatch: degrees of mismatch between recorded actuator and surface positions
    :type mismatch: numpy masked array.
    
    :Note: mismatch is zero for autopilot not engaged, and is computed for
    the engaged channel only.
    '''
    mismatch = np_ma_zeros_like(ap)
    act = np.ma.where(ap_l == 1, act_l, act_r) * scaling
    
    ap_engs = np.ma.clump_unmasked(np.ma.masked_equal(ap, 0))
    for ap_eng in filter_slices_duration(ap_engs, 4, frequency):
        # Allow the actuator two seconds to settle after engagement.
        check = slice(ap_eng.start + (3 * frequency), ap_eng.stop)

        # We compute a transient mismatch to avoid long term scaling errors.
        mismatch[check] = first_order_washout(surf[check] - act[check], 30.0,
                                              1.0)

    # Square to ensure always positive, and take moving average to smooth.
    mismatch = moving_average(mismatch ** 2.0)
    
    '''
    # This plot shows how the fitted straight sections match the recorded data.
    import matplotlib.pyplot as plt
    plt.plot(surf)
    plt.plot(act)
    plt.plot(mismatch)
    plt.show()
    '''
    
    return mismatch    


def all_of(names, available):
    '''
    Returns True if all of the names are within the available list.
    i.e. names is a subset of available
    '''
    return all(name in available for name in names)


def any_of(names, available):
    '''
    Returns True if any of the names are within the available list.

    NB: Was called "one_of" but that implies ONLY one name is available.
    '''
    return any(name in available for name in names)


def air_track(lat_start, lon_start, lat_end, lon_end, spd, hdg, frequency):
    """
    Computation of the air track for cases where recorded latitude and longitude
    are not available but the origin and destination airport locations are known.

    Note that as the data will be "stretched" to match the origin and
    destination coordinates, either groundspeed or airspeed may be used, as
    the stretching function effectively determines the average wind.

    :param lat_start: Fixed latitude point at the origin.
    :type lat_start: float, latitude degrees.
    :param lon_start: Fixed longitude point at the origin.
    :type lon_start: float, longitude degrees.
    :param lat_end: Fixed latitude point at the destination.
    :type lat_end: float, latitude degrees.
    :param lon_end: Fixed longitude point at the destination.
    :type lon_end: float, longitude degrees.
    :param spd: Speed (air or ground) in knots
    :type gspd: Numpy masked array.
    :param hdg: Heading (ideally true) in degrees.
    :type hdg: Numpy masked array.
    :param frequency: Frequency of the groundspeed and heading data
    :type frequency: Float (units = Hz)

    :returns
    :param lat_track: Latitude of computed ground track
    :type lat_track: Numpy masked array
    :param lon_track: Longitude of computed ground track
    :type lon_track: Numpy masked array.

    :error conditions
    :Fewer than 5 valid data points, returns None, None
    :Invalid mode fails with ValueError
    :Mismatched array lengths fails with ValueError
    """
    # First check that the gspd/hdg arrays are sensible.
    if len(spd) != len(hdg):
        raise ValueError('Ground_track requires equi-length speed and '
                         'heading arrays')

    # It's not worth doing anything if there is too little data
    if np.ma.count(spd) < 5:
        return None, None

    # Prepare arrays for the outputs
    lat = np_ma_masked_zeros_like(spd)
    lon = np_ma_masked_zeros_like(spd)

    repair_mask(spd, repair_duration=None)
    repair_mask(hdg, repair_duration=None)

    valid_slice = np.ma.clump_unmasked(spd)[0]

    hdg_rad = hdg[valid_slice] * deg2rad
    spd_north = spd[valid_slice] * np.ma.cos(hdg_rad)
    spd_east = spd[valid_slice] * np.ma.sin(hdg_rad)

    # Compute displacements in metres north and east of the starting point.
    north = integrate(spd_north, frequency, scale=KTS_TO_MPS)
    east = integrate(spd_east, frequency, scale=KTS_TO_MPS)

    brg, dist = bearing_and_distance(lat_start, lon_start, lat_end, lon_end)
    north_final = dist * np.cos(brg * deg2rad)
    east_final = dist * np.sin(brg * deg2rad)

    # The delta U north and east (dun & due) correct for the integration over
    # (N-1) sample intervals.
    closest_north = closest_unmasked_value(north, -1)
    closest_east = closest_unmasked_value(east, -1)
    
    dun = (north_final - closest_north.value) / ((closest_north.index-1) * KTS_TO_MPS)
    due = (east_final - closest_east.value) / ((closest_east.index-1) * KTS_TO_MPS)

    north = integrate(spd_north+dun, frequency, scale=KTS_TO_MPS)
    east = integrate(spd_east+due, frequency, scale=KTS_TO_MPS)

    bearings = np.ma.array(np.rad2deg(np.arctan2(east, north)))
    distances = np.ma.array(np.ma.sqrt(north**2 + east**2))

    lat[valid_slice],lon[valid_slice] = latitudes_and_longitudes(
        bearings, distances, {'latitude':lat_start, 'longitude':lon_start})

    repair_mask(lat, repair_duration=None, extrapolate=True)
    repair_mask(lon, repair_duration=None, extrapolate=True)
    return lat, lon


def is_power2(number):
    """
    States if a number is a power of two. Forces floats to Int.
    Ref: http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    """
    if number % 1:
        return False
    num = int(number)
    return num > 0 and ((num & (num - 1)) == 0)


def align(slave, master, interpolate=True):
    """
    This function takes two parameters which will have been sampled at
    different rates and with different measurement offsets in time, and
    aligns the slave parameter's samples to match the master parameter. In
    this way the master and aligned slave data may be processed without
    timing errors.

    The values of the returned array will be those of the slave parameter,
    aligned to the master and adjusted by linear interpolation. The initial
    or final values will be masked zeros if they lie outside the timebase of
    the slave parameter (i.e. we do not extrapolate). The offset and hz for
    the returned masked array will be those of the master parameter.

    MappedArray slave parameters (discrete/multi-state) will not be
    interpolated, even if interpolate=True.

    Anything other than discrete or multi-state will result in interpolation
    of the data across each sample period.

    WARNING! Not tested with ASCII arrays.

    :param slave: The parameter to be aligned to the master
    :type slave: Parameter objects
    :param master: The master parameter
    :type master: Parameter objects
    :param interpolate: Whether to interpolate parameters (multistates exempt)
    :type interpolate: Bool

    :raises AssertionError: If the arrays and sample rates do not equate to the same overall data duration.

    :returns: Slave array aligned to master.
    :rtype: np.ma.array
    """
    slave_array = slave.array # Optimised access to attribute.
    if isinstance(slave_array, MappedArray):  # Multi-state array.
        # force disable interpolate!
        slave_array = slave_array.raw
        interpolate = False
        _dtype = int
    elif isinstance(slave_array, np.ma.MaskedArray):
        _dtype = float
    else:
        raise ValueError('Cannot align slave array of unknown type: '
            'Slave: %s, Master: %s.', slave.name, master.name)

    if len(slave_array) == 0:
        # No elements to align, avoids exception being raised in the loop below.
        return slave_array
    if slave.frequency == master.frequency and slave.offset == master.offset:
        # No alignment is required, return the slave's array unchanged.
        return slave_array

    # Get the sample rates for the two parameters
    wm = master.frequency
    ws = slave.frequency
    slowest = min(wm, ws)

    # The timing offsets comprise of word location and possible latency.
    # Express the timing disparity in terms of the slave parameter sample interval
    delta = (master.offset - slave.offset) * slave.frequency

    # If the slowest sample rate is less than 1 Hz, we extend the period and
    # so achieve a lowest rate of one per period.
    if slowest < 1:
        wm /= slowest
        ws /= slowest

    # Check the values are in ranges we have tested
    assert is_power2(wm), \
           "master = '%s' @ %sHz; wm=%s" % (master.name, master.hz, wm)
    assert is_power2(ws), \
           "slave = '%s' @ %sHz; ws=%s" % (slave.name, slave.hz, ws)

    # Compute the sample rate ratio:
    r = wm / float(ws)

    # Here we create a masked array to hold the returned values that will have
    # the same sample rate and timing offset as the master
    len_aligned = int(len(slave_array) * r)
    if len_aligned != (len(slave_array) * r):
        raise ValueError("Array length problem in align. Probable cause is flight cutting not at superframe boundary")

    slave_aligned = np.ma.zeros(len(slave_array) * r, dtype=_dtype)

    # Where offsets are equal, the slave_array recorded values remain
    # unchanged and interpolation is performed between these values.
    # - and we do not interpolate mapped arrays!
    if not delta and interpolate:
        slave_aligned.mask = True
        if master.frequency > slave.frequency:
            # populate values and interpolate
            slave_aligned[0::r] = slave_array[0::1]
            # Interpolate and do not extrapolate masked ends or gaps
            # bigger than the duration between slave samples (i.e. where
            # original slave data is masked).
            # If array is fully masked, return array of masked zeros
            dur_between_slave_samples = 1.0 / slave.frequency
            return repair_mask(slave_aligned, frequency=master.frequency,
                               repair_duration=dur_between_slave_samples,
                               zero_if_masked=True)

        else:
            # step through slave taking the required samples
            return slave_array[0::1/r]

    # Each sample in the master parameter may need different combination parameters
    for i in range(int(wm)):
        bracket = (i / r) + delta
        # Interpolate between the hth and (h+1)th samples of the slave array
        h = int(floor(bracket))
        h1 = h + 1

        # Compute the linear interpolation coefficients, b & a
        b = bracket - h

        # Cunningly, if we are interpolating (working with mapped arrays e.g.
        # discrete or multi-state parameters), by reverting to 1,0 or 0,1
        # coefficients we gather the closest value in time to the master
        # parameter.
        if not interpolate:
            b = round(b)

        # Either way, a is the residual part.
        a = 1 - b

        if h < 0:
            if h<-ws:
                raise ValueError('Align called with excessive timing mismatch')
            # slave_array values do not exist in aligned array
            if ws==1:
                slave_aligned[i+wm::wm] = a*slave_array[h+ws:-ws:ws] + b*slave_array[h1+ws::ws]
            else:
                slave_aligned[i+wm::wm] = a*slave_array[h+ws:-ws:ws] + b*slave_array[h1+ws:1-ws:ws]
            # We can't interpolate the inital values as we are outside the
            # range of the slave parameters.
            # Treat ends as "padding"; Value of 0 and Masked.
            slave_aligned[i] = 0
            slave_aligned[i] = np.ma.masked
        elif h1 >= ws:
            slave_aligned[i:-wm:wm] = a*slave_array[h:-ws:ws] + b*slave_array[h1::ws]
            # At the other end, we run out of slave parameter values so need to
            # pad to the end of the array.
            # Treat ends as "padding"; Value of 0 and Masked.
            slave_aligned[i-wm] = 0
            slave_aligned[i-wm] = np.ma.masked
        else:
            # Sheer bliss. We can compute slave_aligned across the whole
            # range of the data without having to take special care at the
            # ends of the array.
            slave_aligned[i::wm] = a*slave_array[h::ws] + b*slave_array[h1::ws]

    return slave_aligned


def align_slices(slave, master, slices):
    '''
    :param slave: The node to align the slices to.
    :type slave: Node
    :param master: The node which the slices are currently aligned to.
    :type master: Node
    :param slices: Slices to align or None values to skip.
    :type slices: [slice or None]
    :returns: Slices aligned to slave.
    :rtype: [slice or None]
    '''
    if slave.frequency == master.frequency and slave.offset == master.offset:
        return slices
    multiplier = slave.frequency / master.frequency
    offset = (master.offset - slave.offset) * slave.frequency
    aligned_slices = []
    for s in slices:
        if s is None:
            aligned_slices.append(s)
            continue
        aligned_slices.append(slice(
            int(ceil((s.start * multiplier) + offset)) if s.start else None,
            int(ceil((s.stop * multiplier) + offset)) if s.stop else None,
            s.step))
    return aligned_slices


def align_slice(slave, master, _slice):
    '''
    :param slave: The node to align the slice to.
    :type slave: Node
    :param master: The node which the slice is currently aligned to.
    :type master: Node
    :param _slice: Slice to align.
    :type _slice: slice or None
    :returns: Slice aligned to slave.
    :rtype: slice or None
    '''
    return align_slices(slave, master, [_slice])[0]


def ambiguous_runway(rwy):
    # There are a number of runway related KPVs that we only create if we
    # know the actual runway we landed on. Where there is ambiguity the
    # runway attribute may be truncated, or the identifier, if present, will
    # end in a "*" character.
    return (rwy is None or rwy.value is None or not 'identifier' in rwy.value or
            rwy.value['identifier'].endswith('*'))


def bearing_and_distance(lat1, lon1, lat2, lon2):
    """
    Simplified version of bearings and distances for a single pair of
    locations. Gives bearing and distance of point 2 from point 1.
    """
    brg, dist = bearings_and_distances(np.ma.array(lat2), np.ma.array(lon2),
                                       {'latitude':lat1, 'longitude':lon1})
    return np.asscalar(brg), np.asscalar(dist)


def bearings_and_distances(latitudes, longitudes, reference):
    """
    Returns the bearings and distances of a track with respect to a fixed point.

    Usage:
    brg[], dist[] = bearings_and_distances(lat[], lon[], {'latitude':lat_ref, 'longitude':lon_ref})

    :param latitudes: The latitudes of the track.
    :type latitudes: Numpy masked array.
    :param longitudes: The latitudes of the track.
    :type longitudes: Numpy masked array.
    :param reference: The location of the second point.
    :type reference: dict with {'latitude': lat, 'longitude': lon} in degrees.

    :returns bearings, distances: Bearings in degrees, Distances in metres.
    :type distances: Two Numpy masked arrays

    Navigation formulae have been derived from the scripts at
    http://www.movable-type.co.uk/scripts/latlong.html
    Copyright 2002-2011 Chris Veness, and altered by Flight Data Services to
    suit the POLARIS project.
    """

    lat_array = latitudes*deg2rad
    lon_array = longitudes*deg2rad
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])

    dlat = lat_ref - lat_array
    dlon = lon_ref - lon_array

    a = np.ma.sin(dlat/2) * np.ma.sin(dlat/2) + \
        np.ma.cos(lat_array) * np.ma.cos(lat_ref) * \
        np.ma.sin(dlon/2) * np.ma.sin(dlon/2)
    dists = 2 * np.ma.arctan2(np.ma.sqrt(a), np.ma.sqrt(1.0 - a))
    dists *= 6371000 # Earth radius in metres


    y = np.ma.sin(dlon) * np.ma.cos(lat_ref)
    x = np.ma.cos(lat_array) * np.ma.sin(lat_ref) \
        - np.ma.sin(lat_array) * np.ma.cos(lat_ref) * np.ma.cos(dlon)
    brgs = np.ma.arctan2(-y,-x)

    joined_mask = np.logical_or(latitudes.mask, longitudes.mask)
    brg_array = np.ma.array(data=np.rad2deg(brgs) % 360,
                            mask=joined_mask)
    dist_array = np.ma.array(data=dists,
                             mask=joined_mask)

    return brg_array, dist_array

"""
Landing stopping distances.

def braking_action(gspd, landing, mu):
    dist = integrate(gspd.array[landing.slice], gspd.hz, scale=KTS_TO_MPS)
    #decelerate = np.power(gspd.array[landing.slice]*KTS_TO_MPS,2.0)\
        #/(2.0*GRAVITY_METRIC*mu)
    mu = np.power(gspd.array[landing.slice]*KTS_TO_MPS,2.0)\
        /(2.0*GRAVITY_METRIC*dist)
    limit_point = np.ma.argmax(mu)
    ##limit_point = np.ma.argmax(dist+decelerate)
    ##braking_distance = dist[limit_point] + decelerate[limit_point]
    return limit_point, mu[limit_point]
"""

def bump(acc, kti):
    """
    This scans an acceleration array for a short period either side of the
    moment of interest. Too wide and we risk monitoring flares and
    post-liftoff motion. Too short and we may miss a local peak.

    :param acc: An acceleration parameter
    :type acc: A Parameter object
    :param kti: A Key Time Instance
    :type kti: A KTI object

    :returns: The peak acceleration within +/- 3 seconds of the KTI
    :type: Acceleration, from the acc.array.
    """
    dt = 3.0 # Half width of range to scan across for peak acceleration.
    from_index = max(ceil(kti.index - dt * acc.hz), 0)
    to_index = min(int(kti.index + dt * acc.hz)+1, len(acc.array))
    bump_accel = acc.array[from_index:to_index]

    # Taking the absoulte value makes no difference for normal acceleration
    # tests, but seeks the peak left or right for lateral tests.
    bump_index = np.ma.argmax(np.ma.abs(bump_accel))

    peak = bump_accel[bump_index]
    return from_index + bump_index, peak


def calculate_timebase(years, months, days, hours, mins, secs):
    """
    Calculates the timestamp most common in the array of timestamps. Returns
    timestamp calculated for start of array by applying the offset of the
    most common timestamp.

    Accepts arrays and numpy arrays at 1Hz.

    WARNING: If at all times, one or more of the parameters are masked, you
    willnot get a valid timestamp and an exception will be raised.

    Note: if uneven arrays are passed in, they are assumed by izip that the
    start is valid and the uneven ends are invalid and skipped over.

    Supports years as a 2 digits - e.g. "11" is "2011"

    :param years, months, days, hours, mins, secs: Appropriate 1Hz time elements
    :type years, months, days, hours, mins, secs: iterable of numeric type
    :returns: best calculated datetime at start of array
    :rtype: datetime
    :raises: InvalidDatetime if no valid timestamps provided
    """
    base_dt = None
    # OrderedDict so if all values are the same, max will consistently take the
    # first val on repeated runs
    clock_variation = OrderedDict()

    if not len(years) == len(months) == len(days) == \
       len(hours) == len(mins) == len(secs):
        raise ValueError("Arrays must be of same length")

    for step, (yr, mth, day, hr, mn, sc) in enumerate(izip(years, months, days, hours, mins, secs)):
        #TODO: Try using numpy datetime functions for speedup?
        #try:
            #date = np.datetime64('%d-%d-%d' % (yr, mth, day), 'D')
        #except np.core._mx_datetime_parser.RangeError  :
            #continue
        # same for time?

        if yr and yr < 100:
            yr = convert_two_digit_to_four_digit_year(yr)

        try:
            dt = datetime(int(yr), int(mth), int(day), int(hr), int(mn), int(sc))
        except (ValueError, TypeError, np.ma.core.MaskError):
            continue
        if not base_dt:
            base_dt = dt # store reference datetime
        # calc diff from base
        diff = dt - base_dt - timedelta(seconds=step)
        ##print "%02d - %s %s" % (step, dt, diff)
        try:
            clock_variation[diff] += 1
        except KeyError:
            # new difference
            clock_variation[diff] = 1
    if base_dt:
        # return most regular difference
        clock_delta = max(clock_variation, key=clock_variation.get)
        return base_dt + clock_delta
    else:
        # No valid datestamps found
        raise InvalidDatetime("No valid datestamps found")


def convert_two_digit_to_four_digit_year(yr):
    """
    Everything below the current year is assume to be in the current
    century, everything above is assumed to be in the previous
    century.
    if current year is 2012

    13 = 1913
    12 = 2012
    11 = 2011
    01 = 2001
    """
    # convert to 4 digit year
    century = int(CURRENT_YEAR[:2]) * 100
    yy = int(CURRENT_YEAR[2:])
    if yr > yy:
        return century - 100 + yr
    else:
        return century + yr


def coreg(y, indep_var=None, force_zero=False):
    """
    Combined correlation and regression line calculation.

    correlate, slope, offset = coreg(y, indep_var=x, force_zero=True)

    :param y: dependent variable
    :type y: numpy float array - NB: MUST be float
    :param indep_var: independent variable
    :type indep_var: numpy float array. Where not supplied, a linear scale is created.
    :param force_zero: switch to force the regression offset to zero
    :type force_zero: logic, default=False

    :returns:
    :param correlate: The modulus of Pearson's correlation coefficient

    Note that we use only the modulus of the correlation coefficient, so that
    we only have to test for positive values when checking the strength of
    correlation. Thereafter the slope is used to identify the sign of the
    correlation.

    :type correlate: float, in range 0 to +1,
    :param slope: The slope (m) in the equation y=mx+c for the regression line
    :type slope: float
    :param offset: The offset (c) in the equation y=mx+c
    :type offset: float

    Example usage:

    corr,m,c = coreg(air_temp.array, indep_var=alt_std.array)

    corr > 0.5 shows weak correlation between temperature and altitude
    corr > 0.8 shows good correlation between temperature and altitude
    m is the lapse rate
    c is the temperature at 0ft
    """
    n = len(y)
    if n < 2:
        raise ValueError('Function coreg called with data of length %s' % n)
    if indep_var is None:
        x = np.ma.arange(n, dtype=float)
    else:
        x = indep_var
        if len(x) != n:
            raise ValueError('Function coreg called with arrays of differing '
                             'length')

    # Need to propagate masks into both arrays equally.
    mask = np.ma.logical_or(x.mask, y.mask)
    x_ = np.ma.array(data=x.data,mask=mask)
    y_ = np.ma.array(data=y.data,mask=mask)

    if x_.ptp() == 0.0 or y_.ptp() == 0.0:
        # raise ValueError, 'Function coreg called with invariant independent variable'
        return None, None, None

    # n_ is the number of useful data pairs for analysis.
    n_ = np.ma.count(x_)
    sx = np.ma.sum(x_)
    sxy = np.ma.sum(x_*y_)
    sy = np.ma.sum(y_)
    sx2 = np.ma.sum(x_*x_)
    sy2 = np.ma.sum(y_*y_)

    # Correlation
    p = abs((n_*sxy - sx*sy)/(sqrt(n_*sx2-sx*sx)*sqrt(n_*sy2-sy*sy)))

    # Regression
    if force_zero:
        m = sxy/sx2
        c = 0.0
    else:
        m = (sxy-sx*sy/n_)/(sx2-sx*sx/n_)
        c = sy/n_ - m*sx/n_

    return p, m, c


def create_phase_inside(array, hz, offset, phase_start, phase_end):
    '''
    This function masks all values of the reference array outside of the phase
    range phase_start to phase_end, leaving the valid phase inside these times.

    :param array: input data
    :type array: masked array
    :param a: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param phase_start: time into the array where we want to start seeking the threshold transit.
    :type phase_start: float
    :param phase_end: time into the array where we want to stop seeking the threshold transit.
    :type phase_end: float
    :returns: input array with samples outside phase_start and phase_end masked.
    '''
    return _create_phase_mask(array,  hz, offset, phase_start, phase_end, 'inside')


def create_phase_outside(array, hz, offset, phase_start, phase_end):
    '''
    This function masks all values of the reference array inside of the phase
    range phase_start to phase_end, leaving the valid phase outside these times.

    :param array: input data
    :type array: masked array
    :param a: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param phase_start: time into the array where we want to start seeking the threshold transit.
    :type phase_start: float
    :param phase_end: time into the array where we want to stop seeking the threshold transit.
    :type phase_end: float
    :returns: input array with samples outside phase_start and phase_end masked.
    '''
    return _create_phase_mask(array, hz, offset, phase_start, phase_end, 'outside')


def _create_phase_mask(array, hz, offset, a, b, which_side):
    # Create Numpy array of same size as array data
    length = len(array)
    m = np.arange(length)

    if a > b:
        a, b = b, a # Swap them over to make sure a is the smaller.

    # Convert times a,b to indices ia, ib and check they are within the array.
    ia = int((a-offset)*hz)
    if ia < (a-offset)*hz:
        ia += 1
    if ia < 0 or ia > length:
        raise ValueError('Phase mask index out of range')

    ib = int((b-offset)*hz) + 1
    if ib < 0 or ib > length:
        raise ValueError('Phase mask index out of range')

    # Populate the arrays to be False where the flight phase is valid.
    # Adjustments ensure phase is intact and not overwritten by True data.
    if which_side == 'inside':
        m[:ia]  = True
        m[ia:ib] = False
        m[ib:]  = True
    else:
        m[:ia]  = False
        m[ia:ib] = True
        m[ib:]  = False

    # Return the masked array containing reference data and the created mask.
    return np.ma.MaskedArray(array, mask = m)


def cycle_counter(array, min_step, max_time, hz, offset=0):
    '''
    Counts the number of consecutive cycles.

    Each cycle must have a period of not more than ``cycle_time`` seconds, and
    have a variation greater than ``min_step``.

    Note: Where two events with the same cycle count arise in the same array,
    the latter is recorded as it is normally the later in the flight that will
    be most hazardous.

    :param array: Array of data to count cycles within.
    :type array: numpy.ma.core.MaskedArray
    :param min_step: Minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param max_time: Maximum time for a complete valid cycle in seconds.
    :type max_time: float
    :param hz: The sample rate of the array.
    :type hz: float
    :param offset: Index offset to start of the provided array.
    :type offset: int
    :returns: A tuple containing the index of the array element at the end of
        the highest number of cycles and the highest number of cycles in the
        array. Note that the value can be a float as we count a half cycle for
        each change over the minimum step.
    :rtype: (int, float)
    '''
    idxs, vals = cycle_finder(array, min_step=min_step)

    if idxs is None:
        return Value(None, None)

    index, half_cycles = None, 0
    max_index, max_half_cycles = None, 0

    # Determine the half cycle times and look for the most cycling:
    half_cycle_times = np.ediff1d(idxs) / hz
    for n, half_cycle_time in enumerate(half_cycle_times):
        # If we are within the max time, keep track of the half cycle:
        if half_cycle_time < max_time:
            half_cycles += 1
            index = idxs[n + 1]
        # Otherwise check if this is the most cycling and reset:
        elif 0 < half_cycles >= max_half_cycles:
            max_index, max_half_cycles = index, half_cycles
            half_cycles = 0
    else:
        # Finally check whether the last loop had most cycling:
        if 0 < half_cycles >= max_half_cycles:
            max_index, max_half_cycles = index, half_cycles

    # Ignore single direction movements (we only want full cycles):
    if max_half_cycles < 2:
        return Value(None, None)

    return Value(offset + max_index, max_half_cycles / 2.0)


def cycle_select(array, min_step, max_time, hz, offset=0):
    '''
    Selects the value difference in the array when cycling.

    Each cycle must have a period of not more than ``cycle_time`` seconds, and
    have a variation greater than ``min_step``.  The selected value is the
    largest peak-to-peak value of the cycles.

    Note: Where two events with the same cycle difference arise in the same
    array, the latter is recorded as it is normally the later in the flight
    that will be most hazardous.

    :param array: Array of data to count cycles within.
    :type array: numpy.ma.core.MaskedArray
    :param min_step: Minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param cycle_time: Maximum time for a complete valid cycle in seconds.
    :type cycle_time: float
    :param hz: The sample rate of the array.
    :type hz: float
    :param offset: Index offset to start of the provided array.
    :type offset: int
    :returns: A tuple containing the index of the array element at the end of
        the highest difference and the highest difference between a peak and a
        trough in the array while cycling.
    :rtype: (int, float)
    '''
    idxs, vals = cycle_finder(array, min_step=min_step)

    if idxs is None:
        return Value(None, None)

    max_index, max_value = None, 0

    # Determine the half cycle times and ptp values for the half cycles:
    half_cycle_times = np.ediff1d(idxs) / hz
    half_cycle_diffs = np.ediff1d(vals)
    half_cycle_pairs = zip(half_cycle_times, half_cycle_diffs)
    for n, (half_cycle_time, value) in enumerate(half_cycle_pairs):
        # If we are within the max time and have max difference, keep it:
        if half_cycle_time < max_time and abs(value) >= max_value:
            max_index, max_value = idxs[n + 1], abs(value)

    if max_index is None:
        return Value(None, None)

    return Value(offset + max_index, max_value)


def cycle_finder(array, min_step=0.0, include_ends=True):
    '''
    Simple implementation of a peak detection algorithm with small cycle
    remover.

    :param array: time series data
    :type array: Numpy masked array
    :param min_step: Optional minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param include_ends: Decides whether the first and last points of the array are to be included as possible turning points
    :type include_ends: logical

    :returns: A tuple containing the list of peak indexes, and the list of peak values.
    '''

    if len(array) == 0:
        # Nothing to do, so return None.
        return None, None

    # Find the peaks and troughs by difference products which change sign.
    x = np.ma.ediff1d(array, to_begin=0.0)
    # Stripping out only the nonzero values ensures we don't get confused with
    # invariant data.
    y = np.ma.nonzero(x)[0]
    z = x[y] # np.ma.nonzero returns a tuple of indices
    peak = -z[:-1] * z[1:] # Here we compute the change in direction.
    # And these are the indeces where the direction changed.
    idxs = y[np.nonzero(np.ma.maximum(peak, 0.0))]
    vals = array.data[idxs] # So these are the local peak and trough values.

    # Optional inclusion of end points.
    if include_ends and np.ma.count(array):
        # We can only extend over the range of valid data, so find the first
        # and last valid samples.
        first, last = np.ma.flatnotmasked_edges(array)
        idxs = np.insert(idxs, 0, first)
        vals = np.insert(vals, 0, array.data[first])
        # If the end two are in line, scrub the middle one.
        try:
            if (vals[2] - vals[1]) * (vals[1] - vals[0]) >= 0.0:
                idxs = np.delete(idxs, 1)
                vals = np.delete(vals, 1)
        except:
            # If there are few vals in the array, there's nothing to tidy up.
            pass
        idxs = np.append(idxs, last)
        vals = np.append(vals, array.data[last])
        try:
            if (vals[-3] - vals[-2]) * (vals[-2] - vals[-1]) >= 0.0:
                idxs = np.delete(idxs, -2)
                vals = np.delete(vals, -2)
        except:
            pass # as before.

    # This section progressively removes reversals smaller than the step size of
    # interest, hence the arrays shrink until just the desired answer is left.
    dvals = np.ediff1d(vals)
    while len(dvals) > 0 and np.min(abs(dvals)) < min_step:
        sort_idx = np.argmin(abs(dvals))
        last = len(dvals)
        if sort_idx == 0:
            idxs = np.delete(idxs, 0)
            vals = np.delete(vals, 0)
            dvals = np.delete(dvals, 0)
        elif sort_idx == last-1:
            idxs = np.delete(idxs, last)
            vals = np.delete(vals, last)
            dvals = np.delete(dvals, last-1) # One fewer dval than val.
        else:
            idxs = np.delete(idxs, slice(sort_idx, sort_idx + 2))
            vals = np.delete(vals, slice(sort_idx, sort_idx + 2))
            dvals[sort_idx - 1] += dvals[sort_idx] + dvals[sort_idx + 1]
            dvals = np.delete(dvals, slice(sort_idx, sort_idx + 2))
    if len(dvals) == 0:
        # All the changes have disappeared, so return the
        # single array peak index and value.
        return idxs, vals
    else:
        return idxs, vals


def datetime_of_index(start_datetime, index, frequency=1):
    '''
    Returns the datetime of an index within the flight at a particular
    frequency.

    :param start_datetime: Start datetime of the flight available as the 'Start Datetime' attribute.
    :type start_datetime: datetime
    :param index: Index within the flight.
    :type index: int
    :param frequency: Frequency of the index.
    :type frequency: int or float
    :returns: Datetime at index.
    :rtype: datetime
    '''
    index_in_seconds = index / frequency
    offset = timedelta(seconds=index_in_seconds)
    return start_datetime + offset


def delay(array, period, hz=1.0):
    '''
    This function introduces a time delay. Used in validation testing where
    correlation is improved by allowing for the delayed response of one
    parameter when compared to another.

    :param array: Masked array of floats
    :type array: Numpy masked array
    :param period: Time delay(sec)
    :type period: int/float
    :param hz: Frequency of the data_array
    :type hz: float

    :returns: array with data shifted back in time, and initial values masked.
    '''
    n = int(period * hz)
    result = np_ma_masked_zeros_like(array)
    if len(result[n:])==len(array[:-n]):
        result[n:] = array[:-n]
        return result
    else:
        if n==0:
            return array
        else:
            return result


def clip(array, period, hz=1.0, remove='peaks_and_troughs'):
    '''
    This function clips the data array such that the
    values are present (or exceeded) in the original data for the period
    defined. After processing with this function, the resulting array can be
    used to detect maxima or minima (in exactly the same way as a non-clipped
    parameter), however the values will have been met or exceeded in the
    original data for the given duration.

    :param array: Masked array of floats
    :type array: Numpy masked array
    :param period: Time for the output values to be sustained(sec)
    :type period: int/float
    :param hz: Frequency of the data_array
    :type hz: float
    :param remove: form of clipping required.
    :type remove: string default is 'peaks_and_troughs', 'peaks' or 'troughs' alternatives.
    '''
    if remove not in ['peaks_and_troughs', 'peaks', 'troughs']:
        raise ValueError('Clip called with unrecognised remove argument')
        
    if hz <= 0.01:
        raise ValueError('Clip called with sample rate outside permitted range')

    half_width = int(period/hz)/2
    # Trap low values. This can occur, for an example, where a parameter has
    # a lower sample rate than expected.
    if half_width < 1:
        logger.warning('Clip called with period too short to have an effect')
        return array
    
    if np.ma.count(array) == 0:
        logger.warning('Clip called with entirely masked data')
        return array
        
    # OK - normal operation here. We repair the mask to avoid propogating
    # invalid samples unreasonably.
    source = np.ma.array(repair_mask(array, frequency=hz, repair_duration=period-(1/hz)))

    if source is None or np.ma.count(source)==0:
        return np_ma_masked_zeros_like(source)
    
    # We are going to compute maximum and minimum values with the required
    # duration, so allocate working spaces...
    local_max = np_ma_zeros_like(source)
    local_min = np_ma_zeros_like(source)
    end = len(source)-half_width
    
    #...and work out these graphs.
    for point in range(half_width,end):
        local_max[point]=np.ma.max(source[point-half_width:point+half_width+1])
        local_min[point]=np.ma.min(source[point-half_width:point+half_width+1])
    
    # For the maxima, find them using the cycle finder and remove the higher
    # maxima (we are interested in using the lower cycle peaks to replace
    # trough values).
    max_index_cycles, max_value_cycles = cycle_finder(local_max, include_ends=False)
    if len(max_value_cycles)<2:
        max_indexes = max_index_cycles
        max_values = max_value_cycles
    else:
        if max_value_cycles[1]>max_value_cycles[0]:
            # Rising initally
            max_indexes = [i for i in max_index_cycles[0::2]]
            max_values = [v for v in max_value_cycles[0::2]]
        else:
            # Falling initally
            max_indexes = [m for m in max_index_cycles[1::2]]    
            max_values = [v for v in max_value_cycles[1::2]]
    
    # Same for minima, which will be used to substitute for peaks.
    min_index_cycles, min_value_cycles = cycle_finder(local_min, include_ends=False)
    if len(min_value_cycles)<2:
        min_indexes = min_index_cycles
        min_values = min_value_cycles
    else:
        if min_value_cycles[1]>min_value_cycles[0]:
            # Rising initally
            min_indexes = [i for i in min_index_cycles[1::2]]
            min_values = [v for v in min_value_cycles[1::2]]
        else:
            # Falling initally
            min_indexes = [i for i in min_index_cycles[0::2]]
            min_values = [v for v in min_value_cycles[0::2]]
        
    
    # Now build the final result.
    result = source
    # There is a fairly crude technique to find where maxima and minima overlap...
    overlap_finder = np_ma_zeros_like(source)
    
    if remove in ['peaks_and_troughs', 'troughs']:
        for i, index in enumerate(max_indexes):
            for j in range(index-half_width, index+half_width+1):
                # Overwrite the local values with the clipped maximum value
                result[j]=max_values[i]
                # Record which indexes were overwritten.
                overlap_finder[j]+=1

    if remove in ['peaks_and_troughs', 'peaks']:
        for i, index in enumerate(min_indexes):
            for j in range(index-half_width, index+half_width+1):
                # Overwrite the local values with the clipped minimum value
                result[j]=min_values[i]
                # Record which indexes were overwritten.
                overlap_finder[j]+=1

    # This is not an ideal solution of how to deal with minima and maxima
    # that sit close to each other. This may need improving at a later date.
    overlaps = np.ma.clump_masked(np.ma.masked_greater(overlap_finder,1))
    for overlap in overlaps:
        for p in range(overlap.start, overlap.stop):
            result[p]=np.ma.average(source[p-half_width:p+half_width+1])

    # Mask the ends as we cannot have long periods at the end of the data.
    result[:half_width+1] = np.ma.masked
    result[-half_width-1:] = np.ma.masked

    return result


def closest_unmasked_value(array, index, _slice=slice(None)):
    '''
    :param array: Array to find the closest unmasked value within.
    :type array: np.ma.array
    :param index: Find the closest unmasked value to this index.
    :type index: int
    :param _slice: Find closest unmasked value within this slice.
    :type _slice: slice
    :returns: The closest index and value of an unmasked value.
    :rtype: Value
    '''
    array = array[_slice]
    if index < 0:
        index = abs(len(array) + index)
    if not np.ma.count(array):
        return None
    indices = np.ma.arange(len(array))
    indices.mask = array.mask
    index = np.ma.abs(indices - index).argmin()
    value = array[index]
    index = index + (_slice.start or 0)
    return Value(index=index, value=value)


def clump_multistate(array, state, _slices, condition=True):
    '''
    This tests a multistate array and returns a classic POLARIS list of slices.

    :param array: data to scan
    :type array: multistate numpy masked array
    :param state: state to be tested
    :type state: string
    :param _slices: slice or list of slices over which to scan the array.
    :type _slices: slice list
    :param condition: selection of true or false (i.e. inverse) test to apply.
    :type condition: boolean

    :returns: list of slices.
    '''
    def add_clump(clumps, _slice, start, stop):
        #Note that the resulting clumps are expanded by half an index so that
        #where significant the errors in timing are minimized. A clamp to avoid
        #-0.5 values is included, but as we don't know the length of the calling
        #array here, a limit on the maximum case is impractical.
        if _slice.start == 0 and start == 0:
            begin = 0
        else:
            begin = start-0.5
        new_slice = slice(begin, stop+0.5)
        clumps.append(shift_slice(new_slice,_slice.start))
        return

    if not state in array.state:
        return None

    try:
        iter_this = iter(_slices)
    except TypeError:
        iter_this = [_slices]

    clumps = []

    for _slice in iter_this:

        if condition==True:
            array_tuple = np.ma.nonzero(array[_slice]==state)
        else:
            array_tuple = np.ma.nonzero(np.ma.logical_not(array[_slice]==state))

        start = None
        stop = None

        for x in array_tuple[0]:
            if start==None:
                start = x
                stop = x + 1
            elif x==stop:
                stop+=1
            else:
                add_clump(clumps, _slice, start, stop)
                start=x
                stop=x+1
        if stop:
            add_clump(clumps, _slice, start, stop)

    return clumps


def filter_vor_ils_frequencies(array, navaid):
    '''
    This function passes valid ils or vor frequency data and masks all other data.

    To quote from Flightgear ~(where the clearest explanation can be found)
    "The VOR uses frequencies in the the Very High Frequency (VHF) range, it
    uses channels between 108.0 MHz and 117.95 MHz. It is spaced with 0.05
    MHz intervals (so 115.00; 115.05; 115.10 etc). The range 108...112 is
    shared with ILS frequencies. To differentiate between them VOR has an
    even number on the 0.1 MHz frequency and the ILS has an uneven number on
    the 0,1 MHz frequency.

    So 108.0; 108.05; 108.20; 108.25; 108.40; 108.45 would be VOR stations.
    and 108.10; 108.15; 108.30; 108.35; 108.50; 108.55 would be ILS stations.

    :param array: Masked array of radio frequencies in MHz
    :type array: Floating point Numpy masked array
    :param navaid: Type of navigation aid
    :type period: string, 'ILS' or 'VOR' only accepted.

    :returns: Numpy masked array. The requested navaid type frequencies will be passed as valid. All other frequencies will be masked.
    '''
    vor_range = np.ma.masked_outside(array, 108.0, 117.95)
    ils_range = np.ma.masked_greater(vor_range, 111.95)

    # This finds the four sequential frequencies, so fours has values:
    #   0 = .Even0, 1 = .Even5, 2 = .Odd0, 3 = .Odd5
    # The round function is essential as using floating point values leads to inexact values.
    fours = np.ma.round(array * 20) % 4

    # Remove frequencies outside the operating range.
    if navaid == 'ILS':
        return np.ma.masked_where(fours < 2.0, ils_range)
    elif navaid == 'VOR':
        return np.ma.masked_where(fours > 1.0, vor_range)
    else:
        raise ValueError('Navaid of unrecognised type %s' % navaid)


def find_app_rwy(app_info, this_loc):
    """
    This function scans through the recorded approaches to find which matches
    the current localizer established phase. This is required because we
    cater for multiple ILS approaches on a single flight.
    """
    for approach in app_info:
        # line up an approach slice
        if slices_overlap(this_loc.slice, approach.slice):
            # we've found a matching approach where the localiser was established
            break
    else:
        logger.warning("No approach found within slice '%s'.",this_loc)
        return None, None

    runway = approach.runway
    if not runway:
        logger.warning("Approach runway information not available.")
        return approach, None

    return approach, runway


def index_of_first_start(bool_array, _slice=slice(0, None), min_dur=0.0,
                         frequency=1):
    '''
    Find the first starting index of a state change.

    Using bool_array allows one to select the filter before hand,
    e.g. index_of_first_start(state.array == 'state', this_slice)

    Similar to "find_edges_on_state_change" but allows a minumum
    duration (in samples)

    Note: applies -0.5 offset to interpolate state transition, so use
    value_at_index() for the returned index to ensure correct values
    are returned from arrays.
    '''
    if _slice.step and _slice.step < 0:
        raise ValueError("Reverse step not supported")
    runs = runs_of_ones(bool_array[_slice])
    if min_dur:
        runs = filter_slices_duration(runs, min_dur, frequency=frequency)
    if runs:
        return runs[0].start + (_slice.start or 0) - 0.5  # interpolate offset
    else:
        return None


def index_of_last_stop(bool_array, _slice=slice(0, None), min_dur=1,
                       frequency=1):
    '''
    Find the first stopping index of a state change.

    Using bool_array allows one to select the filter before hand,
    e.g. index_of_first_stop(state.array != 'state', this_slice)

    Similar to "find_edges_on_state_change" but allows a minumum
    duration (in samples)

    Note: applies +0.5 offset to interpolate state transition, so use
    value_at_index() for the returned index to ensure correct values
    are returned from arrays.
    '''
    if _slice.step and _slice.step < 0:
        raise ValueError("Reverse step not supported")
    runs = runs_of_ones(bool_array[_slice])
    if min_dur:
        runs = filter_slices_duration(runs, min_dur, frequency=frequency)
    if runs:
        return runs[-1].stop + (_slice.start or 0) - 0.5
    else:
        return None


def find_edges(array, _slice, direction='rising_edges'):
    '''
    Edge finding low level routine, called by create_ktis_at_edges (and
    historically create_kpvs_at_edges). Also useful within algorithms
    directly.

    :param array: array of values to scan for edges
    :type array: Numpy masked array
    :param _slice: slice to be examined
    :type _slice: slice
    :param direction: Optional edge direction for sensing. Default 'rising_edges'
    :type direction: string, one of 'rising_edges', 'falling_edges' or 'all_edges'.

    :returns edge_list: Indexes for the appropriate edge transitions.
    :type edge_list: list of floats.

    Note: edge_list values are always integer+0.5 as it is assumed that the
    transition took place (with highest probability) midway between the two
    recorded states.
    '''
    # Find increments. Extrapolate at start to keep array sizes straight.
    deltas = np.ma.ediff1d(array[_slice], to_begin=array[_slice][0])
    deltas[0]=0 # Ignore the first value
    if direction == 'rising_edges':
        edges = np.ma.nonzero(np.ma.maximum(deltas, 0))
    elif direction == 'falling_edges':
        edges = np.ma.nonzero(np.ma.minimum(deltas, 0))
    elif direction == 'all_edges':
        edges = np.ma.nonzero(deltas)
    else:
        raise ValueError('Edge direction not recognised')

    # edges is a tuple catering for multi-dimensional arrays, but we
    # are only interested in 1-D arrays, hence selection of the first
    # element only.
    # The -0.5 shifts the value midway between the pre- and post-change
    # samples.
    edge_list = edges[0] + int(_slice.start or 0) - 0.5
    return list(edge_list)


def find_edges_on_state_change(state, array, change='entering', phase=None):
    '''
    Version of find_edges tailored to suit multi-state parameters.

    :param state: multistate parameter condition e.g. 'Ground'
    :type state: text, from the states for that parameter.
    :param array: the multistate parameter array
    :type array: numpy masked array with state attributes.

    :param change: condition for detecting edge. Default 'entering', 'leaving' and 'entering_and_leaving' alternatives
    :type change: text
    :param phase: flight phase or list of slices within which edges will be detected.
    :type phase: list of slices, default=None

    :returns: list of indexes

    :raises: ValueError if change not recognised
    :raises: KeyError if state not recognised
    '''
    def state_changes(state, array, change, _slice=slice(0, -1)):

        length = len(array[_slice])
        # The offset allows for phase slices and puts the transition midway
        # between the two conditions as this is the most probable time that
        # the change took place.
        offset = _slice.start - 0.5
        state_periods = np.ma.clump_unmasked(
            np.ma.masked_not_equal(array[_slice], array.state[state]))
        edge_list = []
        for period in state_periods:
            if change == 'entering':
                if period.start > 0:
                    edge_list.append(period.start + offset)

            elif change == 'leaving':
                if period.stop < length:
                    edge_list.append(period.stop + offset)

            elif change == 'entering_and_leaving':
                if period.start > 0:
                    edge_list.append(period.start + offset)
                if period.stop < length:
                    edge_list.append(period.stop + offset)
            else:
                raise  ValueError("Change '%s'in find_edges_on_state_change not recognised" % change)

        return edge_list

    if phase is None:
        return state_changes(state, array, change)

    edge_list = []
    for period in phase:
        if hasattr(period, 'slice'):
            _slice = period.slice
        else:
            _slice = period
        edges = state_changes(state, array, change, _slice)
        edge_list.extend(edges)
    return edge_list


def first_valid_sample(array, start_index=0):
    '''
    Returns the first valid sample of data from a point in an array.

    :param array: array of values to scan
    :type array: Numpy masked array
    :param start_index: optional initial point for the scan. Must be positive.
    :type start_index: integer

    :returns index: index for the first valid sample at or after start_index.
    :type index: Integer or None
    :returns value: the value of first valid sample.
    :type index: Float or None
    '''
    # Trap to ensure we don't stray into the far end of the array and that the
    # sliced array is not empty.
    if not 0 <= start_index < len(array):
        return Value(None, None)

    clumps = np.ma.clump_unmasked(array[start_index:])
    if clumps:
        index = clumps[0].start + start_index
        return Value(index, array[index])
    else:
        return Value(None, None)


def last_valid_sample(array, end_index=None):
    '''
    Returns the last valid sample of data before a point in an array.

    :param array: array of values to scan
    :type array: Numpy masked array
    :param end_index: optional initial point for the scan. May be negative.
    :type end_index: integer

    :returns index: index for the last valid sample at or before end_index.
    :type index: Integer or None
    :returns value: the value of last valid sample.
    :type index: Float or None
    '''
    if end_index is None:
        end_index = len(array)
    elif end_index > len(array):
        return Value(None, None)

    clumps = np.ma.clump_unmasked(array[:end_index+1])
    if clumps:
        index = clumps[-1].stop - 1
        return Value(index, array[index])
    else:
        return Value(None, None)


def first_order_lag(param, time_constant, hz, gain=1.0, initial_value=None):
    '''
    Computes the transfer function
            x.G
    y = -----------
         (1 + T.s)
    where:
    x is the input function
    G is the gain
    T is the timeconstant
    s is the Laplace operator
    y is the output

    Basic example:
    first_order_lag(param, time_constant=5) is equivalent to
    array[index] = array[index-1] * 0.8 + array[index] * 0.2.

    :param param: input data (x)
    :type param: masked array
    :param time_constant: time_constant for the lag function (T)(sec)
    :type time_constant: float
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param gain: gain of the transfer function (non-dimensional)
    :type gain: float
    :param initial_value: initial value of the transfer function at t=0
    :type initial_value: float
    :returns: masked array of values with first order lag applied
    '''
    #input_data = np.copy(array.data)

    # Scale the time constant to allow for different data sample rates.
    tc = time_constant * hz

    # Trap the condition for stability
    if tc < 0.5:
        raise ValueError('Lag timeconstant too small')

    x_term = []
    x_term.append (gain / (1.0 + 2.0*tc)) #b[0]
    x_term.append (gain / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)

    y_term = []
    y_term.append (1.0) #a[0]
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)

    return masked_first_order_filter(y_term, x_term, param, initial_value)


def masked_first_order_filter(y_term, x_term, param, initial_value):
    """
    This provides access to the scipy filter function processed across the
    unmasked data blocks, with masked data retained as masked zero values.
    This is a better option than masking all subsequent values which would be
    the mathematically correct thing to do with infinite response filters.

    :param y_term: Filter denominator terms.
    :type param: list
    :param x_term: Filter numerator terms.
    :type x_term: list
    :param param: input data array
    :type param: masked array
    :param initial_value: Value to be used at the start of the data
    :type initial_value: float (or may be None)
    """
    # import locally to speed up imports of library.py
    from scipy.signal import lfilter, lfilter_zi
    z_initial = lfilter_zi(x_term, y_term) # Prepare for non-zero initial state
    # The initial value may be set as a command line argument, mainly for testing
    # otherwise we set it to the first data value.

    result = np.ma.zeros(len(param))  # There is no zeros_like method.
    good_parts = np.ma.clump_unmasked(param)
    for good_part in good_parts:

        if initial_value is None:
            initial_value = param[good_part.start]
        # Tested version here...
        answer, z_final = lfilter(x_term, y_term, param[good_part], zi=z_initial*initial_value)
        result[good_part] = np.ma.array(answer)

    # The mask should last indefinitely following any single corrupt data point
    # but this is impractical for our use, so we just copy forward the original
    # mask.
    bad_parts = np.ma.clump_masked(param)
    for bad_part in bad_parts:
        # The mask should last indefinitely following any single corrupt data point
        # but this is impractical for our use, so we just copy forward the original
        # mask.
        result[bad_part] = np.ma.masked

    return result


def first_order_washout(param, time_constant, hz, gain=1.0, initial_value=None):
    '''
    Computes the transfer function
         x.G.(T.s)
    y = -----------
         (1 + T.s)
    where:
    x is the input function
    G is the gain
    T is the timeconstant
    s is the Laplace operator
    y is the output

    :param param: input data (x)
    :type param: masked array
    :param time_constant: time_constant for the lag function (T)(sec)
    :type time_constant: float
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param gain: gain of the transfer function (non-dimensional)
    :type gain: float
    :param initial_value: initial value of the transfer function at t=0
    :type initial_value: float
    :returns: masked array of values with first order lag applied
    '''
    #input_data = np.copy(param.data)

    # Scale the time constant to allow for different data sample rates.
    tc = time_constant * hz

    # Trap the condition for stability
    if tc < 0.5:
        raise ValueError('Lag timeconstant too small')

    x_term = []
    x_term.append (gain*2.0*tc  / (1.0 + 2.0*tc)) #b[0]
    x_term.append (-gain*2.0*tc / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)

    y_term = []
    y_term.append (1.0) #a[0]
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)

    return masked_first_order_filter(y_term, x_term, param, initial_value)


def _dist(lat1_d, lon1_d, lat2_d, lon2_d):
    """
    Haversine formula for calculating distances between coordinates.

    :param lat1_d: latitude of first point
    :type lat1_d: float, units = degrees latitude
    :param lon1_d: longitude of first point
    :type lon1_d: float, units = degrees longitude
    :param lat2_d: latitude of second point
    :type lat2_d: float, units = degrees latitude
    :param lon2_d: longitude of second point
    :type lon2_d: float, units = degrees longitude

    :return _dist: distance between the two points
    :type _dist: float (units=metres)

    """
    if (lat1_d == 0.0 and lon1_d == 0.0) or (lat2_d == 0.0 and lon2_d == 0.0):
        # Being asked for a distance from nowhere point on the Atlantic.
        # Decline to get sucked into this trap !
        return None

    lat1 = radians(lat1_d)
    lon1 = radians(lon1_d)
    lat2 = radians(lat2_d)
    lon2 = radians(lon2_d)

    dlat = lat2-lat1
    dlon = lon2-lon1

    a = sin(dlat/2) * sin(dlat/2) + \
        sin(dlon/2) * sin(dlon/2) * cos(lat1) * cos(lat2)
    return 2 * atan2(sqrt(a), sqrt(1-a)) * 6371000


def runway_distance_from_end(runway, *args, **kwds):
    """
    Distance from the end of the runway to any point. The point is first
    snapped onto the runway centreline and then the distance from the runway
    end is taken. This is a convenient startingpoint for measuring runway
    landing distances.

    Note: If high accuracy is required, compute the latitude and longitude
    using the value_at_index function rather than just indexing into the
    latitude and longitude array. Alternatively use KPVs 'Latitude Smoothed At
    Touchdown' and 'Longitude Smoothed At Touchdown' which are the most
    accurate locations we have available for touchdown.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    *args if supplied are the latitude and longitude of a point.
    :param lat: Latitude of the point of interest
    :type lat: float
    :param lon: Longitude of the point of interest
    :type lon: float

    **kwds if supplied are a point in the runway dictionary
    :param point: dictionary name of the point of reference, e.g. 'glideslope'
    :type point: String

    :return distance: Distance from runway end to the point of interest, along runway centreline.
    :type distance: float (units=metres)
    """
    if args:
        new_lat, new_lon = runway_snap(runway, args[0], args[1])
    else:
        try:
            # if kwds['point'] in ['localizer', 'glideslope', 'start']:
            new_lat, new_lon = runway_snap(runway, runway[kwds['point']]['latitude'], runway[kwds['point']]['longitude'])
        except (KeyError, ValueError):
            logger.warning ('Runway_distance_from_end: Unrecognised or missing'\
                            ' keyword %s for runway id %s',
                            kwds['point'], runway['id'])
            return None

    if new_lat and new_lon:
        return _dist(new_lat, new_lon,
                     runway['end']['latitude'], runway['end']['longitude'])
    else:
        return None

def runway_deviation(array, runway={}, heading=None):
    '''
    Computes an array of heading deviations from the selected runway
    centreline calculated from latitude/longitude coordinates. For use with
    True Heading.

    If you use heading, it allows one to use magnetic heading
    comparisons.

    NOTE: Uses heading supplied in preference to coordinates.

    :param array: array or Value of TRUE heading values
    :type array: Numpy masked array (usually already sliced to relate to the landing in question).
    :param runway: runway details.
    :type runway: dict (runway.value if this is taken from an attribute).
    :param heading: heading to use, in preference to runway.
    :type heading: Int/Float

    :returns dev: array of heading deviations
    :type dev: Numpy masked array.
    '''
    if heading is not None:
        rwy_hdg = heading
    else:
        rwy_hdg = runway_heading(runway)
    dev = (array - rwy_hdg) % 360
    return np.ma.where(dev>180.0, dev-360.0, dev)

def runway_distances(runway):
    '''
    Projection of the ILS antenna positions onto the runway
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['localizer']['latitude'] ILS localizer antenna position
    ['localizer']['longitude']
    ['glideslope']['latitude'] ILS glideslope antenna position
    ['glideslope']['longitude']

    :return
    :param start_loc: distance from start of runway to localizer antenna
    :type start_loc: float, units = metres
    :param gs_loc: distance from projected position of glideslope antenna on runway centerline to the localizer antenna
    :type gs_loc: float, units = metres
    :param end_loc: distance from end of runway to localizer antenna
    :type end_loc: float, units = metres
    :param pgs_lat: projected position of glideslope antenna on runway centerline
    :type pgs_lat: float, units = degrees latitude
    :param pgs_lon: projected position of glideslope antenna on runway centerline
    :type pgs_lon: float, units = degrees longitude
    '''
    start_lat = runway['start']['latitude']
    start_lon = runway['start']['longitude']
    end_lat = runway['end']['latitude']
    end_lon = runway['end']['longitude']
    lzr_lat = runway['localizer']['latitude']
    lzr_lon = runway['localizer']['longitude']
    gs_lat = runway['glideslope']['latitude']
    gs_lon = runway['glideslope']['longitude']

    #a = _dist(gs_lat, gs_lon, lzr_lat, lzr_lon)
    #b = _dist(gs_lat, gs_lon, start_lat, start_lon)
    #c = _dist(end_lat, end_lon, lzr_lat, lzr_lon)
    #d = _dist(start_lat, start_lon, lzr_lat, lzr_lon)

    #r = (1.0+(a**2 - b**2)/d**2)/2.0
    #g = r*d

    start_2_loc = _dist(start_lat, start_lon, lzr_lat, lzr_lon)
    # The projected glideslope antenna position is given by this formula
    pgs_lat, pgs_lon = runway_snap(runway, gs_lat, gs_lon)
    gs_2_loc = _dist(pgs_lat, pgs_lon, lzr_lat, lzr_lon)
    end_2_loc = _dist(end_lat, end_lon, lzr_lat, lzr_lon)

    return start_2_loc, gs_2_loc, end_2_loc, pgs_lat, pgs_lon  # Runway distances to start, glideslope and end.


def runway_length(runway):
    '''
    Calculation of only the length for runways with no glideslope details
    and possibly no localizer information. In these cases we assume the
    glideslope is near end of runway and the beam is 700ft wide at the
    threshold.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']

    :return
    :param start_end: distance from start of runway to end
    :type start_loc: float, units = metres.

    :error conditions
    :runway without adequate information fails with ValueError
    '''

    try:
        start_lat = runway['start']['latitude']
        start_lon = runway['start']['longitude']
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']

        return _dist(start_lat, start_lon, end_lat, end_lon)
    except:
        raise ValueError("runway_length unable to compute length of runway id='%s'" %runway['id'])


def runway_heading(runway):
    '''
    Computation of the runway heading from endpoints.
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']

    :return
    :param rwy_hdg: true heading of runway centreline.
    :type rwy_hdg: float, units = degrees, facing from start to end.

    :error conditions
    :runway without adequate information fails with ValueError
    '''
    try:
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']

        brg, dist = bearings_and_distances(np.ma.array(end_lat),
                                           np.ma.array(end_lon),
                                           runway['start'])
        return float(brg.data)
    except:
        if runway:
            raise ValueError("runway_heading unable to resolve heading for runway id='%s'" %runway['id'])
        else:
            raise ValueError("runway_heading unable to resolve heading; no runway")


def runway_snap_dict(runway, lat, lon):
    """
    This function snaps any location onto the closest point on the runway centreline.

    :param runway: Dictionary containing the runway start and end points.
    :type dict
    :param lat: Latitude of the point to snap
    :type lat: float
    :param lon: Longitude of the point to snap
    :type lon: float

    :returns dictionary {['latitude'],['longitude']}
    :type dict.
    """
    lat, lon = runway_snap(runway, lat, lon)
    to_return = {}
    to_return['latitude'] = lat
    to_return['longitude'] = lon
    return to_return


def runway_snap(runway, lat, lon):
    """
    This function snaps any location onto the closest point on the runway centreline.

    :param runway: Dictionary containing the runway start and end points.
    :type dict
    :param lat: Latitude of the point to snap
    :type lat: float
    :param lon: Longitude of the point to snap
    :type lon: float

    :returns new_lat, new_lon: Amended position now on runway centreline.
    :type float, float.
    """
    try:
        start_lat = runway['start']['latitude']
        start_lon = runway['start']['longitude']
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']
    except:
        # Can't do the sums without endpoints.
        return None, None

    a = _dist(lat, lon, end_lat, end_lon)
    b = _dist(lat, lon, start_lat, start_lon)
    d = _dist(start_lat, start_lon, end_lat, end_lon)

    if not a or not b:
        return lat, lon

    if d:
        r = (1.0+(a**2 - b**2)/d**2)/2.0

        # The projected glideslope antenna position is given by this formula
        new_lat = end_lat + r*(start_lat - end_lat)
        new_lon = end_lon + r*(start_lon - end_lon)

        return new_lat, new_lon

    else:
        return None, None


def ground_track(lat_fix, lon_fix, gspd, hdg, frequency, mode):
    """
    Computation of the ground track assuming no slipping.
    :param lat_fix: Fixed latitude point at one end of the data.
    :type lat_fix: float, latitude degrees.
    :param lon_fix: Fixed longitude point at the same time as lat_fix.
    :type lat_fix: float, longitude degrees.
    :param gspd: Groundspeed in knots
    :type gspd: Numpy masked array.
    :param hdg: True heading in degrees.
    :type hdg: Numpy masked array.
    :param frequency: Frequency of the groundspeed and heading data (used for integration scaling).
    :type frequency: Float (units = Hz)
    :param mode: type of calculation to be completed.
    :type mode: String, either 'takeoff' or 'landing' accepted.

    :returns
    :param lat_track: Latitude of computed ground track
    :type lat_track: Numpy masked array
    :param lon_track: Longitude of computed ground track
    :type lon_track: Numpy masked array.

    :error conditions
    :Fewer than 5 valid data points, returns None, None
    :Invalid mode fails with ValueError
    :Mismatched array lengths fails with ValueError
    """

    # We are going to extend the lat/lon_fix point by the length of the gspd/hdg arrays.
    # First check that the gspd/hdg arrays are sensible.
    if len(gspd) != len(hdg):
        raise ValueError('Ground_track requires equi-length groundspeed and '
                         'heading arrays')

    # Dummy masked array to join the invalid data arrays
    result=np.ma.array(np.zeros_like(gspd))
    result.mask = np.ma.logical_or(np.ma.getmaskarray(gspd),
                                   np.ma.getmaskarray(hdg))
    # It's not worth doing anything if there is too little data
    if np.ma.count(result) < 5:
        return None, None

    # Force a copy of the result array, as the repair_mask functions will
    # otherwise overwrite the result mask.
    result = np.ma.copy(result)

    repair_mask(gspd, repair_duration=None)
    repair_mask(hdg, repair_duration=None)

    if mode == 'takeoff':
        direction = 'backwards'
    elif mode == 'landing':
        direction = 'forwards'
    else:
        raise ValueError('Ground_track only recognises takeoff or landing '
                         'modes')

    hdg_rad = hdg * deg2rad
    delta_north = gspd * np.ma.cos(hdg_rad)
    delta_east = gspd * np.ma.sin(hdg_rad)

    north = integrate(delta_north, frequency, scale=KTS_TO_MPS,
                      direction=direction)
    east = integrate(delta_east, frequency, scale=KTS_TO_MPS,
                     direction=direction)

    bearing = np.ma.array(np.rad2deg(np.arctan2(east, north)))
    distance = np.ma.array(np.ma.sqrt(north**2 + east**2))
    distance.mask = result.mask

    lat, lon = latitudes_and_longitudes(bearing, distance,
                                        {'latitude':lat_fix,
                                         'longitude':lon_fix})
    return lat, lon

def gtp_weighting_vector(speed, straight_ends, weights):
    # Compute the speed weighted error
    speed_weighting = np_ma_masked_zeros_like(speed)

    for idx, point in enumerate(straight_ends):
        index = point
        if index == len(speed_weighting):
            index =- 1
        speed_weighting[index] = weights[idx]

    # We ensure the endpoint scaling is unchanged, to avoid a sudden jump in speed.
    speed_weighting[0] = 1.0
    speed_weighting[-1] = 1.0
    speed_weighting = interpolate(speed_weighting)

    return speed_weighting

def gtp_compute_error(weights, *args):
    straights = args[0]
    straight_ends = args[1]
    lat = args[2]
    lon = args[3]
    speed = args[4]
    hdg = args[5]
    frequency = args[6]
    mode = args[7]
    return_arg_set = args[8]
    
    if len(speed)==0:
        if return_arg_set == 'iterate':
            return 0.0
        else:
            return lat, lon, 0.0

    speed_weighting  = gtp_weighting_vector(speed, straight_ends, weights)
    if mode == 'takeoff':
        lat_est, lon_est = ground_track(lat[-1], lon[-1],
                                        speed * speed_weighting,
                                        hdg, frequency, mode)
    else:
        lat_est, lon_est = ground_track(lat[0], lon[0],
                                        speed * speed_weighting,
                                        hdg, frequency, mode)

    # Although we compute the whole track (it's easy) we only compute the
    # error over the track_slice range to ignore the static ends of the
    # data, which often contain spurious data.
    errors = np.arange(len(straights), dtype=float)
    for n, straight in enumerate(straights):
        x_track_errors = ((lon[straight]-lon_est[straight])*np.cos(np.radians(hdg[straight])) -
                          (lat[straight]-lat_est[straight])*np.sin(np.radians(hdg[straight])))
        errors[n] = np.nansum(x_track_errors**2.0) \
            * 1.0E09 # Just to make the numbers easy to read !

    error = np.nansum(errors) # Treats nan as zero, in case masked values present.

    # The optimization process expects a single error term in response, but
    # it is convenient to use this function to return the latitude and
    # longitude as well when asking for the final result, hence two
    # alternative endings to this story.
    if return_arg_set == 'iterate':
        return error
    else:
        return lat_est, lon_est, error


def ground_track_precise(lat, lon, speed, hdg, frequency, mode):
    """
    Computation of the ground track.
    :param lat: Latitude for the duration of the ground track.
    :type lat: Numpy masked array, latitude degrees.
    :param lon: Longitude for the duration of the ground track.
    :type lat: Numpy masked array, longitude degrees.

    :param gspd: Groundspeed for the duration of the ground track.
    :type gspd: Numpy masked array in knots.
    :param hdg: True heading for the duration of the ground track.
    :type hdg: Numpy masked array in degrees.

    :param frequency: Frequency of the array data (required for integration scaling).
    :type frequency: Float (units = Hz)
    :param mode: type of calculation to be completed.
    :type mode: String, either 'takeoff' or 'landing' accepted.

    :returns
    :param lat_track: Latitude of computed ground track
    :type lat_track: Numpy masked array
    :param lon_track: Longitude of computed ground track
    :type lon_track: Numpy masked array.

    :error conditions
    :Fewer than 5 valid data points, returns None, None
    :Invalid mode fails with ValueError
    :Mismatched array lengths fails with ValueError
    """
    # Build arrays to return the computed track.
    lat_return = np_ma_masked_zeros_like(lat)
    lon_return = np_ma_masked_zeros_like(lat)

    # We are going to extend the lat/lon_fix point by the length of the gspd/hdg arrays.
    # First check that the gspd/hdg arrays are sensible.
    if (len(speed) != len(hdg)) or (len(speed) != len(lat)) or (len(speed) != len(lon)):
        raise ValueError('Ground_track_precise requires equi-length arrays')

    # We are going to use the period from the runway to the last point where
    # the speed was over 1kn, to stop the aircraft appearing to wander about
    # on the stand.
    track_edges = np.ma.flatnotmasked_edges(np.ma.masked_less(speed, 1.0))

    # In cases where the data starts with no useful groundspeed data, throw in the towel now.
    if track_edges==None:
        return lat_return, lon_return, 0.0

    # Increment to allow for Python indexing, but don't step over the edge.
    track_edges[1] = min(track_edges[1]+1, len(speed))

    if mode == 'landing':
        track_slice=slice(0, track_edges[1])
    elif mode == 'takeoff':
        track_slice=slice(track_edges[0], len(speed))
    else:
        raise 'unknown mode in ground_track_precise'

    rot = np.ma.abs(rate_of_change_array(hdg[track_slice], frequency, width=8.0))
    straights = np.ma.clump_unmasked(np.ma.masked_greater(rot, 2.0)) # 2deg/sec

    straight_ends = []

    for straight in straights:
        straight_ends.append(straight.start)
        straight_ends.append(straight.stop)

    # unable to optimize track if we have too few curves
    if len(straight_ends) <= 4:
        logger.warning('Ground_track_precise needs at least two curved sections to operate.')
        # Substitute a unity weight vector.
        weights_opt = [np.array([1.0]*len(speed))]

    else:
        # We aren't interested in the first and last
        del straight_ends[0]
        del straight_ends[-1]

        # Initialize the weights for no change.
        weight_length = len(straight_ends)
        weights = np.ma.ones(weight_length)

        # Adjust the speed during each leg to reduce cross track errors.
        speed_bound = (0.5,1.5) # Restict the variation in speeds to 50%.
        boundaries = [speed_bound]*weight_length

        # Then iterate until optimised solution has been found. We use a dull
        # algorithm for reliability, rather than the more exciting forms which
        # can go astray and give less predictable results.
        weights_opt = optimize.fmin_l_bfgs_b(gtp_compute_error, weights,
                                             fprime=None,
                                             args = (straights,
                                                     straight_ends,
                                                     lat[track_slice],
                                                     lon[track_slice],
                                                     speed[track_slice],
                                                     hdg[track_slice],
                                                     frequency,
                                                     mode, 'iterate'),
                                             approx_grad=True, epsilon=1.0E-4,
                                             bounds=boundaries, maxfun=10)
        """
        fmin_l_bfgs_b license: This software is freely available, but we expect that all publications describing work using this software, or all commercial products using it, quote at least one of the references given below. This software is released under the BSD License.
        R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
        C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
        J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        """

    args = (straights, straight_ends, lat[track_slice], lon[track_slice],
            speed[track_slice], hdg[track_slice], frequency, mode, 'final_answer')
    lat_est, lon_est, wt = gtp_compute_error(weights_opt[0], *args)


    """
    # Outputs for debugging and inspecting operation of the optimization algorithm.
    print weights_opt[0]

    for num, weighting in enumerate(weights_opt[0]):
        if weighting == speed_bound[0] or weighting == speed_bound[1]:
            ref = straight_ends[num]
            print 'Mode=',mode, ' Wt[',num, ']=',weighting, 'Index',ref, 'Hdg',hdg[ref], 'Gs',speed[ref]

    # This plot shows how the fitted straight sections match the recorded data.
    import matplotlib.pyplot as plt
    for straight in straights:
        plt.plot(lon_est[straight], lat_est[straight])
    plt.plot(lon[track_slice], lat[track_slice])
    plt.show()
    """

    if mode == 'takeoff':
        lat_return[track_edges[0]:] = lat_est
        lon_return[track_edges[0]:] = lon_est
    else:
        lat_return[:track_edges[1]] = lat_est
        lon_return[:track_edges[1]] = lon_est
    return lat_return, lon_return, wt


def hash_array(array, sections, min_samples):
    '''
    Creates a sha256 hash from the array's tostring() method .
    '''
    checksum = sha256()
    for section in sections:
        if section.stop - section.start < min_samples:
            continue
        checksum.update(array[section].tostring())

    return checksum.hexdigest()


def hysteresis(array, hysteresis):
    """
    Applies hysteresis to an array of data. The function applies half the
    required level of hysteresis forwards and then backwards to provide a
    phase neutral result.

    :param array: Input data for processing
    :type array: Numpy masked array
    :param hysteresis: Level of hysteresis to apply.
    :type hysteresis: Float
    """
    if np.ma.count(array) == 0: # No unmasked elements.
        return array

    quarter_range = hysteresis / 4.0
    # Length is going to be used often, so prepare here:
    length = len(array)
    half_done = np.zeros(length)
    result = np.zeros(length)
    length = length-1 #  To be used for array indexing next

    # get a list of the unmasked data - allow for array.mask = False (not an array)
    if array.mask is np.False_:
        notmasked = np.arange(length+1)
    else:
        notmasked = np.ma.where(array.mask == False)[0]
    # The starting point for the computation is the first notmasked sample.
    old = array[notmasked[0]]
    for index in notmasked:
        new = array[index]

        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        half_done[index] = old

    # Repeat the process in the "backwards" sense to remove phase effects.
    for index in notmasked[::-1]:
        new = half_done[index]
        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        result[index] = old

    # At the end of the process we reinstate the mask, although the data
    # values may have affected the result.
    return np.ma.array(result, mask=array.mask)


def ils_glideslope_align(runway):
    '''
    Projection of the ILS glideslope antenna onto the runway centreline
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['glideslope']['latitude'] ILS glideslope antenna position
    ['glideslope']['longitude']

    :returns dictionary containing:
    ['latitude'] ILS glideslope position aligned to start and end of runway
    ['longitude']

    :error: if there is no glideslope antenna in the database for this runway, returns None
    '''
    try:
        new_lat, new_lon = runway_snap(runway,
                                       runway['glideslope']['latitude'],
                                       runway['glideslope']['longitude'])
        return {'latitude':new_lat, 'longitude':new_lon}
    except KeyError:
        return None


def ils_localizer_align(runway):
    '''
    Projection of the ILS localizer antenna onto the runway centreline
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['localizer']['latitude'] ILS localizer antenna position
    ['localizer']['longitude']

    :returns dictionary containing:
    ['latitude'] ILS localizer position aligned to start and end of runway
    ['longitude']
    '''
    try:
        new_lat, new_lon = runway_snap(runway,
                                   runway['localizer']['latitude'],
                                   runway['localizer']['longitude'])
    except KeyError:
        new_lat, new_lon = runway['end']['latitude'], runway['end']['longitude']
        logger.warning('Localizer not found for this runway, so endpoint substituted')

    return {'latitude':new_lat, 'longitude':new_lon}


def integrate(array, frequency, initial_value=0.0, scale=1.0,
              direction="forwards", contiguous=False):
    """
    Trapezoidal integration

    Usage example:
    feet_to_land = integrate(airspeed[:touchdown], scale=KTS_TO_FPS, direction='reverse')

    :param array: Integrand.
    :type array: Numpy masked array.
    :param frequency: Sample rate of the integrand.
    :type frequency: Float
    :param initial_value: Initial value for the integral
    :type initial_value: Float
    :param scale: Scaling factor, default = 1.0
    :type scale: float
    :param direction: Optional integration sense, default = 'forwards'
    :type direction: String - ['forwards', 'backwards', 'reverse']
    :param contiguous: Option to restrict the output to the single longest contiguous section of data
    :type contiguous: Logical

    Note: Reverse integration does not include a change of sign, so positive
    values have a negative slope following integration using this function.
    Backwards integration DOES include a change of sign, so positive
    values have a positive slope following integration using this function.

    :returns integral: Result of integration by time
    :type integral: Numpy masked array.

    Note: Masked values will be "repaired" before integration. If errors longer
    than the repair limit exist, subsequent values in the array will all be
    masked.
    """
    if np.ma.count(array)==0:
        return np_ma_masked_zeros_like(array)

    result = np.ma.copy(array)

    if contiguous:
        blocks = np.ma.clump_unmasked(array)
        longest_index = None
        longest_slice = 0
        for n, block in enumerate(blocks):
            slice_length = block.stop-block.start
            if slice_length > longest_slice:
                longest_slice = slice_length
                longest_index = n
        integrand = np_ma_masked_zeros_like(array)
        integrand[blocks[longest_index]] = array[blocks[longest_index]]
    else:
        integrand = array

    if direction.lower() == 'forwards':
        d = +1
        s = +1
    elif direction.lower() == 'reverse':
        d = -1
        s = +1
    elif direction.lower() == 'backwards':
        d = -1
        s = -1
    else:
        raise ValueError("Invalid direction '%s'" % direction)

    k = (scale * 0.5)/frequency
    to_int = k * (integrand + np.roll(integrand, d))
    edges = np.ma.flatnotmasked_edges(to_int)
    if direction == 'forwards':
        if edges[0] == 1:
            to_int[0] = initial_value
        else:
            to_int[edges[0]] = initial_value
    else:
        if edges[1] == -1:
            to_int[-1] = initial_value * s
        else:
            to_int[edges[1]] = initial_value * s
            # Note: Sign of initial value will be reversed twice for backwards case.

    result[::d] = np.ma.cumsum(to_int[::d] * s)

    return result


def interpolate(array, extrapolate=True):
    """
    This will replace all masked values in an array with linearly
    interpolated values between unmasked point pairs, and extrapolate first
    and last unmasked values to the ends of the array by default.

    See Derived Parameter Node 'Magnetic Deviation' for the prime example of
    use.

    In the special case where all source data is masked, the algorithm
    returns an unmasked array of zeros.

    :param array: Array of data with masked values to be interpolated over.
    :type array: numpy masked array
    :param extrapolate: Option to extrapolate the first and last masked values
    :type extrapolate: Bool

    :returns interpolated: array of all valid data
    :type interpolated: Numpy masked array, with all masks False.
    """
    # Where do we need to use the raw data?
    blocks = np.ma.clump_masked(array)
    last = len(array)
    if len(blocks)==1:
        if blocks[0].start == 0 and blocks[0].stop == last:
            logger.warn('No unmasked data to interpolate')
            return np_ma_zeros_like(array)

    for block in blocks:
        # Setup local variables
        a = block.start
        b = block.stop

        if a == 0:
            if extrapolate:
                array[:b] = array[b]
            else:
                # leave masked values at start untouched
                continue
        elif b == last:
            if extrapolate:
                array[a:] = array[a-1]
            else:
                # leave masked values at end untouched
                continue
        else:
            join = np.linspace(array[a - 1], array[b], num=b - a + 2)
            array[a:b] = join[1:-1]

    return array


def interleave(param_1, param_2):
    """
    Interleaves two parameters (usually from different sources) into one
    masked array. Maintains the mask of each parameter.

    :param param_1:
    :type param_1: Parameter object
    :param param_2:
    :type param_2: Parameter object

    """
    # Check the conditions for merging are met
    if param_1.frequency != param_2.frequency:
        raise ValueError('Attempt to interleave parameters at differing sample '
                         'rates')

    dt = param_2.offset - param_1.offset
    # Note that dt may suffer from rounding errors,
    # hence rounding the value before comparison.
    if 2 * abs(round(dt, 6)) != 1 / param_1.frequency:
                raise ValueError('Attempt to interleave parameters that are '
                                 'not correctly aligned')

    merged_array = np.ma.zeros((2, len(param_1.array)))
    if dt > 0:
        merged_array = np.ma.column_stack((param_1.array, param_2.array))
    else:
        merged_array = np.ma.column_stack((param_2.array, param_1.array))

    return np.ma.ravel(merged_array)

"""
Superceded by blend routines.

def interleave_uneven_spacing(param_1, param_2):
    '''
    This interleaves samples that are not quote equi-spaced.
       |--------dt---------|
       |   x             y |
       |          m        |
       |   |------a------| |
       |     o         o   |
       |   |b|         |b| |

    Over a period dt two samples x & y will be merged to an equi-spaced new
    parameter "o". x & y are a apart, while samples o are displaced by b from
    their original positions.

    There is a second case where the samples are close together and the
    interpolation takes place not between x > y, but across the y > x interval.
    Hence two sections of code. Also, we don't know at the start whether x is
    parameter 1 or 2, so there are two options for the basic interleaving stage.
    '''
    # Check the conditions for merging are met
    if param_1.frequency != param_2.frequency:
        raise ValueError('Attempt to interleave parameters at differing sample rates')

    mean_offset = (param_2.offset + param_1.offset) / 2.0
    #result_offset = mean_offset - 1.0/(2.0 * param_1.frequency)
    dt = 1.0/param_1.frequency

    merged_array = np.ma.zeros((2, len(param_1.array)))

    if mean_offset - dt > 0:
        # The larger gap is between the two first samples
        merged_array = np.ma.column_stack((param_1.array,param_2.array))
        offset_0 = param_1.offset
        offset_1 = param_2.offset
        a = offset_1 - offset_0
    else:
        # The larger gap is between the second and third samples
        merged_array = np.ma.column_stack((param_2.array,param_1.array))
        offset_0 = param_2.offset
        offset_1 = param_1.offset
        a = dt - (offset_1 - offset_0)
    b = (dt - a)/2.0

    straight_array = np.ma.ravel(merged_array)
    if a < dt:
        straight_array[0] = straight_array[1] # Extrapolate a little at start
        x = straight_array[1::2]
        y = straight_array[2::2]
    else:
        x = straight_array[0::2]
        y = straight_array[1::2]
    # THIS WON'T WORK !!!
    x = (y - x)*(b/a) + x
    y = (y-x) * (1.0 - b) / a + x

    #return straight_array
    return None # to force a test error until this is fixed to prevent extrapolation
"""
"""
def interpolate_params(*params):
    '''
    Q: Should we mask indices which are being interpolated in masked areas of
       the input arrays.
    '''
    param_frequencies = [param.frequency for param in params]
    max_frequency = max(param_frequencies)
    out_frequency = sum(param_frequencies)

    data_arrays = []
    index_arrays = []

    for param in sorted(params, key=attrgetter('frequency')):
        multiplier = out_frequency / param.frequency
        offset = (param.offset * multiplier)
        # Will not create interpolation points for masked indices.
        unmasked_indices = np.where(param.array.mask == False)[0]
        index_array = unmasked_indices.astype(np.float_) * multiplier + offset
        # Take only unmasked values to match size with index_array.
        data_arrays.append(param.array.data[unmasked_indices])
        index_arrays.append(index_array)
    # param assigned within loop has the maximum frequency.

    data_array = np.concatenate(data_arrays)
    index_array = np.concatenate(index_arrays)
    record = np.rec.fromarrays([index_array, data_array],
                               names='indices,values')
    record.sort()
    # Masked values will be NaN.
    interpolator = interp1d(record.indices, record.values, bounds_error=False,
                            fill_value=np.NaN)
    # Ensure first interpolated value is within range.
    out_offset = np.min(record.indices)
    out_indices = np.arange(out_offset, len(param.array) * multiplier,
                            param.frequency / out_frequency)
    interpolated_array = interpolator(out_indices)
    masked_array = np.ma.masked_array(interpolated_array,
                                      mask=np.isnan(interpolated_array))
    return masked_array, out_frequency, out_offset
"""


def index_of_datetime(start_datetime, index_datetime, frequency, offset=0):
    '''
    :param start_datetime: Start datetime of data file.
    :type start_datetime: datetime
    :param index_datetime: Datetime of which to calculate the index.
    :type index_datetime: datetime
    :param frequency: Frequency of index.
    :type frequency: float or int
    :param offset: Optional offset of the parameter in seconds.
    :type offset: float
    :returns: The index of index_datetime relative to start_datetime and frequency.
    :rtype: int or float
    '''
    difference = index_datetime - start_datetime
    return (difference.total_seconds() - offset) * frequency


def is_index_within_slice(index, _slice):
    '''
    :type index: int or float
    :type _slice: slice
    :returns: whether index is within the slice.
    :rtype: bool
    '''
    if _slice.start is None and _slice.stop is None:
        return True
    elif _slice.start is None:
        return index < _slice.stop
    elif _slice.stop is None:
        return index >= _slice.start
    return _slice.start <= index < _slice.stop


def filter_slices_duration(slices, duration, frequency=1):
    '''
    Q: Does this need to be updated to use Sections?
    :param slices: List of slices to filter.
    :type slices: [slice]
    :param duration: Minimum duration of slices in seconds.
    :type duration: int or float
    :param frequency: Frequency of slice start and stop.
    :type frequency: int or float
    :returns: List of slices greater than duration.
    :rtype: [slice]
    '''
    return [s for s in slices if (s.stop - s.start) >= (duration * frequency)]


def find_slices_containing_index(index, slices):
    '''
    :type index: int or float
    :type slices: a list of slices to search through

    :returns: the first slice which contains index or None
    :rtype: [slice]
    '''
    return [s for s in slices if is_index_within_slice(index, s)]


def is_slice_within_slice(inner_slice, outer_slice, within_use='slice'):
    '''
    inner_slice is considered to not be within outer slice if its start or
    stop is None.

    :type inner_slice: slice
    :type outer_slice: slice
    :returns: Whether inner_slice is within the outer_slice.
    :rtype: bool
    '''

    def entire_slice_within_slice():
        if outer_slice.start is None and outer_slice.stop is None:
            return True
        elif inner_slice.start is None and outer_slice.start is not None:
            return False
        elif inner_slice.stop is None and outer_slice.stop is not None:
            return False
        elif inner_slice.start is None and outer_slice.start is None:
            return inner_slice.stop < outer_slice.stop
        elif outer_slice.stop is None and outer_slice.stop is None:
            return inner_slice.start >= outer_slice.start
        else:
            start_within = outer_slice.start <= inner_slice.start <= outer_slice.stop
            stop_within = outer_slice.start <= inner_slice.stop <= outer_slice.stop
            return start_within and stop_within

    if within_use == 'slice':
        return entire_slice_within_slice()
    elif within_use == 'start':
        return is_index_within_slice(inner_slice.start, outer_slice)
    elif within_use == 'stop':
        return is_index_within_slice(inner_slice.stop, outer_slice)
    elif within_use == 'any':
        return slices_overlap(inner_slice, outer_slice)


def slices_overlap(first_slice, second_slice):
    '''
    Logical check for an overlap existing between two slices.
    Requires more than one value overlapping

    :param slice1: First slice
    :type slice1: Python slice
    :param slice2: Second slice
    :type slice2: Python slice

    :returns boolean
    '''
    if first_slice.step is not None and first_slice.step < 1 \
       or second_slice.step is not None and second_slice.step < 1:
        raise ValueError("Negative step not supported")
    return ((first_slice.start < second_slice.stop) or
            (second_slice.stop is None)) and \
           ((second_slice.start < first_slice.stop) or
            (first_slice.stop is None))

def slices_and(first_list, second_list):
    '''
    This is a simple AND function to allow two slice lists to be merged.

    Note: This currently has a trap for reverse slices, although this could be
    extended.

    :param first_list: First list of slices
    :type first_list: List of slices
    :param second_list: Second list of slices
    :type second_list: List of slices

    :returns: List of slices where first and second lists overlap.
    '''
    result_list = []
    for first_slice in first_list:
        for second_slice in second_list:
            if (first_slice.step is not None and first_slice.step < 0) or \
               (second_slice.step is not None and second_slice.step < 0):
                raise ValueError('slices_and will not work with reverse slices')
            if slices_overlap(first_slice, second_slice):
                result_list.append(
                    slice(max(first_slice.start, second_slice.start),
                          min(first_slice.stop, second_slice.stop)))
    return result_list

def slices_and_not(first, second):
    '''
    It is surprisingly common to need one condition but not a second.
    Airborne but not Approach And Landing, for example. This little routine
    makes this simple.

    :param first: First Section - values to be included
    :type first: Section
    :param second: Second Section - values to be excluded
    :type second: Section

    :returns: List of slices in the first but outside the second lists.
    '''
    return slices_and([s.slice for s in first],
                      slices_not([s.slice for s in second],
                                 begin_at=min([s.slice.start for s in first]),
                                 end_at=max([s.slice.stop for s in first])))


def slices_not(slice_list, begin_at=None, end_at=None):
    '''
    Inversion of a list of slices. Currently does not cater for reverse slices.

    :param slice_list: list of slices to be inverted.
    :type slice_list: list of Python slices.
    :param begin_at: optional starting index value, slices before this will be ignored
    :param begin_at: integer
    :param end_at: optional ending index value, slices after this will be ignored
    :param end_at: integer

    :returns: list of slices. If begin or end is specified, the range will extend to these points. Otherwise the scope is within the end slices.
    '''
    if not slice_list:
        return [slice(begin_at, end_at)]

    a = min([s.start for s in slice_list])
    b = min([s.stop for s in slice_list])
    c = max([s.step for s in slice_list])
    if c>1:
        raise ValueError("slices_not does not cater for non-unity steps")

    startpoint = a if b is None else min(a,b)

    if begin_at is not None and begin_at < startpoint:
        startpoint = begin_at
    if startpoint is None:
        startpoint = 0

    c = max([s.start for s in slice_list])
    d = max([s.stop for s in slice_list])
    endpoint = max(c,d)
    if end_at is not None and end_at > endpoint:
        endpoint = end_at

    workspace = np.ma.zeros(endpoint)
    for each_slice in slice_list:
        workspace[each_slice] = 1
    workspace=np.ma.masked_equal(workspace, 1)
    return shift_slices(np.ma.clump_unmasked(workspace[startpoint:endpoint]), startpoint)


def slices_or(*slice_lists, **kwargs):
    '''
    "OR" function for a list of slices.

    :param slice_list: list of slices to be combined.
    :type slice_list: list of Python slices.
    :param begin_at: optional starting index value, slices before this will be ignored
    :type begin_at: integer
    :param end_at: optional ending index value, slices before this will be ignored
    :type end_at: integer

    :returns: list of slices. If begin or end is specified, the range will
    extend to these points. Otherwise the scope is within the end slices.
    
    :error: raises ValueError in the case where None has been passed in. This
    can arise with TAWS Alert derived parameter if a new LFL carries the
    wrong text string for a TAWS signal, so forms a "backstop" error trap.
    '''
    if len(slice_lists) == 0:
        return

    a = None
    b = None
    for slice_list in slice_lists:
        if slice_list==None:
            raise ValueError('slices_or called with slice list of None')
        for each_slice in slice_list:
            if not each_slice:
                break

            a = each_slice.start or 0 if a is None else min(a, each_slice.start)

            if each_slice.stop is None:
                break
            b = each_slice.stop if b is None else max(b, each_slice.stop)

    if kwargs.has_key('begin_at'):
        startpoint = kwargs['begin_at']
    else:
        startpoint = 0

    if kwargs.has_key('end_at'):
        endpoint = kwargs['end_at']
    else:
        endpoint = b

    if startpoint>=0 and endpoint>0:
        workspace = np.ma.zeros(b)
        for slice_list in slice_lists:
            for each_slice in slice_list:
                workspace[each_slice] = 1
        workspace=np.ma.masked_equal(workspace, 1)
        return shift_slices(np.ma.clump_masked(workspace[startpoint:endpoint]), startpoint)


def slices_remove_small_gaps(slice_list, time_limit=10, hz=1):
    '''
    Routine to remove small gaps in a list of slices. Typically when a list
    of flight phases have been computed and we don't want to drop out for
    trivial periods, this will create a single slice across what were two
    slices with a small gap.

    :param slice_list: list of slices to be processed
    :type slice_list: list of Python slices.
    :param time_limit: Tolerance below which slices will be joined.
    :type time_limit: integer (sec)
    :param hz: sample rate for the parameter
    :type hz: float

    :returns: slice list.
    '''
    sample_limit = time_limit * hz
    if slice_list is None or len(slice_list) < 2:
        return slice_list
    new_list = [slice_list[0]]
    for each_slice in slice_list[1:]:
        if each_slice.start - new_list[-1].stop < sample_limit:
            new_list[-1] = slice(new_list[-1].start, each_slice.stop)
        else:
            new_list.append(each_slice)
    return new_list
            

def slices_remove_small_slices(slice_list, time_limit=10, hz=1, count=None):
    '''
    Routine to remove small slices in a list of slices.

    :param slice_list: list of slices to be processed
    :type slice_list: list of Python slices.
    
    :param time_limit: Tolerance below which slice will be rejected.
    :type time_limit: integer (sec)
    :param hz: sample rate for the parameter
    :type hz: float

    :param count: Tolerance based on count, not time
    :type count: integer (default = None)
    
    :returns: slice list.
    '''
    if count:
        sample_limit = count
    else:
        sample_limit = time_limit * hz

    if slice_list is None :
        return slice_list
    new_list = []
    for each_slice in slice_list:
        if each_slice.stop - each_slice.start > sample_limit:
            new_list.append(each_slice)
    return new_list
            

"""
def section_contains_kti(section, kti):
    '''
    Often want to check that a KTI value is inside a given slice.
    '''
    if len(kti)!=1 or len(section)!=2:
        return False
    return section.slice.start <= kti[0].index <= section.slice.stop
"""


def latitudes_and_longitudes(bearings, distances, reference):
    """
    Returns the latitudes and longitudes of a track given true bearing and
    distances with respect to a fixed point.

    Usage:
    lat[], lon[] = latitudes_and_longitudes(brg[], dist[],
                   {'latitude':lat_ref, 'longitude', lon_ref})

    :param bearings: The bearings of the track in degrees.
    :type bearings: Numpy masked array.
    :param distances: The distances of the track in metres.
    :type distances: Numpy masked array.
    :param reference: The location of the reference point in degrees.
    :type reference: dict with {'latitude': lat, 'longitude': lon} in degrees.

    :returns latitude, longitude: Latitudes and Longitudes in degrees.
    :type latitude, longitude: Two Numpy masked arrays

    Navigation formulae have been derived from the scripts at
    http://www.movable-type.co.uk/scripts/latlong.html
    Copyright 2002-2011 Chris Veness, and altered by Flight Data Services to
    suit the POLARIS project.
    """
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])
    brg = bearings * deg2rad
    dist = distances.data / 6371000.0 # Scale to earth radius in metres

    lat = np.arcsin(sin(lat_ref)*np.ma.cos(dist) +
                   cos(lat_ref)*np.ma.sin(dist)*np.ma.cos(brg))
    lon = np.arctan2(np.ma.sin(brg)*np.ma.sin(dist)*np.ma.cos(lat_ref),
                      np.ma.cos(dist)-sin(lat_ref)*np.ma.sin(lat))
    lon += lon_ref

    joined_mask = np.logical_or(bearings.mask, distances.mask)
    lat_array = np.ma.array(data = np.rad2deg(lat),mask = joined_mask)
    lon_array = np.ma.array(data = np.rad2deg(lon),mask = joined_mask)
    return lat_array, lon_array

def localizer_scale(runway):
    """
    Compute the ILS localizer scaling factor from runway or nominal data.
    """
    try:
        # Compute the localizer scale factor (degrees per dot)
        # Half the beam width is 2.5 dots full scale
        scale = (runway['runway']['localizer']['beam_width']/2.0) / 2.5
    except:
        try:
            length = runway_length(runway)
        except:
            length = None

        if length == None:
            length = 8000 / METRES_TO_FEET # Typical length

        # Normal scaling of a localizer gives 700ft width at the threshold,
        # so half of this is 350ft=106.68m. This appears to be full 2-dots
        # scale.
        scale = np.degrees(np.arctan2(106.68, length)) / 2.0
    return scale

def mask_inside_slices(array, slices):
    '''
    Mask slices within array.

    :param array: Masked array to mask.
    :type array: np.ma.masked_array
    :param slices: Slices to mask.
    :type slices: list of slice
    :returns: Array with masks applied.
    :rtype: np.ma.masked_array
    '''
    mask = np.zeros(len(array), dtype=np.bool_) # Create a mask of False.
    for slice_ in slices:
        mask[slice_] = True
    return np.ma.array(array, mask=np.ma.mask_or(mask, array.mask))


def mask_outside_slices(array, slices):
    '''
    Mask areas outside of slices within array.

    :param array: Masked array to mask.
    :type array: np.ma.masked_array
    :param slices: The areas outside these slices will be masked..
    :type slices: list of slice
    :returns: Array with masks applied.
    :rtype: np.ma.masked_array
    '''
    mask = np.ones(len(array), dtype=np.bool_) # Create a mask of True.
    for slice_ in slices:
        mask[slice_] = False
    return np.ma.array(array, mask=np.ma.mask_or(mask, array.mask))


def max_continuous_unmasked(array, _slice=slice(None)):
    """
    Returns the max_slice
    """
    if _slice.step and _slice.step != 1:
        raise ValueError("Step not supported")
    clumps = np.ma.clump_unmasked(array[_slice])
    if not clumps or clumps == [slice(0,0,None)]:
        return None

    _max = None
    for clump in clumps:
        dur = clump.stop - clump.start
        if not _max or _max.stop-_max.start < dur:
            _max = clump
    offset = _slice.start or 0
    return slice(_max.start + offset, _max.stop + offset)


def max_abs_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the value of the maximum absolute value in the array.
    Return value is NOT the absolute value (i.e. may be negative)

    Note, if all values are masked, it will return the value at the first index
    (which will be masked!)

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max absolute value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    index, value = max_value(np.ma.abs(array), _slice)
    # If start or stop edges are given, check these extreme (interpolated) values.
    if start_edge:
        edge_value = abs(value_at_index(array, start_edge) or 0)
        if edge_value and edge_value > value:
            index = start_edge
            value = edge_value
    if stop_edge:
        edge_value = abs(value_at_index(array, stop_edge) or 0)
        if edge_value and edge_value > value:
            index = stop_edge
            value = edge_value
    return Value(index, array[index]) # Recover sign of the value.


def max_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the maximum value in the array and its index relative to the array and
    not the _slice argument.

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    index, value = _value(array, _slice, np.ma.argmax)
    # If start or stop edges are given, check these extreme (interpolated) values.
    if start_edge:
        edge_value = value_at_index(array, start_edge)
        if edge_value and edge_value > value:
            index = start_edge
            value = edge_value
    if stop_edge:
        edge_value = value_at_index(array, stop_edge)
        if edge_value and edge_value > value:
            index = stop_edge
            value = edge_value
    return Value(index, value)


def merge_masks(masks, min_unmasked=1):
    '''
    :type masks: [mask]
    :type min_unmasked: int
    :returns: Array of merged masks.
    :rtype: np.array(dtype=np.bool_)
    '''
    if len(masks) == 1:
        return masks[0]
    # Q: What if min_unmasked is less than one?
    mask_sum = np.sum(np.array(masks), axis=0)
    return mask_sum >= min_unmasked


def min_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the minimum value in the array and its index.

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return min value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    index, value = _value(array, _slice, np.ma.argmin)
    # If start or stop edges are given, check these extreme (interpolated) values.
    if start_edge:
        edge_value = value_at_index(array, start_edge)
        if edge_value and edge_value < value:
            index = start_edge
            value = edge_value
    if stop_edge:
        edge_value = value_at_index(array, stop_edge)
        if edge_value and edge_value < value:
            index = stop_edge
            value = edge_value
    return Value(index, value)


def minimum_unmasked(array1, array2):
    """
    Get the minimum value between two arrays. Differs from the Numpy minimum
    in that is there are masked values in one array, these are ignored and
    data from the other array is used.

    :param array_1: masked array
    :type array_1: np.ma.array
    :param array_2: masked array
    :type array_2: np.ma.array
    """
    a1_masked = np.ma.getmaskarray(array1)
    a2_masked = np.ma.getmaskarray(array2)
    neither_masked = np.logical_not(np.logical_or(a1_masked,a2_masked))
    one_masked = np.logical_xor(a1_masked,a2_masked)
    # Data for a1 is good when only one is masked and the mask is on a2.
    a1_good = np.logical_and(a2_masked, one_masked)

    return np.ma.where(neither_masked, np.ma.minimum(array1, array2),
                       np.ma.where(a1_good, array1, array2))


def merge_two_parameters(param_one, param_two):
    '''
    Use: merge_two_parameters is intended for discrete and multi-state
    parameters. Use blend_two_parameters for analogue parameters.

    This process merges two parameter objects. They must be recorded at the
    same frequency. They are interleaved without smoothing, and then the
    offset and frequency are computed as though the finished item was
    equispaced.

    If the two parameters are recorded less than half the sample interval
    apart, a value error is raised as the synthesized parameter cannot
    realistically be described by an equispaced result.

    :param param_one: Parameter object
    :type param_one: Parameter
    :param param_two: Parameter object
    :type param_two: Parameter

    :returns array, frequency, offset
    '''
    assert param_one.frequency  == param_two.frequency
    assert len(param_one.array) == len(param_two.array)

    delta = (param_one.offset - param_two.offset) * param_one.frequency
    off = (param_one.offset+param_two.offset-(1/(2.0*param_one.frequency)))/2.0
    if -0.75 < delta < -0.25:
        # merged array should be monotonic (always increasing in time)
        array = merge_sources(param_one.array, param_two.array)
        return array, param_one.frequency * 2, off
    elif 0.25 < delta < 0.75:
        array = merge_sources(param_two.array, param_one.array)
        return array, param_two.frequency * 2, off
    else:
        raise ValueError("merge_two_parameters called with offsets too similar. %s : %.4f and %s : %.4f" \
                         % (param_one.name, param_one.offset, param_two.name, param_two.offset))

def merge_sources(*arrays):
    '''
    This simple process merges the data from multiple sensors where they are
    sampled alternately. Unlike blend_alternate_sensors or the parameter
    level option blend_two_parameters, this procedure does not make any
    allowance for the two sensor readings being different.

    :param array: sampled data from an alternate signal source
    :type array: masked array
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    result = np.ma.empty((len(arrays[0]),len(arrays)))
    for dim, array in enumerate(arrays):
        result[:,dim] = array
    return np.ma.ravel(result)


def blend_equispaced_sensors(array_one, array_two):
    '''
    This process merges the data from two sensors where they are sampled
    alternately. Where one sensor is invalid, the process substitutes from
    the other sensor where possible, maintaining a higher level of data
    validity.

    :param array_one: sampled data from one signal source
    :type array_one: masked array
    :param array_two: sampled data from one signal source
    :type array_two: masked array
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    assert len(array_one) == len(array_two)
    both = merge_sources(array_one, array_two)
    both_mask = np.ma.getmaskarray(both)

    av_other = np_ma_masked_zeros_like(both)
    av_other[1:-1] = (both[:-2] + both[2:])/2.0
    av_other[0] = both[1]
    av_other[-1] = both[-2]
    av_other_mask = np.ma.getmaskarray(av_other)

    best = (both + av_other)/2.0
    best_mask = np.ma.getmaskarray(best)

    # We build up the best available data starting from the worst case, where
    # we have no valid data, so return a masked zero
    result = np_ma_masked_zeros_like(both)

    # If the other channel is valid, use the average of the before and after
    # samples of the other channel.
    result = np.ma.where(av_other_mask, result, av_other)

    # Better - if the channel sampled at the right moment is valid, use this.
    result = np.ma.where(both_mask, result, both)

    # Best option is this channel averaged with the mean of the other channel
    # before and after samples.
    result = np.ma.where(best_mask, result, best)

    return result


def blend_nonequispaced_sensors(array_one, array_two, padding):
    '''
    Where there are timing differences between the two samples, this
    averaging process computes the average value between alternate pairs of
    samples. This has the effect of removing sensor mismatch and providing
    equispaced data points. The disadvantage is that in the presence of one
    sensor malfunction, all resulting data is invalid.

    :param array_one: sampled data from one signal source
    :type array_one: masked array
    :param array_two: sampled data from one signal source
    :type array_two: masked array
    :param padding: where to put the padding value in the array
    :type padding: String "Precede" or "Follow"
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    assert len(array_one) == len(array_two)
    both = merge_sources(array_one, array_two)
    # A simpler technique than trying to append to the averaged array.
    av_pairs = np.ma.empty_like(both)
    if padding == 'Follow':
        av_pairs[:-1] = (both[:-1]+both[1:])/2
        av_pairs[-1] = av_pairs[-2]
        av_pairs[-1] = np.ma.masked
    else:
        av_pairs[1:] = (both[:-1]+both[1:])/2
        av_pairs[0] = av_pairs[1]
        av_pairs[0] = np.ma.masked
    return av_pairs


def blend_two_parameters(param_one, param_two):
    '''
    Use: blend_two_parameters is intended for analogue parameters. Use
    merge_two_parameters for discrete and multi-state parameters.

    This process merges the data from two sensors where they are sampled
    alternately. Often pilot and co-pilot attitude and air data signals are
    stored in alternate locations to provide the required sample rate while
    allowing errors in either to be identified for investigation purposes.

    For FDM, only a single parameter is required, but mismatches in the two
    sensors can lead to, taking pitch attitude as an example, apparent "nodding"
    of the aircraft and errors in the derived pitch rate.

    This process merges two parameter arrays of the same frequency.
    Smoothes and then computes the offset and frequency appropriately.

    Two alternative processes are used, depending upon whether the samples
    are equispaced or not.

    :param param_one: Parameter object
    :type param_one: Parameter
    :param param_two: Parameter object
    :type param_two: Parameter

    :returns array, frequency, offset
    :type array: Numpy masked array
    :type frequency: Float (Hz)
    :type offset: Float (sec)

    '''
    if param_one == None:
        return param_two.array, param_two.frequency, param_two.offset

    if param_two == None:
        return param_one.array, param_one.frequency, param_one.offset

    assert param_one.frequency == param_two.frequency

    # A common problem is that one sensor may be unserviceable, and has been
    # identified already by parameter validity testing. Trap this case and
    # deal with it first, raising a warning and dropping back to the single
    # reliable source of information.
    a = np.ma.count(param_one.array)
    b = np.ma.count(param_two.array)
    if a+b == 0:
        logger.warning("Neither '%s' or '%s' has valid data available.",
                       param_one.name, param_two.name)
        # Return empty space of the right shape...
        return np_ma_masked_zeros_like(param_one.array), param_one.frequency, param_one.offset

    if a < b*0.8:
        logger.warning("Little valid data available for %s (%d), using only %s (%d)data.", param_one.name, float(a)/len(param_one.array)*100, param_two.name, float(b)/len(param_two.array)*100)
        return param_two.array, param_two.frequency, param_two.offset

    elif b < a*0.8:
        logger.warning("Little valid data available for %s (%d), using only %s (%d) data.", param_two.name, float(b)/len(param_two.array)*100, param_one.name, float(a)/len(param_one.array)*100)
        return param_one.array, param_one.frequency, param_one.offset

    # A second problem is where both sensor may appear to be serviceable but
    # one is invariant. If the parameters were similar, a/(a+b)=0.5 so we are
    # looking for one being less than 20% of its normal level.
    c = float(np.ma.ptp(param_one.array))
    d = float(np.ma.ptp(param_two.array))

    if c+d == 0.0:
        logger.warning("No variation in %s or %s, returning %s.", param_one.name, param_two.name, param_one.name)
        return param_one.array, param_one.frequency, param_one.offset

    if c/(c+d) < 0.1:
        logger.warning("No variation in %s, using only %s.", param_one.name, param_two.name)
        return param_two.array, param_two.frequency, param_two.offset

    elif d/(c+d) < 0.1:
        logger.warning("No variation in %s, using only %s.", param_two.name, param_one.name)
        return param_one.array, param_one.frequency, param_one.offset

    else:
        frequency = param_one.frequency * 2.0

        # Are the parameters equispaced?
        if abs(param_one.offset - param_two.offset) * frequency == 1.0:
            # Equispaced process
            if param_one.offset < param_two.offset:
                offset = param_one.offset
                array = blend_equispaced_sensors(param_one.array, param_two.array)
            else:
                offset = param_two.offset
                array = blend_equispaced_sensors(param_two.array, param_one.array)

        else:
            # Non-equispaced process
            offset = (param_one.offset + param_two.offset)/2.0
            padding = 'Follow'

            if offset > 1.0/frequency:
                offset = offset - 1.0/frequency
                padding = 'Precede'

            if param_one.offset <= param_two.offset:
                # merged array should be monotonic (always increasing in time)
                array = blend_nonequispaced_sensors(param_one.array, param_two.array, padding)
            else:
                array = blend_nonequispaced_sensors(param_two.array, param_one.array, padding)

        return array, frequency, offset

def blend_parameters_weighting(array, wt):
    '''
    A small function to relate masks to weights.
    
    :param array: array to compute weights for
    :type array: numpy masked array
    :param wt: weighting factor =  ratio of sample rates
    :type wt: float
    '''
    mask = np.ma.getmaskarray(array)
    param_weight = (1.0-mask)
    result_weight = np_ma_masked_zeros_like(np.ma.arange(len(param_weight)*wt))
    final_weight = np_ma_masked_zeros_like(np.ma.arange(len(param_weight)*wt))
    result_weight[0]=param_weight[0]/wt
    result_weight[-1]=param_weight[-1]/wt

    for i in range(1, len(param_weight)-1):
        if param_weight[i]==0.0:
            result_weight[i*wt]=0.0
            continue
        if param_weight[i-1]==0.0 or param_weight[i+1]==0.0:
            result_weight[i*wt]=0.1 # Low weight to tail of valid data. Non-zero to avoid problems of overlapping invalid sections.
            continue
        result_weight[i*wt]=1.0/wt

    for i in range(1, len(result_weight)-1):
        if result_weight[i-1]==0.0 or result_weight[i+1]==0.0:
            final_weight[i]=result_weight[i]/2.0
        else:
            final_weight[i]=result_weight[i]
    final_weight[0]=result_weight[0]
    final_weight[-1]=result_weight[-1]

    return repair_mask(final_weight, repair_duration=None)


def blend_parameters(params, offset=0.0, frequency=1.0, debug=False):
    '''
    This most general form of the blend options allows for multiple sources
    to be blended together even though the spacing, validity and even sample
    rate may be different. Furthermore the offset and frequency of the output
    parameter can be selected if required.
    
    This uses cubic spline interpolation for each of the component
    parameters, then applies weighting to reflect both the frequency of
    samples of the parameter and it's mask. The multiple cubic splines are
    then summed at the points where new samples are required.
    
    We may change to use a different form of interpolation in the
    future, allowing for control of the first derivative at the ends of
    the data, but that's in the future...

    :param params: the parameters to be merged
    :type params: tuple of parameters 
    :param offset: the offset of the resulting parameter
    :type offset: float (sec)
    :param frequency: the frequency of the resulting parameter
    :type frequency: float (Hz)
    
    :param debug: flag to plot graphs for ease of testing
    :type debug: boolean, default to False
    '''
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
    assert frequency>0.0
    
    p_valid_slices = []
    p_offset = []
    p_freq = []
    
    # Prepare a place for the output signal
    length = len(params[0].array) * frequency / params[0].frequency
    result = np_ma_masked_zeros(length)
    # Ensure mask is expanded for slicing.
    result.mask = np.ma.getmaskarray(result)
    
    # Find out about the parameters we have to deal with...
    for seq, param in enumerate(params):
        p_freq.append(param.frequency)
        p_offset.append(param.offset)
    min_ip_freq = min(p_freq)
    
    # Slices of valid data are scaled to the lowest timebase and then or'd
    # to find out when any valid data is available.
    for seq, param in enumerate(params):
        # We can only work on non-trivial slices which have four or more
        # samples as below this level it's not possible to compute a cubic
        # spline.
        nts=slices_remove_small_slices(np.ma.clump_unmasked(param.array),
                                       count=4)
        # Now scale these non-trivial slices into the lowest timebase for
        # collation.
        p_valid_slices.append(slices_multiply(nts, min_ip_freq / p_freq[seq]))
        
    # To find the valid ranges I need to 'or' the slices at a high level, hence
    # this list of lists of slices needs to be flattened. Don't ask me what
    # this does, go to http://stackoverflow.com/questions/952914 for an
    # explanation !
    any_valid = slices_or([item for sublist in p_valid_slices for item in sublist])
    
    if any_valid is None:
        # No useful chunks of data to process, so give up now.
        return
    
    # Now we can work through each period of valid data.
    for this_valid in any_valid:
        
        result_slice = slice_multiply(this_valid, frequency/min_ip_freq)
        
        new_t = np.linspace(result_slice.start / frequency,
                            result_slice.stop / frequency,
                            num=(result_slice.stop - result_slice.start),
                            endpoint=False) + offset
        
        # Make space for the computed curves
        curves=[]
        weights=[]
        resampled_masks = []

        # Compute the individual splines
        for seq, param in enumerate(params):
            # The slice and timebase for this parameter...
            my_slice = slice_multiply(this_valid, p_freq[seq] / min_ip_freq)
            resampled_masks.append(
                resample(np.ma.getmaskarray(param.array)[my_slice],
                         param.frequency, frequency))
            timebase = np.linspace(my_slice.start/p_freq[seq],
                                   my_slice.stop/p_freq[seq],
                                   num=my_slice.stop-my_slice.start,
                                   endpoint=False) + p_offset[seq]
            my_time = np.ma.array(
                data=timebase, mask=np.ma.getmaskarray(param.array[my_slice]))
            if len(my_time.compressed()) < 4:
                continue
            my_curve = scipy_interpolate.splrep(
                my_time.compressed(), param.array[my_slice].compressed(), s=0)
            # my_curve is the spline knot array, now compute the values for
            # the output timebase.
            curves.append(
                scipy_interpolate.splev(new_t, my_curve, der=0, ext=0))

            # Compute the weights 
            weights.append(blend_parameters_weighting(
                param.array[my_slice], frequency/param.frequency))
            
            if debug:
                plt.plot(my_time,param.array[my_slice], 'o')
                plt.plot(new_t,curves[seq], '-.')
                plt.plot(new_t,weights[seq])
                
        if curves==[]:
            continue
        a = np.vstack(tuple(curves))
        result[result_slice] = np.average(a, axis=0, weights=weights)
        # Q: Is this the right place? Should it be applied to this_valid slice?
        result.mask[result_slice] = merge_masks(resampled_masks,
                                                min_unmasked=2)
        # The endpoints of a cubic spline are generally unreliable, so trim
        # them back.
        result[result_slice][0] = np.ma.masked
        result[result_slice][-1] = np.ma.masked
        
        if debug:
            plt.plot(new_t,result[result_slice], '-')
            plt.show()

    return result
    

def most_points_cost(coefs, x, y):
    '''
    This cost function computes a value which is minimal for points clost to
    a "best fit" line. It differs from normal least squares optimisation in
    that points a long way from the line have almost the same error as points
    a little way off the line.
    
    The function is used as a form of correlation function where we are
    looking to find the largest number of points on a certain line, with less
    regard to points that lie off that line.
    
    :param coefs: line coefficients, m and c, to be adjusted to minimise this cost function.
    :type coefs: list of floats, containing [m, c]
    :param x: independent variable
    :type x: numpy masked array
    :param y: dependent variable
    :type y: numpy masked array
    
    :returns: cost function; most negative value represents best fit.
    :type: float
    '''
    # Wrote "assert len(x) == len(y)" but can't find how to test this, so verbose equivalent is...
    if len(x) != len(y):
        raise ValueError('most_points_cost called with x & y of unequal length')
    if len(x) < 2:
        raise ValueError('most_points_cost called with inadequate samples')
    # Conventional y=mx+c equation for the "bet fit" line
    m=coefs[0]
    c=coefs[1]
    
    # We compute the distance of each point from the line
    d = np.ma.sqrt((m*x+c-y)**2.0/(m**2.0+1))
    # and work out the maximum distance
    d_max = np.ma.max(d)
    if d_max == 0.0:
        raise ValueError('most_points_cost called with colinear data')
    # The error for each point is computed as a nonlinear function of the
    # distance, tailored to make points on the line give a small error, and
    # those away from the line progressively greater, but reaching a limit
    # value of 0 so that points at a great distance do not contribute more to
    # the weighted error.
    
    # width sets the width of the channel created by this function. Larger
    # values make the channel wider, but this opens up the function to
    # settling on minima away from the optimal line. Too narrow a width and,
    # again, the function can latch onto few points and determine a local
    # minimum. The value of 0.003 was chosen from analysis of fuel flow vs
    # altitude plots where periods of level flight in the climb create low
    # fuel readings which are not part of the climb performance we are trying
    # to detect. Values 3 times greater or smaller gave similar results,
    # while values 10 times greater or smaller led to erroneous results.
    width=0.003
    e = 1.0 -1.0/((d/d_max)**2 + width)
    return np.ma.sum(e)


def moving_average(array, window=9, weightings=None, pad=True):
    """
    Moving average over an array with window of n samples. Weightings allows
    customisation of the importance of each position's value in the average.

    Recommend odd lengthed moving windows as the result is positioned
    centrally in the window offset.

    :param array: Masked Array
    :type array: np.ma.array
    :param window: Size of moving average window to use
    :type window: Integer
    :param pad: Pad the returned array to the same length of the input, using masked 0's
    :type pad: Boolean
    :param weightings: Apply uneven weightings across the window - the same length as window.
    :type weightings: array-like object consisting of floats

    Ref: http://argandgahandapandpa.wordpress.com/2011/02/24/python-numpy-moving-average-for-data/
    """
    if len(array)==0:
        return None
    
    if weightings is None:
        weightings = np.repeat(1.0, window) / window
    elif len(weightings) != window:
        raise ValueError("weightings argument (len:%d) must equal window (len:%d)" % (
            len(weightings), window))
    # repair mask
    repaired = repair_mask(array, repair_duration=None,
                           raise_duration_exceedance=False)
    # if start of mask, ignore this section and remask at end
    start, end = np.ma.notmasked_edges(repaired)
    stop = end+1
    # slice array with these edges
    unmasked_data = repaired.data[start:stop]

    averaged = np.convolve(unmasked_data, weightings, 'valid')
    if pad:
        # mask the new stuff
        pad_front = np.ma.zeros(window/2 + start)
        pad_front.mask = True
        pad_end = np.ma.zeros(len(array)-1 + ceil(window/2.0) - stop)
        pad_end.mask = True
        return np.ma.hstack([pad_front, averaged, pad_end])
    else:
        return averaged


def nearest_neighbour_mask_repair(array, copy=True):
    """
    WARNING: Currently wraps, so masked items at start will be filled with
    values from end of array.

    TODO: Avoid wrapping from start /end and use first value to preceed values.

    Ref: http://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
    """
    if copy:
        array = array.copy()
    def next_neighbour(start=1):
        """
        Generates incrementing positive and negative pairs from start
        e.g. start = 1
        yields 1,-1, 2,-2, 3,-3,...
        """
        x = start
        while True:
            yield x
            yield -x
            x += 1
    # if first or last masked, repair now
    start, stop = np.ma.notmasked_edges(array)
    if start > 0:
        array[:start] = array[start]
    if stop+1 < len(array):
        array[stop+1:] = array[stop]

    neighbours = next_neighbour()
    a_copy = array.copy()
    for shift in neighbours:
        if not np.any(array.mask):
            break
        a_shifted = np.roll(a_copy,shift=shift)
        idx = ~a_shifted.mask * array.mask
        array[idx] = a_shifted[idx]
    return array


def normalise(array, normalise_max=1.0, scale_max=None, copy=True, axis=None):
    """
    Normalise an array between 0 and normalise_max.

    :param normalise_max: Upper limit of normalised result. Default range is between 0 and 1.
    :type normalise_max: float
    :param scale_max: Maximum value to normalise against. If None, the maximum value will be sourced from the array.
    :type scale_max: int or float or None
    :param copy: Returns a copy of the array, leaving input array untouched
    :type copy: bool
    :param axis: default to normalise across all axis together. Only supports None, 0 and 1!
    :type axis: int or None
    :returns: Array containing normalised values.
    :rtype: np.ma.masked_array
    """
    if copy:
        array = array.copy()
    scaling = normalise_max / (scale_max or array.max(axis=axis))
    if axis == 1:
        # transpose
        scaling = scaling.reshape(scaling.shape[0],-1)
    array *= scaling
    ##array *= normalise_max / array.max() # original single axis version
    return array

def np_ma_concatenate(arrays):
    """
    Derivative of the normal concatenate function which handles mapped discrete arrays.
    :param arrays: list of arrays, which may have mapped values.
    :type arrays: list of numpy masked arrays

    :returns: single numpy masked array, which may have mapped values.

    :raises: ValueError if mapped arrays carry different mappings.
    """
    if len(arrays) == 0:
        return None # Nothing to concatenate !

    if hasattr(arrays[0], 'values_mapping'):
        # Handle mapped arrays here.
        mapping = arrays[0].values_mapping
        for each_array in arrays[1:len(arrays)+1]:
            if each_array.values_mapping != mapping:
                raise ValueError('Attempt to concatenate differing multistate arrays')
        array = np.ma.concatenate(arrays)
        array.values_mapping = mapping
        return array
    else:
        # Numeric only arrays.
        return np.ma.concatenate(arrays)


def np_ma_zeros_like(array, mask=False):
    """
    The Numpy masked array library does not have equivalents for some array
    creation functions. These are provided with similar names which may be
    replaced should the Numpy library be extended in future.

    :param array: array of length to be replicated.
    :type array: A Numpy masked array - can be masked or not.

    TODO: Confirm operation with normal Numpy array. The reference to array.data probably fails.

    :returns: Numpy masked array of unmasked zero values, length same as input array.
    """
    return np.ma.array(np.zeros_like(array.data), mask=mask, dtype=float)


def np_ma_ones_like(array):
    """
    Creates a masked array filled with ones. See also np_ma_zeros_like.

    :param array: array of length to be replicated.
    :type array: A Numpy array - can be masked or not.

    :returns: Numpy masked array of unmasked 1.0 float values, length same as input array.
    """
    return np_ma_zeros_like(array) + 1.0


def np_ma_ones(length):
    """
    Creates a masked array filled with ones.

    :param length: length of the array to be created.
    :type length: integer.

    :returns: Numpy masked array of unmasked 1.0 float values, length as specified.
    """
    return np_ma_zeros_like(np.ma.arange(length)) + 1.0


def np_ma_masked_zeros(length):
    """
    Creates a masked array filled with masked values. The unmasked data
    values are all zero. The very klunky code here is to circumvent Numpy's
    normal response which is to return random data values where it knows the
    data is masked. In this case we want to ensure zero values as we may be
    lifting the mask in due course and we don't want to reveal random data.

    See also np_ma_zeros_like.

    :param length: array length to be replicated.
    :type length: int

    :returns: Numpy masked array of masked 0.0 float values of length equal to
    input.
    """
    return np.ma.array(data=np.zeros(length), mask=True)


def np_ma_masked_zeros_like(array):
    """
    Creates a masked array filled with masked values. The unmasked data
    values are all zero. The very klunky code here is to circumvent Numpy's
    normal response which is to return random data values where it knows the
    data is masked. In this case we want to ensure zero values as we may be
    lifting the mask in due course and we don't want to reveal random data.

    See also np_ma_zeros_like.

    :param array: array of length to be replicated.
    :type array: A Numpy array - can be masked or not.

    :returns: Numpy masked array of masked 0.0 float values, length same as
    input array.
    """
    return np.ma.array(data = np_ma_zeros_like(array).data,
                       mask = np_ma_ones_like(array).data)


def truck_and_trailer(data, ttp, overall, trailer, curve_sense, _slice):
    '''
    See peak_curvature procedure for details of parameters.
    '''
    # Trap for invariant data
    if np.ma.ptp(data) == 0.0:
        return None

    # Set up working arrays
    x = np.arange(ttp) + 1 #  The x-axis is always short and constant
    sx = np.sum(x)
    r = sx/float(x[-1]) #
    trucks = len(data) - ttp + 1 #  How many trucks fit this array length?

    sy = np.empty(trucks) #  Sigma y
    sy[0]=np.sum(data[0:ttp]) #  Initialise this array with just y values

    sxy = np.empty(trucks) #  Sigma x.y
    sxy[0]=np.sum(data[0:ttp]*x[0:ttp]) #  Initialise with xy products

    for back in range(trucks-1):
        # We compute the values for the least squares formula, using the
        # numerator only (the denominator is constant and we're not really
        # interested in the answer).

        # As we move the back of the truck forward, the trailer front is a
        # little way ahead...
        front = back + ttp
        sy[back+1] = sy[back] - data [back] + data[front]
        sxy[back+1] = sxy[back] - sy[back] + ttp*data[front]

    m = np.empty(trucks) # Resulting least squares slope (best fit y=mx+c)
    m = sxy - r*sy

    #  How many places can the truck and trailer fit into this data set?
    places=len(data) - overall + 1
    #  The angle between the truck and trailer at each place it can fit
    angle=np.empty(places)

    for place in range(places):
        angle[place] = m[place+trailer] - m[place]

    # Normalise array and prepare for masking operations
    angle_max = np.max(np.abs(angle))
    if angle_max == 0.0:
        return None # All data in a straight line, so no curvature to find.

    angle=np.ma.array(angle/angle_max)

    # Default curve sense of Concave has a positive angle. The options are
    # adjusted to allow us to use positive only tests hereafter.
    if curve_sense == 'Bipolar':
        angle = np.ma.abs(angle)
    elif curve_sense == 'Convex':
        angle = -angle
    else:  # curve_sense == 'Concave'
        # angle remains as is
        pass

    # Find peak - using values over 50% of the highest allows us to operate
    # without knowing the data characteristics.
    peak_slice=np.ma.clump_unmasked(np.ma.masked_less(angle,0.5))

    if peak_slice:
        index = peak_index(angle.data[peak_slice[0]])+\
            peak_slice[0].start+(overall/2.0)-0.5
        return index*(_slice.step or 1) + (_slice.start or 0)
    else:
        # Data curved in wrong sense or too weakly to find corner point.
        return None


def offset_select(mode, param_list):
    """
    This little piece of code finds the offset from a list of possibly empty
    parameters. This is used in the collated engine parameters where
    allowance is made for four engines, but only two or three may be
    installed and we don't know which order the parameters are recorded in.

    :param mode: which type of offset to compute.
    :type mode: string 'mean', 'first', 'last'

    :return: offset
    :type: float
    """
    least = None
    for p in param_list:
        if p:
            if not least:
                least = p.offset
                most = p.offset
                total = p.offset
                count = 1
            else:
                least = min(least, p.offset)
                most = max(most, p.offset)
                total = total + p.offset
                count += 1
    if mode == 'mean':
        return total / float(count)
    if mode == 'first':
        return least
    if mode == 'last':
        return most
    raise ValueError ("offset_select called with unrecognised mode")


def peak_curvature(array, _slice=slice(None), curve_sense='Concave',
                   gap = TRUCK_OR_TRAILER_INTERVAL,
                   ttp = TRUCK_OR_TRAILER_PERIOD):
    """
    :param array: Parameter to be examined
    :type array: Numpy masked array
    :param _slice: Range of index values to be scanned.
    :type _slice: Python slice. May be indexed in reverse to scan backwards in time.
    :param curve_sense: Optional operating mode. Default 'Concave' has
                        positive curvature (concave upwards when plotted).
                        Alternatives 'Convex' for curving downwards and
                        'Bi-polar' to detect either sense.
    :type curve_sense: string

    :returns peak_curvature: The index where the curvature first peaks in the required sense.
    :rtype: integer

    Note: Although the range to be inspected may be restricted by slicing,
    the peak curvature index relates to the whole array, not just the slice.

    This routine uses a "Truck and Trailer" algorithm to find where a
    parameter changes slope. In the case of FDM, we are looking for the point
    where the airspeed starts to increase (or stops decreasing) on the
    takeoff and landing phases. This is more robust than looking at
    longitudinal acceleration and complies with the POLARIS philosophy that
    we should provide analysis with only airspeed, altitude and heading data
    available.
    """
    curve_sense = curve_sense.title()
    if curve_sense not in ('Concave', 'Convex', 'Bipolar'):
        raise ValueError('Curve Sense %s not supported' % curve_sense)
    if gap%2 - 1:
        gap -= 1  #  Ensure gap is odd
    trailer = ttp+gap
    overall = 2*ttp + gap

    input_data = array[_slice]
    if np.ma.count(input_data)==0:
        return None

    valid_slices = np.ma.clump_unmasked(input_data)
    for valid_slice in valid_slices:
        # check the contiguous valid data is long enough.
        if (valid_slice.stop - valid_slice.start) <= 3:
            # No valid segment data is not long enough to process
            continue
        elif np.ma.ptp(input_data[valid_slice]) == 0:
            # No variation to scan in current valid slice.
            continue
        elif valid_slice.stop - valid_slice.start > overall:
            # Use truck and trailer as we have plenty of data
            data = array[_slice][valid_slice]
            # The normal path is to go and process this data.
            corner = truck_and_trailer(data, ttp, overall, trailer, curve_sense, _slice)  #Q: What is _slice going to do if we've already subsliced it?
            if corner:
                # Found curve
                return corner + valid_slice.start
            # Look in next slice
            continue
        else:
            if _slice.step not in (None, 1, -1):
                raise ValueError("Index returned cannot handle big steps!")
            # Simple methods for small data sets.
            data = input_data[valid_slice]
            curve = data[2:] - 2.0*data[1:-1] + data[:-2]
            if curve_sense == 'Concave':
                curve_index, val = max_value(curve)
                if val <= 0:
                    # No curve or Curved wrong way
                    continue
            elif curve_sense == 'Convex':
                curve_index, val = min_value(curve)
                if val >= 0:
                    # No curve or Curved wrong way
                    continue
            else:  #curve_sense == 'Bipolar':
                curve_index, val = max_abs_value(curve)
                if val == 0:
                    # No curve
                    continue
            # Add 1 to move into middle of 3 element curve and add slice positions back on
            index = curve_index + 1 + valid_slice.start + (_slice.start or 0)
            if _slice.step is not None and _slice.step < 0:
                # stepping backwards through data, change index
                return len(array) - index
            else:
                return index
        #endif
    else:  #endfor
        # did not find curve in valid data
        return None

def peak_index(a):
    '''
    Scans an array and returns the peak, where possible computing the local
    maximum assuming a quadratic curve over the top three samples.

    :param a: array
    :type a: list of floats

    '''
    if len(a) == 0:
        raise ValueError('No data to scan for peak')
    elif len(a) == 1:
        return 0
    elif len(a) == 2:
        return np.argmax(a)
    else:
        loc=np.argmax(a)
        if loc == 0:
            return 0
        elif loc == len(a)-1:
            return len(a)-1
        else:
            denominator = (2.0*a[loc-1]-4.0*a[loc]+2.0*a[loc+1])
            if abs(denominator) < 0.001:
                return loc
            else:
                peak=(a[loc-1]-a[loc+1])/denominator
                return loc+peak


def rate_of_change_array(to_diff, hz, width=2.0):
    '''
    Lower level access to rate of change algorithm. See rate_of_change for description.

    :param to_diff: input data
    :type to_diff: Numpy masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param width: the differentiation time period (sec)
    :type width: float

    :returns: masked array of values with differentiation applied

    '''
    hw = int(width * hz / 2.0)
    if hw < 1:
        raise ValueError('Rate of change called with inadequate width.')
    if len(to_diff) <= 2*hw:
        logger.info("Rate of change called with short data segment. Zero rate "
                    "returned")
        return np_ma_zeros_like(to_diff)

    # Set up an array of masked zeros for extending arrays.
    slope = np.ma.copy(to_diff)
    slope[hw:-hw] = (to_diff[2*hw:] - to_diff[:-2*hw])/width
    slope[:hw] = (to_diff[1:hw+1] - to_diff[0:hw]) * hz
    slope[-hw:] = (to_diff[-hw:] - to_diff[-hw-1:-1])* hz
    return slope

def rate_of_change(diff_param, width):
    '''
    @param to_diff: Parameter object with .array attr (masked array)

    Differentiation using the xdot(n) = (x(n+hw) - x(n-hw))/w formula.
    Half width hw=w/2 and this provides smoothing over a w second period,
    without introducing a phase shift.

    :param diff_param: input Parameter
    :type diff_param: Parameter object
    :type diff_param.array : masked array
    :param diff_param.frequency : sample rate for the input data (sec-1)
    :type diff_param.frequency: float
    :param width: the differentiation time period (sec)
    :type width: float

    :returns: masked array of values with differentiation applied
    '''
    hz = diff_param.frequency
    to_diff = diff_param.array
    return rate_of_change_array(to_diff, hz, width)


def repair_mask(array, frequency=1, repair_duration=REPAIR_DURATION,
                raise_duration_exceedance=False, copy=False, extrapolate=False,
                zero_if_masked=False, repair_above=None):
    '''
    This repairs short sections of data ready for use by flight phase algorithms
    It is not intended to be used for key point computations, where invalid data
    should remain masked.

    :param copy: If True, returns modified copy of array, otherwise modifies the array in-place.
    :param zero_if_masked: If True, returns a fully masked zero-filled array if all incoming data is masked.
    :param repair_duration: If None, any length of masked data will be repaired.
    :param raise_duration_exceedance: If False, no warning is raised if there are masked sections longer than repair_duration. They will remain unrepaired.
    :param extrapolate: If True, data is extrapolated at the start and end of the array.
    :param repair_above: If value provided only masked ranges where first and last unmasked values are this value will be repaired.
    :raises ValueError: If the entire array is masked.
    '''
    if not np.ma.count(array):
        if zero_if_masked:
            return np_ma_zeros_like(array, mask=True)
        else:
            raise ValueError("Array cannot be repaired as it is entirely masked")
    if copy:
        array = array.copy()
    if repair_duration:
        repair_samples = repair_duration * frequency
    else:
        repair_samples = None

    masked_sections = np.ma.clump_masked(array)
    for section in masked_sections:
        length = section.stop - section.start
        if repair_samples and (length) > repair_samples:
            if raise_duration_exceedance:
                raise ValueError("Length of masked section '%s' exceeds "
                                 "repair duration '%s'." % (length * frequency,
                                                            repair_duration))
            else:
                continue # Too long to repair
        elif section.start == 0:
            if extrapolate:
                # TODO: Does it make sense to subtract 1 from the section stop??
                #array.data[section] = array.data[section.stop - 1]
                array.data[section] = array.data[section.stop]
                array.mask[section] = False
            else:
                continue # Can't interpolate if we don't know the first sample

        elif section.stop == len(array):
            if extrapolate:
                array.data[section] = array.data[section.start - 1]
                array.mask[section] = False
            else:
                continue # Can't interpolate if we don't know the last sample
        else:
            start_value = array.data[section.start - 1]
            end_value = array.data[section.stop]
            if repair_above is None or (start_value > repair_above and end_value > repair_above):
                array.data[section] = np.interp(np.arange(length) + 1,
                                                [0, length + 1],
                                                [start_value, end_value])
                array.mask[section] = False

    return array


def resample(array, orig_hz, resample_hz):
    '''
    Upsample or downsample an array for it to match resample_hz.
    Offset is maintained because the first sample is always returned.
    '''
    if orig_hz == resample_hz:
        return array
    modifier = resample_hz / float(orig_hz)
    if modifier > 1:
        return np.ma.repeat(array, modifier)
    else:
        return array[::1 / modifier]


def round_to_nearest(array, step):
    """
    Rounds to nearest step value, so step 5 would round as follows:
    1 -> 0
    3.3 -> 5
    7.5 -> 10
    10.5 -> 10 # np.round drops to nearest even number(!)

    :param array: Array to be rounded
    :type array: np.ma.array
    :param step: Value to round to
    :type step: int or float
    """
    step = float(step) # must be a float
    return np.ma.round(array / step) * step


def rms_noise(array, ignore_pc=None):
    '''
    :param array: input parameter to measure noise level
    :type array: numpy masked array
    :param ignore_pc: percent to ignore (see below)
    :type integer: % value in range 0-100

    :returns: RMS noise level
    :type: Float, units same as array

    :exception: Should all the difference terms include masked values, this
    function will return None.

    This computes the rms noise for each sample compared with its neighbours.
    In this way, a steady cruise at 30,000 ft will yield no noise, as will a
    steady climb or descent.

    The rms noise may be used to examine parameter reasonableness, in which
    case the occasional spike is not considered background noise levels. The
    ignore_pc value allows the highest spike readings to be ignored and the
    rms is then the level for the normal operation of the parameter.
    '''
    # The difference between one sample and the ample to the left is computed
    # using the ediff1d algorithm, then by rolling it right we get the answer
    # for the difference between this sample and the one to the right.
    diff_left = np.ma.ediff1d(array, to_end=0)
    diff_right = np.ma.array(data=np.roll(diff_left.data,1),
                             mask=np.roll(diff_left.mask,1))
    local_diff = (diff_left - diff_right)/2.0
    diffs = local_diff[1:-1]
    if np.ma.count(diffs) == 0:
        return None
    elif ignore_pc == None or ignore_pc/100.0*len(array)<1.0:
        to_rms = diffs
    else:
        monitor = slice(0, floor(len(diffs) * (1-ignore_pc/100.0)))
        to_rms = np.ma.sort(np.ma.abs(diffs))[monitor]
    return sqrt(np.ma.mean(np.ma.power(to_rms,2))) # RMS in one line !


def runs_of_ones(bits):
    '''
    Q: This function used to have a min_len kwarg which was a result of its
    implementation. If there is a use case for only returning sections greater
    than a minimum length, would it be better to specify time based on a
    frequency rather than samples?
    TODO: Update to return Sections?
    :returns: S
    :rtype: [slice]
    '''
    return np.ma.clump_unmasked(np.ma.masked_not_equal(bits, 1))


def shift_slice(this_slice, offset):
    """
    This function shifts a slice by an offset. The need for this arises when
    a phase condition has been used to limit the scope of another phase
    calculation.

    :type this_slice: slice
    :type offset: int or float
    :rtype: slice
    """
    if not offset:
        return this_slice

    start = None if this_slice.start is None else this_slice.start + offset
    stop = None if this_slice.stop is None else this_slice.stop + offset

    if start is None or stop is None or (stop - start) >= 1:
        ### This traps single sample slices which can arise due to rounding of
        ### the iterpolated slices.
        return slice(start, stop, this_slice.step)
    else:
        return None


def shift_slices(slicelist, offset):
    """
    This function shifts a list of slices by a common offset, retaining only
    the valid (not None) slices.

    :type slicelist: [slice]
    :type offset: int or float
    :rtype [slice]

    """
    if offset:
        newlist = []
        for each_slice in slicelist:
            if each_slice and offset:
                new_slice = shift_slice(each_slice,offset)
                if new_slice: newlist.append(new_slice)
        return newlist
    else:
        return slicelist


def slice_duration(_slice, hz):
    '''
    Gets the duration of a slice in taking the frequency into account. While
    the calculation is simple, there were instances within the code of slice
    durations being compared against values in seconds without considering
    the frequency of the slice indices.

    :param _slice: Slice to calculate the duration of.
    :type _slice: slice
    :param hz: Frequency of slice.
    :type hz: float or int
    :returns: Duration of _slice in seconds.
    :rtype: float
    '''
    if _slice.stop is None:
        raise ValueError("Slice stop '%s' is unsupported by slice_duration.",
                         _slice.stop)
    return (_slice.stop - (_slice.start or 0)) / hz

def slice_multiply(_slice, f):
    '''
    :param _slice: Slice to rescale
    :type _slice: slice
    :param f: Rescale factor
    :type f: float

    :returns: slice rescaled by factor f
    :rtype: integer
    '''
    """
    Original version replaced by less tidy version to maintain start=0 cases
    and to ensure rounding for reductions in frequency does not extend into
    earlier samples than those intended.
    """
    if _slice.start is None:
        _start = None
    else:
        _start = ceil(_slice.start*f)

    return slice(_start,
                 int(_slice.stop*f) if _slice.stop else None,
                 int(_slice.step*f) if _slice.step else None)

def slices_multiply(_slices, f):
    '''
    :param _slices: List of slices to rescale
    :type _slice: slice
    :param f: Rescale factor
    :type f: float

    :returns: List of slices rescaled by factor f
    :rtype: integer
    '''
    result=[]
    for s in _slices:
        result.append(slice_multiply(s,f))
    return result

def slice_samples(_slice):
    '''
    Gets the number of samples in a slice.

    :param _slice: Slice to count sample length.
    :type _slice: slice
    :returns: Number of samples in _slice.
    :rtype: integer
    '''
    step = 1 if _slice.step is None else _slice.step

    if _slice.start is None or _slice.stop is None:
        return 0
    else:
        return (abs(_slice.stop - _slice.start) - 1) / abs(step) + 1


def slices_above(array, value):
    '''
    Get slices where the array is above value. Repairs the mask to avoid a
    large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param value: Value to create slices above.
    :type value: float or int
    :returns: Slices where the array is above a certain value.
    :rtype: list of slice
    '''
    if len(array) == 0:
        return array, []
    repaired_array = repair_mask(array)
    if repaired_array is None: # Array length is too short to be repaired.
        return array, []
    band = np.ma.masked_less(repaired_array, value)
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_below(array, value):
    '''
    Get slices where the array is below value. Repairs the mask to avoid a
    large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param value: Value to create slices below.
    :type value: float or int
    :returns: Slices where the array is below a certain value.
    :rtype: list of slice
    '''
    if len(array) == 0:
        return array, []
    repaired_array = repair_mask(array)
    if repaired_array is None: # Array length is too short to be repaired.
        return array, []
    band = np.ma.masked_greater(repaired_array, value)
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_between(array, min_, max_):
    '''
    Get slices where the array's values are between min_ and max_. Repairs
    the mask to avoid a large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param min_: Minimum value within slices.
    :type min_: float or int
    :param max_: Maximum value within slices.
    :type max_: float or int
    :returns: Slices where the array is above a certain value.
    :rtype: list of slice
    '''
    if len(array) == 0:
        return array, []
    repaired_array = repair_mask(array)
    if repaired_array is None: # Array length is too short to be repaired.
        return array, []
    # Slice through the array at the top and bottom of the band of interest
    band = np.ma.masked_outside(repaired_array, min_, max_)
    # Remove the equality cases as we don't want these. (The common issue
    # here is takeoff and landing cases where 0ft includes operation on the
    # runway. As the array samples here are not coincident with the parameter
    # being tested in the KTP class, by doing this we retain the last test
    # parameter sample before array parameter saturated at the end condition,
    # and avoid testing the values when the array was unchanging.
    band = np.ma.masked_equal(band, min_)
    band = np.ma.masked_equal(band, max_)
    # Group the result into slices - note that the array is repaired and
    # therefore already has small masked sections repaired, so no allowance
    # is needed here for minor data corruptions.
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_from_to(array, from_, to):
    '''
    Get slices of the array where values are between from_ and to, and either
    ascending or descending depending on whether from_ is greater than or less
    than to. For instance, slices_from_to(array, 1000, 1500) is ascending and
    requires will only return slices where values are between 1000 and 1500 if
    the value in the array at the start of the slice is less than the value at
    the stop. The opposite condition would be applied if the arguments are
    descending, e.g. slices_from_to(array, 1500, 1000).

    :param array:
    :type array: np.ma.masked_array
    :param from_: Value from.
    :type from_: float or int
    :param to: Value to.
    :type to: float or int
    :returns: Slices of the array where values are between from_ and to and either ascending or descending depending on comparing from_ and to.
    :rtype: list of slice
    '''

    if from_ == to:
        raise ValueError('From and to values should not be equal.')

    def condition(s):
        start_v = rep_array[s.start]
        mid_v = rep_array[(s.start+s.stop)/2]
        end_v = array[s.stop - 1]

        if len(array[s]) == 1:
            if s.start:
                start_v = array[s.start - 1]
            if s.stop and s.stop < len(array):
                end_v = array[s.stop]

        if from_ > to:
            return start_v >= mid_v >= end_v
        else:
            return start_v <= mid_v <= end_v

    if len(array) == 0:
        return array, []
    rep_array, slices = slices_between(array, from_, to)
    # Midpoint conditions added to lambda to prevent data that just dips into
    # a band triggering.

    filtered_slices = filter(condition, slices)
    return rep_array, filtered_slices

"""
Spline function placeholder

At some time we are likely to want to add interpolation, and this scrap of
code was used to prove the principle. Easy to do and the results are really
close to the recorded data in the case used for testing.

See 'Pitch rate computation at 4Hz and 1Hz with interpolation.xls'

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

y=np.array([0.26,0.26,0.79,0.35,-0.26,-0.04,1.23,4.57,4.75,1.93,0.44,1.14,0.97,1.14,0.79])
x=np.array(range(236,251,1))
f = interp.interp1d(x, y, kind='cubic')
xnew = np.linspace(236,250,57)
plt.plot(x,y,'o',xnew,f(xnew),'-')
plt.legend(['data', 'cubic'], loc='best')
plt.show()
for i in xnew:
    print f(i)
"""

def step_values(array, steps):
    """
    Rounds each value in array to nearest step. Maintains the
    original array mask.

    :param array: Masked array to step
    :type array: np.ma.array
    :param steps: Steps to round to nearest value
    :type steps: list of integers
    :returns: Stepped masked array
    :rtype: np.ma.array
    """
    stepping_points = np.ediff1d(steps, to_end=[0])/2.0 + steps
    stepped_array = np.zeros_like(array.data)
    low = None
    for level, high in zip(steps, stepping_points):
        if low is None:
            stepped_array[(-high < array) & (array <= high)] = level
        else:
            stepped_array[(low < array) & (array <= high)] = level
        low = high
    else:
        # all values above the last
        stepped_array[low < array] = level
    return np.ma.array(stepped_array, mask=array.mask)


def touchdown_inertial(land, roc, alt):
    """
    For aircraft without weight on wheels switches, or if there is a problem
    with the switch for this landing, we do a local integration of the
    inertial rate of climb to estimate the actual point of landing. This is
    referenced to the available altitude signal, Altitude AAL, which will
    have been derived from the best available source. This technique leads on
    to the rate of descent at landing KPV which can then make the best
    calculation of the landing ROD as we know more accurately the time where
    the mainwheels touched.

    :param land: Landing period
    :type land: slice
    :param roc: inertial rate of climb
    :type roc: Numpy masked array
    :param alt: altitude aal
    :type alt: Numpy masked array

    :returns: index, rod
    :param index: index within landing period
    :type index: integer
    :param rod: rate of descent at touchdown
    :type rod: float, units fpm
    """

    # Time constant of 6 seconds.
    tau = 1/6.0
    # Make space for the integrand
    startpoint = land.start_edge
    endpoint = land.stop_edge
    sm_ht = np_ma_zeros_like(roc.array[startpoint:endpoint])
    # Repair the source data (otherwise we propogate masked data)
    my_roc = repair_mask(roc.array[startpoint:endpoint])
    my_alt = repair_mask(alt.array[startpoint:endpoint])

    # Start at the beginning...
    sm_ht[0] = alt.array[startpoint]
    #...and calculate each with a weighted correction factor.
    for i in range(1, len(sm_ht)):
        sm_ht[i] = (1.0-tau)*sm_ht[i-1] + tau*my_alt[i-1] + my_roc[i]/60.0/roc.hz

    '''
    # Plot for ease of inspection during development.
    from analysis_engine.plot_flight import plot_parameter
    plot_parameter(alt.array[startpoint:endpoint], show=False)
    plot_parameter(roc.array[startpoint:endpoint]/100.0, show=False)
    plot_parameter(on_gnd.array[startpoint:endpoint], show=False)
    plot_parameter(sm_ht)
    '''

    # Find where the smoothed height touches zero and hence the rod at this
    # point. Note that this may differ slightly from the touchdown measured
    # using wheel switches.
    index = index_at_value(sm_ht, 0.0)
    if index:
        roc_tdn = my_roc[index]
        return Value(index + startpoint, roc_tdn)
    else:
        return Value(None, None)


def track_linking(pos, local_pos):
    """
    Obtain corrected tracks from takeoff phase, final approach and landing
    phase and possible intermediate approach and go-around phases, and
    compute error terms to align the recorded lat&long with each partial data
    segment.

    Takes an array of latitude or longitude position data and the equvalent
    array of local position data from ILS localizer and synthetic takeoff
    data.

    :param pos: Flight track data (latitude or longitude) in degrees.
    :type pos: np.ma.masked_array, masked from data validity tests.
    :param local_pos: Position data relating to runway or ILS.
    :type local_pos: np.ma.masked_array, masked where no local data computed.

    :returns: Position array using local_pos data where available and interpolated pos data elsewhere.

    TODO: Include last valid sample style functions to avoid trap of adjusting at a masked value.
    """
    # Where do we need to use the raw data?
    blocks = np.ma.clump_masked(local_pos)
    last = len(local_pos)

    for block in blocks:
        # Setup local variables
        a = block.start
        b = block.stop
        adj_a = 0.0
        adj_b = 0.0
        link_a = 0
        link_b = 0

        # Look at the first edge
        if a==0:
            link_a = 1
        else:
            adj_a = local_pos[a-1] - pos[a-1]

        # now the other end
        if b==last:
            link_b = 1
        else:
            adj_b = local_pos[b] - pos[b]

        fix_a = adj_a + link_a*adj_b
        fix_b = adj_b + link_b*adj_a

        if link_a ==1 or link_b == 1:
            fix = np.linspace(fix_a, fix_b, num=b-a)
        else:
            fix = np.linspace(fix_a, fix_b, num=b-a+2)[1:-1]
        local_pos[block] = pos[block] + fix
    return local_pos


def smooth_track_cost_function(lat_s, lon_s, lat, lon, hz):
    # Summing the errors from the recorded data is easy.
    from_data = np.sum((lat_s - lat)**2)+np.sum((lon_s - lon)**2)

    # The errors from a straight line are computed swiftly using convolve.
    slider=np.array([-1,2,-1])
    from_straight = np.sum(np.convolve(lat_s,slider,'valid')**2) + \
        np.sum(np.convolve(lon_s,slider,'valid')**2)

    if hz == 1.0:
        weight = 1000
    elif hz == 0.5:
        weight = 300
    elif hz == 0.25:
        weight = 100
    else:
        raise ValueError('Lat/Lon sample rate not recognised in smooth_track_cost_function.')

    cost = from_data + weight*from_straight
    return cost


def smooth_track(lat, lon, hz):
    """
    Input:
    lat = Recorded latitude array
    lon = Recorded longitude array
    hz = sample rate

    Returns:
    lat_last = Optimised latitude array
    lon_last = optimised longitude array
    Cost = cost function, used for testing satisfactory convergence.
    """

    if len(lat) <= 5:
        return lat, lon, 0.0 # Polite return of data too short to smooth.

    lat_s = np.ma.copy(lat)
    lon_s = np.ma.copy(lon)

    # Set up a weighted array that will slide past the data.
    r = 0.7
    # Values of r alter the speed to converge; 0.7 seems best.
    slider = np.ma.ones(5)*r/4
    slider[2] = 1-r

    cost_0 = float('inf')
    cost = smooth_track_cost_function(lat_s, lon_s, lat, lon, hz)

    while cost < cost_0:  # Iterate to an optimal solution.
        lat_last = np.ma.copy(lat_s)
        lon_last = np.ma.copy(lon_s)

        # Straighten out the middle of the arrays, leaving the ends unchanged.
        lat_s.data[2:-2] = np.convolve(lat_last,slider,'valid')
        lon_s.data[2:-2] = np.convolve(lon_last,slider,'valid')

        cost_0 = cost
        cost = smooth_track_cost_function(lat_s, lon_s, lat, lon, hz)

    if cost>0.1:
        logger.warn("Smooth Track Cost Function closed with cost %f.3",cost)

    return lat_last, lon_last, cost_0

def straighten_altitudes(fine_array, coarse_array, limit, copy=False):
    '''
    Like straighten headings, this takes an array and removes jumps, however
    in this case it is the fine altimeter rollovers that get corrected. We
    keep the signal in step with the coarse altimeter signal without relying
    upon that for accuracy.
    '''
    return straighten(fine_array, coarse_array, limit, copy)

def straighten_headings(heading_array, copy=True):
    '''
    We always straighten heading data before checking for spikes.
    It's easier to process heading data in this format.

    :param heading_array: array/list of numeric heading values
    :type heading_array: iterable
    :returns: Straightened headings
    :rtype: Generator of type Float
    '''
    return straighten(heading_array, None, 360.0, copy)
    #if copy:
        #heading_array = heading_array.copy()
    #head_prev = None
    #diff_prev = None
    #for clump in np.ma.clump_unmasked(heading_array):
        #head_start = heading_array.data[clump.start]
        #if diff_prev is not None:
            ## Account for rollovers within masked sections.
            #if (head_start - head_prev) > 180:
                #diff_prev -= 360  + head_prev
            #elif (head_start - head_prev) < -180:
                #diff_prev += 360 + (360 - head_prev)
            #else:
                #diff_prev += head_start - head_prev
            #head_start += diff_prev
        ## Store the last unmasked value of heading to compare with
        ## the start of the next unmasked section.
        #head_prev = heading_array[clump][-1]
        
        #diff = np.ediff1d(heading_array.data[clump])
        #diff = diff - 360.0 * np.trunc(diff / 180.0)
        #diff = np.cumsum(diff)
        #diff_prev = diff[-1]
        
        #heading_array[clump][0] = head_start
        #heading_array[clump][1:] = diff + head_start
        
    #return heading_array

def straighten(array, estimate, limit, copy):
    '''
    Basic straightening routine, used by both heading and altitude signals.
    
    :param array: array of numeric of overflowing values
    :type array: numpy masked array
    :param limit: limit value for overflow.
    :type limit: float
    :returns: Straightened parameter
    :rtype: numpy masked array
    '''
    if copy:
        array = array.copy()
    last_value = None
    for clump in np.ma.clump_unmasked(array):
        starting_value = array[clump.start]
        if estimate is not None and estimate[clump.start]:
            # Make sure we are close to the estimate at the start of each block.
            offset = estimate[clump.start] - starting_value
            if offset>0.0:
                starting_value += floor(offset / limit + 0.5) * limit
            else:
                starting_value += ceil(offset / limit - 0.5) * limit
        else:
            if last_value is not None:
                # Check that we start this section within +/- limit/2 of the
                # previous section. This situation arises when data has been
                # masked at a rollover point.
                last_half = np.trunc(last_value / (limit / 2))
                starting_half = np.trunc(starting_value / (limit / 2))
                if last_half > starting_half:
                    starting_value += limit
                elif last_half < starting_half:
                    starting_value -= limit

        diff = np.ediff1d(array[clump])
        diff = diff - limit * np.trunc(diff * 2.0 / limit)
        array[clump][0] = starting_value
        array[clump][1:] = np.cumsum(diff) + starting_value
        last_value = array[clump][-1]
    return array

def subslice(orig, new):
    """
    a = slice(2,10,2)
    b = slice(2,2)
    c = subslice(a, b)
    assert range(100)[c] == range(100)[a][b]

    See tests for capabilities.
    """
    step = (orig.step or 1) * (new.step or 1)

    # FIXME: asks DJ
    # Inelegant fix for one special case. Sorry, Glen.
    if new.start == 0:
        start = orig.start
    else:
        start = (orig.start or 0) + (new.start or orig.start or 0) * (orig.step or 1)

    stop = orig.stop if new.stop is None else \
        (orig.start or 0) + (new.stop or orig.stop or 0) * (orig.step or 1) # the bit after "+" isn't quite right!!

    return slice(start, stop, None if step == 1 else step)


def index_closest_value(array, threshold, _slice=slice(None)):
    '''
    This function seeks the moment when the parameter in question gets
    closest to a threshold. It works both forwards and backwards in time. See
    index_at_value for further details.
    '''
    return index_at_value(array, threshold, _slice, endpoint='closing')


def index_at_value(array, threshold, _slice=slice(None), endpoint='exact'):
    '''
    This function seeks the moment when the parameter in question first crosses
    a threshold. It works both forwards and backwards in time. To scan backwards
    pass in a slice with a negative step. This is really useful for finding
    things like the point of landing.

    For example, to find 50ft Rad Alt on the descent, use something like:
       idx_50 = index_at_value(alt_rad, 50.0, slice(on_gnd_idx,0,-1))

    :param array: input data
    :type array: masked array
    :param threshold: the value that we expect the array to cross in this slice.
    :type threshold: float
    :param _slice: slice where we want to seek the threshold transit.
    :type _slice: slice
    :param endpoint: type of end condition being sought.
    :type endpoint: string 'exact' requires array to pass through the threshold,
    while 'closing' seeks the last point where the array is closing on the
    threshold and 'nearest' seeks the point nearest to the threshold.

    :returns: interpolated time when the array values crossed the threshold. (One value only).
    :returns type: Float or None
    '''
    step = _slice.step or 1
    max_index = len(array)

    # Arrange the limits of our scan, ensuring that we stay inside the array.
    if step == 1:
        begin = max(int(round(_slice.start or 0)), 0)
        end = min(int(round(_slice.stop or max_index)), max_index)
        left, right = slice(begin, end - 1, step), slice(begin + 1, end,step)

    elif step == -1:
        begin = min(int(round(_slice.start or max_index)), max_index-1)
        # Indexing from the end of the array results in an array length
        # mismatch. There is a failing test to cover this case which may work
        # with array[:end:-1] construct, but using slices appears insoluble.
        end = max(int(_slice.stop or 0),0)
        left = slice(begin, end, step)
        right = slice(begin - 1, end - 1 if end > 0 else None, step)

    else:
        raise ValueError('Step length not 1 in index_at_value')

    if begin == end:
        logger.warning('No range for seek function to scan across')
        return None
    elif abs(begin - end) < 2:
        # Requires at least two values to find if the array crosses a
        # threshold.
        return None

    # When the data being tested passes the value we are seeking, the
    # difference between the data and the value will change sign.
    # Therefore a negative value indicates where value has been passed.
    value_passing_array = (array[left] - threshold) * (array[right] - threshold)
    test_array = np.ma.masked_greater(value_passing_array, 0.0)

    if len(test_array) == 0:
        # Q: Does this mean that value_passing_array is also empty?
        return None

    if (_slice.stop == _slice.start) and (_slice.start is not None):
        # No range to scan across. Special case of slice(None, None, None)
        # covers the whole array so is allowed.
        return None

    elif not np.ma.count(test_array):
        # The parameter does not pass through threshold in the period in
        # question, so return empty-handed.
        if endpoint == 'closing':
            # Rescan the data to find the last point where the array data is
            # closing.
            diff = np.ma.ediff1d(array[_slice])
            try:
                value = closest_unmasked_value(array, _slice.start or 0,
                                               _slice=_slice)[1]
            except:
                return None
            if threshold >= value:
                diff_where = np.ma.where(diff < 0)
            else:
                diff_where = np.ma.where(diff > 0)
            try:
                return (_slice.start or 0) + (step * diff_where[0][0])
            except IndexError:
                return (_slice.stop - step) if _slice.stop else len(array) - 1
        elif endpoint == 'nearest':
            closing_array = abs(array-threshold)
            return begin + step * np.ma.argmin(closing_array[_slice])
        else:
            return None

    else:
        n, dummy = np.ma.flatnotmasked_edges(test_array)
        a = array[begin + (step * n)]
        b = array[begin + (step * (n + 1))]
        # Force threshold to float as often passed as an integer.
        # Also check for b=a as otherwise we get a divide by zero condition.
        if (a is np.ma.masked or b is np.ma.masked or a == b):
            r = 0.5
        else:
            r = (float(threshold) - a) / (b - a)

    return (begin + step * (n + r))


def _value(array, _slice, operator):
    """
    Applies logic of min_value and max_value across the array slice.
    """
    if _slice.step and _slice.step < 0:
        raise ValueError("Negative step not supported")
    if np.ma.count(array[_slice]):
        # floor the start position as it will have been floored during the slice
        index = operator(array[_slice]) + floor(_slice.start or 0) * (_slice.step or 1)
        value = array[index]
        return Value(index, value)
    else:
        return Value(None, None)


def value_at_time(array, hz, offset, time_index):
    '''
    Finds the value of the data in array at the time given by the time_index.

    :param array: input data
    :type array: masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param time_index: time into the array where we want to find the array value.
    :type time_index: float
    :returns: interpolated value from the array
    :raises ValueError: From value_at_index if time_index is outside of array range.
    '''
    # Timedelta truncates to 6 digits, therefore round offset down.
    time_into_array = time_index - round(offset-0.0000005, 6)
    location_in_array = time_into_array * hz

    # Trap overruns which arise from compensation for timing offsets.
    diff = location_in_array - len(array)
    if location_in_array < 0:
        location_in_array = 0
    if diff > 0:
        location_in_array = len(array)-1

    return value_at_index(array, location_in_array)


def value_at_datetime(start_datetime, array, hz, offset, value_datetime):
    '''
    Finds the value of the data in array at the time given by value_datetime.

    :param start_datetime: Start datetime of data.
    :type start_datetime: datetime
    :param array: input data
    :type array: masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param value_datetime: Datetime to fetch the value for.
    :type value_datetime: datetime
    :returns: interpolated value from the array
    :raises ValueError: From value_at_index if value_datetime is outside of array range.
    '''
    value_timedelta = value_datetime - start_datetime
    seconds = value_timedelta.total_seconds()
    return value_at_time(array, hz, offset, seconds)


def value_at_index(array, index):
    '''
    Finds the value of the data in array at a given index.

    Samples outside the array boundaries are permitted, as we need this to
    allow for offsets within the data frame.

    :param array: input data
    :type array: masked array
    :param index: index into the array where we want to find the array value.
    :type index: float
    :returns: interpolated value from the array
    '''

    if index < 0.0:  # True if index is None
        return array[0]
    elif index > len(array)-1:
        return array[-1]

    low = int(index)
    if (low==index):
        # I happen to have arrived at exactly the right value by a fluke...
        return None if np.ma.is_masked(array[low]) else array[low]
    else:
        high = low + 1
        r = index - low
        low_value = array.data[low]
        high_value = array.data[high]
        # Crude handling of masked values. Must be a better way !
        if array.mask.any(): # An element is masked
            if array.mask[low] == True:
                if array.mask[high] == True:
                    return None
                else:
                    return high_value
            else:
                if array.mask[high] == True:
                    return low_value
        # In the cases of no mask, or neither sample masked, interpolate.
        return r*high_value + (1-r) * low_value


def vstack_params(*params):
    '''
    Create a multi-dimensional masked array with a dimension per param.

    :param params: Parameter arguments as required. Allows some None values.
    :type params: np.ma.array or Parameter object or None
    :returns: Each parameter stacked onto a new dimension
    :rtype: np.ma.array
    :raises: ValueError if all params are None (concatenation of zero-length sequences is impossible)
    '''
    return np.ma.vstack([getattr(p, 'array', p) for p in params if p is not None])


def vstack_params_where_state(*param_states):
    '''
    Create a multi-dimensional masked array with a dimension for each param,
    where the state is equal to that provided.

    :param param_states: tuples containing params or array and multistate value to match with. Allows None parameters.
    :type param_states: np.ma.array or Parameter object or None
    :returns: Each parameter stacked onto a new dimension
    :rtype: np.ma.array
    :raises: ValueError if all params are None (concatenation of zero-length sequences is impossible)
    '''
    param_arrays = []
    for param, state in param_states:
        if param is None:
            continue
        if state in param.array.state:
            array = getattr(param, 'array', param)
            param_arrays.append(array == state)
    return np.ma.vstack(param_arrays)


def second_window(array, frequency, seconds):
    '''
    Only include values which are maintained for a number of seconds, shorter
    exceedances are excluded.
    
    e.g. [0, 1, 2, 3, 2, 1, 2, 3] -> [0, 1, 2, 2, 2, 2, 2, 2]
    
    :type array: np.ma.masked_array
    '''
    if int(seconds) != seconds:
        raise ValueError('Only whole seconds are currently supported.')
    if ((seconds % 2 == 0 and not frequency % 2 == 1) or
        (seconds % 2 == 1 and not frequency % 2 == 0)):
        raise ValueError('Invalid seconds for frequency')
    
    samples = (seconds * frequency) + 1
    # TODO: Fix for frequency..
    arrays = [array]
    for roll_value in range((samples / 2) + 1):
        positive_roll = np.roll(array, roll_value)
        positive_roll[:roll_value] = np.ma.masked
        negative_roll = np.roll(array, -roll_value)
        negative_roll[-roll_value:] = np.ma.masked
        arrays.append(positive_roll)
        arrays.append(negative_roll)
    combined_array = np.ma.array(arrays)
    min_array = np.ma.min(combined_array, axis=0)
    max_array = np.ma.max(combined_array, axis=0)
    window_array = np_ma_masked_zeros_like(array)
    unmasked_slices = np.ma.clump_unmasked(array)
    for unmasked_slice in unmasked_slices:
        last_value = array[unmasked_slice.start]
        algo_slice = slice(unmasked_slice.start + (samples / 2),
                           unmasked_slice.stop)
        zipped_arrays = zip(array[algo_slice],
                            min_array[algo_slice],
                            max_array[algo_slice])
        for index, (array_value,
                    min_window,
                    max_window) in enumerate(zipped_arrays,
                                             start=unmasked_slice.start):
            if array_value is np.ma.masked:
                continue
            if min_window < last_value < max_window:
                # Mixed
                window_array[index] = last_value
            elif max_window > last_value:
                # All greater than.
                window_array[index] = last_value = min_window
            elif min_window < last_value:
                # All less than
                window_array[index] = last_value = max_window
            else:
                window_array[index] = last_value
        #try:
            #first_index = np.ma.clump_unmasked(array)[0].start
        #except IndexError:
            ## array is entirely masked?
            #return window_array
        ##np.ma.array([array, max_array, min_array])
        
        #window_array[first_index] = last_value = array[first_index]
        
        #for index, (array_value,
                    #min_window,
                    #max_window) in enumerate(zip(array[first_index + 1:],
                                                 #min_array[first_index + 1:],
                                                 #max_array[first_index + 1:]),
                                             #start=first_index):
        ##stacked_array = np.ma.array([array, max_array, min_array])
        ###for index, values in enumerate(tacked_array[], start=first_index):
        ##for index in xrange(first_index, stacked_array.shape[1]):
        ##values = stacked_array[...,index]
        ##array_value, max_window, min_window = values.tolist()
    
    return np.ma.array(window_array)


#---------------------------------------------------------------------------
# Air data calculations adapted from AeroCalc V0.11 to suit POLARIS Numpy
# data format. For increased speed, only standard POLARIS units used.
#
# AeroCalc is Copyright (c) 2008, Kevin Horton and used under open source
# license with permission. For copyright notice and disclaimer, please see
# airspeed.py source code in AeroCalc.
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Initialise constants used by the air data algorithms
#---------------------------------------------------------------------------
P0 = 1013.25       # Pressure at sea level, mBar
Rhoref = 1.2250    # Density at sea level, kg/m**3
A0 = 340.2941      # Speed of sound at sea level, m/s
T0 = 288.15        # Sea level temperature 15 C = 288.15 K
L0 = -0.0019812    # Lapse rate C/ft
g = 9.80665        # Acceleration due to gravity, m/s**2
Rd = 287.05307     # Gas constant for dry air, J/kg K
H1 = 36089.0       # Transition from Troposphere to Stratosphere

# Values at 11km:
T11 =  T0 + 11000 * L0
PR11 = (T11 / T0) ** ((-g) / (Rd * L0))
P11 = PR11 * P0

#---------------------------------------------------------------------------
# Computation modules use AeroCalc structure and are called from the Derived
# Parameters as required.
#---------------------------------------------------------------------------

def alt2press(alt_ft):
    press = P0  * alt2press_ratio(alt_ft)
    return press

def alt2press_ratio(alt_ft):
    return np.ma.where(alt_ft <= H1, \
                       _alt2press_ratio_gradient(alt_ft),
                       _alt2press_ratio_isothermal(alt_ft))

def cas2dp(cas_kt):
    """
    Convert corrected airspeed to pressure rise (includes allowance for
    compressibility)
    """
    if np.ma.max(cas_kt) > 661.48:
        raise ValueError('Supersonic airspeed compuations not included')
    cas_mps = np.ma.masked_greater(cas_kt, 661.48) * KTS_TO_MPS
    p = P0*100 # pascal not mBar inside the calculation
    return P0 * (((Rhoref * cas_mps*cas_mps)/(7.* p) + 1.)**3.5 - 1.)

def cas_alt2mach(cas, alt_ft):
    """
    Return the mach that corresponds to a given CAS and altitude.
    """
    dp = cas2dp(cas)
    p = alt2press(alt_ft)
    dp_over_p = dp / p
    mach = dp_over_p2mach(dp_over_p)
    return mach

def dp_over_p2mach(dp_over_p):
    """
    Return the mach number for a given delta p over p. Supersonic results masked as invalid.
    """
    mach = np.sqrt(5.0 * ((dp_over_p + 1.0) ** (2.0/7.0) - 1.0))
    return np.ma.masked_greater_equal(mach, 1.0)

def _dp2speed(dp, P, Rho):

    p = P*100 # pascal not mBar inside the calculation
    # dp / P not changed as we use mBar for pressure dp.
    speed_mps = np.ma.sqrt(((7. * p) * (1. / Rho)) * (
        np.ma.power((dp / P + 1.), 2./7.) - 1.))
    speed_kt = speed_mps / KTS_TO_MPS

    # Mask speeds over 661.48 kt
    return np.ma.masked_greater(speed_kt, 661.48)

def dp2cas(dp):
    return np.ma.masked_greater(_dp2speed(dp, P0, Rhoref), 661.48)

def dp2tas(dp, alt_ft, sat):
    P = alt2press(alt_ft)
    press_ratio = alt2press_ratio(alt_ft)
    temp_ratio = (sat + 273.15) / 288.15
    # FIXME: FloatingPointError: underflow encountered in multiply
    density_ratio = press_ratio / temp_ratio
    Rho = Rhoref * density_ratio
    tas = _dp2speed(dp, P, Rho)
    return tas

def alt2sat(alt_ft):
    """ Convert altitude to temperature using lapse rate"""
    return np.ma.where(alt_ft <= H1, 15.0 + L0 * alt_ft, -56.5)

def machtat2sat(mach, tat, recovery_factor=0.995):
    """
    Return the ambient temp, given the mach number, indicated temperature and the
    temperature probe's recovery factor.

    Recovery factor is taken from the BF Goodrich Model 101 and 102 Total
    Temperature Sensors data sheet. As "...the world's leading supplier of
    total temperature sensors" it is likely that a sensor of this type, or
    comparable, will be installed on monitored aircraft.
    """
    # Default fill of zero produces runtime divide by zero errors in Numpy.
    # Hence force fill to >0.
    denominator = np.ma.array(1.0 + (0.2*recovery_factor) * mach * mach, fill_value=1.0)
    ambient_temp = (tat + 273.15) / denominator
    sat = ambient_temp - 273.15
    return sat

def _alt2press_ratio_gradient(H):
    # From http://www.aerospaceweb.org/question/atmosphere/q0049.shtml
    # Faster to compute than AeroCalc formulae, and pass AeroCalc tests.
    return np.ma.power(1 - H/145442.0, 5.255876)

def _alt2press_ratio_isothermal(H):
    # FIXME: FloatingPointError: overflow encountered in exp
    return 0.223361 * np.ma.exp((36089.0-H)/20806.0)

def is_day(when, latitude, longitude, twilight='civil'):
    """
    This simple function takes the date, time and location of any point on
    the earth and return True for day and False for night.

    :param when: Date and time in datetime format
    :param longitude: Longitude in decimal degrees, east is positive
    :param latitude: Latitude in decimal degrees, north is positive
    :param twilight: optional twilight setting. Default='civil', None, 'nautical' or 'astronomical'.

    :raises ValueError if twilight not recognised.

    :returns boolean True = daytime (including twilight), False = nighttime.

    This function is drawn from Jean Meeus' Astronomial Algorithms as
    implemented by Michel J. Anders. In accordance with his Collective
    Commons license, the reworked function is being released under the OSL
    3.0 license by FDS as a part of the POLARIS project.

    For FDM purposes, the actual time of sunrise and sunset is of no
    interest, so function 12.6 is adapted to give just the day/night
    decision, with allowance for different, generally recognised, twilight
    tolerances.

    FAA Regulation FAR 1.1 defines night as: "Night means the time between
    the end of evening civil twilight and the beginning of morning civil
    twilight, as published in the American Air Almanac, converted to local
    time.

    EASA EU OPS 1 Annex 1 item (76) states: 'night' means the period between
    the end of evening civil twilight and the beginning of morning civil
    twilight or such other period between sunset and sunrise as may be
    prescribed by the appropriate authority, as defined by the Member State;

    CAA regulations confusingly define night as 30 minutes either side of
    sunset and sunrise, then include a civil twilight table in the AIP.

    With these references, it was decided to make civil twilight the default.
    """
    if latitude is np.ma.masked or longitude is np.ma.masked:
        return np.ma.masked
    day = when.toordinal() - (734124-40529)
    t = when.time()
    time = (t.hour + t.minute/60.0 + t.second/3600.0)/24.0
    # Julian Day
    Jday     = day+2415019.5 + time
    # Julian Century
    Jcent    = (Jday-2451545.0)/36525  # (24.1)
    # Siderial time at Greenwich (11.4)
    Gstime   = (280.46061837 + 360.98564736629*(Jday-2451545.0) + (0.0003879331-Jcent/38710000) * Jcent * Jcent)%360.0
    # Geom Mean Long Sun (deg)
    Mlong    = (280.46645+Jcent*(36000.76983+Jcent*0.0003032))%360 # 24.2
    # Geom Mean Anom Sun (deg)
    Manom    = 357.52910+Jcent*(35999.05030-Jcent*(0.0001559+0.00000048*Jcent)) # 24.3
    # Eccent Earth Orbit
    ##### XXX: The following line is unused. Remove?
    ####Eccent   = 0.016708617-Jcent*(0.000042037+0.0000001236*Jcent) # 24.4 (significantly changed from web version)
    # Sun Eq of Ctr
    Seqcent  = sin(radians(Manom))*(1.914600-Jcent*(0.004817+0.000014*Jcent))+sin(radians(2*Manom))*(0.019993-0.000101*Jcent)+sin(radians(3*Manom))*0.000290 # p152
    # Sun True Long (deg)
    Struelong= Mlong+Seqcent # Theta on p152
    # Mean Obliq Ecliptic (deg)
    Mobliq   = 23+(26+((21.448-Jcent*(46.815+Jcent*(0.00059-Jcent*0.001813))))/60)/60  # 21.2
    # Obliq Corr (deg)
    obliq    = Mobliq + 0.00256*cos(radians(125.04-1934.136*Jcent))  # 24.8
    # Sun App Long (deg)
    Sapplong = Struelong-0.00569-0.00478*sin(radians(125.04-1934.136*Jcent)) # Omega, Lambda p 152.
    # Sun Declin (deg)
    declination = degrees(asin(sin(radians(obliq))*sin(radians(Sapplong)))) # 24.7
    # Sun Rt Ascen (deg)
    rightasc = degrees(atan2(cos(radians(Mobliq))*sin(radians(Sapplong)),cos(radians(Sapplong))))

    elevation = degrees(asin(sin(radians(latitude))*sin(radians(declination)) +
                    cos(radians(latitude))*cos(radians(declination))*cos(radians(Gstime+longitude-rightasc))))

    # Solar diamteter gives an adjustment of 0.833 deg, as the rim of the sun
    # appears before the centre of the disk.
    if twilight == None:
        limit = -0.8333 # Allows for diameter of sun's disk
    # For civil twilight, allow 6 deg
    elif twilight == 'civil':
        limit = -6.0
    # For nautical twilight, allow 12 deg
    elif twilight == 'nautical':
            limit = -12.0
    # For astronomical twilight, allow 18 deg
    elif twilight == 'astronomical':
            limit = -18.0
    else:
        raise ValueError('is_day called with unrecognised twilight zone')

    if elevation > limit:
        return True # It is Day
    else:
        return False # It is Night
