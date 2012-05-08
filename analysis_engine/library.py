import numpy as np
import logging

from math import floor, sqrt, sin, cos, atan2, radians
from collections import OrderedDict, namedtuple
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import izip
from settings import KTS_TO_MPS, METRES_TO_FEET
##from scipy.optimize import fmin, fmin_bfgs, fmin_tnc
# TODO: Inform Enthought that fmin_l_bfgs_b dies in a dark hole at _lbfgsb.setulb

from settings import REPAIR_DURATION, TRUCK_OR_TRAILER_INTERVAL, TRUCK_OR_TRAILER_PERIOD

Value = namedtuple('Value', 'index value')

class InvalidDatetime(ValueError):
    pass


def align(slave, master, interval='Subframe', data_type=None):
    """
    This function takes two parameters which will have been sampled at different
    rates and with different offsets, and aligns the slave parameter's samples
    to match the master parameter. In this way the master and aligned slave data
    may be processed without timing errors.
    
    The values of the returned array will be those of the slave 
    parameter, aligned to the master and adjusted by linear interpolation. The initial
    or final values will be extended from the first or last values if they lie 
    outside the timebase of the slave parameter (i.e. we do not extrapolate).
    The offset and hz for the returned masked array will be those of the 
    master parameter.
    
    The slave's data_type is used to determine the method of interpolation.
    Anything other than discrete or multi-state will result in interpolation
    of the data across each sample period
 
    WARNING! Not tested with ascii data_type.
    
    :param slave: The parameter to be aligned to the master
    :type slave: Parameter objects
    :param master: The master parameter
    :type master: Parameter objects    
    :param interval: Has possible values 'Subframe' or 'Frame'.  #TODO: explain this!
    :type interval: String
    :param data_type: Overrides the slave data_type for interpolation method. 
    :type data_type: str
    :type interval: String
    
    :raises AssertionError: If the interval is neither 'Subframe' or 'Frame'
    :raises AssertionError: If the arrays and sample rates do not equate to the same overall data duration.
    
    :returns: Slave array aligned to master.
    :rtype: np.ma.array
    """
    slave_array = slave.array # Optimised access to attribute.
    if len(slave_array) == 0:
        # No elements to align, avoids exception being raised in the loop below.
        return slave_array
    if slave.frequency == master.frequency and slave.offset == master.offset:
        # No alignment is required, return the slave's array unchanged.
        return slave_array
    
    # Check the interval is one of the two forms we recognise
    assert interval in ['Subframe', 'Frame']
    
    if data_type is None and slave.data_type:
        data_type = slave.data_type
    
    # Get the timing offsets, comprised of word location and possible latency.
    tp = master.offset
    ts = slave.offset

    # Get the sample rates for the two parameters
    wm = master.frequency
    ws = slave.frequency
    slowest = min(wm,ws)
    
    #---------------------------------------------------------------------------
    # Section to handle superframe parameters, in master, slave or both.
    # Note: We do not interpolate at these low sample rates.
    #---------------------------------------------------------------------------
    if slowest < 0.25:
        # One or both parameters is in a superframe. Handle without
        # interpolation as these parameters are recorded at too low a rate
        # for interpolation to be meaningful.
        if wm==ws:
            # All we need do is copy the data across as they are at the same
            # sample rate.
            slave_aligned=np.ma.copy(slave.array)
            return slave_aligned

        if wm>ws:
            # Increase samples in slave accordingly
            r = wm/ws
            assert r in [2,4,8,16,32,64,128,256]
            slave_aligned = np.ma.repeat(slave.array, r)
            return slave_aligned

        else:
            # Reduce samples in slave.
            r = ws/wm
            assert r in [2,4,8,16,32,64,128,256]
            slave_aligned=np.ma.empty_like(master.array)
            slave_aligned=slave_array[::r]
            return slave_aligned
    #---------------------------------------------------------------------------
            

    # Express the timing disparity in terms of the slave paramter sample interval
    delta = (tp-ts)*slave.frequency

    # If the slowest sample rate is less than 1 Hz, we extend the period and
    # so achieve a lowest rate of one per period.
    if slowest<1:
        wm /= slowest
        ws /= slowest
        
    # Check the values are in ranges we have tested
    assert wm in [1,2,4,8,16,32,64]
    assert ws in [1,2,4,8,16,32,64]
           
    # Compute the sample rate ratio:
    r = wm/float(ws)
    
    # Here we create a masked array to hold the returned values that will have 
    # the same sample rate and timing offset as the master
    slave_aligned = np.ma.empty(len(slave_array) * r)

    # Each sample in the master parameter may need different combination parameters
    for i in range(int(wm)):
        bracket=(i/r+delta)
        # Interpolate between the hth and (h+1)th samples of the slave array
        h=int(floor(bracket))
        h1 = h+1

        # Compute the linear interpolation coefficients, b & a
        b = bracket-h
        
        # Cunningly, if we are working with discrete or multi-state parameters, 
        # by reverting to 1,0 or 0,1 coefficients we gather the closest value
        # in time to the master parameter.
        if data_type and data_type.lower() in ('discrete', 'multi-state'):
            b = round(b)
            
        # Either way, a is the residual part.    
        a=1-b
        
        if h<0:
            slave_aligned[i+wm::wm]=a*slave_array[h+ws:-ws:ws]+b*slave_array[h1+ws::ws]
            # We can't interpolate the inital values as we are outside the 
            # range of the slave parameters. Take the first value and extend to 
            # the end of the data.
            slave_aligned[i] = slave_array[0]
        elif h1>=ws:
            slave_aligned[i:-wm:wm]=a*slave_array[h:-ws:ws]+b*slave_array[h1::ws]
            # At the other end, we run out of slave parameter values so need to
            # extend to the end of the array.
            slave_aligned[i-wm] = slave_array[-1]
        else:
            # Sheer bliss. We can compute slave_aligned across the whole
            # range of the data without having to take special care at the
            # ends of the array.
            slave_aligned[i::wm]=a*slave_array[h::ws]+b*slave_array[h1::ws]

    return slave_aligned

def alt_rad_non_linear(array, sensor_type):
    """
    Nonlinear conversion for radio altimeters.
    :param array: Input data array, as raw binary count from FDR data file.
    :type array: Numpy masked array, unsigned integers stored as floats.
    :param sensor_type: Identifier for the type of radio altimeter installed.
    :type sensor_type: String - see code for details of sensors recognised.
    
    :returns altitude_radio
    :type Numpy masked array of converted altitudes, in feet.
    """
    if sensor_type in ['D226A101_1_16D']:
        # Reference Boeing document D226A101-1, Note 16D.
        # This type uses an ARINC 573 HLDC input where 32V=4095 bits full scale.
        # Interestingly, this input is not listed in the ARINC 717 spec.
        ratio = 32.0/4095.0
        volts = array * ratio
        height = np.ma.where(volts<10.0,
                             (volts-0.4)/0.02,
                             500*np.ma.exp(volts/10.0)/np.exp(1.0)-20.0)
        return np.ma.masked_outside(height, -20, 2500)
    
    raise ValueError,'alt_rad_non_linear called with unrecognised sensor_type'

def bearings_and_distances(latitudes, longitudes, reference):
    """
    Returns the bearings and distances of a track with respect to a fixed point.
    
    Usage: 
    brg[], dist[] = bearings_and_distances(lat[], lon[], {'latitude':lat_ref, 'longitude', lon_ref})
    
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

    lat_array = np.ma.array(data=np.deg2rad(latitudes.data),mask=latitudes.mask)
    lon_array = np.ma.array(data=np.deg2rad(longitudes.data),mask=longitudes.mask)
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])
    
    dlat = lat_ref-lat_array
    dlon = lon_ref-lon_array
    
    a = np.ma.sin(dlat/2) * np.ma.sin(dlat/2) + \
        np.ma.cos(lat_array) * np.ma.cos(lat_ref) * np.ma.sin(dlon/2) * np.ma.sin(dlon/2)
    dists = 2 * np.ma.arctan2(np.ma.sqrt(a), np.ma.sqrt(1.0-a))
    dists *= 6371000 # Earth radius in metres

    
    y = np.ma.sin(dlon) * np.ma.cos(lat_ref)
    x = np.ma.cos(lat_array) * np.ma.sin(lat_ref) \
        - np.ma.sin(lat_array) * np.ma.cos(lat_ref) * np.ma.cos(dlon)
    brgs = np.ma.arctan2(-y,-x)
    
    joined_mask = np.logical_or(latitudes.mask, longitudes.mask)
    brg_array = np.ma.array(data = np.rad2deg(brgs),mask = joined_mask)
    dist_array = np.ma.array(data = dists,mask = joined_mask)

    return brg_array, dist_array


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
    clock_variation = OrderedDict() # Ordered so if all values are the same, max will consistently take the first val
    
    #22.02.2012 - below no longer seems to be required.
    ##length = len(mins)
    ##if years is None or not len(years):
        ##logging.warning("Year not supplied, filling in with 1970")
        ##years = np.repeat([1970], length) # force invalid year
    ##if months is None or not len(months):
        ##logging.warning("Month not supplied, filling in with 01")
        ##months = np.repeat([01], length) # force invalid month
    ##if days is None or not len(days):
        ##logging.warning("Day not supplied, filling in with 01")
        ##days = np.repeat([01], length) # force invalid days
        
    for step, (yr, mth, day, hr, mn, sc) in enumerate(izip(years, months, days, hours, mins, secs)):
        
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

# calculate here to avoid repitition
CURRENT_YEAR = str(datetime.now().year)
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
    :param x: independent variable
    :type x: numpy float array. Where not supplied, a linear scale is created.
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
        raise ValueError, 'Function coreg called with data of length 1 or null'
    if indep_var == None:
        x = np.ma.arange(n, dtype=float)
    else:
        x = indep_var
        if len(x) != n:
            raise ValueError, 'Function coreg called with arrays of differing length'
    if x.ptp() == 0.0 or y.ptp() == 0.0:
        # raise ValueError, 'Function coreg called with invariant independent variable'
        return 0.0, 0.0, 0.0
    
    # Need to propagate masks into both arrays equally.
    mask = np.ma.logical_or(x.mask, y.mask)
    x_ = np.ma.array(data=x.data,mask=mask)
    y_ = np.ma.array(data=y.data,mask=mask)

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
        raise ValueError, 'Phase mask index out of range'
            
    ib = int((b-offset)*hz) + 1
    if ib < 0 or ib > length:
        raise ValueError, 'Phase mask index out of range'

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


def cycle_counter(array, min_step, cycle_time, hz, array_offset):
    '''
    Counts the number of consecutive cycles, each with a period not more than
    cycle_time seconds, and with variation greater than min_step.

    :param array: time series data 
    :type array: Numpy masked array
    :param min_step: Optional minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param cycle_time: Maximum time for a complete valid cycle 
    :type cycle_time: float, seconds
    :param hz: array sample rate
    :type hz: float, Hz
    :param array_offset: Index to start of array
    :type array_offset: integer
    
    :returns max_index: index of the array element at the end of the highest number of cycles
    :type max_iundex: integer
    :returns max_cycles: the highest number of cycles in the array.
    :type max_cycles: float (counts a half cycle for each change over min_step)
    
    Note - Where two events with the same cycle count arise in the same
    array, the latter is recorded as it is normally the later in the flight
    that will be most hazardous.
    '''
    idxs, vals = cycle_finder(array, min_step=min_step)
    if idxs == None:
        return None, None
    
    half_cycle_times = np.ediff1d(idxs) / hz
    half_cycles = 0
    max_half_cycles = 0
    for num, half_cycle_time in enumerate(half_cycle_times):
        if half_cycle_time < cycle_time:
            half_cycles += 1
            index = idxs[num+1]
        else:
            if 0 < half_cycles >= max_half_cycles:
               max_half_cycles = half_cycles
               max_index = index
               half_cycles = 0
               
    if 0 < half_cycles >= max_half_cycles:
       max_half_cycles = half_cycles
       max_index = index
    
    if max_half_cycles == 0:
        return None, None
    return array_offset+max_index, max_half_cycles/2.0 
    
    
def cycle_finder(array, min_step=0.0, include_ends=True):
    '''
    Simple implementation of a peak detection algorithm with small cycle remover.

    :param array: time series data 
    :type array: Numpy masked array
    :param min_step: Optional minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param include_ends: Decides whether the first and last points of the array are to be included as possible turning points
    :type include_ends: logical
    
    :returns: List of peak value, index tuples.
    '''
    
    if len(array) == 0:
        # Nothing to do, so return None.
        return None, None
    
    # Find the peaks and troughs by difference products which change sign.
    x = np.ma.ediff1d(array, to_begin=0.0)
    peak = -x[:-1]*x[1:]
    idxs = np.nonzero(np.ma.maximum(peak,0.0))
    vals = array.data[idxs]
    
    # Optional inclusion of end points
    if include_ends:
        idxs = np.insert(idxs, 0, 0)
        vals = np.insert(vals,0,array.data[0])
        idxs = np.append(idxs, len(array))
        vals = np.append(vals,array.data[-1])

    dvals = np.ediff1d(vals)
    while len(dvals) > 0 and np.min(abs(dvals)) < min_step:
        sort_idx = np.argmin(abs(dvals))
        last = len(dvals)
        step = dvals[sort_idx]
        if sort_idx == 0:
            idxs = np.delete(idxs, 0)
            vals = np.delete(vals, 0)
            dvals = np.delete(dvals, 0)
        elif sort_idx == last-1:
            idxs = np.delete(idxs, last)
            vals = np.delete(vals, last)
            dvals = np.delete(dvals, last-1) # One fewer dval than val
        else:
            idxs = np.delete(idxs, slice(sort_idx,sort_idx+2))
            vals = np.delete(vals, slice(sort_idx,sort_idx+2))
            dvals[sort_idx-1] += dvals[sort_idx] + dvals[sort_idx+1]
            dvals = np.delete(dvals, slice(sort_idx,sort_idx+2))
    if len(dvals) == 0:
        # All the changes have disappeared, so return None rather than a
        # single residual array value.
        return None, None
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
    

# Previously known as Duration
def clip(array, period, hz=1.0, remove='peaks'):
    '''
    This function clips the maxima of a data array such that the 
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
    :param remove: type of data to clip.
    :type remove: string, default to 'peaks' option 'troughs'
    '''

    if hz <= 0.01:
        raise ValueError('Duration called with sample rate outside permitted range')

    delay = (int(period * hz)-1)/2
    # Trap low values. This can occur, for an example, where a parameter has
    # a lower sample rate than expected.
    if delay < 1:
        raise ValueError('Duration called with period too short to have an effect')
    if remove not in ['peaks', 'troughs']:
        raise ValueError('Clip called with unrecognised removal mode')
    
    # Preparation of convenient numbers and data to process
    width = len(array) - 2*delay
    source = repair_mask(array, frequency=hz, repair_duration=period-(1/hz))
    result = np.ma.copy(source)
    
    for step in range(2*delay+1):
        if remove == 'peaks':
            result[delay:-delay] = np.ma.minimum(result[delay:-delay], source[step:step+width])
        else:
            result[delay:-delay] = np.ma.maximum(result[delay:-delay], source[step:step+width])

    # Stretch the ends out and return the answer.
    result[:delay] = result[delay]
    result[-delay:] = result[-(delay+1)]
    return result

"""
    # Compute an array of differences across period, such that each maximum or
    # minimum results in a negative result.
    b = (a[:-delay]-a[delay-1:-1]) * (a[1:1-delay]-a[delay:])
    
    # We now remove the positive values (as these are of no interest), sort the 
    # list to put the negative values first, then index through the arguments: 
    for index in b.clip(max = 0).argsort():
        if b[index]<0: # Data has passed through this level for delay samples.
            if a[index+1] > a[index]:
                # We are truncating a peak, so find the higher end value
                #  TODO: could interpolate ends.
                level = min(a[index], a[index+delay])
                # Replace the values with the preceding value to trim the maxima to those
                # values which are present for at least the required period.
                a[index:index+delay+1] = level
            else:
                # We are truncating a trough, so find the lower end value
                #  TODO: could interpolate ends as above.
                level = max(a[index], a[index+delay])
                a[index:index+delay+1] = level
        else:
            break # No need to process the rest of the array.
    return a
    """

def find_edges(array, start_index, direction='rising_edges'):
    '''
    Edge finding low level routine, called by create_ktis_at_edges and
    create_kpvs_at_edges. Also useful within algorithms directly.
    
    :param array: array of values to scan for edges
    :type array: Numpy masked array (what else?!)
    :param start_index: offset to the start of the array
    :type start_index: float
    :param direction: Optional edge direction for sensing. Default 'rising_edges'
    :type direction: string, one of 'rising_edges', 'falling_edges' or 'all_edges'.
    
    :returns edge_list: Indexes for the appropriate edge transitions. 
    :type edge_list: list of floats. 
    
    Note: edge_list values are always integer+0.5 as it is assumed that the
    transition took place (with highest probability) midway between the two
    recorded states.
    '''
    # Find increments. Extrapolate at start to keep array sizes straight.
    deltas = np.ma.ediff1d(array, to_begin=array[0])
    deltas[0]=0 # Ignore the first value 
    if direction == 'rising_edges':
        edges = np.ma.nonzero(np.ma.maximum(deltas,0))
    elif direction == 'falling_edges':
        edges = np.ma.nonzero(np.ma.minimum(deltas,0))
    elif direction == 'all_edges':
        edges = np.ma.nonzero(deltas)
    else:
        raise ValueError, 'Edge direction not recognised'
    
    # edges is a tuple catering for multi-dimensional arrays, but we
    # are only interested in 1-D arrays, hence selection of the first
    # element only. 
    # The -0.5 shifts the value midway between the pre- and post-change
    # samples.
    edge_list = edges[0] + int(start_index) - 0.5
    return list(edge_list)

def first_order_lag (in_param, time_constant, hz, gain = 1.0,
                     initial_value = None):
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
    
    :param in_param: input data (x)
    :type in_param: masked array
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
        raise ValueError, 'Lag timeconstant too small'
    
    x_term = []
    x_term.append (gain / (1.0 + 2.0*tc)) #b[0]
    x_term.append (gain / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)
    
    y_term = []
    y_term.append (1.0) #a[0] 
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)
    
    return masked_first_order_filter(y_term, x_term, in_param, initial_value)

def masked_first_order_filter(y_term, x_term, in_param, initial_value):
    """
    This provides access to the scipy filter function processed across the
    unmasked data blocks, with masked data retained as masked zero values.
    This is a better option than masking all subsequent values which would be
    the mathematically correct thing to do with infinite response filters.
    
    :param y_term: Filter denominator terms. 
    :type in_param: list 
    :param x_term: Filter numerator terms.
    :type x_term: list
    :param in_param: input data array
    :type in_param: masked array
    :param initial_value: Value to be used at the start of the data
    :type initial_value: float (or may be None)
    """
    # import locally to speed up imports of library.py
    from scipy.signal import lfilter, lfilter_zi
    z_initial = lfilter_zi(x_term, y_term) # Prepare for non-zero initial state
    # The initial value may be set as a command line argument, mainly for testing
    # otherwise we set it to the first data value.
    
    result = np.ma.zeros(len(in_param))  # There is no zeros_like method.
    good_parts = np.ma.clump_unmasked(in_param)
    for good_part in good_parts:
        
        if initial_value == None:
            initial_value = in_param[good_part.start]
        # Tested version here...
        answer, z_final = lfilter(x_term, y_term, in_param[good_part], zi=z_initial*initial_value)
        result[good_part] = np.ma.array(answer)
        
    # The mask should last indefinitely following any single corrupt data point
    # but this is impractical for our use, so we just copy forward the original
    # mask.
    bad_parts = np.ma.clump_masked(in_param)
    for bad_part in bad_parts:
        # The mask should last indefinitely following any single corrupt data point
        # but this is impractical for our use, so we just copy forward the original
        # mask.
        result[bad_part] = np.ma.masked
        
    return result
    
def first_order_washout (in_param, time_constant, hz, gain = 1.0,
                         initial_value = None):
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
    
    :param in_param: input data (x)
    :type in_param: masked array
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
    #input_data = np.copy(in_param.data)
    
    # Scale the time constant to allow for different data sample rates.
    tc = time_constant * hz
    
    # Trap the condition for stability
    if tc < 0.5:
        raise ValueError, 'Lag timeconstant too small'
    
    x_term = []
    x_term.append (gain*2.0*tc  / (1.0 + 2.0*tc)) #b[0]
    x_term.append (-gain*2.0*tc / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)
    
    y_term = []
    y_term.append (1.0) #a[0] 
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)
    
    return masked_first_order_filter(y_term, x_term, in_param, initial_value)


def _dist(lat1_d, lon1_d, lat2_d, lon2_d):
    """
    Haversine formula for calculating distances between coordinates.
    """
    lat1 = radians(lat1_d)
    lon1 = radians(lon1_d)
    lat2 = radians(lat2_d)
    lon2 = radians(lon2_d)

    dlat = lat2-lat1
    dlon = lon2-lon1

    a = sin(dlat/2) * sin(dlat/2) + cos(lat2) \
        * (lat2) * (dlon/2) * sin(dlon/2)
    return 2 * atan2(sqrt(a), sqrt(1-a)) * 6371000

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
    :type start_loc: float, units = feet.
    :param gs_loc: distance from projected position of glideslope antenna on runway centerline to the localizer antenna
    :type gs_loc: float, units = ??
    :param end_loc: distance from end of runway to localizer antenna
    :type end_loc: float, units = ??
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
    
    a = _dist(gs_lat, gs_lon, lzr_lat, lzr_lon)
    b = _dist(gs_lat, gs_lon, start_lat, start_lon)
    c = _dist(end_lat, end_lon, lzr_lat, lzr_lon)
    d = _dist(start_lat, start_lon, lzr_lat, lzr_lon)
    
    r = (1.0+(a**2 - b**2)/d**2)/2.0
    g = r*d
    
    # The projected glideslope antenna position is given by this formula
    pgs_lat = lzr_lat + r*(start_lat - lzr_lat)
    pgs_lon = lzr_lon + r*(start_lon - lzr_lon)
    
    return d, g, c, pgs_lat, pgs_lon  # Runway distances to start, glideslope and end.

def runway_length(runway):
    '''
    Calculation of only the length for runways with no glideslope details
    and possibly no localizer information. In these cases we assume the
    localizer is near end of runway and the beam is 700ft wide at the
    threshold.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
        
    :return
    :param start_end: distance from start of runway to end
    :type start_loc: float, units = feet.
    '''
    
    start_lat = runway['start']['latitude']
    start_lon = runway['start']['longitude']
    end_lat = runway['end']['latitude']
    end_lon = runway['end']['longitude']
    
    return _dist(start_lat, start_lon, end_lat, end_lon)

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
    '''
    end_lat = runway['end']['latitude']
    end_lon = runway['end']['longitude']
    
    brg, dist = bearings_and_distances(np.ma.array(end_lat),
                                       np.ma.array(end_lon),
                                       runway['start'])
    return brg.data    

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
    :param frequency: Frequency of the groundspeed and heading data
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
        raise ValueError, 'Ground_track requires equi-length groundspeed and heading arrays'
    
    # Dummy masked array to join the invalid data arrays
    result=np.ma.array(np.zeros_like(gspd))    
    result.mask = np.ma.logical_or(np.ma.getmaskarray(gspd),
                                   np.ma.getmaskarray(hdg))
    # It's not worth doing anything if there is too little data
    if np.ma.count(result) < 5:
        return None, None
    
    repair_mask(gspd, repair_duration=None)
    repair_mask(hdg, repair_duration=None)
    
    if mode == 'takeoff':
        direction = 'backwards'
    elif mode == 'landing':
        direction = 'forwards'
    else:
        raise ValueError,'Ground_track only recognises takeoff or landing modes'
    
    delta_north = np.ma.array(gspd*np.cos(np.deg2rad(hdg.data)))
    delta_east = np.ma.array(gspd*np.sin(np.deg2rad(hdg.data)))
    
    north = integrate(delta_north, frequency, scale=KTS_TO_MPS, direction=direction)
    east = integrate(delta_east, frequency, scale=KTS_TO_MPS, direction=direction)
    
    bearing = np.ma.array(np.rad2deg(np.arctan2(east, north)))
    distance = np.ma.array(np.sqrt(north**2 + east**2))
    distance.mask = result.mask
    
    lat, lon = latitudes_and_longitudes(bearing, distance, 
                                        {'latitude':lat_fix, 'longitude':lon_fix})
    return lat, lon
         
def hash_array(array):
    '''
    Creates a sha256 hash from the array's tostring() method.
    '''
    checksum = sha256()
    checksum.update(array.tostring())
    return checksum.hexdigest()

def hysteresis (array, hysteresis):
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
    
    start_lat = runway['start']['latitude']
    start_lon = runway['start']['longitude']
    end_lat = runway['end']['latitude']
    end_lon = runway['end']['longitude']
    lzr_lat = runway['localizer']['latitude']
    lzr_lon = runway['localizer']['longitude']
    
    a = _dist(lzr_lat, lzr_lon, end_lat, end_lon)
    b = _dist(lzr_lat, lzr_lon, start_lat, start_lon)
    d = _dist(start_lat, start_lon, end_lat, end_lon)
    
    r = (1.0+(a**2 - b**2)/d**2)/2.0
    
    # The projected glideslope antenna position is given by this formula
    new_lat = end_lat + r*(start_lat - end_lat)
    new_lon = end_lon + r*(start_lon - end_lon)
    
    return {'latitude':new_lat, 'longitude':new_lon}  # Runway distances to start, glideslope and end.
    
    
def integrate (array, frequency, initial_value=0.0, scale=1.0, direction="forwards"):
    """
    Trapezoidal integration
    
    Usage example: 
    feet_to_land = integrate(airspeed[:touchdown], scale=KTS_TO_FPS, direction='reverse')

    :param array: Integrand.
    :type array: Numpy masked array.
    :param frequency: Sample rate of the integrand.
    :type frequency: Float
    :param initial_value: Initial falue for the integral
    :type initial_value: float
    :param scale: Scaling factor, default = 1.0
    :type scale: float
    :param direction: Optional integration sense, default = 'forwards'
    
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
    result = np.copy(array)
    
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
        
    k = (scale*0.5)/frequency
    to_int = k*(array + np.roll(array,d))
    if direction == 'forwards':
        to_int[0] = initial_value
    else:
        to_int[-1] = initial_value * s 
        # Note: Sign of initial value will be reversed twice for backwards case.
        
    result[::d] = np.ma.cumsum(to_int[::d] * s)
    
    return result
    
def interpolate_and_extend (array):
    """ 
    This will replace all masked values in an array with linearly
    interpolated values between unmasked point pairs, and extrapolate first
    and last unmasked values to the ends of the array.
    
    See Derived Parameter Node 'Magnetic Deviation' for an example of use.
    
    :param array: Array of data to be interpolated
    :type array: numpy masked array
    
    :returns interpolated: array of all valid data
    :type interpolated: Numpy masked array, with all masks (normally) False.
    """
    
    # Where do we need to use the raw data?
    blocks = np.ma.clump_masked(array)
    last = len(array)
    if len(blocks)==0:
        raise ValueError,'No masked data in interpolate_and_extend'
    
    for block in blocks:
        # Setup local variables
        a = block.start
        b = block.stop

        if a == 0:
            # First data can only be extension of the first valid sample.
            if b == last:
                raise ValueError,'No unmasked data in interpolate_and_extend'
            array[:b] = array[b]
        elif b == last:
            array[a:] = array[a-1]
        else:
            join = np.linspace(array[a-1], array[b], num=b-a+2)
            array[a:b] = join[1:-1]
            
    return array
        
    
    
def interleave (param_1, param_2):
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
        raise ValueError, 'Attempt to interleave parameters at differing sample rates'

    dt = param_2.offset - param_1.offset
    # Note that dt may suffer from rounding errors, 
    # hence rounding the value before comparison.
    if 2*abs(round(dt,6)) != 1/param_1.frequency: 
                raise ValueError, 'Attempt to interleave parameters that are not correctly aligned'
    
    merged_array = np.ma.zeros((2, len(param_1.array)))
    if dt > 0:
        merged_array = np.ma.column_stack((param_1.array,param_2.array))
    else:
        merged_array = np.ma.column_stack((param_2.array,param_1.array))
        
    return np.ma.ravel(merged_array)
            
"""
Superceded by blend routines.

def interleave_uneven_spacing (param_1, param_2):
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
        raise ValueError, 'Attempt to interleave parameters at differing sample rates'

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
        multiplier = out_frequency / float(param.frequency)
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
                            param.frequency / float(out_frequency))
    interpolated_array = interpolator(out_indices)
    masked_array = np.ma.masked_array(interpolated_array,
                                      mask=np.isnan(interpolated_array))
    return masked_array, out_frequency, out_offset
"""

def index_of_datetime(start_datetime, index_datetime, frequency):
    '''
    :param start_datetime: Start datetime of data file.
    :type start_datetime: datetime
    :param index_datetime: Datetime to calculate the index of.
    :type index_datetime: datetime
    :param frequency: Frequency of index.
    :type frequency: float or int
    :returns: The index of index_datetime relative to start_datetime and frequency.
    '''
    difference = index_datetime - start_datetime
    return difference.total_seconds() * frequency

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

def is_slice_within_slice(inner_slice, outer_slice):
    '''
    inner_slice is considered to not be within outer slice if its start or 
    stop is None.
    
    :type inner_slice: slice
    :type outer_slice: slice
    :returns: Whether inner_slice is within the outer_slice.
    :rtype: bool
    '''
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
    if first_slice.step != None and first_slice.step < 1 \
       or second_slice.step != None and second_slice.step < 1:
        raise ValueError("Negative step not supported")
    return first_slice.start < second_slice.stop \
           and second_slice.start < first_slice.stop

def slices_overlay(first_list, second_list):
    '''
    This is a simple function to allow two slice lists to be merged.
    
    Note: This currently has a trap for reverse slices, although this could be extended.
    
    :param first_list: First list of slices
    :type first_list: List of slices
    :param second_list: Second list of slices
    :type second_list: List of slices
    
    :returns: List of slices where first and second lists overlap.
    '''
    result_list = []
    for first_slice in first_list:
        for second_slice in second_list:
            if (first_slice.step != None and first_slice.step < 0) or \
               (second_slice.step != None and second_slice.step < 0):
                raise ValueError, 'slices_overlay will not work with reverse slices'
            if slices_overlap(first_slice, second_slice):
                result_list.append(slice(max(first_slice.start, second_slice.start),
                                         min(first_slice.stop, second_slice.stop)))
    return result_list

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
    Returns the bearings and distances of a track with respect to a fixed point.
    
    Usage: 
    lat[], lon[] = latitudes_and_longitudes(brg[], dist[], {'latitude':lat_ref, 'longitude', lon_ref})
    
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
    Copyright 2002-2011 Chris Veness, and altered by Flight dAta Services to 
    suit the POLARIS project.
    """
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])
    brg = np.deg2rad(bearings.data)
    dist = distances.data / 6371000.0 # Scale to earth radius in metres

    lat = np.arcsin(sin(lat_ref)*np.cos(dist) + 
                   cos(lat_ref)*np.sin(dist)*np.cos(brg))
    lon = np.arctan2(np.sin(brg)*np.sin(dist)*np.cos(lat_ref), 
                      np.cos(dist)-sin(lat_ref)*np.sin(lat))
    lon += lon_ref 
 
    joined_mask = np.logical_or(bearings.mask, distances.mask)
    lat_array = np.ma.array(data = np.rad2deg(lat),mask = joined_mask)
    lon_array = np.ma.array(data = np.rad2deg(lon),mask = joined_mask)
    return lat_array, lon_array

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

def max_abs_value(array, _slice=slice(None)):
    """
    Get the value of the maximum absolute value in the array. 
    Return value is NOT the absolute value (i.e. may be negative)
    
    Note, if all values are masked, it will return the value at the first index 
    (which will be masked!)
    
    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max value relative to
    :type _slice: slice
    """
    index, value = max_value(np.ma.abs(array), _slice)
    return Value(index, array[index])
    
def max_value(array, _slice=slice(None)):
    """
    Get the maximum value in the array and its index relative to the array and
    not the _slice argument.
    
    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max value relative to
    :type _slice: slice
    """
    return _value(array, _slice, np.ma.argmax)

def min_value(array, _slice=slice(None)):
    """
    Get the minimum value in the array and its index.
    
    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max value relative to
    :type _slice: slice
    """
    return _value(array, _slice, np.ma.argmin)
            
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

def merge_two_parameters (param_one, param_two):
    '''
    This process merges two parameter arrays of the same frequency.
    without smoothing, and then computes the offset and frequency appropriately.
    
    Note: There is no check for the parameters being equi-spaced.
    
    See also blend_two_parameters which compensates for differences between the two sensors.
    
    :param param_one: Parameter object
    :type param_one: Parameter
    '''
    assert param_one.frequency  == param_two.frequency
    
    if param_one.offset <= param_two.offset:
        # merged array should be monotonic (always increasing in time)
        array = merge_sources(param_one.array, param_two.array)
        offset = param_one.offset
    else:
        array = merge_sources(param_two.array, param_one.array)
        offset = param_two.offset
    return array, param_one.frequency * 2, offset


def merge_sources(*arrays):
    '''
    This simple process merges the data from multiple sensors where they are
    sampled alternately. Unlike blend_alternate_sensors or the parameter
    level option blend_teo_parameters, this procedure does not make any
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


def blend_alternate_sensors (array_one, array_two, padding):
    '''
    This simple process merges the data from two sensors where they are sampled
    alternately. Often pilot and co-pilot attitude and air data signals are
    stored in alternate locations to provide the required sample rate while
    allowing errors in either to be identified for investigation purposes.
    
    For FDM, only a single parameter is required, but mismatches in the two 
    sensors can lead to, taking pitch attitude as an example, apparent "nodding"
    of the aircraft and errors in the derived pitch rate.
    
    Mismatches can also occur when there are timing differences between the
    two samples, in which case this averaging process combined with
    corrections to the offset and sampling interval are effective.
    
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

def blend_two_parameters (param_one, param_two):
    '''
    This process merges two parameter arrays of the same frequency.
    Smoothes and then computes the offset and frequency appropriately.
    
    :param param_one: Parameter object
    :type param_one: Parameter
    '''
    assert param_one.frequency  == param_two.frequency
    offset = (param_one.offset + param_two.offset)/2.0
    frequency = param_one.frequency * 2
    padding = 'Follow'
    
    if offset > 1/frequency:
        offset = offset - 1/frequency
        padding = 'Precede'
        
    if param_one.offset <= param_two.offset:
        # merged array should be monotonic (always increasing in time)
        array = blend_alternate_sensors(param_one.array, param_two.array, padding)
    else:
        array = blend_alternate_sensors(param_two.array, param_one.array, padding)
    return array, param_one.frequency * 2, offset

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

def np_ma_zeros_like(array):
    """
    The Numpy masked array library does not have equivalents for some array
    creation functions. These are provided with similar names which may be
    replaced should the Numpy library be extended in future.
    """
    return np.ma.array(np.zeros_like(array), mask=False)


def np_ma_masked_zeros_like(array):
    """
    The Numpy masked array library does not have equivalents for some array
    creation functions. These are provided with similar names which may be
    replaced should the Numpy library be extended in future.
    """
    return np.ma.masked_all_like(np.zeros_like(array))


def peak_curvature(array, _slice=slice(None), curve_sense='Concave'):
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
    
    :returns peak_curvature: The index within this array where the curvature first peaks in the required sense.
    :rtype: integer
    
    This routine uses a "Truck and Trailer" algorithm to find where a
    parameter changes slope. In the case of FDM, we are looking for the point
    where the airspeed starts to increase (or stops decreasing) on the
    takeoff and landing phases. This is more robust than looking at
    longitudinal acceleration and complies with the POLARIS philosophy that
    we should provide analysis with only airspeed, altitude and heading data
    available.
    """
    data = array[_slice].data
    gap = TRUCK_OR_TRAILER_INTERVAL
    if gap%2-1:
        gap-=1  #  Ensure gap is odd
    ttp = TRUCK_OR_TRAILER_PERIOD
    trailer = ttp+gap
    overall = 2*ttp + gap 
    # check the array is long enough.
    if len(data) < overall:
        raise ValueError, 'Peak curvature called with too short a sample'

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
    if curve_sense == 'Convex':
        angle = -angle
    
    # Find peak - using values over 50% of the highest allows us to operate
    # without knowing the data characteristics.
    peak_slice=np.ma.clump_unmasked(np.ma.masked_less(angle,0.5))
        
    if peak_slice:
        index = peak_index(angle.data[peak_slice[0]])+\
            peak_slice[0].start+(overall/2.0)-0.5
        return index*(_slice.step or 1) + (_slice.start or 0)
    else:
        return None
    
def peak_index(a):
    '''
    Scans an array and returns the peak, where possible computing the local
    maximum assuming a quadratic curve over the top three samples.
    
    :param a: array
    :type a: list of floats
    
    '''
    if len(a) == 0:
        raise ValueError, 'No data to scan for peak'
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
    
    hw = width * hz / 2.0
    if hw < 1:
        raise ValueError
    
    # Set up an array of masked zeros for extending arrays.
    slope = np.ma.copy(to_diff)
    slope[hw:-hw] = (to_diff[2*hw:] - to_diff[:-2*hw])/width
    slope[:hw] = (to_diff[1:hw+1] - to_diff[0:hw]) * hz
    slope[-hw:] = (to_diff[-hw:] - to_diff[-hw-1:-1])* hz
    return slope

def repair_mask(array, frequency=1, repair_duration=REPAIR_DURATION,
                raise_duration_exceedance=False):
    '''
    This repairs short sections of data ready for use by flight phase algorithms
    It is not intended to be used for key point computations, where invalid data
    should remain masked. Modifies the array in-place.
    
    :param repair_duration: If None, any length of masked data will be repaired.
    '''
    if repair_duration:
        repair_samples = repair_duration * frequency
        if len(array) < repair_samples/3:
            # TODO: Better handling of trivial data segments
            return None
    else:
        repair_samples = None
    masked_sections = np.ma.clump_masked(array)
    for section in masked_sections:
        length = section.stop - section.start
        if repair_samples and (length) > repair_samples:
            if raise_duration_exceedance:
                raise ValueError("Length of masked section '%s' exceeds "
                                 "repair_samples '%s'." % (length,
                                                           repair_samples))
            else:
                continue # Too long to repair
        elif section.start == 0:
            continue # Can't interpolate if we don't know the first sample
        elif section.stop == len(array):
            continue # Can't interpolate if we don't know the last sample
        else:
            array[section] = np.interp(np.arange(length) + 1,
                                       [0, length + 1],
                                       [array.data[section.start - 1],
                                        array.data[section.stop]])
    return array
   

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


def rms_noise(array):
    '''
    :param array: input parameter to measure noise level
    :type array: numpy masked array
    :returns: RMS noise level
    :type: Float
    
    :exception: Should all the difference terms include masked values, this
    function will return None.
    
    This computes the rms noise for each sample compared with its neighbours.
    In this way, a steady cruise at 30,000 ft will yield no noise, as will a
    steady climb or descent.
    '''
    # The difference between one sample and the ample to the left is computed
    # using the ediff1d algorithm, then by rolling it right we get the answer
    # for the difference between this sample and the one to the right.
    diff_left = np.ma.ediff1d(array, to_end=0)
    diff_right = np.ma.array(data=np.roll(diff_left.data,1), 
                             mask=np.roll(diff_left.mask,1))
    local_diff = (diff_left - diff_right)/2.0
    if np.ma.count(local_diff[1:-1]) == 0:
        return None
    else:
        return sqrt(np.ma.mean(np.ma.power(local_diff[1:-1],2)))  # RMS in one line !
    
def shift_slice(this_slice, offset):
    """
    This function shifts a slice by an offset. The need for this arises when
    a phase condition has been used to limit the scope of another phase
    calculation.
    """
    a = (this_slice.start or 0) + offset
    b = (this_slice.stop or 0) + offset
    c = this_slice.step
    if (b-a)>1:
        # This traps single sample slices which can arise due to rounding of
        # the iterpolated slices.
        return(slice(a,b,c))
    
def shift_slices(slicelist, offset):
    """
    This function shifts a list of slices by a common offset, retaining only
    the valid (not None) slices.
    """
    newlist = []
    for each_slice in slicelist:
        if each_slice:
            new_slice = shift_slice(each_slice,offset)
            if new_slice: newlist.append(new_slice)
    return newlist

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
    return (_slice.stop - _slice.start) / hz

def slice_samples(_slice):
    '''
    Gets the number of samples in a slice.
    
    :param _slice: Slice to count sample length.
    :type _slice: slice
    :returns: Number of samples in _slice.
    :rtype: integer
    '''
    if _slice.step == None:
        step = 1
    else:
        step = _slice.step
        
    if _slice.start == None or _slice.stop == None:
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
    if len(array) == 0:
        return array, []
    rep_array, slices = slices_between(array, from_, to)
    if from_ > to:
        condition = lambda s: rep_array[s.start] > rep_array[s.stop-1]
    elif from_ < to:
        condition = lambda s: rep_array[s.start] < rep_array[s.stop-1]
    else:
        raise ValueError('From and to values should not be equal.')
    filtered_slices = filter(condition, slices)
    return rep_array, filtered_slices

"""
Spline finction placeholder

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
        stepped_array[(low < array) & (array <= high)] = level
        low = high
    else:
        # all values above the last
        stepped_array[low < array] = level
    return np.ma.array(stepped_array, mask=array.mask)
            

def smooth_track_cost_function(lat_s, lon_s, lat, lon):
    # Summing the errors from the recorded data is easy.
    from_data = np.sum((lat_s - lat)**2)+np.sum((lon_s - lon)**2)
    
    # The errors from a straight line are computed swiftly using convolve.
    slider=np.array([-1,2,-1])
    from_straight = np.sum(np.convolve(lat_s,slider,'valid')**2) + \
        np.sum(np.convolve(lon_s,slider,'valid')**2)
    
    cost = from_data + 100*from_straight
    return cost

def track_linking(pos, local_pos):
    """
    Obtain corrected tracks from takeoff phase, final approach and landing
    phase and possible intermediate approach and go-around phases, and
    compute error terms to align the recorded lat&long with each partial data
    segment. 
    
    Takes an array of latitude or longitude position data and the equvalent
    array of local position data from ILS localizer and synthetic takeoff
    data.
    
    This is done by computing linearly varying adjustment factors between
    each computed section, a process that was found to be unnecessarily
    complex, but as it gives good results and is already programmed it was
    decided to leave this in place.
    
    :param pos: Flight track data (latitude or longitude) in degrees.
    :type pos: np.ma.masked_array, masked from data validity tests.
    :param local_pos: Position data relating to runway or ILS.
    :type local_pos: np.ma.masked_array, masked where no local data computed.
    
    :returns: Position array using local_pos data where available and interpolated pos data elsewhere.
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
        if a<2:
            link_a = 1
        else:
            adj_a = (3 * local_pos.data[a-1] - local_pos.data[a-2])/2 -\
                (3 * pos.data[a] - pos.data[a+1])/2
    
        # now the other end
        if b>last-2:
            link_b=1
        else:
            adj_b = (3 * local_pos.data[b] - local_pos.data[b+1])/2 -\
                (3 * pos.data[b-1] - pos.data[b-2])/2

        fix_a = adj_a + link_a*adj_b
        fix_b = adj_b + link_b*adj_a
        
        fix = np.linspace(fix_a, fix_b, num=b-a)
        local_pos[a:b] = pos[a:b] + fix
    return local_pos
        
        
def smooth_track(lat, lon):
    """
    Input:
    lat = Recorded latitude array
    lon = Recorded longitude array
    
    Returns:
    lat_last = Optimised latitude array
    lon_last = optimised longitude array
    Cost = cost function, used for testing satisfactory convergence.
    """
    
    if len(lat) <= 5:
        return lat, lon, 0.0 # Polite return of data too short to smooth.
    
    # This routine used to index through the arrays. By using np.convolve (in
    # both the iteration and cost functions) the same algorithm runs 350
    # times faster !!!
    
    lat_s = np.ma.copy(lat)
    lon_s = np.ma.copy(lon)
    
    # Set up a weighted array that will slide past the data.
    r = 0.7  
    # Values of r alter the speed to converge; 0.7 seems best.
    slider=np.ma.ones(5)*r/4
    slider[2]=1-r

    cost_0 = float('inf')
    cost = smooth_track_cost_function(lat_s, lon_s, lat, lon)
    
    while cost < cost_0:  # Iterate to an optimal solution.
        lat_last = np.ma.copy(lat_s)
        lon_last = np.ma.copy(lon_s)

        # Straighten out the middle of the arrays, leaving the ends unchanged.
        lat_s.data[2:-2] = np.convolve(lat_last,slider,'valid')
        lon_s.data[2:-2] = np.convolve(lon_last,slider,'valid')

        cost_0 = cost
        cost = smooth_track_cost_function(lat_s, lon_s, lat, lon)
    return lat_last, lon_last, cost_0

            
def straighten_headings(heading_array, copy=True):
    '''
    We always straighten heading data before checking for spikes. 
    It's easier to process heading data in this format.
    
    :param heading_array: array/list of numeric heading values
    :type heading_array: iterable
    :returns: Straightened headings
    :rtype: Generator of type Float
    '''
    if copy:
        heading_array = heading_array.copy()
    for clump in np.ma.clump_unmasked(heading_array):
        head_prev = heading_array.data[clump.start]
        diff = np.ediff1d(heading_array.data[clump])
        diff = diff - 360.0 * np.trunc(diff/180.0)
        heading_array[clump][0] = head_prev
        heading_array[clump][1:] = np.cumsum(diff) + head_prev
    return heading_array

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

    if new.stop == None:
        stop = orig.stop
    else:
        stop = (orig.start or 0) + (new.stop or orig.stop or 0) * (orig.step or 1) # the bit after "+" isn't quite right!!

    return slice(start, stop, None if step == 1 else step)

def index_closest_value(array, threshold, _slice=slice(None)):
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
    threshold.
    :returns: interpolated time when the array values crossed the threshold. (One value only).
    :returns type: float
    '''
    step = _slice.step or 1
    max_index = len(array)
    # Arrange the limits of our scan, ensuring that we stay inside the array.
    if step == 1:
        begin = max(int(round(_slice.start or 0)),0)
        end = min(int(round(_slice.stop or max_index)),max_index)

        # A "let's get the logic right and tidy it up afterwards" bit of code...
        # TODO: Refactor this algorithm
        if begin >= len(array):
            begin = max_index
        elif begin < 0:
            begin = 0
        if end > len(array):
            end = max_index
        elif end < 0:
            end = 0
            
        left, right = slice(begin,end-1,step), slice(begin+1,end,step)
        
    elif step == -1:
        begin = min(int(round(_slice.start or max_index)),max_index)
        end = max(int(round(_slice.stop or 0)),0)

        # More "let's get the logic right and tidy it up afterwards" bit of code...
        if begin >= len(array):
            begin = max_index - 1
        elif begin < 0:
            begin = 0
        if end > len(array):
            end = max_index
        elif end < 0:
            end = 0
            
        left, right = slice(begin,end+1,step), slice(begin-1,end,step)
        
    else:
        raise ValueError, 'Step length not 1 in index_at_value'
    
    if begin == end:
        raise ValueError, 'No range for seek function to scan across'
    elif abs(begin - end) < 2:
        # Requires at least two values to find if the array crosses a
        # threshold.
        return None

    # When the data being tested passes the value we are seeking, the 
    # difference between the data and the value will change sign.
    # Therefore a negative value indicates where value has been passed.
    value_passing_array = (array[left]-threshold) * (array[right]-threshold)
    test_array = np.ma.masked_greater(value_passing_array, 0.0)
    
    if np.all(test_array.mask):
        # The parameter does not pass through threshold in the period in question, so return empty-handed.
        if endpoint=='closing':
            # Rescan the data to find the last point where the array data is closing.
            closing_array = abs(array-threshold)
            i=begin
            while (closing_array [i+step]<=closing_array [i]):
                i=i+step
                if i==end:
                    return end
            return i
        else:
            return None
    else:
        n,dummy=np.ma.flatnotmasked_edges(np.ma.masked_greater(value_passing_array, 0.0))
        a = array[begin+step*n]
        b = array[begin+step*(n+1)]
        # Force threshold to float as often passed as an integer.
        r = (float(threshold) - a) / (b-a) 
        #TODO: Could test 0 < r < 1 for completeness
    return (begin + step * (n+r))

def _value(array, _slice, operator):
    """
    Applies logic of min_value and max_value
    """
    if _slice.step and _slice.step < 0:
        raise ValueError("Negative step not supported")
    if np.ma.count(array[_slice]):
        index = operator(array[_slice]) + (_slice.start or 0) * (_slice.step or 1)
        return Value(index, array[index])
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
    
    Samples within one index outside the array boundaries are permitted to
    allow for offsets within the data frame. Beyond this a ValueError is
    raised.
    
    :param array: input data
    :type array: masked array
    :param index: index into the array where we want to find the array value.
    :type index: float
    :returns: interpolated value from the array
    :raises ValueError: If index is outside of array range.
    '''
    
    if index < -1.0 or index > len(array):
        raise ValueError, 'Seeking value outside data time range'
    elif index < 0.0:
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
        raise ValueError, 'Supersonic airspeed compuations not included'
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
    density_ratio = press_ratio / temp_ratio
    Rho = Rhoref * density_ratio
    tas = _dp2speed(dp, P, Rho)
    return tas

def alt2sat(alt_ft):
    """ Convert altitude to temperature using lapse rate"""
    return np.ma.where(alt_ft <= H1, 15.0 + L0 * alt_ft, -56.5)
    
def machtat2sat(mach, tat, recovery_factor=0.9):
    """
    Return the ambient temp, given the mach number, indicated temperature and
    the temperature probe's recovery factor. Default recovery factor is fine
    for most cases, and allows for inherited test case which used a lower
    value.
    """
    ambient_temp = (tat + 273.15) / (1. + (0.2*recovery_factor) * mach** 2.)
    sat = ambient_temp - 273.15
    return sat

def _alt2press_ratio_gradient(H):
    # From http://www.aerospaceweb.org/question/atmosphere/q0049.shtml
    # Faster to compute than AeroCalc formulae, and pass AeroCalc tests.
    return np.ma.power(1 - H/145442.0, 5.255876)

def _alt2press_ratio_isothermal(H):
    return 0.223361 * np.ma.exp((36089.0-H)/20806.0)

