import numpy as np

from collections import OrderedDict, namedtuple
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import izip
from math import floor
from scipy.signal import lfilter, lfilter_zi

from settings import REPAIR_DURATION, TRUCK_OR_TRAILER_INTERVAL, TRUCK_OR_TRAILER_PERIOD

Value = namedtuple('Value', 'index value')

class InvalidDatetime(ValueError):
    pass


def align(slave, master, interval='Subframe', signaltype='Analogue'):
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
    
    :param slave: The parameter to be aligned to the master
    :type slave: Parameter objects
    :param master: The master parameter
    :type master: Parameter objects    
    :param interval: Has possible values 'Subframe' or 'Frame'.  #TODO: explain this!
    :type interval: String
    :param mode: Has possible values 'Analogue' or 'Discrete'. TODO: 'Multistate' mode as those parameters should be shifted similar to Discrete (or use Multistate for discrete)
    :signaltype = Analogue results in interpolation of the data across each sample period
    :signaltype = Discrete or Multi-State results in shifting to the closest data sample, without interpolation.
    :Note: Multistate is a type of discrete in this case.
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
    
    # Check the type of signal is one of those we recognise
    assert signaltype in ['Analogue', 'Discrete', 'Multi-State']
    
    ## slave_aligned[:] = 0.0
    ## Clearing the slave_aligned array is unnecessary, but can make testing easier to follow.

    # Get the timing offsets, comprised of word location and possible latency.
    tp = master.offset
    ts = slave.offset

    # Get the sample rates for the two parameters
    wm = master.frequency
    ws = slave.frequency

    # Express the timing disparity in terms of the slave paramter sample interval
    delta = (tp-ts)*slave.frequency

    # If we are working across a complete frame, the number of samples in each case
    # is four times greater.
    if interval == 'Frame':
        wm = int(wm * 4)
        ws = int(ws * 4)
    assert wm in [1,2,4,8,16,32,64]
    assert ws in [1,2,4,8,16,32,64]
    ##assert len(master.array.data) * ws == len(slave_array.data) * wm
           
    # Compute the sample rate ratio (in range 10:1 to 1:10 for sample rates up to 10Hz)
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
        if signaltype != 'Analogue':
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

def calculate_timebase(years, months, days, hours, mins, secs):
    """
    Calculates the timestamp most common in the array of timestamps. Returns
    timestamp calculated for start of array by applying the offset of the
    most common timestamp.
    
    Accepts arrays and numpy arrays at 1Hz.
    
    Note: if uneven arrays are passed in, they are assumed by izip that the
    start is valid and the uneven ends are invalid and skipped over.
    
    TODO: Support year as a 2 digits - e.g. "11" is "2011"
    
    :param years, months, days, hours, mins, secs: Appropriate 1Hz time elements
    :type years, months, days, hours, mins, secs: iterable of numeric type
    :returns: best calculated datetime at start of array
    :rtype: datetime
    :raises: InvalidDatetime if no valid timestamps provided
    """
    base_dt = None
    clock_variation = OrderedDict() # so if all values are the same, take the first
    for step, (yr, mth, day, hr, mn, sc) in enumerate(izip(years, months, days, hours, mins, secs)):
        try:
            dt = datetime(yr, mth, day, hr, mn, sc)
        except (ValueError, TypeError):
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
    index_in_seconds = index * frequency
    offset = timedelta(seconds=index_in_seconds)
    return start_datetime + offset
    
def duration(a, period, hz=1.0):
    '''
    This function clips the maxima and minima of a data array such that the 
    values are present (or exceeded) in the original data for the period
    defined. After processing with this function, the resulting array can be 
    used to detect maxima or minima (in exactly the same way as a non-clipped 
    parameter), however the values will have been met or exceeded in the 
    original data for the given duration.
        
    :param a: Masked array of floats
    :type a: Numpy masked array
    :param period: Time for the output values to be sustained(sec)
    :type period: int/float
    :param hz: Frequency of the data_array
    :type hz: float
    '''
    if period <= 0.01:
        raise ValueError('Duration called with period outside permitted range')

    if hz <= 0.01:
        raise ValueError('Duration called with sample rate outside permitted range')

    delay = period * hz

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
    
    '''
    Original version using Fortranesque style :o)
    i = 1 # return of the i!
    while i < len(data_array) - delay:
        # check if ...??
        if (result[i]-result[i-1])*(result[i+delay]-result[i]) < 0: #why?
            for j in range (delay-1):
                result[i] = result[i-1]
                i = i + 1 #argh (i += 1 is only slightly nicer)
        else:
            i = i + 1
    '''
    return a

def first_order_lag (array, time_constant, hz, gain = 1.0, initial_value = None):
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
    
    :param array: input data (x)
    :type array: masked array
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
    input_data = np.copy(array.data)
    
    # Scale the time constant to allow for different data sample rates.
    tc = time_constant / hz
    
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
    
    z_initial = lfilter_zi(x_term, y_term) # Prepare for non-zero initial state
    # The initial value may be set as a command line argument, mainly for testing
    # otherwise we set it to the first data value.
    if initial_value == None:
        initial_value = input_data[0]
    answer, z_final = lfilter(x_term, y_term, input_data, zi=z_initial*initial_value)
    masked_result = np.ma.array(answer)
    # The mask should last indefinitely following any single corrupt data point
    # but this is impractical for our use, so we just copy forward the original
    # mask.
    masked_result.mask = array.mask
    return masked_result

def first_order_washout (in_param, time_constant, hz, gain = 1.0, initial_value = None):
    '''
    Computes the transfer function
          x.G.s
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
    input_data = np.copy(in_param.data)
    
    # Scale the time constant to allow for different data sample rates.
    tc = time_constant / hz
    
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
    
    z_initial = lfilter_zi(x_term, y_term)
    if initial_value == None:
        initial_value = input_data[0]
    # Tested version here...
    answer, z_final = lfilter(x_term, y_term, input_data, zi=z_initial*initial_value)
    masked_result = np.ma.array(answer)
    # The mask should last indefinitely following any single corrupt data point
    # but this is impractical for our use, so we just copy forward the original
    # mask.
    masked_result.mask = in_param.mask
    return masked_result

def hash_array(array):
    '''
    Creates a sha256 hash from the array's tostring() method.
    '''
    checksum = sha256()
    checksum.update(array.tostring())
    return checksum.hexdigest()

def hysteresis (array, hysteresis):

    quarter_range = hysteresis / 4.0
    # Length is going to be used often, so prepare here:
    length = len(array)
    half_done = np.empty(length)
    result = np.empty(length)
    length = length-1 #  To be used for array indexing next

    # The starting point for the computation is the first sample.
    old = array[0]

    # Index through the data storing the answer in reverse order
    for index, new in enumerate(array.data):
        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        half_done[length-index] = old

    # Repeat the process in the "backwards" sense to remove phase effects.
    for index, new in enumerate(half_done):
        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        result[length-index] = old

    return np.ma.array(result, mask=array.mask)


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

def is_index_within_slice(index, _slice):
    '''
    Tests whether index is within the slice.
    
    :type index: int or float
    :type _slice: slice
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
            
def merge_alternate_sensors (array):
    '''
    This simple process merges the data from two sensors where they are sampled
    alternately. Often pilot and co-pilot attitude and air data signals are
    stored in alternate locations to provide the required sample rate while
    allowing errors in either to be identified for investigation purposes.
    
    For FDM, only a single parameter is required, but mismatches in the two 
    sensors can lead to, taking pitch attitude as an example, apparent "nodding"
    of the aircraft and errors in the derived pitch rate.
    
    :param array: sampled data from an alternate signal source
    :type array: masked array
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    result = np.ma.empty_like(array)
    result[1:-1] = (array[:-2] + array[1:-1]*2.0 + array[2:]) / 4.0
    result[0] = (array[0] + array[1]) / 2.0
    result[-1] = (array[-2] + array[-1]) / 2.0
    return result


def peak_curvature(array, _slice=slice(None), search_for='Concave'):
    """
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
    angle=np.ma.array(angle/np.max(np.abs(angle)))
    if search_for == 'Bipolar':
        angle = np.ma.abs(angle)
    
    # Find peak - using values over 50% of the highest allows us to operate
    # without knowing the data characteristics.
    peak_slice=np.ma.clump_unmasked(np.ma.masked_less(angle,0.5))
        
    if peak_slice:
        index = peak_index(angle.data[peak_slice[0]])+\
            peak_slice[0].start+(overall/2.0)-0.5
        return index + (_slice.start or 0)
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
    
def rate_of_change(diff_param, half_width):
    '''
    @param to_diff: Parameter object with .array attr (masked array)
    
    Differentiation using the xdot(n) = (x(n+hw) - x(n-hw))/w formula.
    Width w=hw*2 and this approach provides smoothing over a w second period,
    without introducing a phase shift.
    
    :param diff_param: input Parameter
    :type diff_param: Parameter object
    :type diff_param.array : masked array
    :param diff_param.frequency : sample rate for the input data (sec-1)
    :type diff_param.frequency: float
    :param half_width: half the differentiation time period (sec)
    :type half_width: float
    :returns: masked array of values with differentiation applied
    
    Note: Could look at adapting the np.gradient function, however this does not
    handle masked arrays.
    '''
    hz = diff_param.frequency
    to_diff = diff_param.array
    
    hw = half_width * hz
    if hw < 1:
        raise ValueError
    
    # Set up an array of masked zeros for extending arrays.
    slope = np.ma.copy(to_diff)
    slope[hw:-hw] = (to_diff[2*hw:] - to_diff[:-2*hw])\
                       / (2.0 * float(half_width))
    slope[:hw] = (to_diff[1:hw+1] - to_diff[0:hw]) * hz
    slope[-hw:] = (to_diff[-hw:] - to_diff[-hw-1:-1])* hz
    return slope

def repair_mask(array):
    '''
    This repairs short sections of data ready for use by flight phase algorithms
    It is not intended to be used for key point computations, where invalid data
    should remain masked.
    '''
    masked_sections = np.ma.clump_masked(array)
    for section in masked_sections:
        length = section.stop - section.start
        if (length) > REPAIR_DURATION:  # TODO: include frequency as length is in samples and REPAIR_DURATION is in seconds
            break # Too long to repair
        elif section.start == 0:
            break # Can't interpolate if we don't know the first sample
        elif section.stop == len(array):
            break # Can't interpolate if we don't know the last sample
        else:
            array[section] = np.interp(np.arange(length) + 1,
                                       [0, length + 1],
                                       [array.data[section.start - 1],
                                        array.data[section.stop]])
    return array
   

def shift_slices(slicelist, offset):
    """
    This function shifts a list of slices by offset. The need for this arises
    when a phase condition has been used to limit the scope of another phase
    calculation.
    """
    newlist = []
    for each_slice in slicelist:
        a = each_slice.start + offset
        b = each_slice.stop + offset
        if (b-a)>1:
            # This traps single sample slices which can arise due to rounding
            # of the iterpolated slices.
            newlist.append(slice(a,b))
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
    return _slice.start - _slice.stop / hz

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

            
def straighten_headings(heading_array):
    '''
    We always straighten heading data before checking for spikes. 
    It's easier to process heading data in this format.
    
    :param heading_array: array/list of numeric heading values
    :type heading_array: iterable
    :returns: Straightened headings
    :rtype: Generator of type Float
    '''
    head_prev = heading_array[0]
    diff = np.ediff1d(heading_array)
    diff = diff - 360.0 * np.trunc(diff/180.0)
    heading_array[1:] = np.cumsum(diff) + head_prev
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
    start = (orig.start or 0) + (new.start or orig.start or 0) * (orig.step or 1)
    stop = (orig.start or 0) + (new.stop or orig.stop or 0) * (orig.step or 1) # the bit after "+" isn't quite right!!
    return slice(start, stop, None if step == 1 else step)

def index_at_value (array, threshold, _slice=slice(None)):
    '''
    This function seeks the moment when the parameter in question first crosses 
    a threshold. It works both forwards and backwards in time. To scan backwards
    pass in a slice with a negative step. This is really useful for finding
    things like the point of landing.
    
    For example, to find 50ft Rad Alt on the descent, use something like:
       altitude_radio.seek(t_approach, t_landing, slice(50,0,-1))
    
    :param array: input data
    :type array: masked array
    :param threshold: the value that we expect the array to cross between scan_start and scan_end.
    :type threshold: float
    :param _slice: slice where we want to seek the threshold transit.
    :type _slice: slice
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
    
    if np.ma.all(test_array.mask):
        # The parameter does not pass through threshold in the period in question, so return empty-handed.
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
    index = operator(array[_slice]) + (_slice.start or 0) * (_slice.step or 1)
    return Value(index, array[index])

def value_at_time (array, hz, offset, time_index):
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
    '''
    time_into_array = time_index - offset
    location_in_array = time_into_array * hz
    return value_at_index(array, location_in_array)

def value_at_index(array, index):
    '''
    Finds the value of the data in array at a given index.
    
    :param array: input data
    :type array: masked array
    :param index: index into the array where we want to find the array value.
    :type index: float
    :returns: interpolated value from the array
    '''
    if index < 0.0 or index > len(array):
        raise ValueError, 'Seeking value outside data time range'
    
    low = int(index)
    if (low==index):
        # I happen to have arrived at exactly the right value by a fluke...
        return None if array.mask[low] else array[low]
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