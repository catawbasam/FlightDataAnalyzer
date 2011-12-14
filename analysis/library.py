import math
import numpy as np

from datetime import datetime, timedelta
from hashlib import sha256
from itertools import izip
from scipy.signal import iirfilter, lfilter, lfilter_zi, filtfilt

from settings import REPAIR_DURATION


#Q: Not sure that there's any point in these? Very easy to define later
#----------------------------------------------------------------------
#def offset(data, offset):
    #return data + offset
    
#def plus(self, offset):
    #self.data = self.data + offset
    #return self

#def minus(self, offset):
    #self.data = self.data - offset
    #return self
    
#def plus (self, to_add):
    #return self.data + shift(self, to_add)

#def times (self, to_multiply):
    #return self.data * shift(self, to_multiply)
#----------------------------------------------------------------------


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
    if slave.frequency == master.frequency and slave.offset == master.offset:
        return slave.array
    
    # Check the interval is one of the two forms we recognise
    assert interval in ['Subframe', 'Frame']
    
    # Check the type of signal is one of those we recognise
    assert signaltype in ['Analogue', 'Discrete', 'Multi-State']
    
    slave_array = slave.array # Optimised access to attribute.
    if len(slave_array) == 0:
        return slave_array # Otherwise would raise in loop.
    if master.frequency == slave.frequency and master.offset == slave.offset:
        # No alignment is required, return the slave's array unchanged.
        return slave.array
    
    # Here we create a masked array to hold the returned values that will have 
    # the same sample rate and timing offset as the master
    slave_aligned = np.ma.empty_like(master.array)
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
    assert len(master.array.data) * ws == len(slave_array.data) * wm
           
    # Compute the sample rate ratio (in range 10:1 to 1:10 for sample rates up to 10Hz)
    r = wm/float(ws)

    # Each sample in the master parameter may need different combination parameters
    for i in range(wm):
        bracket=(i/r+delta)
        # Interpolate between the hth and (h+1)th samples of the slave array
        h=int(math.floor(bracket))
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
    :type years, months, days, hours, mins, secs: iterable
    """
    base_dt = None
    clock_variation = {}
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
        return None #Q: Invent base datetime, such as taking 1000 years off?!?

    
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

def first_order_lag (in_param, time_constant, hz, gain = 1.0, initial_value = None):
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
    masked_result.mask = in_param.mask
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
    """
    Hysteresis is a process used to prevent noisy data from triggering 
    an unnecessary number of events or state changes when the parameter is 
    close to a threshold.
        
    :param array: data values to process
    :type array: masked array
    :param hysteresis: hysteresis range to apply
    :type hysteresis: float
    :returns: masked array of values with hysteresis applied
    """

    # This routine accepts the usual masked array but only processes the
    # data part of the array as the hysteresis process cannot make the
    # values invalid.
   
    half_range = hysteresis / 2.0
    result = np.ma.copy(array)

    for i in xrange(len(result)-1):
        if result.data[i+1] - result.data[i] > half_range:
            result.data[i+1] = result.data[i+1] - half_range
        elif result.data[i+1] - result.data[i] < -half_range:
            result.data[i+1] = result.data[i+1] + half_range
        else:
            result.data[i+1] = result.data[i]

    return result

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
    result_offset = mean_offset - 1.0/(2.0 * param_1.frequency)
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
    '''
    
    result = np.ma.empty_like(array)
    result[1:-1] = (array[:-2] + array[1:-1]*2.0 + array[2:]) / 4.0
    result[0] = (array[0] + array[1]) / 2.0
    result[-1] = (array[-2] + array[-1]) / 2.0
    return result


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
                       / (2 * float(half_width))
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
            array[section] =np.interp(np.arange(length)+1,[0,length+1],[array.data[section.start - 1],array.data[section.stop]])
    return array            
            
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

def time_at_value_wrapped(parameter, section, value, direction='Forwards'):
    '''
    This function makes it easier to access the time_at_value function 
    when using POLARIS parameter and section components.

    :param parameter: input data
    :type parameter: parameter object
    :param section: the section to be used for finding the value
    :type section: section object
    :param value: the threshold being sought
    :type value: float
    :param direction : the direction of travel for the search.
    :type value: text. Permitted values 'Forwards' (the default) or 'Backwards'
    '''
    data = parameter.array[section.slice]
    if direction == 'Forwards':
        result = time_at_value (repair_mask(data) , parameter.frequency, 
                                parameter.offset, 0, len(data)-1, value)
    else:
        result = time_at_value (repair_mask(data) , parameter.frequency, 
                                parameter.offset, len(data)-1, 0, value)
    return result
            
def time_at_value (array, hz, offset, scan_start, scan_end, threshold):
    '''
    This function seeks the moment when the parameter in question first crosses 
    a threshold. It works both forwards and backwards in time. To scan backwards
    just make the start point later than the end point. This is really useful
    for finding things like the point of landing.
    
    For example, to find 50ft Rad Alt on the descent, use something like:
       altitude_radio.seek(t_approach, t_landing, 50)
    
    :param array: input data
    :type array: masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param scan_start: time into the array where we want to start seeking the threshold transit.
    :type scan_start: float
    :param scan_end: time into the array where we want to stop seeking the threshold transit.
    :type scan_end: float
    :param threshold: the value that we expect the array to cross between scan_start and scan_end.
    :type threshold: float
    :returns: interpolated time when the array values crossed the threshold. (One value only).
    :returns type: float
    '''

    if scan_start == scan_end:
        raise ValueError, 'No range for seek function to scan across'
    
    begin = int((scan_start - offset) * hz)
    cease = int((scan_end   - offset) * hz)
    
    step = 1
    if cease > begin : # Normal increasing scan
        cease = cease + 1
    else:
        # Allow for traversing the data backwards
        step = -1
        cease = max(cease-1, 0)

    if begin < 0 or begin > len(array) or cease < 0 or cease > len(array):
        raise ValueError, 'Attempt to seek outside data range'
        
    # When the data being tested passes the value we are seeking, the 
    # difference between the data and the value will change sign.
    # Therefore a negative value indicates where value has been passed.
    value_passing_array = (array[begin:cease-step:step]-threshold) * (array[begin+step:cease:step]-threshold)
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
    return (begin + step * (n + r)) / hz

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

    if location_in_array < 0.0 or location_in_array > len(array):
        raise ValueError, 'Seeking value outside data time range'
    
    low = int(location_in_array)
    if (low==location_in_array):
        # I happen to have arrived at exactly the right value by a fluke...
        return array.data[low]
    else:
        high = low + 1
        r = location_in_array - low
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
