import math
import numpy as np
# Scipy routines used for transfer functions
import scipy.signal as signal

from datetime import datetime, timedelta
from itertools import izip

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


#2011-10-20 CJ: Redundant - replaced by calculate_timebase
##class ClockWatcher():
    ##def __init__(self):
        ##self.old_datetime = None

    ##def checktime(self, line):
        ### year = try_int(line['Year'], year) - missing from this test file
        ##year = 2010        
        ##month = try_int(line['Month'])
        ##day = try_int(line['Day'])
        ##hr = try_int(line['Hour'])

        #### The data now has minsec as a single parameter
        ####x = try_int(line['MinSec'])
        ####mn = int(x/60)    
        ####sc = x%60
        ### may need this format if time stored in two parameters
        ##mn = try_int(line['Minute'])
        #### sc = try_int(line['Second'])
        ##sc = 0 # Not recorded on ATRs
        
        ##try:
            ##new_datetime = datetime.datetime(year,month,day,hr,mn,sc)
        ##except ValueError:
            ### Trap for missing data from CSV and the 32/1/2011 and 1/13/2012 types of date.
            ##return None
        
        ##if new_datetime == self.old_datetime:
            ### no change
            ##return None
        ##else: #self.old_datetime == None or different
            ##self.old_datetime = new_datetime
            ##return new_datetime

#2011-10-20 CJ: Redundant - replaced by calculate_timebase
##def clock_reading(clock_variation, step):
    ### d = (i - base - count)%4096
    ### Have we seen this value before?
    ##if step in clock_variation:
        ### Yes, so increment our counter.
        ##clock_variation[step] += 1
    ##else:
        ### No, so create a new dictionary entry and start counting from one.
        ##clock_variation[step] = 1
    ##return clock_variation


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

    

def create_phase_inside(reference, a, b):
    """
    Create phase procedures take a reference parameter and return the same data
    Inside is valid inside the range a-b (masked outside)
    """
    m=np.arange(len(reference))
    m[:a]  = True
    m[a:b] = False
    m[b:]  = True
    return np.ma.MaskedArray(reference, mask = m)


def create_phase_outside(reference, a, b):
    """
    Create phase procedures take a reference parameter and return the same data
    Outside is valid outside the range a-b (masked inside)
    """
    m=np.arange(len(reference))
    m[:a]  = False
    m[a:b] = True
    m[b:]  = False
    return np.ma.MaskedArray(reference, mask = m)


def powerset(iterable):
    """
    Ref: http://docs.python.org/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def rate_of_change(to_diff, half_width):
    '''
    @param to_diff: Parameter object with .data attr (masked array)
    
    Differentiation using the xdot(n) = (x(n+hw) - x(n-hw))/w formula.
    Width w=hw*2 and this approach provides smoothing over a w second period,
    without introducing a phase shift.
    '''
    if half_width > 20:
        raise ValueError
    
    if half_width < 1:
        raise ValueError
    
    # Set up an array of masked zeros for extending arrays.    
    pad = np.ma.arange(20)
    pad[:] = np.ma.masked
    slope = (to_diff[2*half_width:] - to_diff[:-2*half_width]) / float((2 * half_width))
    return np.ma.concatenate([pad[:half_width],slope[:],pad[:half_width]])

'''
Not used at this time, and superceded by hysteresis as a technique for removing
unwanted state changes from noisy data.

def running_average(data, half_width=5):
    ## avg.param_name = to_avg.param_name+'_averaged'
    # Set up a masked result array of the right size.
    #   Ideally we'd like to use the command
    #   avg.data = np.ma.zeros_like(to_avg.data)
    #   but this isn't in the np library :-(

    #   Using empty_like then subtracting from itself didn't work as we had
    #   errors where non-numeric data was in the allocated memory. Here's what I tried:
    #     avg.data = np.ma.empty_like(to_avg.data)
    #     # Set the values to zero as we'll be summing into this array.
    #     avg.data  -= avg.data 
    
    # So make a copy and fill it with zeros is the second plan:
    temp = np.ma.copy(data)
    temp.fill(0)

    # The average is performed over an odd number of values 
    # centred on the result point.
    width = (2*half_width) + 1
    length = len(data)
    # This iteration is only for the number of points being averaged.
    for i in range(width):
        temp [half_width:-half_width] += data[i:length-width+i+1]
    # Divide the result to form the average    
    temp  = temp /float(width)
    # Mark the ends as invalid.
    for i in range(half_width):
        temp [i] = np.ma.masked
        temp [-(i+1)] = np.ma.masked
    return temp
'''

def align(master, slave):
    """
    This function takes two parameters which will have been sampled at different
    rates and with different offsets, and aligns the slave parameter's samples
    to match the master parameter. In this way the two may be processed without 
    timing errors.
    
    :type master, slave: Parameter objects with attributes:
    :type master.data: masked array
    :type masger.fdr_offset: float
    :type master.hz: float
    :type slave.data: masked array
    :type slave.fdr_offset: float
    :type slave.hz: float
    
    :returns masked array.
    The offset and hz for the returned masked array will be those of the 
    master parameter.
    The values of the returned array will be those of the slave parameter, 
    aligned to the master and adjusted by linear interpolation. The initial
    or final values may be masked if they lie outside the timebase of the 
    slave parameter (i.e. we do not extrapolate).    
    """
    # Here we create a masked array to hold the returned values that will have 
    # the same sample rate and timing offset as the master
    result = np.ma.empty_like(master.data)
    ## result[:] = 0.0
    ## Clearing the result array is unnecessary, but makes testing easier to follow.

    # Get the timing offsets, comprised of word location and possible latency.
    tp = master.fdr_offset
    ts = slave.fdr_offset

    # Get the sample rates for the two parameters
    wp = master.hz
    ws = slave.hz

    # Express the timing disparity in terms of the slaveary paramter sample interval
    delta = (tp-ts)*ws

    # Compute the sample rate ratio (in range 10:1 to 1:10 for sample rates up to 10Hz)
    r = wp/float(ws)

    # Each sample in the master (primary) parameter may need different combination parameters
    for i in range(wp):
        bracket=(i/r+delta)
        # Interpolate between the hth and (h+1)th samples of the slaveary
        h=int(math.floor(bracket))
        h1 = h+1
        # Linear interpolation coefficients
        b = bracket-h
        a=1-b

        if h<0:
            # We can't compute the inital values as the slaveary parameters we need
            # are out of range, so not available. Mask the result and work on the
            # later seconds of data to the end.
            result[i] = 0.0 
            # Allows unassigned values to be tested as np.ma.empty_like does not write values to the array.
            result[i] = np.ma.masked
            result[i+wp::wp]=a*slave.data[h+ws:-ws:ws]+b*slave.data[h1+ws::ws]
        elif h1>=ws:
            # We can't compute the final values as the secondary runs out of data.
            # Again, mask the final values and run from the beginning to almost the end
            # of the arrays.
            result[i-wp] = 0.0
            result[i-wp]=np.ma.masked
            result[i:-wp:wp]=a*slave.data[h:-ws:ws]+b*slave.data[h1::ws]
        else:
            # Sheer bliss. We can compute results across the whole range of the data.
            result[i::wp]=a*slave.data[h::ws]+b*slave.data[h1::ws]

    return result



def straighten_headings(heading_array):
    """
    We always straighten heading data before checking for spikes. 
    It's easier to process heading data in this format.
    
    If the spike is over 180 diff, that will count as a jump, but it will then
    jump back again on the next sample.
    
    TODO: could return a new numpy array?
    
    :param heading_array: array/list of numeric heading values
    :type heading_array: iterable
    :returns: Straightened headings
    :rtype: Generator of type Float
    """
    # Ref: flight_analysis_library.py
    
    head_prev = heading_array[0]
    yield head_prev
    offset = 0.0
    for heading in heading_array[1:]:
        diff = heading - head_prev
        if diff > 180:
            offset -= 360
        elif diff < -180:
            offset += 360
        else:
            pass # no change to offset
        head_prev = heading
        yield heading + offset
        

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
    s is the Laplace operator (think differentiation with time d/dt)
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

    result = np.copy(in_param.data)
    if initial_value is not None:
        result[0] = initial_value
    
    # Scale the time constant to allow for different data sample rates.
    tc = time_constant / hz

    x_term = []
    x_term.append (gain * 2.0 * tc) #b[0]
    x_term.append (-gain * 2.0 * tc) #b[1]
    
    y_term = []
    y_term.append (1.0 + 2.0 * tc) #a[0]
    y_term.append (1.0 - 2.0 * tc) #a[1]
    
    #TODO: Sort out what happens if the in_param array contains masked data.
    # May be OK if we can be sure the masked values aren't silly.

    result = signal.lfilter(x_term, y_term, result)
    masked_result = np.ma.array(result)
    masked_result.mask = in_param.mask
    return masked_result
