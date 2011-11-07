import math
import numpy as np

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


def rate_of_change(data, half_width=5):
    '''
    @param to_diff: Parameter object with .data attr (masked array)
    
    Differentiation using the xdot(n) = (x(n+hw) - x(n-hw))/w formula.
    Width w=hw*2 and this approach provides smoothing over a w second period.
    The default 10-second period is suitable for flight phase computations where
    a high degree of smoothing minimises unnecessary phase changes.
    '''
    # Set up an array of masked zeros for extending arrays.    
    pad=np.ma.arange(20)
    pad[:]=np.ma.masked
    # Initialise 
    temp = []
    # Process the data array
    temp = (data[2*half_width:] - data[:-2*half_width])/float((2 * half_width))
    temp = np.ma.concatenate([pad[:half_width],temp[:],pad[:half_width]])

    #if period=='sec':
        #self.units = self.units+'/sec'
    #elif period=='min':
        #self.units = self.units+'/min'
        #self.data *= 60
    #else:
        #print 'Error - invalid rate of change'

    return temp

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


def shift(first, second):
    """
    TODO: This Docstring!
    
    TODO: need a more in-your-face name than shift, also
    rename arguments to be more meaningful - first is the "source_param" ...
    
    :type first, second: Parameter objects with attributes:
    :type first.data: masked array
    :type first.fdr_offset: float
    :type first.hz: int
    
    :returns
    """
    # Here we create a masked array to hold the second term of a function
    # that has the same sample rate and timing offset as the first term,
    # in preparation for later combination.
    result = np.ma.empty_like(first.data)
    # Get the timing offsets, comprised of word location and possible latency.
    tp = first.fdr_offset
    ts = second.fdr_offset
    # Get the sample rates for the two parameters
    wp = first.hz
    ws = second.hz
    # Express the timing disparity in terms of the secondary paramter sample interval
    delta = (tp-ts)*ws
    # Compute the sample rate ratio (in range 10:1 to 1:10 for sample rates up to 10Hz)
    R = wp/float(ws)

    # Each sample in the first (primary) parameter may need different combination parameters
    for i in range(wp):
        bracket=(i/R+delta)
        # Interpolate between the hth and (h+1)th samples of the secondary
        h=int(math.floor(bracket))
        h1 = h+1
        # Linear interpolation coefficients
        b = bracket-h
        a=1-b
        ##for testing: print wp, ws, i, bracket, h, h1, a, b

        if h<0:
            # We can't compute the inital values as the secondary parameters we need
            # are out of range, so not available. Mask the result and work on the
            # later seconds of data to the end.
            result[i] = np.ma.masked
            result[i+wp::wp]=a*second.data[h+ws:-ws:ws]+b*second.data[h1+ws::ws]
        elif h1>=ws:
            # We can't compute the final values as the secondary runs out of data.
            # Again, mask the final values and run from the beginning to almost the end
            # of the arrays.
            result[i-R]=np.ma.masked
            result[i:-wp:wp]=a*second.data[h:-ws:ws]+b*second.data[h1::ws]
        else:
            # Sheer bliss. We can compute results across the whole range of the data.
            result[i::wp]=a*second.data[h::ws]+b*second.data[h1::ws]
            ##self.path = filenameandpath -- CJ commented out - what's this here for?

    ##result.fdr_word_rate = first.word_rate -- CJ commented out - adding an attr to np.ma is bad practice
    return result


def slope (parameter, half_width):
    '''
    Differentiation using the xdot(n) = x(n+1) - x(n-1) formula.
    '''
    # Set up an array of masked zeros for extending arrays.    
    pad = np.ma.arange(20)
    pad[:] = np.ma.masked
    slope = (parameter[2*half_width:] - parameter[:-2*half_width]) / float((2 * half_width))
    return np.ma.concatenate([pad[:half_width],slope[:],pad[:half_width]])
    


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
        



