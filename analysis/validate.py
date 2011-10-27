import numpy as np

class Limit(object):
    """
    Limit class used for storage of parameter limits.
    """
    def __init__(self, min_limit=None, max_limit= None, roc_limit = None):
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.roc_limit = roc_limit
        
        
def arinc_mask(data):
    """
    FIXME: Do not edit array in-place, return new masked array / copy.
    """
    # Invalid ARINC 429 standard data is characterised by frame lengh toggling
    # from one value to another, so this detects the 4 on / 4 off sample
    # patterns.
    
    # Compute an array of differences across the 4-second interval
    step4 = abs(data[:-4] - data[4:])
    i=0
    end = len(data)-4
    for j in range(end):
        if (step4[i] == step4[j]) and (step4[i]>50.0): #Need to make this a variable, but 50kts will do for now.
            pass
        else:
            if (j-i) > 8:
                data[i:j+4] = np.ma.masked
            i=j
    if (j-i) > 8:
        data[i:j+5] = np.ma.masked
    return data


def min_max_limits(param, lower_bound, upper_bound, spike_limit, spike_spread, rate):
    '''
    This takes a masked array and calculates the average of the points
    +/- spread seconds apart. In computing the average, points at the ends 
    of the data are lost, so replaced by null values to maintain the
    data length. Values that differ from the average of the
    points before and after by greater than limit are then masked before
    returning the array.

    The default spike_spread allows for single frame corruption to be eliminated.

    '''
    # Set up an array of masked zeros for extending arrays.    
    pad=np.ma.arange(20)
    pad[:]=np.ma.masked

    first_mask = np.ma.masked_outside(param, lower_bound, upper_bound)    

    # The number of samples to look across is the time x sample rate (Hz)
    N = spike_spread * rate
    ### TODO:  if (N>20) raise error condition ###

    second_mask = (first_mask[2*N:] + first_mask[:-2*N])/2

    # Restore the original data length, as N points have been lost in this process at both ends.
    restored_mask = np.ma.concatenate([pad[:N],second_mask[:],pad[:N]])

    # The spike height is the difference between the original data and the local means.
    spike_height=abs(first_mask-restored_mask)

    return np.ma.masked_where(spike_height>spike_limit,param)



        
def check_for_change(data):
    """
    Checks for any variation in the data. Raises exception if data is empty.
    
    :param data: a numpy masked array of floats
    :type data: numpy.ma.array
    :returns: True if variation in data
    :rtype: bool
    """
    
    return min(data) - max(data) != 0
    #return len(set(data)) != 1
    #first_value = data[0]
    #for value in data[1:]:
        #if value != first_value:
            #return True
    #return False
        

def validate(param, limit, test_arinc=False):
    """
    Calls a series of validation functions based on the limits provided and
    whether requested to test for ARINC behaviours.
    
    Each test appends to the masked array. Returns a masked array with the
    original data.
    
    :param param: Numpy masked_array
    :type param: np.ma.array([data], [mask])
    :param limit: Accessor with attributes min_limit, max_limit, roc_limit
    :type limit: class Limit
    :param test_arinc: Whether to test for ARINC behaviour (flag in LFL)
    :type test_arinc: Boolean
    :return: Numpy masked array
    :rtype: np.ma.array([data], [modified_mask])
    """
    changed = check_for_change(param)
    if not changed:
        # nothing to do!
        return param
    
    if test_arinc:
        param = validate_arinc(param)
        
    if limit.min_limit or limit.max_limit:
        param = validate_min_max(param, limit.min_limit, limit.max_limit)
        
    if limit.roc_limit:
        param = validate_rate_of_change(param, limit.roc_limit)
        
    return param


        
