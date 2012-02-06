.. _NumpyTips:

NumpyTips
=========

* masking using array[:4] = np.ma.masked
* 


Moving Average:
<http://argandgahandapandpa.wordpress.com/2011/02/24/python-numpy-moving-average-for-data/>

In-place conversion of a numpy array
<http://stackoverflow.com/questions/4389517/in-place-type-conversion-of-a-numpy-array>

Find nearest value
<http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array>

Fill in masked values with nearest neighbour
<http://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays>

Slow pattern matching for multi-axis array using apply_along_axis

.. code-block:: python
    :linenos:
    
    # alternative pattern matching (slow)
    def equal_to(a, values):
        return np.ma.allequal(a, values, fill_value=False)
    
    stacked = vstack_params(flap, slat, aileron)
    for state, values in map_a330:
        # apply_along_axis is very slow!!
        match = np.ma.apply_along_axis(equal_to, axis=0, arr=stacked, test=values)
        self.array[match] = state
  
