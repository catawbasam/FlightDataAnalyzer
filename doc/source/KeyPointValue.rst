.. _KeyPointValue:

=====================
Key Point Value (KPV)
=====================

-----------
Description
-----------

**Value** measured at a point during a flight.

--------
Benefits
--------

measurement technique is consistent accros all aircraft
allows direct comparisons(not dependent on thresolds/limits)
can be used to detect events with the use of thresolds/limits
distributions/histograms

-------
Example
-------

A simple example of a Key Point Value is AltitudeMax, The maximum altitude
during the flight.

Here is the code for AltitudeMax::

    class AltitudeMax(KeyPointValueNode):
        '''
        Maximum Altitude STD
        '''
    
        units = 'ft'
    
        def derive(self,
                   alt_std=P('Altitude STD Smoothed'),
                   airborne=S('Airborne')):
    
            self.create_kpvs_within_slices(alt_std.array, airborne, max_value)

----------------
Helper Functions
----------------

 * create_kpv
 * create_kpvs_at_ktis
 * create_kpvs_from_slice
 * create_kpvs_where_state
 * create_kpv_within_slices
 * create_kpvs_outside_slices