.. _KeyPointValue:

=====================
Key Point Value (KPV)
=====================

-----------
Description
-----------

A Key Point Value is a **Value** of interest measured at a point during a
flight. There may be between 0 and many instances of a KPV throughout a
flight.

--------
Benefits
--------

 * ensures a consistent measurement technique is employed accros all aircraft
 * allows direct comparisons(not dependent on thresolds/limits)
 * can be used to detect events with the use of thresolds/limits
 * distributions/histograms

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

AltitudeMax no can operate defined so both "Altitude STD Smoothed" and
"Airborne" sections are required for AltitudeMax to run.
to return max value, prevents data spikes whilst on the
ground. aligned to Altitude STD Smoothed. uses create_kpvs_within_slices.
resulting Key Point Value in feet ("ft")

----------------
Helper Functions
----------------

 * create_kpv
 * create_kpvs_at_ktis
 * create_kpvs_from_slice
 * create_kpvs_where_state
 * create_kpv_within_slices
 * create_kpvs_outside_slices

--------
Tutorial
--------


