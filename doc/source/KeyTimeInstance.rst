.. _KeyTimeInstances:

=======================
Key Time Instance (KTI)
=======================

-----------
Description
-----------

A key time instance is an **index** into a flight

--------
Benefits
--------

Allows one time identification of an instance in a flight eg. Touchdown

Can be used in Key Point Value (KPV) to look up value at that point
Can be used to create phase eg. x -> touchdown

-------
Example
-------

The resulting Acceleration Vertical is scaled in g, and retains the 1.0 datum and positive upwards. The code looks like this::
    
    pitch_rad = np.radians(pitch.array)
    roll_rad = np.radians(roll.array)
    resolved_in_roll = acc_norm.array*np.ma.cos(roll_rad) - acc_lat.array * np.ma.sin(roll_rad)
    self.array = resolved_in_roll * np.ma.cos(pitch_rad) + acc_long.array * np.ma.sin(pitch_rad)

----------------
Helper Functions
----------------

 * create_kti
 * create_ktis_on_state_change
 * create_ktis_at_edges