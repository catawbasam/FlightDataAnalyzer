.. _SplitSegments:

=====================
Split HDF to Segments
=====================

It's so much easier to work on a single flight segment!

Core Parameters
---------------

These are the required minimum parameters for flight data analysis:

1. Airspeed
2. Altitude STD
3. Heading

If the Logical Frame Layout has a `reliable frame counter` this is included too:

4. Frame Counter

Any more parameters than these are a bonus - oh and probably mandated (e.g.
Pitch, Normal Acceleration, etc.) ;)


The Process
-----------

#. Validate Aircraft
#. Split Segments

   #. Find Split boundaries between flights
   #. Identify the segment type
   
#. Write Segment to new HDF file
#. Append Segment Info

   #. Calculate Timebase to establish Start Datetime
   #. Establish Go Fast Datetime
   #. Generate Airspeed Hash - Unique Identifier


Validate Aircraft
~~~~~~~~~~~~~~~~~

.. note::
    Not currently implemented.

Using information known about the aircraft, validate that the data being
processed has the same information. If not, `raise AircraftMismatch`.

There are various parameters in the logical frame layouts which can assist in
uniquely identifying the correct aircraft such as:

* Aircraft Tail (Registration number)
* Aircraft Identifier
* Fleet Identifier
* Manufacturer Serial Number (MSN)
* Manufacturer Code
* Mandatory Software Part Number Code
* QAR Serial Number
* etc...


Split Segments
~~~~~~~~~~~~~~

Essentially we want to separate flights This is easiest to understand in Pseudo code::

    start = None
    # Loop over any slow sections to split the segments within
    for slow_section in airspeeds_below(AIRSPEED_THRESHOLD):
        split_index = find_split(slow_section)
        segments.append(slice(start, split_index))
        start = split_index
    # Add left over data
    segments.append(slice(start, None))
        
    def find_split(slow_section):
        #1. Use frame counter jumps (if reliable)
        if reliable_frame_counter:
            frame_jump = find_frame_counter_jump_within(slow_section)
            if frame_jump:
                return frame_jump
                
        #2. Search where Engines (N1, N2, NP) are at their minimums
        min_overall_value = find_minimum_of_values(ENGINE_SPLIT_PARAMS)
        if min_overall_value < MINIMUM_SPLIT_PARAM_VALUE:
            return min_overall_value
            
        #3. Find a point where the aircraft was not turning (parked)
        not_turning = rate_of_turn('Heading') == 0 
        return not_turning / 2
        



Calculate Timebase
~~~~~~~~~~~~~~~~~~

Calculate the most regular reoccurring timebase throughout the flight. Uses the following parameters:

* Year
* Month
* Day
* Hour
* Minute
* Second

Where a parameter from the above list does not exist, it will populate it
from the `fallback_dt` datetime object. This is a powerful tool; for example
if you know something about when this datafile was flown such as the Month
and Year, this will dramatically improve the accuracy of the calculated date
than relying on a system paramter such as the current date/time which will
mean that each time you reprocess the same flight you will get a different
datestamp.

There are a few restrictions; the datetime found cannot be in the future -
Flight Data Analysis occurs `after` the flight, not before! If
`settings.MAX_TIMEBASE_AGE` is set, the data cannot be older than
`settings.MAX_TIMEBASE_AGE` many days ago.

`Strengths`: You get a reliable timestamp that increases linearly through the dataset.

`Weakness`: Data corruption can disturb the timestamp. For example, should
the dataframe counter reset you may get superframe padding to the end of the
current superframe and up to the the recorded position in the next
superframe. This can mean up to 63 seconds of additional padding in a 64
second superframe.


Airspeed Hash
~~~~~~~~~~~~~

To uniquely identify a flight we create a unique identifier from the flight's
airspeed where it was above the threshold `settings.AIRSPEED_THRESHOLD`. All
the airspeed values are compiled into a `sha256` checksum.

No two flights are likely to have exactly the same airspeed values for every
second in the flight, but the recorded airspeed will be identical no matter
where it was sourced.

A possible weakness here is that if the scaling of the Airspeed paramter is
changed then this unique idenfier will be different for the same flight
processed before and after the change. However one can assume that as Airspeed
is one of the core parameters, this will be correctly defined in the Logical
Frame Layout.
