.. _Approach:

================
Approaches (App)
================

Description
===========

An ApproachNode stores information about approaches during the flight.

An ApproachItem object represents an individual approach and stores the following information if available:

* type — The type of the approach, either APPROACH, LANDING, GO_AROUND, TOUCH_AND_GO.
* slice — The slice of the approach at the Node's frequency.
* airport — Information about the target airport.
* runway — Information about the target runway.
* gs_est — The slice when the ILS Glideslope was established.
* loc_est — The slice when the ILS Localiser was established.
* ils_freq — The ILS Frequency during the approach.
* turnoff — The index which the aircraft turned off the runway of the approach was of type LANDING.
* lowest_lat — Latitude at the lowest point during the approach.
* lowest_lon — Longitude at the lowest point during the approach.
* lowest_lon — Heading at the lowest point during the approach.

ApproachItems are created by calling the ApproachNode's `create_approach` method.

There is only a single ApproachNode within the FlightDataAnalyzer named `Approach Information`.

ApproachNodes provide a list-like interface.

.. code-block:: python
   
   class ApproachTurbulence(DerivedParameterNode):
       def derive(apps=App('Approach Information'):
           for app in apps:
               ...

