.. _Attribute:

==============
Attributes (A)
==============

Description
===========

An Attribute is an object which stores a single value of any type. The Attribute class's constructor receives two arguments—name and value.

   .. code-block:: python
   
   >>> from analysis_engine.node import Attribute
   >>> attr = Attribute('Flight Number', '8608')
   >>> print attr
   Attribute('Flight Number', '8608')
   >>> print attr.name
   Flight Number
   >>> print attr.value
   8608

FlightAttributeNode (A)
-----------------------

A FlightAttributeNode is a Node which also stores a single value of any type. Unlike other Node types, Attributes do not store time-indexed data and therefore cannot be aligned.

Two attributes are created by the `NodeManager` and universally available as dependencies:

* `Start Datetime` — The start datetime of the flight data within the HDF file.
* `HDF Duration` —  The duration of flight data stored within the HDF file.

The `set_flight_attr` method assigns the value of a FlightAttributeNode:

.. code-block:: python
   
   from datetime import timedelta
   from analysis_engine.node import FlightAttributeNode
   
   class FiveMinutesAfterStartOfData(FlightAttributeNode)
       def derive(self, start_datetime=A('Start Datetime'):
           five_minutes_after = start_datetime.value + timedelta(5 * 60)
           self.set_flight_attr(five_minutes_after)

Other attributes are created from optional aircraft and Achieved Flight Record (AFR) information provided to process_flight:

* `Tail Number` — The tail number of the aircraft.
* `Identifier` — The identifier of the aircraft.
* `Manufacturer` — The manufacturer of the aircraft.
* `Manufacturer Serial Number` — The serial number of the aircraft.
* `Model` — The model of the aircraft.
* `Series` — The series of the aircraft.
* `Frame` — The logical frame layout used to process the raw flight data file with the FlightDataConverter.
* `Family` — The family of the aircraft.
* `Main Gear To Altitude Radio` — The distance in metres from the main gear to the Altitude Radio measuring device.
* `Wing Span` — The wing span of the aircraft in metres.

* `AFR Flight ID` — The flight ID as provided within the Achieved Flight Record.
* `AFR Flight Number` — The flight number as provided within the Achieved Flight Record.
* `AFR Type` — The type of flight as provided within the Achieved Flight Record.
* `AFR On Blocks Datetime` — The on blocks datetime of the flight as provided within the Achieved Flight Record.
* `AFR Off Blocks Datetime` — The off blocks datetime of the flight as provided within the Achieved Flight Record.
* `AFR Takeoff Datetime` — The takeoff datetime of the flight as provided within the Achieved Flight Record.
* `AFR Takeoff Pilot` — The takeoff pilot of the flight as provided within the Achieved Flight Record.
* `AFR Takeoff Gross Weight` — The takeoff gross weight in kilograms as provided within the Achieved Flight Record.
* `AFR Takeoff Fuel` — The takeoff fuel weight in kilograms as provided within the Achieved Flight Record.
* `AFR Landing Datetime` — The landing datetime of the flight as provided within the Achieved Flight Record.
* `AFR Landing Pilot` — The landing pilot of the flight as provided within the Achieved Flight Record.
* `AFR Landing Gross Weight` — The landing gross weight in kilograms as provided within the Achieved Flight Record.
* `AFR Landing Fuel` — The landing fuel weight in kilograms as provided within the Achieved Flight Record.
* `AFR V2` — The V2 used in knots as provided within the Achieved Flight Record.
* `AFR Vapp` — The Vapp used in knots as provided within the Achieved Flight Record.
* `AFR Vref` — The Vref used in knots as provided within the Achieved Flight Record.
* `AFR Takeoff Airport` — Takeoff airport information as provided within the Achieved Flight Record.
* `AFR Landing Airport` — Landing airport information as provided within the Achieved Flight Record.
* `AFR Destination Airport` — Destination airport information as provided within the Achieved Flight Record.
* `AFR Takeoff Runway` — Takeoff runway information as provided within the Achieved Flight Record.
* `AFR Landing Runway` — Landing runway information as provided within the Achieved Flight Record.
