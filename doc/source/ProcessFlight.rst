.. _ProcessFlight:

==============
Process Flight
==============

:py:func:`analysis_engine.process_flight.process_flight`

The process is as follows:

#. Take a list of available Nodes by finding Classes within modules listed in settings.NODE_MODULES 
#. Get the requested parameters and establish their dependency tree
#. Establish the parameter process order
#. Determine which parameters are used most often and therefore best to cache in memory
#. For each parameter in the process order:

   #. Align parameters and offsets (interpolation used - see :ref:`aligning`)
   #. Call get_derived method
   #. Check return was correct (basic checks for within dataset)
   #. Cache / write to HDF

#. GeoLocate and Timestamp KPV / KTI data.


Required Params
---------------

Used for requesting a subset of parameters to be processed.

If no required_params are provided, it will assume all Nodes are required!

A log messages warns when requested parameters are not available. If you
requested all parameters, this is likely to be a long list!

To process a subset of parameters, send a list of parameters to the
**required_params** keyword argument.::

   >>> process_flight('FlightDataAnalyzer-server/tests/test_data/Specimen_Flight.hdf5', aircraft_info={}, required_params=['Mach Max'], include_flight_attributes=False)


process_flight stores all derived parameters as new series within the HDF
file and returns all other objects within a dictionary::

   {
      'approach': [], 
      'phases': [], 
      'flight': [], 
      'kti': [], 
      'kpv': [],
   }
