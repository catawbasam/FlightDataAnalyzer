. _ProcessFlight:

==============
Process Flight
==============

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


