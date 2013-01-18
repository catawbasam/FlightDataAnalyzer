Introduction
============

A tool for analysing HDF5 files containing engineering unit representation of 
flight data. It provides the following utilities:

* Flight phase detection.
* Split flight data into multiple flight segments.
* Deriving parameters from others.
* Creation of flight attributes, such as take and landing runways.
* Calculation and creation of KPVs and KTIs.

Project sponsored by `Flight Data Services`_ and released under the Open 
Software License (`OSL-3.0`_).

NOTE! It is not currently possible to run FlightDataAnalyzer from source since
it is dependant on the `Utilities` module which contains proprietary code we
can't release. We are in the process of liberating the `Utilities` module and
will release `FlightDataUtilities` in Q1 2013 as a fully Open Source
replacement. At which time we will the publish "run from source" documentation
which will be available via the `FlightDataCommunity_` website.

Installation
------------

Package requires ``pip`` for installation.
::

    pip install git+https://github.com/FlightDataServices/FlightDataAnalyser.git

Source Code
-----------

Source code is available from `GitHub`_:

* https://github.com/organizations/FlightDataServices/FlightDataAnalyser

Documentation
-------------

Documentation is available from the `Python Package Index`_:

* http://packages.python.org/FlightDataAnalyser/

.. _Flight Data Services: http://www.flightdataservices.com/
.. _Flight Data Community: http://www.flightdatacommunity.com/
.. _OSL-3.0: http://www.opensource.org/licenses/osl-3.0.php
.. _GitHub: https://github.com/
.. _Python Package Index: http://pypi.python.org/

.. image:: https://cruel-carlota.pagodabox.com/9932acf5231d508d118026b0e621d296
    :alt: githalytics.com
    :target: http://githalytics.com/FlightDataServices/FlightDataAnalyzer
