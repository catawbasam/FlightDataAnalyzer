.. _Testing:

==============================
Testing the FlightDataAnalyser
==============================

----------------------------------
Preparing a Stripped Down HDF File
----------------------------------

To improve the AnalysisEngine test suite we should create assertions against
real data. Since nodes may require a complex dependency tree, creating nodes
and loading multiple numpy arrays from .npy files is a very slow process. It is
much easier to create tests against HDF files instead.

Since we only require a subset of parameters within an HDF file for our tests,
files should be stripped of unnecessary parameters to avoid storing large
data files within revision control. There is a utility within the HDFAccess
repository designed for this purpose. The following command will display help
information::

    # python HDFAccess/hdfaccess/utils.py strip -h

    usage: utils.py strip [-h]
                          input_file_path output_file_path parameters
                          [parameters ...]

    positional arguments:
      input_file_path   Input hdf filename.
      output_file_path  Output hdf filename.
      parameters        Store this list of parameters into the output hdf file.
                        All other parameters will be stripped.

    optional arguments:
      -h, --help        show this help message and exit

Example::

    python HDFAccess/hdfaccess/utils.py strip input.hdf5 output.hdf5 Airspeed "Altitude STD"

*input.hdf5* is the input filename, *output.hdf5* is the output filename and
the parameter names which follow will be copied to the output file - all other
parameters will be stripped. Parameter names which contain spaces must be 
wrapped in double quotes.

The utility will not raise an error if specified parameters are missing from
the input file, but will instead output which parameters are in the output
file::

    The following parameters are in the output hdf file:
     * Airspeed
     * Altitude STD

If no parameters are successfully copied then the following message will be 
displayed::

    No matching parameters were found in the hdf file.

Please check that the output hdf file is an acceptable size before adding it
to the repository and that it has been de-identified.