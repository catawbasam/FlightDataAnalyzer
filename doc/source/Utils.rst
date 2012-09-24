.. _Utils:

Utils
=====

.. _trimmer:

trimmer
-------

It is generally a good principle to keep large binary files outside of a
repository's version control system, otherwise the revision history
will grow and become unmanageable.

*trimmer* is a utility within AnalysisEngine.analysis_engine.utils which is
designed to help avoid this problem when preparing HDF test data::

    # python utils.py trimmer -h
    
    usage: utils.py trimmer [-h]
                            input_file_path output_file_path nodes [nodes ...]
    
    positional arguments:
      input_file_path   Input hdf filename.
      output_file_path  Output hdf filename.
      nodes             Keep dependencies of the specified nodes within the output
                        hdf file. All other parameters will be stripped.
    
    optional arguments:
      -h, --help        show this help message and exit

Example::

    # python utils.py trimmer input.hdf5 output.hdf5 "Altitude AAL" "Heading Continuous"

    The following parameters are in the output hdf file:
     * Altitude Radio (B)
     * Altitude Radio (A)
     * Altitude Radio (C)
     * Airspeed
     * Altitude STD
     * Heading

*input.hdf5* is the input filename, *output.hdf5* is the output filename and
both *"Altitude AAL"* and *"Heading Continuous"* are node names. Parameter names
which contain spaces must be wrapped in double quotes. The output of the
command shows that the dependencies of these nodes have been copied into
*output.hdf5*.