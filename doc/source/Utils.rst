.. _NumpyTips:

Utils
=====

trimmer
-------

It is generally a good principle to keep large binary files outside of a
repository's version control system, otherwise the history of the repository
will grow and become unmanageable.

*trimmer* is a utility within AnalysisEngine.analysis_engine.utils which is
designed to help avoid this problem when preparing HDF test data.

derived_trimmer() takes
the following as arguments:

 * Input HDF file path which contains parameters.
 * List of Node names which are required. Parameters which these Nodes are dependent upon will be kept, while all others will be trimmed.
 * Output HDF file path containing only those parameters required by the Nodes specified.

derived_trimmer() can be run from the utils.py file directly by providing the trimmer argument.

Example usage:

 ``python utils.py trimmer source_path.hdf5 "True Heading" "FDR Takeoff Pilot" output_path.hdf5

