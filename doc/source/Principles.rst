.. _Principles:

Principles of Flight Data Processing
====================================

* Offsets
* Align of all dependencies to the first available dependency
* Dependency tree
* Testing
* Inputs and Outputs

Flight Data Characteristics
---------------------------

POLARIS is a data analysis suite designed specifically to perform computations on recorded flight data. 
Flight data recorded on aircraft Flight Data Recorders (FDRs) has specific characteristics which we have to 
address in such a system.

The first characteristic of flight data is that the data is recorded in a compressed binary format with the 
accuracy of some parameters degraded to reduce the length of the recorded binary number. 
An initial step in processing is therefore to convert the binary values into their engineering units, 
and this is not a trivial process. 
Different aircraft types, variants and even recording equipment manufacturers introduce different data 
formats and the permutations of data format run into the many hundreds. 
Each needs a specific set of decoding algorithms and while many features can be standardised, it is 
often necessary to develop data format-specific algorithms to cater for special cases. 
This is indeed an area where special cases are common !

The second characteristic of flight data is that the parameters are not all sampled at the same 
rate or latency and so before any two recorded parameters can be combined we need to look at the 
timing issues in some detail.

Data Latency
------------

When computing the sample point for a conventional parameter, the time from the start of the data frame is given by::

 T = n.dt

where n is the number of samples from the start of the frame and dt is the sample interval. 

For a parameter with latency, the time from the start of the data frame is given by::
    
 T = n.dt – L

Where L is the latency.

It follows that parameters with large latency sampled close to the start of the frame can have been sampled during the 
preceding frame.

A typical value of L might be 50 mS and with data sampled at 256 wps, this corresponds to shifting a data 
point 1/20th second or 13 samples earlier than its position implies.

Computing With Data Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The problem of computing in the presence of data latency is illustrated in the diagram below. Here two parameters 
with differing sample rates and latency have been represented by 'a' and 'b' and a calculation has been performed
at times representing 0, 1, 2 seconds into the data. This is typical of analysis systems that perform periodic computations.

.. image:: images/data processing before POLARIS.jpg

The computed values, represented by the green squares, do not lie on the correct result path and these errors 
can build surprisingly rapidly. As an example, FDS had one algorithm for computing the takeoff where the compuation lag 
was so bad that the radio altimeter readings had reached almost 70ft at the point of computed takeoff.

It is possible to keep such errors under control, but it would be better not to have such errors in the first place.

By contrast, POLARIS calculations are exact in time for the primary parameter in a computation and accept an interpolated
estimate of the secondary parameter(s). Here the results can be seen to lie exactly on the optimal solution line.

.. image:: images/data processing with POLARIS.jpg

One mathematical issue here is that the results are non-communtative. That is to say::
    
    a + b != b + a
and::

    a * b != b * a

We will see later how this affects the implementation of the algorithms, but suffice to say for now that the 
implementation takes the effort away from the programmer and end users will just see more accurate answers.

