.. _SectionNode:

=============================
Section Nodes (Flight Phases)
=============================

.. _interval-theory:
---------------
Interval Theory
---------------

A mathematical `interval` represents all the numbers between two given
values. Much like a python slice represents a subset of data, only interval's
bounds can be decimal and optionally include or exclude the bound values.

Sections are single intervals which allow us many options:

.. code-block:: none

    left-closed, right-open: [a, b)  ≡  a <= x < b  ≡  slice(a, b)
    left-bounded, right-unbounded, left-closed: [a, inf)  ≡  x >= a  ≡  slice(a, None)
    left-unbounded and right-bounded, right-open: (-inf, b)  ≡  x <= b  ≡  slice(None, b)
    
Including the ability to do fully closed intervals which slices cannot afford us:

.. code-block:: none

    proper and bounded, closed: [a, b]  ≡  a <= x <= b

To define a right-closed interval (inclusive of b) use endpoint=True 
(default) as used by `numpy linspace <http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_

    

Python slicing uses half-open (left-closed, right-open) intervals to make
arithmetics easier (see comment in link). Slicing lists raises a
TypeError if floats are used as the slice start or stop, but in numpy the 
start and stop are floored to their lower integer. See this well answered 
`question on Stackoverflow <http://stackoverflow.com/questions/9421057/numpy-indexing-questions-on-odd-behavior-inconsistencies#answer-9421268>`_

.. 
    It is difficult to define a flight phase as half-open as you often
    determine the closed boundaries of the phase based on information from
    the available data. In addition we often work on parameters at different
    frequencies which means the start and stop positions must be easily
    aligned to other frequencies and offsets requiring that the start/stop
    positions become decimal values.



Section
-------

A `Section` is a subclass of `Interval`, adding functionality with additional
methods such as size, duration, and slice.


.. code-block:: python
    :linenos:
    
    Section()


-----------
IntervalSet
-----------

Multiple `Intervals` can be represented by a single `IntervalSet`.

.. code-block:: python

    >>> from interval import IntervalSet, Interval
    >>> IntervalSet(items=[Interval(2, 10), Interval(15, 18)])
    IntervalSet([Interval(2, 10, lower_closed=True, upper_closed=True), Interval(15, 18, lower_closed=True, upper_closed=True)])
    >>> print IntervalSet(items=[Interval(2, 10), Interval(15, 18)])
    [2..10],[15..18]

The set concept is used (rather than a list) as no two `Intervals` contained
within the set can overlap. If an Interval is added to the set which does
overlap, it is extended:

    >>> iset = IntervalSet([Interval(2, 10)])
    >>> iset.add(Interval(5, 20))
    >>> iset
    IntervalSet([Interval(2, 20, lower_closed=True, upper_closed=True)])
    >>> print iset
    [2..20]


SectionNode
-----------

Inheriting from IntervalSet, these provide the required base `Node`
functionality including the `get_aligned` method.


.. _FlightPhaseNode:

FlightPhaseNode
---------------

One can be in multiple Flight Phases at the same time and each flight phase
can occur multiple times during a flight.

SectionNodes are the basis for all Flight Phases.

One be `Climbing` multiple times during a flight, but you cannot overlap two
Climbing intervals as one cannot be climbing twice at the same time, alas one
is either Climbing or not Climbing (`~climbing`). This suits the set theory
perfectly.

    >>> climbing = FlightPhaseNode([Section(60, 1000), Section(1200, 1800)])
    >>> print climbing
    [60..1000],[1200..1800]
    >>> print ~climbing
    (...60),(1000..1200),(1800...)

However, one can be Climbing and Turning at the same time. Therefore another
Flight Phase is created for Turning which is independant of the Climbing phase.

    >>> turning = FlightPhaseNode([Section(100, 500), Section(1025, 1050), Section(1150, 1400)])
    >>> print turning
    [100..500],[1025..1050],[1150..1400]

Note that there is both overlap in the intervals and segregation.

To find when one is both Climbing and Turning:

    >>> print climbing & turning
    [100..500],[1200..1400]

To find when one is Turning but not Climbing:

    >>> print turning & ~climbing
    [1025..1050],[1150..1200)
    
Note that the 1200 is open, therefore all the values from 1150 up to but
excluding 1200 are encompassed by this interval.
