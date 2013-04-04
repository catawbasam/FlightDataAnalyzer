.. _DerivedParameter:

=================
Derived Parameter
=================

Description
===========

A derived parameter is a **calculated** parameter derived from one or more other parameters. 

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class NodeName(DerivedParameterNode):
        '''
        Docstring
        '''
    
        units = 'unit'
    
        def derive(self, param1=P('Parameter One'), ...):
            ...

Units is the unit which gives the values stored in the parameter meaning.

Aligning and First Dependancies
-------------------------------

By default the derive method dependencies are aligned to the the first
available dependency. The frequency and offset can be forced by providing
align_frequency and/or align_offset attribtes to the Derived Parameters class.

For example to force all dependancies to a frequency of one htz with a zero offset.

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class NodeName(DerivedParameterNode):
        '''
        Docstring
        '''
    
        align_frequency = 1
        align_offset = 0
        units = 'unit'
    
        def derive(self, param1=P('Parameter One'), ...):
            ...

If you do not wish to align dependencies provide set the align attribute on
the Derived Parameters class to False. The Node will inherit the first
dependency's frequency/offset but self.frequency and self.offset can be
overidden within the derive method.

For example to not align dependancies and set the Derived Parameter to a frequency of one htz with a zero offset.

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class NodeName(DerivedParameterNode):
        '''
        Docstring
        '''
    
        align = false
        units = 'unit'
    
        def derive(self, param1=P('Parameter One'), ...):
            self.frequency = 1
            self.offset = 0
            ...

see :ref:`aligning-of-parameters` for more details of Aligning

As a rule of thumb always align the most important part. for example if you want a measurement align to the parameter you wish to take the measurement from. If you want a value at a specific time, align to KTI or Section.
There are exceptions to this rule.


Tutorial
========
A simple example of a Derived Parameter would be the true track heading of the aircraft which we can calculate by by adding Drift from track to the aircraft Heading.

We will start by creating the class with a suitable name, in this case TrackTrue. We will provide a docstring and the units the Value will be recorded in (deg for heading).

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class TrackTrue(DerivedParameterNode):
        '''
        True Track Heading of the Aircraft by adding Drift from track to the
        aircraft Heading.

        Range 0 to 360
        '''
        units = 'deg'

We now need a derive method which will create the array of values based on some dependancies. As we have already identified we will use the 'Heading True Continuous' and the 'Drift' parameters.
As we require both 'Heading True Continuous' and the 'Drift' parameters we do not require a can_operate method as the default behaviour is to require all dependancies are avaliable to run???.
Heading is primary parameter we are interested in so we will use this as the first dependancy which other dependancies will be aligned to.

.. note::
    We use a wrapper (**P()** here) to assist the programmer with IDE
    auto-completion of the **first** keyword argument, providing it with the
    attributes available on the expected data type being used.
    
    The name of the dependency is provided as a String.

.. code-block:: python

    def derive(self, heading=P('Heading True Continuous'), drift=P('Drift')):
        ...

All that is left is to assign self.array to the heading array plus the drift array. We use % (modulus) 360 as headings have a range of 0-360 degrees. It is good practive to add an inline comment here to inform other users of the reason for adding the arrays.

.. code-block:: python

        #Note: drift is to the right of heading, so: Track = Heading + Drift
        self.array = (heading.array + drift.array) % 360

The completed node will look as follows.

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class TrackTrue(DerivedParameterNode):
        '''
        True Track Heading of the Aircraft by adding Drift from track to the
        aircraft Heading.

        Range 0 to 360
        '''
        units = 'deg'
    
        def derive(self, heading=P('Heading True Continuous'), drift=P('Drift')):
            #Note: drift is to the right of heading, so: Track = Heading + Drift
            self.array = (heading.array + drift.array) % 360


Helper Functions
================

Can operate
-----------

Below are some helpful ways to implement the can operate methods.

:py:func:`analysis_engine.library.all_of`
    Returns True if all of the names are within the available list.
    
    for example if we need Altitude AAL and either Flap (L) or Flap (R)

    .. code-block:: python

        from analysis_engine.library import all_of
        
        @classmethod
        def can_operate(cls, available):
            return all_of(('Altitude AAL', 'Flap (L)'), available) or \
                   all_of(('Altitude AAL', 'Flap (R)'), available)

:py:func:`analysis_engine.library.any_of`
    Returns True if any of the names are within the available list.
    
    using the same example as above we could use

    .. code-block:: python

        from analysis_engine.library import any_of
        
        @classmethod
        def can_operate(cls, available):
            return 'Altitude AAL' in available and \
                   any_of(('Flap (L)', 'Flap (R)'), available)

As you can see in this example we can accoumplish the same goal using either Functions. The correct function for the job therefore comes down to readablillity. For this example we would use the 'any_of' piece of code.
