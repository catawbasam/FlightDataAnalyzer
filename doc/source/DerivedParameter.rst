.. _DerivedParameter:

=================
Derived Parameter
=================

Description
===========

A derived parameter is a **calculated** parameter derived from one or more
other parameters.

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class NodeName(DerivedParameterNode):
        '''
        Docstring
        '''
    
        units = 'unit'
    
        def derive(self, param1=P('Parameter One'), ...):
            ...

.. note::
    We use a wrapper (**P()** here) to assist the programmer with IDE
    auto-completion of the **first** keyword argument, providing it with the
    attributes available on the expected data type being used.
    
    The name of the dependency is provided as a String.

Units is the unit which gives the values stored in the parameter meaning.

Aligning and First Dependancies
-------------------------------

By default the derive method dependencies are aligned to the the first
available dependency. The frequency and offset can be forced by providing
align_frequency and/or align_offset attribtes to the Derived Parameters class.

For example to force all dependancies to a frequency of one Hz with a zero offset.

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

For example to not align dependancies and set the Derived Parameter to a
frequency of one htz with a zero offset.

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

.. warning::
   Do not return anything from a derive method as this will raise a UserWarning exception.

see :ref:`aligning` for more details on Aligning

As a rule of thumb always align the most important part. for example if you want a measurement align to the parameter you wish to take the measurement from. If you want a value at a specific time, align to KTI or Section.
There are exceptions to this rule.


.. _can-operate:

Can operate
-----------

The can_operate method is used to determine if the node (in this case the
Derived Parameter) has all the dependancies it requires to operate. The
default behaviour is to require all dependancies are avaliable to operate.

The following code returns true if 'Altitude AAL' is in the list of available
parameters, this will allow the Derived Parameter to operate.

.. code-block:: python

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL' in available


Here is a more complex example which uses python sets with loical operators
to handle multiple different combinations of parameters.

.. code-block:: python

    @classmethod
    def can_operate(cls, available):
        available = set(available)
        afr = 'AFR V2' in available and 'Airspeed' in available
        base_for_lookup = ['Airspeed', 'Gross Weight At Liftoff', 'Series',
                           'Family']
        airbus = set(base_for_lookup + ['Configuration']).issubset(available)
        boeing = set(base_for_lookup + ['Flap']).issubset(available)
        return afr or airbus or boeing

See :ref:`can-operate-helpers` for some usefull functions when working with
can_operate methods along with some more examples.

Multistate
----------

Multistate Derived Parameter Nodes have an additional values_mapping
attribute which is used to map values in the array to states. An example of
this is GearDown which which maps 0 to 'Up' and 1 to 'Down'.

.. code-block:: python

    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

Here is a simple example of a Multistate Parameter, as we may come accross in
a derive method. (using a M() wrapper here)

.. code-block:: python

    import numpy as np
    from analysis_engine.node import M
    
    spd_brk = M(name='Speedbrake Selected',
                array=np.ma.array([0, 1, 2, 0, 0] * 3),
                values_mapping={
                    0: 'Stowed',
                    1: 'Armed/Cmd Dn',
                    2: 'Deployed/Cmd Up',
                },)

We can look up both states and the values used by index.

.. code-block:: python

    >>> spd_brk.array[2]
    'Deployed/Cmd Up'
    
    >>> spd_brk.array.data[2]
    2

We can view the values mapping at any time by looking at the values_mapping
attribute of a Multistate Parameter

.. code-block:: python

    >>> spd_brk.values_mapping
    {0: 'Stowed', 1: 'Armed/Cmd Dn', 2: 'Deployed/Cmd Up'}

We can also see the reverse of the mapping by looking at the state attribe,
and of course look up the raw value used for a state.

.. code-block:: python

    >>> spd_brk.state
    {'Deployed/Cmd Up': 2, 'Stowed': 0, 'Armed/Cmd Dn': 1}
    
    >>> spd_brk.state['Armed/Cmd Dn']
    1

States can be used in combination with numpy functions. In this example
finding the locations in the array where the state is 'Deployed/Cmd Up'

.. code-block:: python

    >>> np.ma.where(spd_brk.array == 'Deployed/Cmd Up')
    (array([ 2,  7, 12]),)

Logging
-------

Each node has a logger attribute which should be used for logging messages.


.. code-block:: python

    self.logger.warning(
            "'AirspeedReference' will be fully masked because "
            "'Gross Weight' array could not be repaired.")

Tutorial
========

A simple example of a Derived Parameter would be the true track heading of
the aircraft which we can calculate by by adding Drift from track to the
aircraft Heading.

We will start by creating the class with a suitable name, in this case
TrackTrue. We will provide a docstring and the units the Value will be
recorded in (deg for heading).

.. code-block:: python

    from analysis_engine.node import DerivedParameterNode, P

    class TrackTrue(DerivedParameterNode):
        '''
        True Track Heading of the Aircraft by adding Drift from track to the
        aircraft Heading.

        Range 0 to 360
        '''
        units = 'deg'

We now need a derive method which will create the array of values based on
some dependancies. As we have already identified we will use the 'Heading
True Continuous' and the 'Drift' parameters. As we require both 'Heading True
Continuous' and the 'Drift' parameters we do not require a can_operate
method. Heading is primary parameter we are interested in so we will use this
as the first dependancy which other dependancies will be aligned to.

.. code-block:: python

    def derive(self, heading=P('Heading True Continuous'), drift=P('Drift')):
        ...

All that is left is to assign self.array to the heading array plus the drift
array. We use % (modulus) 360 as headings have a range of 0-360 degrees. It
is good practice to add an inline comment here to inform other users of the
reason for adding the arrays.

.. code-block:: python

        #Note: drift is to the right of heading, so: Track = Heading + Drift
        self.array = (heading.array + drift.array) % 360

.. warning::
    derive methods must set self.array.

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

.. _can-operate-helpers:

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

Here is the earlier example using all_of instead of sets to accoumplish the
same results.

.. code-block:: python

    from analysis_engine.library import all_of
    
    @classmethod
    def can_operate(cls, available):
        base_for_lookup = ['Airspeed', 'Gross Weight At Liftoff', 'Series',
                       'Family']
    
        afr = all_of(('AFR V2', 'Airspeed'), available)
        airbus = all_of(base_for_lookup + ['Configuration'], available)
        boeing = all_of(base_for_lookup + ['Flap'], available)
        return afr or airbus or boeing


As you can see in these examples we can accoumplish the same goal using either
Functions or sets or combinations of all of them. The correct function for the job therefore comes down to
readablillity and personal preference.

Derive
------

:py:func:`analysis_engine.library.np_ma_masked_zeros_like`
    Creates a masked array filled with masked values. The unmasked data values
    are all zero. The array is the same length as the arrray passed in. This is
    very useful for setting self.array in derive methods for derived parameters
    which have no valid values.

    .. code-block:: python

        >>> import numpy as np
        >>> from analysis_engine.library import np_ma_masked_zeros_like
        
        >>> an_array = np.ma.arange(10, 33)
        masked_array(data = [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32],
                     mask = False,
               fill_value = 999999)
        
        >>> np_ma_masked_zeros_like(an_array)
        masked_array(data = [-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --],
                     mask = [ True  True  True  True  True  True  True  True  True  True  True  True
          True  True  True  True  True  True  True  True  True  True  True],
               fill_value = 999999)