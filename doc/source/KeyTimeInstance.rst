.. _KeyTimeInstance:

=======================
Key Time Instance (KTI)
=======================


Description
===========

A Key Time Instance is a point of interest on a flight. A KTI allows one time
identification of an instance in a (flight eg. Touchdown) allowing quick
access to this point in the flight and ensureing a consistent index is used.
The key attribute of a KTI is the **index** which indicates how far into the
flight in seconds the moment of interest occurred.

Boiler plate code

.. code-block:: python

    from analysis_engine.node import KeyTimeInstanceNode

    class NodeName(KeyTimeInstanceNode):
        '''
        Docstring
        '''
        def derive(self, param1=P('Parameter One'), ...):
            ...


.. _ListNode

List nodes
----------

As Key Time Instance Nodes inhierit from List Nodes they are capable of
storing multiple occurrences of KTI's. An example would be creating indexes
at Gear Down selections. This can occur several times per flight so the Key Time Instance Node will contain 

.. code-block:: python

    class GearDownSelection(KeyTimeInstanceNode):
        '''
        Instants at which gear down was selected while airborne.
        '''
    
        def derive(self,
                   gear_dn_sel=M('Gear Down Selected'),
                   airborne=S('Airborne')):
    
            self.create_ktis_on_state_change('Down', gear_dn_sel.array,
                                             change='entering', phase=airborne)


.. _FormattedNameNode

Formatted name nodes
--------------------

KeyTimeInstanceNode Formatted Name Nodes creates a name from applying a combination of
replace_values and kwargs as string formatting arguments to self.NAME_FORMAT.

For example if you want to create indexes as a countdown on the minute for
the last 5 minutes of the flight instead of creating a KTI node for each of
the 5 indexes required can use single node by providing a NAME_FORMAT
attribute with a string template and passing a key work argument (kwarg) into
the create_kti function. These use python string substitution to create the
correct name for the minute to touchdown.

.. code-block:: python

    class MinsToTouchdown(KeyTimeInstanceNode):
        NAME_FORMAT = "%(time)d Mins To Touchdown"
        NAME_VALUES = {'time': [5, 4, 3, 2, 1]}
    
        def derive(self, touchdowns=KTI('Touchdown')):
            for t in self.NAME_VALUES['time']:
                ...
                self.create_kti(index, time=t)
            

Benefits
========

* ensures a consistent index value is used accross all derived parameters
* Can be used in Key Point Value (KPV) to look up value at that point
* Can be used to create phase eg. x -> touchdown


Tutorial
========

A simple example of a Key Time Instance is LocalizerEstablishedStart, The index
where ILS Localizer is first Established.

.. code-block:: python

    class LocalizerEstablishedStart(KeyTimeInstanceNode):
        '''
        The index where ILS Localizer is first Established.
        '''
        def derive(self, ilss=S('ILS Localizer Established')):
            for ils in ilss:
                self.create_kti(ils.slice.start)

.. warning::
   do not return anything from a derive method as this will raise a UserWarning exception.


Helper Functions
================

:py:meth:`analysis_engine.node.KeyTimeInstanceNode.create_kti`
    Creates a KeyTimeInstance with the supplied index.

:py:meth:`analysis_engine.node.KeyTimeInstanceNode.create_ktis_on_state_change`
    Create KTIs from multistate parameters where data reaches and leaves given state.

    Its logic operates on string representation of the multistate parameter, not on the raw data value.

:py:meth:`analysis_engine.node.KeyTimeInstanceNode.create_ktis_at_edges`
    Create one or more key time instances where a parameter rises or falls. Usually used with discrete parameters, e.g. Event marker pressed, it is suitable for multi-state or analogue parameters such as flap selections.
