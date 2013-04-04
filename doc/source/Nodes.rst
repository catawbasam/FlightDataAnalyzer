. _Nodes:

====================
Nodes and Subclasses
====================

----
Node
----

Each derived parameter and key point value etc. is a Node. Each Node forms a part of the dependency tree :ref:`DependencyTree`

There are certain attributes which all nodes have in common.


* frequency
* offset

* derive method
* get_name

Node.get_name()
---------------

The **get_name** method returns the class defined **name** or if that hasn't
been created it generates the name from the class name by converting the
"ClassName" to "Class Name"






•	Naming convention and get_name functionality
•	dump
–	Derived Parameters
–	ListNode & FormattedNameNode
•	Names % format
•	Key Time Instances
•	Key Point Values
–	SectionNode
•	Flight Phase
–	Attributes










First Available Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default we align all parameters to the first available dependency.::

    from analysis_engine.node import P, Node
    
    class NewParameter(Node):
        ##align = True  # default
        def derive(self, a=P('A')):
            pass
            
A fresh instance of NewParameter has the default Node frequency (1.0 Hz) and offset (0 secs)::

    >>> new = NewParameter()
    >>> new
    NewParameter('New Parameter', 1.0, 0)
    
The **get_derived** method takes the list of dependencies and prepares them
for use (aligning them as required) for the Node's **derive** method. Now the
resulting new parameter has the first parameter's frequency and offset::

    >>> a = P('A', frequency=2, offset=0.123)
    >>> new.get_derived([a])
    NewParameter('New Parameter', 2.0, 0.123)


This next block demonstrates how all parameters are aligned to the first available::

    >>> class NewParameter(Node):
    ...     def derive(self, a=P('A'), b=P('B'), c=P('C')):
    ...         print 'A frequency:%.2f offset:%.2f' % (a.frequency, a.offset) if a else 'A'
    ...         print 'B frequency:%.2f offset:%.2f' % (b.frequency, b.offset)
    ...         print 'C frequency:%.2f offset:%.2f' % (c.frequency, c.offset)

    >>> new = NewParameter()
    >>> a = P('A', frequency=2, offset=0.123)
    >>> b = P('B', frequency=4, offset=0.001)
    >>> c = P('C', frequency=0.25, offset=1.101)
    >>> new.get_derived([a, b, c])
    A frequency:2.00 offset:0.12
    B frequency:2.00 offset:0.12
    C frequency:2.00 offset:0.12
    NewParameter('New Parameter', 2.0, 0.123)
    
    
When '**a**' is not avialable the parameters are aligned to '**b**':

    >>> new.get_derived([None, b, c])
    A
    B frequency:4.00 offset:0.00
    C frequency:4.00 offset:0.00
    NewParameter('New Parameter', 4.0, 0.001)


Forcing Frequency and Offset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes up-sampling all dependencies to a higher frequency can be
beneficial to improve the accuracy of a derived parameter.::

    class NewParameter(Node):
        align_frequency = 4  #  Hz
        
Another useful feature is to force the offset, which is quite handy for
Flight Phases.::
        
    class NewParameter(Node):
        align_offset = 0


Turning off alignment
~~~~~~~~~~~~~~~~~~~~~

Aligning can be turned off, which means that one needs to account for the
dependencies having different frequencies and offsets.::

    class NewParameter(Node):
        align = False
        
The Node will default to the first available dependency's frequency and
offset. The typical use-case for not aligning parameters is when performing
customised merging of upsampling of the dependencies. In which case, it is
common to see the resulting frequency and offset being set on the class
within the derive method.::
        
    class NewParameter(Node):
        align = False
        def derive(self, a=P('A'), b=P('B')):
            # merge two signals
            self.array = merge(a, b)
            # set frequency and offset to be the average of a and b
            self.frequency = (a.frequency + b.frequency) / 2
            self.offset = (a.offset + b.offset) / 2

