.. _DependencyTree:

===============
Dependency Tree
===============


Concept
-------

In brief: `To dynamically determine the processing order of dependencies to
satisfy the list of required parameters.`

Aircraft flight principles are generally the same across different aicraft
types. However the parameter set available differs considerably between
aircraft types and across different dataframes. This problem continues to
individual aircraft which may not have optional hardware installed or may
have a failed sensor resulting in a recorded parameter being invalid.

Often this can involve writing Aircraft Tail specific code exceptions when
deriving parameters to account for all these exceptional cases.

The dependency tree takes away those issues by establishing which Nodes are
able to operate and in which order they need to be processed in order that
the hierarchical set of dependencies will be met.


What are the dependencies?
--------------------------

The dependencies are defined as keyword arguments to the derive method which
exists on every Node ::

    class NewParameter(Node):
        def derive(self, first=P('First Dependency')):
            pass

We use the `wrapper` (**P()** here) to assist the programmer with IDE
auto-completion of the **first** keyword argument, providing it with the
attributes available on the expected data type being used.

The name of the dependency is provided as a String within the keyword
argument `wrapper`.

By adding the dependencies into the **derive** method, the dependency tree
can establish a link between the Node and the parameters it can use for
derivation.


Mach Example
~~~~~~~~~~~~

Many aircraft record `Mach`, but for those who do not we can dynamically
establish the Mach from the `Airspeed` and pressure altitude `Altitude STD`.
If Mach is recorded, the Mach DerivedParameterNode will never be executed.

:py:class:`analysis_engine.derived_parameters.Mach`

If one requests a Key Point Value of **Mach Max**, the dependency tree will
establish whether it can meet it's dependencies recursively through each
dependency's dependency.

The dependency tree will establish that Mach is a requirement of say the Key
Point Value **Mach Max** and if Mach is recorded, no further calculations are
performed. If Mach is not recorded it will establish whether the Mach
dependencies are met (Airspeed and Altitude STD)


Can Operate
~~~~~~~~~~~

It may be possible to conduct the Node derivation (within the **derive**)
method without a full set of dependencies being available. The
**can_operate** class method allows the programmer to insert basic logic to
determine which of the available parameters (a subset of the dependencies
within the derive declaration) the Node can operate successfully with.::

    class NewParameter(Node):
        @classmethod
        def can_operate(cls, available):
            # New Parameter can work if the following two are available
            return ('Airspeed', 'Altitude AAL') in available
            
        def derive(self, aspd=P('Airspeed'), gspd=P('Groundspeed'), alt=P('Altitude AAL')):
            # Check if Groundspeed is available
            if gspd:
                ...
            ...

See :ref:`can_operate` for more usage examples.


The :py:meth:`~analysis_engine.Node.NodeManager.operational` method of the
NodeManager calls the **can_operate** method on the classes when traversing
the dependency graph.

When a requested Node is not operating (does not execute the derive method)
you can quickly establish why without having to refer to the dependency tree
by inserting a breakpoint into the can_operate method. If there is no
can_operate method, temporarily add one with a pass statement to breakpoint
upon.::

    class NewParameter(Node):
        @classmethod
        def can_operate(cls, available):
            pass  # add a breakpoint here to inspect "available"
        
        def derive(self, ...

.. 
    As an example, one may calculate a smoothed latitude and longitude location
    of the aircraft from the recorded Latitude and Longitude which may not have a
    very high resolution (causing a steppy track). Latitude Smoothed will depend
    on Latitude:
    
        Latitude Smoothed
        requires: Latitude
        
    In order to better increase the accuracy of the aircraft, some information about the Takeoff and Landing runway will help to pin-point the track onto the runway:
    
        Latitude Smoothed
        requires: Latitude
        optional: Takeoff Runway, Landing Runway
        
    The derived parameter will make the most out of the parameters provided - so
    if the Takeoff Runway isn't known, it will be smoothed without pinpointing
    the track to the runway.
    
    Some aircraft don't record their location, so instead we can use Heading and Airspeed to derive a track and then pinpoint this onto the runways:
    
        Latitude Smoothed
        requires: Latitude or (Heading and Airspeed and Latitude At Takeoff and Latitude At Landing)
        optional: 
    

Graph Theory
------------

Derived Parameter Nodes, Attribute Nodes, Key Time Instance Nodes, Key Point
Value Nodes and Section Nodes are all objects which can have dependencies
upon other Nodes or LFL Parameters.
 
Each of these objects is a Node within a directional graph (`DiGraph`). The
edges of the graph represents the dependency of one Node upon another.

The `root` node is a special node which defines the starting point of the
DiGraph for traversal of the dependency tree. It points to the top level
parameters (those which have no predecessors).


Processing Order
~~~~~~~~~~~~~~~~

The processing order is established by recursively traversing down the
DiGraph using Breadth First Search. 

:py:func:`analysis_engine.dependency_graph.dependencies3`

As each Node is encountered, if it has dependencies we recurse into each
dependency to determine whether the level below is operational. If deemed
operational, the Node is added to the set of active_nodes (so that we do not
process the node again) and appended to the processing order.


Spanning Tree
~~~~~~~~~~~~~

The Spanning Tree is the original Graph, excluding the inactive Nodes. It
represents the actual tree to be used for analysis. These may be inactive due
to being inoperable (the dependencies do not satisfy the can_operate method)
or not being available (the NodeManager does not contain them, normally due
to not being recorded in the LFL but possibly due to a naming error).


Visualising the Tree
~~~~~~~~~~~~~~~~~~~~

The graph can be visualised using the
**:py:func:`~analysis_engine.dependency_graph.draw_graph`** function. This
requires `pygraphviz` and therefore `Graphviz` to be installed.

:py:func:`analysis_engine.dependency_graph.draw_graph`


The `FlightDataParameterTree` tool can also be used to visualise the
dependency tree. This can be easier to understand when your tree is quite
large (often the case!).

The numeric before the Node name represents the Nodes position in the
processing order.




Colours are used to represent the different types of parameters. ??????????????????????

.. note::

    Networkx was chosen over pygraph due to its more pythonic implementation. 


.. warning::

    A RuntimeError will be raised if there is a circular dependency found
    within the digraph (which will cause infinite recursion when resolving
    the depenency tree!).


How to view / identify problems