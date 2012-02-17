.. _Extending:

=============================
Extending the Analysis Engine
=============================

---------------
Custom Settings
---------------

Create a '''custom_settings.py''' file (not within the repository) to
override analysis settings or add modules for adding analysis of additional
Derived Parameters, Key Point Values, Key Time Instances and Sections.


Parameter Analysis
------------------

Override the values of variables used for data analysis settings.


Splitting into Segments
-----------------------

Override variable which determines the parameters used for splitting a file 
into Segments


Modules
-------

If you create your own Nodes which you would like to be included in the
parameter dependency tree created by the Analysis Engine at runtime, add a
variable ending in "_MODULES" with a list of paths to the modules to import.

.. code-block:: python
    :linenos:
    
    MY_CUSTOM_MODULES = ['path.to.module_1', 'path.to.module_2']


------------
Custom Hooks
------------

Hooks are used to execute external code which may be required for additional
processes specific to implementation requirements.

Hooks are added to a '''custom_hooks.py''' file (not within the repository).

See '''hooks.py''' for available hooks and the API for each function.
