import logging
import re

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product

from analysis.library import powerset

# Define named tuples for KPV and KTI and FlightPhase
KeyPointValue = namedtuple('KeyPointValue', 'index value name')
KeyTimeInstance = namedtuple('KeyTimeInstance', 'index state')
GeoKeyTimeInstance = namedtuple('GeoKeyTimeInstance', 'index state latitude longitude')
FlightPhase = namedtuple('FlightPhase', 'mask') #Q: rename mask -> slice

# Ref: django/db/models/options.py:20
# Calculate the verbose_name by converting from InitialCaps to "lowercase with spaces".
get_verbose_name = lambda class_name: re.sub('(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', ' \\1', class_name).lower().strip()


### Parameter Names
##ALTITUDE_STD = "Pressure Altitude"
##ALTITUDE_STD_SMOOTHED = "Pressure Altitude Smoothed"
##AIRSPEED = "Indicated Airspeed"
##MACH = "MACH"
##RATE_OF_TURN = ""
##SAT = "SAT"
##TAT = "TAT"

### KPV Names
##MAX_MACH_CRUISE = "Max Mach Cruise"

### KTI Names
##TOP_OF_CLIMB = "Top of Climb"
##TOP_OF_DESCENT = "Top of Descent"
##TAKEOFF_START = ""
##TAKEOFF_END = ""
##LANDING_START = ""
##LANDING_END = ""



### Aircraft States
##AIRBORNE = "Airborne"
##TURNING = "Turning"
##LEVEL_FLIGHT = "Level Flight"
##CLIMBING = "Climbing"
##DESCENDING = "Descending"

### Flight Phases
##PHASE_ENGINE_RUN_UP = slice(1)
##PHASE_TAXI_OUT = slice(1)
##PHASE_CLIMB = slice(1)
##PHASE_CRUISE = slice(1)
##PHASE_APPROACH = slice(1)
##PHASE_DESCENT = slice(1)
##PHASE_TAXI_IN = slice(1)

#-------------------------------------------------------------------------------
# Abstract Classes
# ================

class Node(object):
    __metaclass__ = ABCMeta

    name = '' # Optional
    dependencies = []
    returns = [] # Move to DerivedParameterNode etc? TODO: Handle dependencies on one of the returns values!!
        
    def __repr__(self):
        return '%s' % self.get_name()
        
    @classmethod
    def get_name(cls):
        """ class My2BNode -> 'My2B Node'
        """
        if cls.name:
            return cls.name
        else:
            # Create name from Class if name not specified!
            return get_verbose_name(cls.__name__).title()
    
    @classmethod
    def get_dependency_names(cls):
        """ Returns list of dependency names
        """
        # TypeError:'ABCMeta' object is not iterable?
        # this probably means dependencies for this class isn't a list!
        return [x if isinstance(x, str) else x.get_name() for x in cls.dependencies]
    
    @classmethod
    def can_operate(cls, available):
        """
        Compares the string names of all dependencies against those available.
        
        Returns true if dependencies is a subset of available. For more
        specific operational requirements, override appropriately.
        
        Strictly this is a classmethod, so please remember to use the
        @classmethod decorator! (if you forget, i don't `think` it will break)
        
        :param available: Available parameters from the dependency tree
        :type available: list of strings
        """
        # ensure all names are strings
        if all([x in available for x in cls.get_dependency_names()]):
            return True
        else:
            return False
        
    @classmethod
    def get_operational_combinations(cls):
        """
        Compute every operational combination of dependencies.
        """
        options = []
        for args in powerset(cls.get_dependency_names()):
            if cls.can_operate(args):
                options.append(args)
        return options
        
    @abstractmethod
    def derive(self, params):
        """
        Note: All params masked arrays can be manipulated as required within
        the scope of this method without affecting any other Node classes.
        This is because we write all results back to the hdf, therefore you
        cannot damage the interim numpy masked arrays.
        
        If an implementation does not adhere to the mask of an array, ensure
        that you document it in the docstring as follows:
        WARNING: Does not adhere to the MASK.
        
        returns namedtuple or list of namedtuples KeyPointValue,
        KeyTimeInstance or numpy.ma masked_array
        
        :param params: 
        :type params: dict key:string value: param or list of kpv/kti or phase
        """
        raise NotImplementedError("Abstract Method")
    
    
    
class DerivedParameterNode(Node):
    frequency = None # Hz
    offset = None # secs  -- established when analysing data
    
    ''' Sample desired usage of frequency (and offset):
    #frequency = dependencies[0].frequency  # Used by default if none provided
    frequency = 'Altitude AAL'  # Take from this param name
    frequency = AltitudeAAL  # As above but from class
    frequency = 8  # Hard coded for this algorithm - NOTE: use with care as you cannot hard code the frequency that the input params are provided in
    def derive(self, params):
        self.frequency = params['Altitude Std'].frequency  # can override at run time? not sure I like this option
        self.frequency = 2
        return result
        
    def _get_derived(self, params, flight_duration):
        res = self.derive(params)
        # Ensure that the frequency has been adhered to!
        assert len(res) == flight_duration * self.frequency
    '''

    pass


class FlightPhaseNode(Node):
    # 1Hz slices
    # TODO: Allow for 8Hz for LiftOff and TouchDown example
    pass

class KeyTimeInstanceNode(Node):
    'TODO: Implement the helper functions like KPV Node below'
    
    # :rtype: KeyTimeInstance or List of KeyTimeInstance or EmptyList
    pass
    
    
class KeyPointValueNode(Node):
    """
    NAME_FORMAT example: 
    'Speed in %(phase)s at %(altitude)d ft'

    RETURN_OPTIONS example:
    {'phase'    : ['ascent', 'descent'],
     'altitude' : [1000,1500],}
    """
    NAME_FORMAT = ""
    RETURN_OPTIONS = {}
    
    def kpv_names(self):
        """        
        :returns: The product of all RETURN_OPTIONS name combinations
        :rtype: list
        """
        # cache option below disabled until required.
        ##if hasattr(self, 'names'):
            ##return self.names
        names = []
        for a in product(*self.RETURN_OPTIONS.values()): 
            name = self.NAME_FORMAT % dict(zip(self.RETURN_OPTIONS.keys(), a))
            names.append(name)
        ##self.names = names  #cache
        return names
    
    
    def _validate_name(self, name):
        """
        Raises ValueError if replace_values are not allowed in RETURN_OPTIONS
        permissive values.
        """
        if name in self.kpv_names():
            return True
        else:
            raise ValueError("invalid KPV name '%s'" % name)

    def create_kpv(self, index, value, replace_values={}, **kwargs):
        """
        Formats FORMAT_NAME with interpolation values and returns a KPV object
        with index and value.
        
        Notes:
        Raises KeyError if required interpolation/replace value not provided.
        Raises TypeError if interpolation value is of wrong type.
        Interpolation values not in FORMAT_NAME are ignored.
        """
        rvals = replace_values.copy()  # avoid re-using static type
        rvals.update(kwargs)
        name = self.NAME_FORMAT % rvals
        # validate name is allowed
        self._validate_name(name)
        return KeyPointValue(index, value, name)
    



    
class NodeManager(object):
    def __repr__(self):
        return 'NodeManager: lfl x%d, requested x%d, derived x%d' % (
            len(self.lfl), len(self.requested), len(self.derived_nodes))
    
    def __init__(self, lfl, requested, derived_nodes):
        """
        :type lfl: list
        :type requested: list
        :type derived_nodes: dict
        """
        self.lfl = lfl
        self.requested = requested
        self.derived_nodes = derived_nodes

        
    def operational(self, name, available):
        """
        Looks up the node and tells you whether it can operate.
        
        :returns: Result of Operational test on parameter.
        :rtype: Boolean
        """
        if name in self.derived_nodes:
            return self.derived_nodes[name].can_operate(available)
        elif name in self.lfl or name == 'root':
            return True
        else:  #elif name in unavailable_deps:
            logging.warning("Confirm - node is unavailable: %s", name)
            return False