#TODO: Rename derived.py to something like base.py or abstract.py
 
 
import re

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product

# Define named tuples for KPV and KTI and FlightPhase
KeyPointValue = namedtuple('KeyPointValue', 'index value name')
KeyTimeInstance = namedtuple('KeyTimeInstance', 'index state')
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
    returns = [] # Move to DerivedParameterNode etc?
        
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
        return [x if isinstance(x, str) else x.get_name() for x in cls.dependencies]
    
    def can_operate(self, available):
        """
        Compares the string names of all dependencies against those available.
        
        Returns true if dependencies is a subset of available. For more
        specific operational requirements, override appropriately.
        
        
        :param available: Available parameters from the dependency tree
        :type available: list of strings
        """
        # ensure all names are strings
        if all([x in available for x in self.get_dependency_names()]):
            return True
        else:
            return False
        
    @abstractmethod
    def derive(self, params):
        """
        returns namedtuple or list of namedtuples KeyPointValue,
        KeyTimeInstance or numpy.ma masked_aray
        """
        raise NotImplementedError("Abstract Method")
    
    
    
class DerivedParameterNode(Node):
    pass


class FlightPhaseNode(Node):
    pass

class KeyTimeInstanceNode(Node):
    'TODO: Implement the helper functions like KPV Node below'
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
        
        Q: Possibly should validate that the formatted value is in list of
        formatted values as this will then allow for small differnces which
        the formatting may disregard. e.g. 0.001 in %d == 0
        """
        if name in self.kpv_names():
            return True
        else:
            raise ValueError("invalid KPV name '%s'" % name)
        
        ##for key, value in replace_values.iteritems():
            ##allowed = self.RETURN_OPTIONS[key]
            ##is_iterable = isinstance(allowed, (list, tuple))
            ##if value == allowed or (is_iterable and value in allowed):
                ##continue  # all good, check next option
            ##else:
                ##raise ValueError("invalid value '%s' for key %s" % (value, key))
        ##return True

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
    



    
 