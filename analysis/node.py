import inspect
import logging
import numpy as np
import re

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product

from analysis.recordtype import recordtype
from analysis.library import powerset

# Define named tuples for KPV and KTI and FlightPhase
KeyPointValue = namedtuple('KeyPointValue', 'index value name')
KeyTimeInstance = namedtuple('KeyTimeInstance', 'index state')
GeoKeyTimeInstance = namedtuple('GeoKeyTimeInstance', 'index state latitude longitude')
FlightPhase = namedtuple('FlightPhase', 'name mask') #Q: rename mask -> slice/section

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
# Parameter container Class
# =========================
class Parameter(object):
    def __init__(self, name, array=[], frequency=1, offset=0):
        '''
        :param name: Parameter name
        :type name: String
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: Int
        :param offset: Offset in Frame.
        :type offset: Float
        '''
        self.name = name
        self.array = array
        self.frequency = self.sample_rate = self.hz = frequency
        self.offset = offset
        
    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)

P = Parameter # shorthand


def get_param_kwarg_names(derive_method):
    args, varargs, varkw, defaults = inspect.getargspec(derive_method)
    if args[:-len(defaults)] != ['self'] or varargs:
        raise ValueError("Only kwargs accepted, cannot accept args: %s %s" % (args[1:], varargs))
    if varkw:
        raise NotImplementedError("One day, could insert all available params as kwargs - but cannot guarentee requirements will work")
    #return dict(zip(defaults, args[-len(defaults):]))
    return defaults


#-------------------------------------------------------------------------------
# Abstract Classes
# ================

class Node(object):
    __metaclass__ = ABCMeta

    name = '' # Optional
    returns = [] # Move to DerivedParameterNode etc? TODO: Handle dependencies on one of the returns values!!
        
    def __init__(self, name='', frequency=1, offset=0):
        """
        Abstract Node. frequency and offset arguments are populated from the
        first available dependency parameter object.
        
        :param name: Name of parameter
        :type params: str
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: Int
        :param offset: Offset in Frame.
        :type offset: Float
        """
        if not self.get_dependency_names():
            raise ValueError("Every Node must have a dependency. Node '%s'" % self.__class__.__name__)
        if name:
            self.name = name + '' # for ease of testing, checks name is string ;-)
        else:
            self.name = self.get_name() # usual option
        self.frequency = self.sample_rate = self.hz = frequency # Hz
        self.offset = offset # secs
        
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
        ##return [x if isinstance(x, str) else x.get_name() for x in cls.dependencies]
        params = get_param_kwarg_names(cls.derive)
        return [d.name or d.get_name() for d in params]
    
    @classmethod
    def can_operate(cls, available):
        """
        Compares the string names of all dependencies against those available.
        
        Returns true if dependencies is a subset of available. For more
        specific operational requirements, override appropriately.
        
        This is a classmethod, so please remember to use the
        @classmethod decorator! (if you forget - it will break)
        
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
    def get_derived(self, args):
        """
        Accessor for derive method's results. Each Node type shall return the
        class attributes appropriate for the Node type.
        
        :param params: Collection of available Parameter objects
        :type params: dict
        """
        raise NotImplementedError("Abstract Method")
        
    @abstractmethod
    def derive(self, **kwargs):
        """
        Accepts keyword arguments where the default determines the derive
        dependencies. Each keyword default must be a Parameter like object
        with attribute name or method get_name() returning a string
        representation of the Parameter.
        
        e.g. def derive(self, first_dep=P('not_available'), first_available=P('available'), another=MyDerivedNode:
                 pass
        
        Note: Although keywords are required to determine the derive method's 
        dependencies, Implementation actually provides the keywords using 
        positional arguments, providing None where the dependency is not 
        available.
        
        e.g. deps = [None, param_obj]
             node.derive(*deps)
             
        Results of derive are saved onto the object's attributes. See each
        implementation of Node.
        
        e.g. self.array = []
        
        Note: All params masked arrays can be manipulated as required within
        the scope of this method without affecting any other Node classes.
        This is because we write all results back to the hdf, therefore you
        cannot damage the interim numpy masked arrays.
        
        If an implementation does not adhere to the mask of an array, ensure
        that you document it in the docstring as follows:
        WARNING: Does not adhere to the MASK.
        
        :param kwargs: Keyword arguments where default is a Parameter object or Node class
        :type kwargs: dict
        :returns: No returns! Save to object attributes.
        :rtype: None
        """
        raise NotImplementedError("Abstract Method")
    
    
    
class DerivedParameterNode(Node):
    """
    """
    def __init__(self, *args, **kwargs):
        # create array results placeholder
        self.array = None # np.ma.array derive result goes here!
        super(DerivedParameterNode, self).__init__(*args, **kwargs)
    
                
    def get_derived(self, args):
        # get results
        res = self.derive(*args)
        if res == NotImplemented:
            ##raise NotImplementedError("Cannot proceed (need self.array)")
            logging.warning("FAKING DATA FOR NotImplemented '%s' - used for test purposes!" % self)
            self.array = np.ma.array(range(10)) #
            pass #TODO: raise error and remove pass
        if self.array is None and res:
            logging.warning("Depreciation Warning: array attribute not set but values returned")
            self.array = res
        ### Ensure that the frequency has been adhered to!
        ##assert len(res) == flight_duration * self.frequency
        # create a simplistic parameter for writing to HDF
        #TODO: Parameter and hdf_access to use available=params.keys()
        return Parameter(self.get_name(), self.array, self.frequency, self.offset)
    
    ##def get_first_param (self, params):
        ##return params[self.get_dependency_names()[0]]

class FlightPhaseNode(Node):
    def __init__(self, *args, **kwargs):
        # place holder
        self._flight_phases = []
        super(FlightPhaseNode, self).__init__(*args, **kwargs)


    def create_phase(self, mask):
        phase = FlightPhase(mask)
        self._flight_phases.append(phase)
        return phase
    
    # 1Hz slices
    # TODO: Allow for 8Hz for LiftOff and TouchDown example
    def get_derived(self, args):
        res = self.derive(*args)
        if res == NotImplemented:
            raise NotImplementedError("Cannot proceed")
        #TODO: Return slice at correct frequency?
        return self.flight_phase
        
    
    #TODO: Accessor for 1Hz slice, 8Hz slice etc.

class KeyTimeInstanceNode(Node):
    """
    TODO: Support 1Hz / 8Hz KTI index locations via accessor on class and
    determine what is required for get_derived to be stored in database
    """
    # :rtype: KeyTimeInstance or List of KeyTimeInstance or EmptyList
    def __init__(self, *args, **kwargs):
        # place holder
        self._kti_list = []
        super(KeyTimeInstanceNode, self).__init__(*args, **kwargs)
        
    def create_kti(self, index, state):
        kti = KeyTimeInstance(index, state)
        self._kti_list.append(kti)
        return kti 
    
    def get_derived(self, args):
        #TODO: Support 1Hz / 8Hz KTI index locations
        self.derive(*args)
        return self._kti_list
    
    
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
    
    def __init__(self, *args, **kwargs):
        self._kpv_list = []
        super(KeyPointValueNode, self).__init__(*args, **kwargs)
        
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
        name = self.NAME_FORMAT % rvals  # common error is to use { inplace of (
        # validate name is allowed
        self._validate_name(name)
        kpv = KeyPointValue(index, value, name)
        self._kpv_list.append(kpv)
        return kpv # return as a confirmation it was successful
    
    def get_derived(self, args):
        res = self.derive(*args)
        if res == NotImplemented:
            #raise NotImplementedError("Cannot proceed")
            pass #TODO: raise error and remove pass
        elif res:
            #Q: store in self._kpv_list to be backward compatible?
            raise RuntimeError("Cannot return from a derive method. Returned '%s'" % res)
        return self._kpv_list


    #TODO: Accessors for first kpv, primary kpv etc.

    
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
        
    def keys(self):
        """
        """
        return self.lfl + self.derived_nodes.keys()

        
    def operational(self, name, available):
        """
        Looks up the node and tells you whether it can operate.
        
        :returns: Result of Operational test on parameter.
        :rtype: Boolean
        """
        if name in self.derived_nodes:
            #NOTE: Raises "Unbound method" here due to can_operate being overridden without wrapping with @classmethod decorator
            return self.derived_nodes[name].can_operate(available)
        elif name in self.lfl or name == 'root':
            return True
        else:  #elif name in unavailable_deps:
            logging.warning("Confirm - node is unavailable: %s", name)
            return False