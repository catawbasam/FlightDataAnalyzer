import inspect
import logging
import numpy as np
import re
import copy

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product
from operator import attrgetter

from hdfaccess.parameter import P, Parameter
from analysis.library import align, is_index_within_slice

from analysis.recordtype import recordtype

# Define named tuples for KPV and KTI and FlightPhase
KeyPointValue = recordtype('KeyPointValue', 'index value name slice datetime', 
                           field_defaults={'slice':slice(None)}, default=None)
KeyTimeInstance = recordtype('KeyTimeInstance', 'index state datetime latitude longitude', 
                             default=None)
Section = namedtuple('Section', 'name slice') #Q: rename mask -> slice/section

# Ref: django/db/models/options.py:20
# Calculate the verbose_name by converting from InitialCaps to "lowercase with spaces".
def get_verbose_name(class_name):
    if re.match('^_\d.*$', class_name):
        # Remove initial underscore to allow class names starting with numbers
        # e.g. '_1000FtInClimb' will become '1000 Ft In Climb'
        class_name = class_name[1:]
    return re.sub('(((?<=[a-z])[A-Z0-9])|([A-Z0-9](?![A-Z0-9]|$)))', ' \\1',
                  class_name).lower().strip()


def powerset(iterable):
    """
    Ref: http://docs.python.org/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_param_kwarg_names(method):
    """
    Inspects a method's arguments and returns the defaults values of keyword
    arguments defined in the method.
    
    Raises ValueError if there are any args defined other than "self".
    
    :param method: Method to be inspected
    :type method: method
    :returns: Ordered list of default values of keyword arguments
    :rtype: list
    """
    args, varargs, varkw, defaults = inspect.getargspec(method)
    if not defaults or args[:-len(defaults)] != ['self'] or varargs:
        raise ValueError("Node '%s' must have kwargs, cannot accept no kwargs or any args other than 'self'. args:'%s' *args:'%s'" % (
            method.im_class.get_name(), args[1:], varargs))
    if varkw:
        # One day, could insert all available params as kwargs - but cannot
        # guarentee requirements will work
        raise NotImplementedError("Cannot define **kwargs")
    # alternative: return dict(zip(defaults, args[-len(defaults):]))
    return defaults

#------------------------------------------------------------------------------
# Abstract Node Classes
# =====================
class Node(object):
    __metaclass__ = ABCMeta

    name = '' # Optional, default taken from ClassName
    align_to_first_dependency = True
        
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
        #NB: removed check for dependencies to allow initialisation in def derive()
        ##if not self.get_dependency_names():
            ##raise ValueError("Every Node must have a dependency. Node '%s'" % self.__class__.__name__)
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
    
    # removed abstract wrapper to allow initialisation within def derive(KTI('a'))
    ##@abstractmethod #TODO: Review removal.
    def get_aligned(self, align_to_param):
        """
        Return a version of self which is aligned to the incoming argument.
        """
        raise NotImplementedError("Abstract Method")
    
    def get_derived(self, args):
        """
        Accessor for derive method which first aligns all parameters to the
        first to ensure parameter data and indices are consistent.
        
        :param args: List of available Parameter objects
        :type args: list
        """
        if self.align_to_first_dependency:
            i, first_param = next(((n, a) for n, a in enumerate(args) if a is not None))
            for n, param in enumerate(args):
                # if param is set and it's after the first dependency
                if param and n > i:
                    # override argument in list in-place
                    args[n] = param.get_aligned(first_param)
        res = self.derive(*args)
        if res is NotImplemented:
            raise NotImplementedError("Class '%s' derive method is not implemented." % \
                                      self.__class__.__name__)
        elif res:
            raise UserWarning("Class '%s' should not have returned anything. Got: %s" % (
                self.__class__.__name__, res))
        return self
        
    # removed abstract wrapper to allow initialisation within def derive(KTI('a'))
    ##@abstractmethod #TODO: Review removal.
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
        
    def get_aligned(self, param):
        """
        Aligns itself to the input parameter and creates a copy
        """
        aligned_array = align(self, param)
        aligned_param = DerivedParameterNode(frequency=param.frequency,
                                             offset=param.offset)
        aligned_param.array = aligned_array
        return aligned_param


class SectionNode(Node, list):
    '''
    Derives from list to implement iteration and list methods.
    '''
    def __init__(self, *args, **kwargs):
        """ List of slices where this phase is active. Has a frequency and offset.
        """
        if 'items' in kwargs:
            self.extend(kwargs['items'])
            del kwargs['items']
        # place holder
        super(SectionNode, self).__init__(*args, **kwargs)

    def create_section(self, section_slice, name=''):
        section = Section(name or self.get_name(), section_slice)
        self.append(section)
        ##return section
        
    def create_sections(self, section_slices, name=''):
        for sect in section_slices:
            self.create_section(sect, name=name)
        
    #TODO: Accessor for 1Hz slice, 8Hz slice etc.
    def get_aligned(self, param):
        '''
        Aligns section slices to the frequency and offset of param.
        
        :param section:
        :type section: SectionNode object
        :param param:
        :type param: Parameter object
        '''
        aligned_node = self.__class__(frequency=param.frequency,
                                      offset=param.offset)
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        for section in self:
            converted_start = (section.slice.start * multiplier) + offset
            converted_stop = (section.slice.stop * multiplier) + offset
            converted_slice = slice(converted_start, converted_stop)
            aligned_node.create_section(converted_slice,
                                        section.name)
        return aligned_node


class FlightPhaseNode(SectionNode):
    """ Is a Section, but called "phase" for user-friendlyness!
    """
    def create_phase(self, phase_slice):
        """
        Creates a Flight Phase using a slice at specific frequency, using the
        classes name.
        
        It's a shortcut to using create_section.
        """
        self.create_section(phase_slice)
        
    def create_phases(self, phase_slices):
        for phase in phase_slices:
            self.create_phase(phase)


class FormattedNameNode(Node, list):
    """
    NAME_FORMAT example: 
    'Speed in %(phase)s at %(altitude)d ft'

    NAME_VALUES example:
    {'phase'    : ['ascent', 'descent'],
     'altitude' : [1000,1500],}
    """
    NAME_FORMAT = ""
    NAME_VALUES = {}
    
    def __init__(self, *args, **kwargs):
        '''
        :param items: Optional keyword argument of initial items to be contained within self.
        :type items: list
        '''
        if 'items' in kwargs:
            self.extend(kwargs['items'])
            del kwargs['items']
        super(FormattedNameNode, self).__init__(*args, **kwargs)
    
    def names(self):
        """        
        :returns: The product of all NAME_VALUES name combinations
        :rtype: list
        """
        # cache option below disabled until required.
        ##if hasattr(self, 'names'):
            ##return self.names
        if not self.NAME_FORMAT and not self.NAME_VALUES:
            return [self.get_name()]
        names = []
        for a in product(*self.NAME_VALUES.values()): 
            name = self.NAME_FORMAT % dict(zip(self.NAME_VALUES.keys(), a))
            names.append(name)
        ##self.names = names  #cache
        return names
    
    def _validate_name(self, name):
        """
        Raises ValueError if replace_values are not allowed in NAME_VALUES
        permissive values.
        """
        if name in self.names():
            return True
        else:
            raise ValueError("invalid name '%s'" % name)
    
    def format_name(self, replace_values={}, **kwargs):
        """
        Formats NAME_FORMAT with interpolation values and returns a KPV object
        with index and value.
        
        Interpolation values not in FORMAT_NAME are ignored.        
        
        :raises KeyError: if required interpolation/replace value not provided.
        :raises TypeError: if interpolation value is of wrong type.
        """
        if not replace_values and not kwargs and not self.NAME_FORMAT:
            # have not defined name format to use, so create using name of node
            return self.get_name()
        rvals = replace_values.copy()  # avoid re-using static type
        rvals.update(kwargs)
        name = self.NAME_FORMAT % rvals  # common error is to use { inplace of (
        # validate name is allowed
        self._validate_name(name)
        return name # return as a confirmation it was successful
    
    def _get_condition(self, within_slice=None, name=None):
        '''
        Returns a condition function which checks if the element is within
        a slice or has a specified name if they are provided.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        if within_slice and name:
            return lambda e: is_index_within_slice(e.index, within_slice) and \
                   e.name == name
        elif within_slice:
            return lambda e: is_index_within_slice(e.index, within_slice)
        elif name:
            return lambda e: e.name == name
        else:
            return None
    
    def get_ordered_by_index(self, within_slice=None, name=None):
        '''
        Gets elements ordered by index (ascending) optionally filter 
        within_slice or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: An object of the same type as self containing elements ordered by index.
        :rtype: self.__class__
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        ordered_by_index = sorted(matching, key=attrgetter('index'))
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=ordered_by_index)
    
    def get_first(self, within_slice=None, name=None):
        '''
        Gets the element with the lowest index optionally filter within_slice or
        by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        return min(matching, key=attrgetter('index')) if matching else None
    
    def get_last(self, within_slice=None, name=None):
        '''
        Gets the element with the lowest index optionally filter within_slice or
        by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        return max(matching, key=attrgetter('index')) if matching else None
    
    def get_named(self, name, within_slice=None):
        '''
        Gets elements with name optionally filtered within_slice.
        
        :param name: Only return elements with this name.
        :type name: str
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :returns: An object of the same type as self containing the filtered elements.
        :rtype: self.__class__
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self)
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=matching)


class KeyTimeInstanceNode(FormattedNameNode):
    """
    TODO: Support 1Hz / 8Hz KTI index locations via accessor on class and
    determine what is required for get_derived to be stored in database
    """
    # :rtype: KeyTimeInstance or List of KeyTimeInstance or EmptyList
    def __init__(self, *args, **kwargs):
        # place holder
        super(KeyTimeInstanceNode, self).__init__(*args, **kwargs)
        
    def create_kti(self, index, name):
        kti = KeyTimeInstance(index, name)
        self.append(kti)
        return kti
    
    def get_aligned(self, param):
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        aligned_node = self.__class__(self.name, param.frequency,
                                      param.offset) 
        for kti in self:
            index_aligned = (kti.index * multiplier) + offset
            aligned_node.create_kti(index_aligned, kti.name)
        return aligned_node


class KeyPointValueNode(FormattedNameNode):
    
    def __init__(self, *args, **kwargs):
        super(KeyPointValueNode, self).__init__(*args, **kwargs)

    def create_kpv(self, index, value, replace_values={}, **kwargs):
        """
        Formats FORMAT_NAME with interpolation values and returns a KPV object
        with index and value.
        
        Interpolation values not in FORMAT_NAME are ignored.        
        
        :raises KeyError: if required interpolation/replace value not provided.
        :raises TypeError: if interpolation value is of wrong type.
        """
        name = self.format_name(replace_values, **kwargs)
        kpv = KeyPointValue(index, value, name)
        self.append(kpv)
        return kpv # return as a confirmation it was successful
    
    def get_aligned(self, param):
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        aligned_node = self.__class__(self.name, param.frequency, param.offset)
        for kpv in self:
            aligned_kpv = copy.copy(kpv)
            aligned_kpv.index = (aligned_kpv.index * multiplier) + offset
            aligned_node.append(aligned_kpv)
        return aligned_node
    #TODO: Accessors for first kpv, primary kpv etc.
    
    def get_max(self, within_slice=None, name=None):
        '''
        Gets the KeyPointValue with the maximum value optionally filter
        within_slice or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        return max(matching, key=attrgetter('value')) if matching else None
    
    def get_min(self, within_slice=None, name=None):
        '''
        Gets the KeyPointValue with the minimum value optionally filter
        within_slice or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        return min(matching, key=attrgetter('value')) if matching else None
    
    def get_ordered_by_value(self, within_slice=None, name=None):
        '''
        Gets the element with the maximum value optionally filter within_slice
        or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        ordered_by_value = sorted(matching, key=attrgetter('value'))
        return KeyPointValueNode(name=self.name, frequency=self.frequency,
                                 offset=self.offset, items=ordered_by_value)
    
        

    # ordered by time (ascending), ordered by value (ascending), 
    
    
    
    
class FlightAttributeNode(Node):
    """
    Can only store a single value per Node, however the value can be any
    object (dict, list, integer etc). The class name serves as the name of the
    attribute.
    """
    def __init__(self, *args, **kwargs):
        self._value = None
        super(FlightAttributeNode, self).__init__(*args, **kwargs)
    
    def set_flight_attribute(self, value):
        self._value = value
    set_flight_attr = set_flight_attribute
    
    def get_aligned(self, deps):
        """
        Cannot align a flight attribute.
        """
        return self


class NodeManager(object):
    def __repr__(self):
        return 'NodeManager: x%d nodes in total' % (
            len(self.lfl) + len(self.requested) + len(self.derived_nodes) + 
            len(self.aircraft_info) + len(self.achieved_flight_record))
    
    def __init__(self, start_datetime, lfl, requested, derived_nodes, aircraft_info, achieved_flight_record):
        """
        Storage of parameter keys and access to derived nodes.
        
        :param start_datetime: datetime of start of data file
        :type start_datetime: datetime
        :type lfl: list
        :type requested: list
        :type derived_nodes: dict
        :type aircraft_info: dict
        :type achieved_flight_record: dict
        """
        self.start_datetime = start_datetime
        self.lfl = lfl
        self.requested = requested
        self.derived_nodes = derived_nodes
        # Attributes:
        self.aircraft_info = aircraft_info
        self.achieved_flight_record = achieved_flight_record
        
    def keys(self):
        """
        Ordered list of all Node names stored within the manager.
        """
        return sorted(['Start Datetime'] \
                      + self.lfl \
                      + self.derived_nodes.keys() \
                      + self.aircraft_info.keys() \
                      + self.achieved_flight_record.keys())
    

    def get_attribute(self, name):
        """
        Get an attribute value from aircraft_info or achieved_flight_record
        dictionaries. If key is None, returns None. If key is present,
        returns an Attribute.
        
        :param name: Attribute name.
        :type name: String
        :returns: Attribute if available.
        :rtype: Attribute object or None
        """
        if name == 'Start Datetime':
            return Attribute(name, value=self.start_datetime)
        if self.aircraft_info.get(name):
            return Attribute(name, value=self.aircraft_info[name])
        elif self.achieved_flight_record.get(name):
            return Attribute(name, value=self.achieved_flight_record[name])
        else:
            return None
    
    def operational(self, name, available):
        """
        Looks up the node and tells you whether it can operate.
        
        :returns: Result of Operational test on parameter.
        :rtype: Boolean
        """
        if name in self.derived_nodes:
            #NOTE: Raises "Unbound method" here due to can_operate being overridden without wrapping with @classmethod decorator
            return self.derived_nodes[name].can_operate(available)
        elif name in self.lfl \
             or self.aircraft_info.get(name) is not None \
             or self.achieved_flight_record.get(name) is not None \
             or name == 'root':
            return True
        else:  #elif name in unavailable_deps:
            logging.warning("Confirm - node is unavailable: %s", name)
            return False

# The following acronyms are intended to be used as placeholder values
# for kwargs in Node derive methods. Cannot instantiate Node subclass without 
# implementing derive.
class Attribute(object):
    def __repr__(self):
        return "Attribute '%s' : %s" % (self.name, self.value)
    
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
    
A = Attribute
P = Parameter
S = SectionNode
KPV = KeyPointValueNode
KTI = KeyTimeInstanceNode
