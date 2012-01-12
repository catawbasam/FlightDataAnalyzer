import inspect
import logging
import numpy as np
import re
import copy

from abc import ABCMeta
from collections import namedtuple
from itertools import product
from operator import attrgetter

from analysis.library import (align, is_index_within_slice,
                              is_slice_within_slice, slices_above,
                              slices_below, slices_between, slices_from_to,
                              value_at_index, value_at_time)
from analysis.recordtype import recordtype

# Define named tuples for KPV and KTI and FlightPhase
KeyPointValue = recordtype('KeyPointValue', 'index value name slice datetime', 
                           field_defaults={'slice':slice(None)}, default=None)
KeyTimeInstance = recordtype('KeyTimeInstance', 'index name datetime latitude longitude', 
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
        #TODO: Add __class__.__name__?
        return "%s %sHz %.2fsecs" % (self.get_name(), self.frequency, self.offset)
        
    @classmethod
    def get_name(cls):
        """ class My2BNode -> 'My2B Node'
        """
        return cls.name or get_verbose_name(cls.__name__).title()
    
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
        
        Sample overrides for "Any deps available":
@classmethod
def can_operate(cls, available):
    # works with any combination of params available
    if any([d in available for d in cls.get_dependency_names()]):
        return True
    else:
        return False

@classmethod
def can_operate(cls, available):
    if set(cls.get_dependency_names()).intersection(available):
        return True  # if ANY are available
    else:
        return False  # we have none available
            
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
        dependencies_powerset = powerset(cls.get_dependency_names())
        return [args for args in dependencies_powerset if cls.can_operate(args)]
    
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
            try:
                i, first_param = next(((n, a) for n, a in enumerate(args) if \
                                       a is not None and a.frequency))
            except StopIteration:
                pass
            else:
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
        :returns: No returns! Sets attributes on self to be accessed after calling derive.
        :rtype: None
        """
        raise NotImplementedError("Abstract Method")


class DerivedParameterNode(Node):
    """
    Base class for DerivedParameters which overide def derive() method.
    
    Also used during processing when creating parameters from HDF files as
    dependencies for other Nodes.
    """
    def __init__(self, name='', array=np.ma.array([]), frequency=1, offset=0, *args, **kwargs):
        # create array results placeholder
        self.array = array # np.ma.array derive result goes here!
        super(DerivedParameterNode, self).__init__(name=name, frequency=frequency, 
                                                   offset=offset, 
                                                   *args, **kwargs)
        
    def at(self, secs):
        """
        Gets the value within the array at time secs. Interpolates to retrieve
        the most accurate value.
        
        :param secs: time delta from start of data in seconds
        :type secs: float or timedelta
        :returns: The interpolated value of the array at time secs.
        :rtype: float
        """
        try:
            # get seconds from timedelta
            secs = float(secs.total_seconds)
        except AttributeError:
            # secs is a float
            secs = float(secs)
        return value_at_time(self.array, self.frequency, self.offset, secs)
        
    def get_aligned(self, param):
        '''
        :param param: Node to align copy to.
        :type param: Node subclass
        :returns: A copy of self aligned to the input parameter.
        :rtype: DerivedParameterNode
        '''
        aligned_array = align(self, param)
        aligned_param = DerivedParameterNode(name=self.name,
                                             frequency=param.frequency,
                                             offset=param.offset)
        aligned_param.array = aligned_array
        return aligned_param 
    
    def slices_above(self, value):
        '''
        Get slices where the parameter's array is above value.
        
        :param value: Value to create slices above.
        :type value: float or int
        :returns: Slices where the array is above a certain value.
        :rtype: list of slice
        '''
        return slices_above(self.array, value)[1]
    
    def slices_below(self, value):
        '''
        Get slices where the parameter's array is below value.
        
        :param value: Value to create slices below.
        :type value: float or int
        :returns: Slices where the array is below a certain value.
        :rtype: list of slice
        '''
        return slices_below(self.array, value)[1]
    
    def slices_between(self, min_, max_):
        '''
        Get slices where the parameter's array values are between min_ and
        max_.
        
        :param min_: Minimum value within slice.
        :type min_: float or int
        :param max_: Maximum value within slice.
        :type max_: float or int
        :returns: Slices where the array is within min_ and max_.
        :rtype: list of slice
        '''
        return slices_between(self.array, min_, max_)[1]
    
    def slices_from_to(self, from_, to):
        '''Get slices of the parameter's array where values are between from_
        and to, and either ascending or descending depending on whether from_ 
        is greater than or less than to. For instance,
        param.slices_from_to(1000, 1500) is ascending and requires will only 
        return slices where values are between 1000 and 1500 if
        the value in the array at the start of the slice is less than the value at
        the stop. The opposite condition would be applied if the arguments are
        descending, e.g. slices_from_to(array, 1500, 1000).
        
        :param array:
        :type array: np.ma.masked_array
        :param from_: Value from.
        :type from_: float or int
        :param to: Value to.
        :type to: float or int
        :returns: Slices of the array where values are between from_ and to and either ascending or descending depending on comparing from_ and to.
        :rtype: list of slice'''
        return slices_from_to(self.array, from_, to)[1]

P = Parameter = DerivedParameterNode # shorthand


class SectionNode(Node, list):
    '''
    Derives from list to implement iteration and list methods.
    
    Is a list of Section namedtuples, each with attributes .name and .slice
    '''
    def __init__(self, *args, **kwargs):
        '''
        List of slices where this phase is active. Has a frequency and offset.
        
        :param items: Optional keyword argument of initial items to be contained within self.
        :type items: list
        '''
        if 'items' in kwargs:
            self.extend(kwargs['items'])
            del kwargs['items']
        super(SectionNode, self).__init__(*args, **kwargs)

    def create_section(self, section_slice, name=''):
        """
        Create a slice of the data.
        
        NOTE: Sections with slice start/ends of None can cause errors later
        when creating KPV/KTIs from a slice. However, they are valid for
        slicing data arrays from.
        """
        if section_slice.start is None or section_slice.stop is None:
            logging.debug("Section %s created %s with None start or stop.", 
                          self.get_name(), section_slice)
        section = Section(name or self.get_name(), section_slice)
        self.append(section)
        ##return section
        
    def create_sections(self, section_slices, name=''):
        for sect in section_slices:
            self.create_section(sect, name=name)
        
    #TODO: Accessor for 1Hz slice, 8Hz slice etc.
    def get_aligned(self, param):
        '''
        Creates a copy with section slices aligned to the frequency and offset
        of param.
        
        :param param: Parameter to align the copy of self to.
        :type param: Parameter object
        :returns: An object of the same type as self containing matching elements.
        :rtype: self.__class__
        '''
        aligned_node = self.__class__(frequency=param.frequency,
                                      offset=param.offset)
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        for section in self:
            if section.slice.start is None:
                converted_start = None
            else:
                converted_start = (section.slice.start * multiplier) + offset
            if section.slice.stop is None:
                converted_stop = None
            else:
                converted_stop = (section.slice.stop * multiplier) + offset
            converted_slice = slice(converted_start, converted_stop)
            aligned_node.create_section(converted_slice, section.name)
        return aligned_node
    
    def _get_condition(self, within_slice=None, name=None):
        '''
        Returns a condition function which checks if the element is within
        a slice or has a specified name if they are provided.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: Either a condition function or None.
        :rtype: func or None
        '''
        if within_slice and name:
            return lambda e: is_slice_within_slice(e.slice, within_slice) and \
                   e.name == name
        elif within_slice:
            return lambda e: is_slice_within_slice(e.slice, within_slice)
        elif name:
            return lambda e: e.name == name
        else:
            return None
    
    def get(self, within_slice=None, name=None):
        '''
        Gets elements either within_slice or with name. Duplicated from
        FormattedNameNode. TODO: Share implementation with NameFormattedNode,
        slight differences between types make it difficult.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: An object of the same type as self containing matching elements.
        :rtype: self.__class__
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=matching)
    
    def get_first(self, within_slice=None, name=None):
        '''
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: First Section matching conditions.
        :rtype: Section
        '''
        matching = self.get(within_slice=within_slice, name=name)
        return min(matching, key=attrgetter('slice.start'))
    
    def get_last(self, within_slice=None, name=None):
        '''
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: Last Section matching conditions.
        :rtype: Section
        '''
        matching = self.get(within_slice=within_slice, name=name)
        return max(matching, key=attrgetter('slice.stop'))
    
    def get_ordered_by_index(self, within_slice=None, name=None):
        '''
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: An object of the same type as self containing elements ordered by index.
        :rtype: self.__class__
        '''
        matching = self.get(within_slice=within_slice, name=name)
        ordered_by_start = sorted(matching, key=attrgetter('slice.start'))
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=ordered_by_start)
    

class FlightPhaseNode(SectionNode):
    """ Is a Section, but called "phase" for user-friendliness!
    """
    # create_phase and create_phases are shortcuts for create_section and 
    # create_sections.
    create_phase = SectionNode.create_section
    create_phases = SectionNode.create_sections


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
        
    def __repr__(self):
        return '%s' % list(self)
    
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
        :raises ValueError: If name is not a combination of self.NAME_FORMAT and self.NAME_VALUES.
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
        :returns: Either a condition function or None.
        :rtype: func or None
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
    
    def get(self, within_slice=None, name=None):
        '''
        Gets elements either within_slice or with name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :returns: An object of the same type as self containing elements ordered by index.
        :rtype: self.__class__
        '''
        condition = self._get_condition(within_slice=within_slice, name=name)
        matching = filter(condition, self) if condition else self # Q: Should this return self.__class__?
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=matching)
    
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
        matching = self.get(within_slice=within_slice, name=name)
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
        :returns: First element matching conditions.
        :rtype: self
        '''
        matching = self.get(within_slice=within_slice, name=name)
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
        matching = self.get(within_slice=within_slice, name=name)
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
        matching = self.get(within_slice=within_slice, name=name)
        return self.__class__(name=self.name, frequency=self.frequency,
                              offset=self.offset, items=matching)


class KeyTimeInstanceNode(FormattedNameNode):
    """
    TODO: determine what is required for get_derived to be stored in database
    """
    def __init__(self, *args, **kwargs):
        # place holder
        super(KeyTimeInstanceNode, self).__init__(*args, **kwargs)
        
    def create_kti(self, index, replace_values={}, **kwargs):
        '''
        Creates a KeyTimeInstance with the supplied index and creates a name
        from applying a combination of replace_values and kwargs as string
        formatting arguments to self.NAME_FORMAT. The KeyTimeInstance is
        appended to self.
        
        :param index: Index of the KeyTimeInstance within the data relative to self.frequency.
        :type index: int or float # Q: is float correct?
        :param replace_values: Dictionary of string formatting arguments to be applied to self.NAME_FORMAT.
        :type replace_values: dict
        :param kwargs: Keyword arguments will be applied as string formatting arguments to self.NAME_FORMAT.
        :type kwargs: dict
        :returns: The created KeyTimeInstance which is now appended to self.
        :rtype: KeyTimeInstance named tuple
        :raises KeyError: If a required string formatting key is not provided.
        :raises TypeError: If a string formatting argument is of the wrong type.
        '''
        if index is None:
            raise ValueError("Cannot create at index None")
        name = self.format_name(replace_values, **kwargs)
        kti = KeyTimeInstance(index, name)
        self.append(kti)
        return kti
    
    def get_aligned(self, param):
        '''
        :param param: Node to align this KeyTimeInstanceNode to.
        :type param: Node subclass
        :returns: An copy of the KeyTimeInstanceNode with its contents aligned to the frequency and offset of param.
        :rtype: KeyTimeInstanceNode
        '''
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        aligned_node = self.__class__(self.name, param.frequency,
                                      param.offset) 
        for kti in self:
            aligned_kti = copy.copy(kti)
            index_aligned = (kti.index * multiplier) + offset
            aligned_kti.index = index_aligned
            aligned_node.append(aligned_kti)
        return aligned_node


class KeyPointValueNode(FormattedNameNode):
    
    def __init__(self, *args, **kwargs):
        super(KeyPointValueNode, self).__init__(*args, **kwargs)

    def create_kpv(self, index, value, replace_values={}, **kwargs):
        '''
        Creates a KeyPointValue with the supplied index and value, and creates
        a name from applying a combination of replace_values and kwargs as 
        string formatting arguments to self.NAME_FORMAT. The KeyPointValue is
        appended to self.
        
        :param index: Index of the KeyTimeInstance within the data relative to self.frequency.
        :type index: int or float # Q: Is float correct?
        :param value: Value sourced at the index.
        :type value: float
        :param replace_values: Dictionary of string formatting arguments to be applied to self.NAME_FORMAT.
        :type replace_values: dict
        :param kwargs: Keyword arguments will be applied as string formatting arguments to self.NAME_FORMAT.
        :type kwargs: dict
        :returns: The created KeyPointValue which is now appended to self.
        :rtype: KeyTimeInstance named tuple
        :raises KeyError: If a required string formatting key is not provided.
        :raises TypeError: If a string formatting argument is of the wrong type.
        '''
        if index is None:
            raise ValueError("Cannot create at index None")
        name = self.format_name(replace_values, **kwargs)
        kpv = KeyPointValue(index, value, name)
        self.append(kpv)
        return kpv
    
    def get_aligned(self, param):
        '''
        :param param: Node to align this KeyPointValueNode to.
        :type param: Node subclass
        :returns: An copy of the KeyPointValueNode with its contents aligned to the frequency and offset of param.
        :rtype: KeyPointValueNode
        '''
        multiplier = param.frequency / self.frequency
        offset = (self.offset - param.offset) * param.frequency
        aligned_node = self.__class__(self.name, param.frequency, param.offset)
        for kpv in self:
            aligned_kpv = copy.copy(kpv)
            aligned_kpv.index = (aligned_kpv.index * multiplier) + offset
            aligned_node.append(aligned_kpv)
        return aligned_node
    
    def get_max(self, within_slice=None, name=None):
        '''
        Gets the KeyPointValue with the maximum value optionally filter
        within_slice or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :rtype: KeyPointValue
        '''
        matching = self.get(within_slice=within_slice, name=name)
        return max(matching, key=attrgetter('value')) if matching else None
    
    def get_min(self, within_slice=None, name=None):
        '''
        Gets the KeyPointValue with the minimum value optionally filter
        within_slice or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :rtype: KeyPointValue
        '''
        matching = self.get(within_slice=within_slice, name=name)
        return min(matching, key=attrgetter('value')) if matching else None
    
    def get_ordered_by_value(self, within_slice=None, name=None):
        '''
        Gets the element with the maximum value optionally filter within_slice
        or by name.
        
        :param within_slice: Only return elements within this slice.
        :type within_slice: slice
        :param name: Only return elements with this name.
        :type name: str
        :rtype: KeyPointValueNode
        '''
        matching = self.get(within_slice=within_slice, name=name)
        ordered_by_value = sorted(matching, key=attrgetter('value'))
        return KeyPointValueNode(name=self.name, frequency=self.frequency,
                                 offset=self.offset, items=ordered_by_value)
    
    def create_kpvs_at_ktis(self, array, ktis):
        '''
        Creates KPVs by sourcing the array at each KTI index. Requires the array
        to be aligned to the KTIs.
        
        :param array: Array to source values from.
        :type array: np.ma.masked_array
        :param ktis: KTIs with indices to source values within the array from.
        :type ktis: KeyTimeInstanceNode
        :returns None:
        :rtype: None
        '''
        for kti in ktis:
            value = value_at_index(array, kti.index)
            if value is None:
                logging.warning("Array is masked at index '%s' and therefore "
                                "KPV '%s' will not be created.", kti.index, self.name)
            else:
                self.create_kpv(kti.index, value)
    create_kpvs_at_kpvs = create_kpvs_at_ktis # both will work the same!
    
    def create_kpvs_within_slices(self, array, slices, function):
        '''
        Shortcut for creating KPVs from a number of slices by retrieving an
        index and value from function (for instance max_value).
        
        :param array: Array to source values from.
        :type array: np.ma.masked_array
        :param slices: Slices to create KPVs within.
        :type slices: SectionNode or list of slices.
        :param function: Function which will return an index and value from the array.
        :type function: function
        :returns: None
        :rtype: None
        '''
        for slice_ in slices:
            if isinstance(slice_, Section): # Use slice within Section.
                slice_ = slice_.slice
            index, value = function(array, slice_)
            self.create_kpv(index, value)


class FlightAttributeNode(Node):
    '''
    Can only store a single value per Node, however the value can be any
    object (dict, list, integer etc). The class name serves as the name of the
    attribute.
    '''
    def __init__(self, *args, **kwargs):
        self._value = None
        super(FlightAttributeNode, self).__init__(*args, **kwargs)
        # FlightAttributeNodes inherit frequency and offset attributes from Node,
        # yet these are not relevant to them. TODO: Change inheritance.
        self.frequency = self.hz = self.sample_rate = None
        self.offset = None
    
    def set_flight_attribute(self, value):
        self.value = value
    set_flight_attr = set_flight_attribute
    
    def get_aligned(self, param):
        """
        Cannot align a flight attribute.
        """
        return self


class NodeManager(object):
    def __repr__(self):
        return 'NodeManager: x%d nodes in total' % (
            len(self.lfl) + len(self.requested) + len(self.derived_nodes) + 
            len(self.aircraft_info) + len(self.achieved_flight_record))
    
    def __init__(self, start_datetime, lfl, requested, derived_nodes,
                 aircraft_info, achieved_flight_record):
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
        :returns: Ordered list of all Node names stored within the manager.
        :rtype: list of str
        """
        return sorted(list(set(['Start Datetime'] \
                               + self.lfl \
                               + self.derived_nodes.keys() \
                               + self.aircraft_info.keys() \
                               + self.achieved_flight_record.keys())))

    def get_attribute(self, name):
        """
        Get an attribute value from aircraft_info or achieved_flight_record
        dictionaries. If key is None, returns None. If key is present,
        returns an Attribute.
        
        :param name: Attribute name.
        :type name: str
        :returns: Attribute if available.
        :rtype: Attribute object or None
        """
        if name == 'Start Datetime':
            return Attribute(name, value=self.start_datetime)
        elif name in self.aircraft_info:
            return Attribute(name, value=self.aircraft_info[name])
        elif name in self.achieved_flight_record:
            return Attribute(name, value=self.achieved_flight_record[name])
        else:
            return None
    
    def operational(self, name, available):
        """
        Looks up the node by name and returns whether it can operate with the
        available dependencies.
        
        :param name: Name of Node.
        :type name: str
        :param available: Available dependencies to be passed into the derive method of the Node instance.
        :type available: list of str
        :returns: Result of Operational test on parameter.
        :rtype: bool
        """
        if name in self.lfl \
             or self.aircraft_info.get(name) is not None \
             or self.achieved_flight_record.get(name) is not None \
             or name == 'root'\
             or name == 'Start Datetime':
            return True
        elif name in self.derived_nodes:
            #NOTE: Raises "Unbound method" here due to can_operate being overridden without wrapping with @classmethod decorator
            res = self.derived_nodes[name].can_operate(available)
            if not res:
                logging.debug("Derived Node %s cannot operate with available nodes: %s",
                              name, available)
            return res
        else:  #elif name in unavailable_deps:
            logging.debug("Node '%s' is unavailable", name)
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
        self.frequency = self.hz = self.sample_rate = None
        self.offset = None

    def __nonzero__(self):
        return self.value != None
    
    def get_aligned(self, param):
        '''
        Attributes do not contain data which can be aligned to other parameters.
        Q: If attributes start storing indices rather than time, this will
        require implementing.
        '''
        return self


A = Attribute
P = Parameter
S = SectionNode
KPV = KeyPointValueNode
KTI = KeyTimeInstanceNode
