from hdfaccess.parameter import Parameter

from analysis.library import align

class Parameter(Parameter):
    '''
    Extends Parameter to provide AnalysisEngine specific functionality.
    '''
    def get_aligned(self, param):
        aligned_array = align(self, param)
        return self.__class__(aligned_array, frequency=param.frequency,
                              offset=param.offset)
