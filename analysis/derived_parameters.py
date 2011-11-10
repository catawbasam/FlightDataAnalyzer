import logging

from analysis.node import DerivedParameterNode
from analysis.library import straighten_headings

#-------------------------------------------------------------------------------
# Derived Parameters


# Q: What do we do about accessing KTIs - params['a kti class name'] is a list of kti's
#   - could have a helper - filter_for('kti_name', take_max=True) # and possibly take_first, take_min, take_last??

class AltitudeQNH(DerivedParameterNode):
    dependencies = ['BAROMB', 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class TrueAirspeed(DerivedParameterNode):
    dependencies = ['SAT', 'VMO', 'MMO', 'Indicated Airspeed', 'Altitude QNH']
    def derive(self, params):
        return NotImplemented
    
class MACH(DerivedParameterNode):
    name = 'MACH'
    dependencies = ['Indicated Airspeed', 'TAT', 'Altitude Std']
    def derive(self, params):
        return NotImplemented
        
    
##class V1V2Vapp(DerivedParameterNode):
    ### URRR?
    ##pass


class StraightHeading(DerivedParameterNode):
    dependencies = ['Heading']
    def derive(self, params):
        hdg = params['Heading']
        return straighten_headings(hdg)

