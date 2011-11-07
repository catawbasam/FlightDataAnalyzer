import logging

from analysis.node import DerivedParameterNode
from analysis.library import straighten_headings

###-------------------------------------------------------------------------------
### Derived Parameters
### ==================


        
##class Sat(DerivedParameterNode):
    ##dependencies = [TAT, ALTITUDE_STD]
    
    ##def derive(self, params):
        ##return sum([params.TAT.value,])
    

##class Mach(DerivedParameterNode):
    ##dependencies = [AIRSPEED, SAT, TAT, ALTITUDE_STD]
    
    ##def can_operate(self, available):
        ##if AIRSPEED in available and (SAT in available or TAT in available):
            ##return True
        ##else:
            ##return False
        
    ##def derive(self, params):
        ##return 12
        
        
class StraightHeading(DerivedParameterNode):
    dependencies = ['Heading']
    def derive(self, params):
        hdg = params['Heading']
        return straighten_headings(hdg)

