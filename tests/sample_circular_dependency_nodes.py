from analysis_engine.node import DerivedParameterNode, P
from analysis_engine.library import any_of

#############################################################################
# Circular Dependency setup

class AirspeedAtGearDownSelected(DerivedParameterNode):
    def derive(self, gd=P('Gear Down Selected'), spd=P('Airspeed')):
        return NotImplemented


class GearDown(DerivedParameterNode):
    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               gl=P('Gear (L) Down'),
               gn=P('Gear (N) Down'),
               gr=P('Gear (R) Down'),
               gear_sel=P('Gear Down Selected')):
        return NotImplemented


class GearDownSelected(DerivedParameterNode):
    @classmethod
    def can_operate(cls, available):
        return 'Gear Down' in available

    def derive(self,
               gear_down=P('Gear Down'),
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning')):
        return NotImplemented