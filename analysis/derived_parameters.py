import logging
import numpy as np

from analysis.node import DerivedParameterNode, Parameter
from analysis.library import (align, hysteresis,
                              rate_of_change, straighten_headings)

from settings import HYSTERESIS_FPIAS, HYSTERESIS_FPROC

#-------------------------------------------------------------------------------
# Derived Parameters


# Q: What do we do about accessing KTIs - params['a kti class name'] is a list of kti's
#   - could have a helper - filter_for('kti_name', take_max=True) # and possibly take_first, take_min, take_last??

# Q: Accessing information like ORIGIN / DESTINATION

# Q: What about V2 Vref etc?


class AccelerationVertical(DerivedParameterNode):
    dependencies = ['Acceleration Normal', 'Acceleration Lateral', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll']
    def derive(self, params):
        # Resolution of three accelerations to compute the vertical
        # acceleration (perpendicular to the earth surface).

        # The Acceleration Normal parameter is referenced many times, so use 
        # this shorthand version:
        az_p = params['Acceleration Normal']
        
        # Align the acceleration and attitude samples to the normal acceleration,
        # ready for combining them.
        # "align" returns an array of the first parameter aligned to the second.
        ax = align(params['Acceleration Longitudinal'], az_p) 
        pch = np.radians(align(params['Pitch'], az_p))
        ay = align(params['Acceleration Lateral'], az_p) 
        rol = np.radians(align(params['Roll'], az_p))
        
        # Simple Numpy algorithm working on masked arrays
        resolved_in_pitch = ax * np.sin(pch) + az_p.array * np.cos(pch)
        resolved_in_roll = resolved_in_pitch * np.cos(rol) - ay * np.sin(rol)
        params['Acceleration Vertical'] = Parameter('Acceleration Vertical',
                                                    resolved_in_roll,
                                                    az_p.hz,
                                                    az_p.offset)

class AltitudeAAL(DerivedParameterNode):
    name = 'Altitude AAL'
    dependencies = ['Altitude Std', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
    
class AltitudeQNH(DerivedParameterNode):
    name = 'Altitude QNH'
    dependencies = ['BAROMB', 'Altitude Std', 'Takeoff Altitude', 'Landing Altitude']
    def derive(self, params):
        return NotImplemented
    
class TrueAirspeed(DerivedParameterNode):
    dependencies = ['SAT', 'VMO', 'MMO', 'Indicated Airspeed', 'Altitude QNH']
    def derive(self, params):
        return NotImplemented
    
class TrueHeading(DerivedParameterNode):
    dependencies = ['Magnetic Heading', 'Magnetic Deviation']
    def derive(self, params):
        return NotImplemented
    
class MACH(DerivedParameterNode):
    name = 'MACH'
    dependencies = ['Indicated Airspeed', 'TAT', 'Altitude Std']
    def derive(self, params):
        return NotImplemented
        
class SmoothedLatitude(DerivedParameterNode):
    dependencies = ['Latitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class SmoothedLongitude(DerivedParameterNode):
    dependencies = ['Longitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class DistanceToLanding(DerivedParameterNode):
    dependencies = ['Altitude AAL', 'Ground Speed', 'Glideslope Deviation', 'LandingAirport']
    def derive(self, params):
        return NotImplemented
    
class FlapCorrected(DerivedParameterNode):
    dependencies = ['Flap']
    def derive(self, params):
        return NotImplemented
    

class FlightPhaseAirspeed(DerivedParameterNode):
    dependencies = ['Airspeed']
    def derive(self, params):
        prime_dep = self.get_first_param(params)
        self.array = hysteresis(prime_dep.array, HYSTERESIS_FPIAS)


class FlightPhaseRateOfClimb(DerivedParameterNode):
    dependencies = ['Altitude STD']
    def derive(self, params):
        prime_dep = self.get_first_param(params)
        self.array = hysteresis(rate_of_change(prime_dep.array, 4, prime_dep.hz),
                                HYSTERESIS_FPROC)


class Relief(DerivedParameterNode):
    # also known as Terrain
    dependencies = ['Altitude AAL', 'Radio Altitude']
    def derive(self, params):
        return NotImplemented
    
class LocaliserGap(DerivedParameterNode):
    dependencies = ['Localiser Deviation', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented

    
class GlideslopeGap(DerivedParameterNode):
    dependencies = ['Glideslope Deviation', 'Altitude AAL']
    def derive(self, params):
        return NotImplemented
 
    
class ILSValLim(DerivedParameterNode):
    # Taken from diagram as: ILS VAL/LIM -- TODO: rename!
    dependencies = [LocaliserGap, GlideslopeGap]
    def derive(self, params):
        return NotImplemented


class RateOfClimb(DerivedParameterNode):
    dependencies = ['Altitude STD', 'Altitude Radio']
    def derive(self, params):
        self.array = rate_of_change(params['Altitude STD'], 1, params['Altitude STD'].hz)

    
class HeadContinuous(DerivedParameterNode):
    dependencies = ['Heading Magnetic']
    def derive(self, params):
        self.array = straighten_headings(params['Heading Magnetic'].array)


class RateOfTurn(DerivedParameterNode):
    dependencies = [HeadContinuous]
    def derive(self, params):
        prime_dep = self.get_first_param(params)
        self.array = rate_of_change(prime_dep.array, 1, prime_dep.hz)

'''
class RateOfTurn(DerivedParameterNode):
    dependencies = ['Head Continuous']
    def derive(self, params):
        p_dep = params[self.get_dependency_names()[0]]
        self.array = rate_of_change(p_dep.array, 1, p_dep.hz)


class RateOfTurn(DerivedParameterNode):
    dependencies = [StraightHeading]
    def derive(self, params):
        shdg = params[StraightHeading.get_name()]
        self.array = rate_of_change(shdg.array, 1, shdg.hz)
'''
