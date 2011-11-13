import logging
import numpy as np

from analysis.node import DerivedParameterNode
from analysis.library import rate_of_change, align, straighten_headings

#-------------------------------------------------------------------------------
# Derived Parameters


# Q: What do we do about accessing KTIs - params['a kti class name'] is a list of kti's
#   - could have a helper - filter_for('kti_name', take_max=True) # and possibly take_first, take_min, take_last??

# Q: Accessing information like ORIGIN / DESTINATION

# Q: What about V2 Vref etc?

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
    dependencies = ['Altitude Std', 'Radio Altitude']
    ##frequency = dependencies[0].frequency
    ##offset = dependencies[0].offset
    ##units = 'ft/min'
    def derive(self, params):
        alt_std = params['Altitude Std']
        alt_radio = params['Radio Altitude']
        
        # do magic in flight_analysis_algorithms.py
        return NotImplemented

    
class StraightHeading(DerivedParameterNode):
    dependencies = ['Heading']
    def derive(self, params):
        hdg = params['Heading']
        return straighten_headings(hdg)
    

class RateOfTurn(DerivedParameterNode):
    dependencies = [StraightHeading]
    ##frequency = StraightHeading.frequency
    ##offset = StraightHeading.offset
    ##units = 'deg/sec'
    def derive(self, params):
        shdg = params[StraightHeading.get_name()]
        return rate_of_change(shdg, 1, 1.0)
    #TODO: Pick up the sample rate and replace the hard-coded 1.0 Hz.
