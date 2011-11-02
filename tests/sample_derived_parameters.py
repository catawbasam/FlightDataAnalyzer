from analysis.node import DerivedParameterNode


#lfl_params = ['Indicated Airspeed', 
              #'Groundspeed', 
              #'Pressure Altitude', 
              #'Radio Altimeter', 
              #'Heading', 'TAT', 
              #'Laititude', 'Longitude', 
              #'Longitudinal g', 'Lateral g', 'Normal g', 
              #'Pitch', 'Roll', 
              #]

class SAT(DerivedParameterNode):
    name = 'SAT' # overide default name to enforce CAPS
    dependencies = ['TAT', 'Indicated Airspeed', 'Pressure Altitude']
    def derive(self): pass
    
class MACH(DerivedParameterNode):
    name = 'MACH'
    dependencies = ['Indicated Airspeed', SAT] # SAT as object reference
    def derive(self): pass

class TrueAirspeed(DerivedParameterNode):
    dependencies = ['Indicated Airspeed', 'Pressure Altitude', 'SAT'] # SAT as string
    def derive(self): pass
    
class SmoothedTrack(DerivedParameterNode):
    dependencies = ['True Airspeed', 'Heading', 'Latitude', 'Longitude']
    def derive(self): pass
    
class VerticalG(DerivedParameterNode):
    name = 'Vertical g'
    dependencies = ['Longitudinal g', 'Lateral g', 'Normal g', 'Pitch', 'Roll']
    def derive(self): pass
    
class HorizontalGAlongTrack(DerivedParameterNode):
    name = 'Horizontal g Along Track'
    dependencies = ['Longitudinal g', 'Lateral g', 'Normal g', 'Pitch', 'Roll']
    def derive(self): pass

class HorizontalGAcrossTrack(DerivedParameterNode):
    name = 'Horizontal g Across Track'
    dependencies = ['Longitudinal g', 'Lateral g', 'Normal g', 'Pitch', 'Roll']
    def derive(self): pass
    
class HeadingRate(DerivedParameterNode):
    dependencies = ['Heading']
    def derive(self): pass
    
class HeightAboveGround(DerivedParameterNode):
    dependencies = [VerticalG, 'Radio Altimeter']
    def derive(self): pass
    
class MomentOfTakeoff(DerivedParameterNode):
    dependencies = [HeightAboveGround]
    def derive(self): pass
    
class SmoothedGroundspeed(DerivedParameterNode):
    dependencies = [HorizontalGAcrossTrack, 'Groundspeed']
    def derive(self): pass
    
class SlipOnRunway(DerivedParameterNode):
    dependencies = [HorizontalGAcrossTrack, 'Heading Rate', 'Smoothed Groundspeed']
    def derive(self): pass
    
class VerticalSpeed(DerivedParameterNode):
    dependencies = ['Pressure Altitude', 'Vertical g']
    def derive(self): pass

    


