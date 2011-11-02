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
    dependencies = ['True Airspeed', 'Heading', 'Latitude', 'Longitude', 'Inertial Latitude', 'Inertial Longitude']
    def derive(self): pass
    
    def can_operate(self, available):
        # Requires matching LAT/LONG pairs to operate - True Airspeed and Heading are a bonus!
        if 'Latitude' in available and 'Longitude' in available:
            # preferred, so lets use this
            #Q: store a flag now?
            ##self.use_ineterial = False
            return True
        elif 'Inertial Latitude' in available and 'Inertial Longitude' in available:
            ##self.use_ineterial = True
            return True
        else:
            return False
    
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

    


