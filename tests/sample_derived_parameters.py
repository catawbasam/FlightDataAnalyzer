from analysis_engine.node import DerivedParameterNode, P


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
    def derive(self, tat=P('TAT'), airspeed=P('Indicated Airspeed'),
               press_alt=P('Pressure Altitude')):
        return NotImplemented
    
class MACH(DerivedParameterNode):
    def derive(self, ias = P('Airspeed'), tat = P('TAT'),
               alt = P('Altitude Std')):
        return NotImplemented

class TrueAirspeed(DerivedParameterNode):
    dependencies = ['Indicated Airspeed', 'Pressure Altitude', 'SAT'] # SAT as string
    def derive(self, airspeed=P('Indicated Airspeed'), 
               press_alt=P('Pressure Altitude'), sat=P('SAT')):
        return NotImplemented
    
class SmoothedTrack(DerivedParameterNode):
    def derive(self, airspeed=P('True Airspeed'), hdg=P('Heading'),
               lat=P('Latitude'), lng=P('Longitude'),
               inertial_lat=P('Inertial Latitude'),
               inertial_lng=P('Inertial Longitude')):
        return NotImplemented
    
    @classmethod
    def can_operate(cls, available):
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
    def derive(self, lng=P('Longitudinal g'), lat=P('Lateral g'),
               pitch=P('Pitch'), roll=P('Roll')):
        return NotImplemented
    
class HorizontalGAlongTrack(DerivedParameterNode):
    name = 'Horizontal g Along Track'
    def derive(self, lng=P('Longitudinal g'), lat=P('Lateral g'),
               norm_g=P('Normal g'), pitch=P('Pitch'), roll=P('Roll')):
        return NotImplemented

class HorizontalGAcrossTrack(DerivedParameterNode):
    name = 'Horizontal g Across Track'
    def derive(self, lng=P('Longitudinal g'), lat=P('Lateral g'),
               norm_g=P('Normal g'), pitch=P('Pitch'), roll=P('Roll')):
        return NotImplemented
    
class HeadingRate(DerivedParameterNode):
    def derive(self, hdg=P('Heading')):
        return NotImplemented
    
class HeightAboveGround(DerivedParameterNode):
    def derive(self, vert_g=P('Vertical g'), rad_alt=P('Radio Altimeter')):
        return NotImplemented
    
class MomentOfTakeoff(DerivedParameterNode):
    def derive(self, height=P('Height Above Ground')):
        return NotImplemented
    
class SmoothedGroundspeed(DerivedParameterNode):
    def derive(self, horiz_g=P('Horizontal g Across Track'),
               groundspeed=P('Groundspeed')):
        return NotImplemented
    
class SlipOnRunway(DerivedParameterNode):
    def derive(self, horiz_g=P('Horizontal g Across Track'),
               hdg_rate=P('Heading Rate'), groundspeed=P('Smoothed Groundspeed')):
        return NotImplemented
    
class VerticalSpeed(DerivedParameterNode):
    def derive(self, press_alt=P('Pressure Altitude'), vert_g=P('Vertical g')):
        return NotImplemented

    


