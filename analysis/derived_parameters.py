import logging
import numpy as np

from hdfaccess.parameter import P, Parameter
from analysis.node import DerivedParameterNode

from analysis.library import (align, 
                              first_order_lag,
                              first_order_washout,
                              hysteresis, 
                              interleave,
                              rate_of_change, 
                              repair_mask,
                              straighten_headings)

from settings import (AZ_WASHOUT_TC,
                      HYSTERESIS_FPALT,
                      HYSTERESIS_FPALT_CCD,
                      HYSTERESIS_FPIAS, 
                      HYSTERESIS_FPROC,
                      GRAVITY,
                      RATE_OF_CLIMB_LAG_TC
                      )

#-------------------------------------------------------------------------------
# Derived Parameters

# Q: Accessing information like ORIGIN / DESTINATION

# Q: What about V2 Vref etc?


class AccelerationVertical(DerivedParameterNode):
    def derive(self, acc_norm=P('Acceleration Normal'), 
               acc_lat=P('Acceleration Lateral'), 
               acc_long=P('Acceleration Longitudinal'), 
               pitch=P('Pitch'), roll=P('Roll')):
        """
        Resolution of three accelerations to compute the vertical
        acceleration (perpendicular to the earth surface).
        """
        # Align the acceleration and attitude samples to the normal acceleration,
        # ready for combining them.
        # "align" returns an array of the first parameter aligned to the second.
        ax = align(acc_long, acc_norm) 
        pch = np.radians(align(pitch, acc_norm))
        ay = align(acc_lat, acc_norm) 
        rol = np.radians(align(roll, acc_norm))
        
        # Simple Numpy algorithm working on masked arrays
        resolved_in_pitch = ax * np.sin(pch) + acc_norm.array * np.cos(pch)
        self.array = resolved_in_pitch * np.cos(rol) - ay * np.sin(rol)
        

class AirspeedMinusVref(DerivedParameterNode):
    
    def derive(self, airspeed=P('Airspeed'), vref=P('Vref')):
        vref_aligned = align(vref, airspeed)
        self.array = airspeed.array - vref_aligned


class AltitudeAAL(DerivedParameterNode):
    name = 'Altitude AAL'
    def derive(self, alt_std=P('Altitude STD'), alt_rad=P('Altitude Radio')):
        return NotImplemented
    
class AltitudeAALForPhases(DerivedParameterNode):
    name = 'Altitude AAL For Phases'
    # This crude parameter is used for flight phase determination of the 
    # Approach phase, and only uses pressure altitude for robustness.
    def derive(self, alt_std=P('Altitude STD'), airs=P('Fast')):
        
        # Initialise the array to zero, so that the altitude above the airfield
        # will be 0ft when the aircraft cannot be airborne.
        self.array = np.ma.zeros(len(alt_std.array))
        
        repair_mask(alt_std.array) # Remove small sections of corrupt data

        for air in airs:
            begin = air.slice.start
            end = air.slice.stop
            peak = np.ma.argmax(alt_std.array[air.slice])
            self.array[begin:begin+peak] = alt_std.array[begin:begin+peak] - alt_std.array[begin]
            self.array[begin+peak:end] = alt_std.array[begin+peak:end] - alt_std.array[end]
    
    
class AltitudeForClimbCruiseDescent(DerivedParameterNode):
    name = 'Altitude For Climb Cruise Descent'
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis ( alt_std.array, HYSTERESIS_FPALT_CCD)
    
    
class AltitudeForPhases(DerivedParameterNode):
    name = 'Altitude For Phases'
    def derive(self, alt_std=P('Altitude STD')):
        self.array = hysteresis ( alt_std.array, HYSTERESIS_FPALT)
    
    
class AltitudeRadio(DerivedParameterNode):
    # This function allows for the distance between the radio altimeter antenna
    # and the main wheels of the undercarriage.

    # The parameter raa_to_gear is measured in feet and is positive if the
    # antenna is forward of the mainwheels.
    def derive(self, alt_rad=P('Altitude Radio Sensor'), pitch=P('Pitch'),
               main_gear_to_alt_rad=P('Main Gear To Altitude Radio')): # TODO: Fix once A (aircraft) has been defined.
        # Align the pitch attitude samples to the Radio Altimeter samples,
        # ready for combining them.
        pitch_aligned = np.radians(align(pitch, alt_rad))
        # Now apply the offset if one has been provided
        self.array = alt_rad.array - np.sin(pitch_aligned) * main_gear_to_alt_rad

        
class AltitudeQNH(DerivedParameterNode):
    name = 'Altitude QNH'
    def derive(self):
        return NotImplemented


class AltitudeTail(DerivedParameterNode):
    # This function allows for the distance between the radio altimeter antenna
    # and the point of the airframe closest to tailscrape.
    
    # The parameter gear_to_tail is measured in feet and is the distance from 
    # the main gear to the point on the tail most likely to scrape the runway.
    def derive(self, alt_rad=P('Altitude Radio'), 
               pitch=P('Pitch'),
               dist_gear_to_tail=None):#A('Dist Gear To Tail')): # TODO: Is this name correct?
        # Align the pitch attitude samples to the Radio Altimeter samples,
        # ready for combining them.
        pitch_aligned = np.radians(align(pitch, alt_rad))
        # Now apply the offset
        self.array = alt_rad.array - np.sin(pitch_aligned) * dist_gear_to_tail
        

class DistanceToLanding(DerivedParameterNode):
    def derive(self, alt_aal=P('Altitude AAL'),
               gspd=P('Ground Speed'),
               ils_gs=P('Glideslope Deviation'),
               ldg=P('LandingAirport')):
        return NotImplemented
    

class FlapCorrected(DerivedParameterNode):
    def derive(self, flap=P('Flap')):
        return NotImplemented
    

class FlightPhaseAirspeed(DerivedParameterNode):  #Q: Rename to AirpseedHysteresis ?
    def derive(self, airspeed=P('Airspeed')):
        self.array = hysteresis(airspeed.array, HYSTERESIS_FPIAS)


class FlightPhaseRateOfClimb(DerivedParameterNode):
    def derive(self, alt=P('Altitude STD')):
        self.array = rate_of_change(alt, 4)
        
        #self.array = hysteresis(rate_of_change(alt, 4),
                                #HYSTERESIS_FPROC)


class HeadContinuous(DerivedParameterNode):
    def derive(self, head_mag=P('Heading Magnetic')):
        self.array = straighten_headings(head_mag.array)


class ILSLocaliserGap(DerivedParameterNode):
    def derive(self, ils_loc=P('Localiser Deviation'),
               alt_aal=P('Altitude AAL')):
        return NotImplemented

    
class ILSGlideslopeGap(DerivedParameterNode):
    def derive(self, ils_gs=P('Glideslope Deviation'),
               alt_aal=P('Altitude AAL')):
        return NotImplemented
 
    
'''

This is old and I don't know what it does or if we need/want this. DJ

class ILSValLim(DerivedParameterNode):
    # Taken from diagram as: ILS VAL/LIM -- TODO: rename!
    dependencies = [LocaliserGap, GlideslopeGap]
    def derive(self, params):
        return NotImplemented
'''

class MACH(DerivedParameterNode):
    name = 'MACH'
    def derive(self, ias=P('Airspeed'),
               tat=P('TAT'), alt=P('Altitude Std')):
        return NotImplemented
        

class RateOfClimb(DerivedParameterNode):
    def derive(self, 
               az=P('Acceleration Vertical'),
               alt_std=P('Altitude STD'),
               alt_rad=P('Altitude_Radio'),
               ige = S('In Ground Effect')
               ):
        roc = rate_of_change(alt_std, 2)
        roc_rad_ma = rate_of_change(alt_rad, 1)
        
        # Use pressure altitude rate outside ground effect and 
        # radio altitude data inside ground effect.
        for ige_sect in ige:
            roc[ige_sect.slice] = roc_rad_ma[ige_sect.slice]
        
        # Lag this rate of climb
        lagged_roc = first_order_lag(roc, RATE_OF_CLIMB_LAG_TC, self.hz)
        az_washout = first_order_washout(az.array, AZ_WASHOUT_TC, az.hz,
                                         initial_value=az.array[0])
        inertial_roc = first_order_lag(az_washout, RATE_OF_CLIMB_LAG_TC, az.hz,
                                       gain=GRAVITY*RATE_OF_CLIMB_LAG_TC*60.0)
        self.array = lagged_roc + inertial_roc



class RateOfClimbForFlightPhases(DerivedParameterNode):
    def derive(self, alt_std = P('Altitude STD')):
        self.array = rate_of_change(repair_mask(alt_std),2)*60


class Relief(DerivedParameterNode):
    # also known as Terrain
    
    # Quickly written without tests as I'm really editing out the old dependencies statements :-(
    def derive(self, alt_aal=P('Altitude AAL'),
               alt_rad=P('Radio Altitude')):
        #WARNING: align is done before hand, but in the reverse order as depicted below:
        ##altitude = align(alt_aal, alt_rad)
        self.array = alt_aal.array - alt_rad.array

'''

Q: Better done together - How?

A: Create a top level class which performs the manipulation using it's own
method or move the method to a library function. Then create two classes to
return each of the individual parameters. It's more important to write the
code once than to do the calculation twice!


class SmoothedLatitude(DerivedParameterNode):
    dependencies = ['Latitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
    
class SmoothedLongitude(DerivedParameterNode):
    dependencies = ['Longitude', 'True Heading', 'Indicated Airspeed'] ##, 'Altitude Std']
    def derive(self, params):
        return NotImplemented
'''

class TrueAirspeed(DerivedParameterNode):
    dependencies = ['SAT', 'VMO', 'MMO', 'Indicated Airspeed', 'Altitude QNH']
    def derive(self, ias=P('Airspeed'),
               alt_std=P('Altitude STD'),
               sat=P('SAT')):
        return NotImplemented
    
    
class MagneticDeviation(DerivedParameterNode):
    def derive(self):
        return NotImplemented
    
    
class TrueHeading(DerivedParameterNode):
    # Requires the computation of a magnetic deviation parameter linearly 
    # changing from the deviation at the origin to the destination.
    def derive(self, head=P('Heading Continuous'),
               mag_dev=P('Magnetic Deviation')):
        self.array = head.array + mag_dev.array
    

class RateOfTurn(DerivedParameterNode):
    def derive(self, head=P('Head Continuous')):
        self.array = rate_of_change(head, 1)


class Pitch(DerivedParameterNode):
    def derive(self, p1=P('Pitch (1)'), p2=P('Pitch (2)')):
        self.hz = p1.hz * 2
        self.offset = min(p1.offset, p2.offset)
        self.array = interleave(p1, p2)