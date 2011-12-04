import logging
import numpy as np

from analysis.node import FlightPhaseNode, KeyTimeInstance, P
from analysis.settings import (AIRSPEED_THRESHOLD,
                               ALTITUDE_FOR_CLB_CRU_DSC,
                               RATE_OF_CLIMB_FOR_CLIMB_PHASE,
                               RATE_OF_CLIMB_FOR_DESCENT_PHASE,
                               RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                               RATE_OF_TURN_FOR_FLIGHT_PHASES,
                               WING_SPAN
                               )

'''
class Airborne(FlightPhaseNode):
    dependencies = ['altitude_std_smoothed', 'takeoff_end', 'landing_start']
    returns = ['Airborne']
    
    def derive(self, altitude_std_smoothed, takeoff_end, landing_start): # TODO: New argument format, what are the Parameter names?
        # Create a simple "Airborne" mask that covers the period between the takeoff and landing phases.
        # We assign it to altitude_std as this makes it easy to plot and check, although it is the mask that is really useful.
        ##airborne_phase = create_phase_inside(altitude_std_smoothed, kpt['TakeoffEndEstimate'], kpt['LandingStartEstimate'])
        airborne_phase = create_phase_inside(altitude_std_smoothed, takeoff_end, landing_start)
        self.create_phase(airborne_phase)
'''

class Airborne(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        level_flight = np.ma.masked_inside(roc.array, 
                                           -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                           RATE_OF_CLIMB_FOR_LEVEL_FLIGHT)
        try:
            a,b = np.ma.flatnotmasked_edges(level_flight)
            self.create_phases([slice(a,b,None)])
        except:
            pass # Just don't create a phase if none exists.
        

class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt=P('Altitude For Phases')):
        ccd = np.ma.masked_less(alt.array, ALTITUDE_FOR_CLB_CRU_DSC)
        self.create_phases(np.ma.clump_unmasked(ccd))


class ClimbFromBottomOfDescent(FlightPhaseNode):
    def derive(self, 
               toc = P('Top Of Climb'),
               eot = P('Climb Start'), # AKA End Of Takeoff
               bod = P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc._kti_list:
            toc_list.append(this_toc.index)
            
        # Now see which follows a takeoff
        for this_eot in eot._kti_list:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc == None)):
                    closest_toc = this_toc

            # Build the slice from what we have found.
            self.create_phase(slice(eot, closest_toc))        
        

        # Now see which follows this minimum
        for this_bod in bod._kti_list:
            bod = this_bod.index
            # Scan the TODs
            closest_toc = None
            for this_toc in toc_list:
                if (bod < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc == None)):
                    closest_toc = this_toc

            # Build the slice from what we have found.
            self.create_phase(slice(bod, closest_toc))        
        return 

        
class Climbing(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb')):
        # Climbing is used for data validity checks and to reinforce regimes.
        climbing = np.ma.masked_less(roc.array, RATE_OF_CLIMB_FOR_CLIMB_PHASE)
        climbing_slices = np.ma.clump_unmasked(climbing)
        self.create_phases(climbing_slices)
        

class Cruise(FlightPhaseNode):
    def derive(self, 
               ccd = P('Climb Cruise Descent'),
               toc = P('Top Of Climb'),
               tod = P('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent 
        # at this time. The problem is that they need not be in tidy 
        # order as the lists may not be of equal lengths.
        
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc._kti_list:
            toc_list.append(this_toc.index)
        tod_list = []
        for this_tod in tod._kti_list:
            tod_list.append(this_tod.index)

        # Now see which fit which Cruise/Climb/Descent phases
        for ccd_phase in ccd._sections:

            # Scan the TOCs
            found_toc = None
            for each_toc in toc_list:
                if (ccd_phase.slice.start <= each_toc and
                    each_toc <= ccd_phase.slice.stop):
                    found_toc = each_toc
                    break

            # Scan the TODs
            found_tod = None
            for each_tod in tod_list:
                if (ccd_phase.slice.start <= each_tod and
                    each_tod <= ccd_phase.slice.stop):
                    found_tod = each_tod
                    break

            # Build the slice from what we have found.
            if found_toc == None and found_tod == None:
                pass
            if found_toc == None and found_tod != None:
                self.create_phase(slice(ccd_phase.slice.start, found_tod))
            if found_toc != None and found_tod == None:
                self.create_phase(slice(found_toc, ccd_phase.slice.stop))
            if found_toc != None and found_tod != None:
                self.create_phase(slice(found_toc, found_tod))
                    

class Descending(FlightPhaseNode):
    """ Descending faster than 800fpm towards the ground
    """
    def derive(self, roc=P('Rate Of Climb')):
        # Rate of climb and descent limits of 800fpm gives good distinction
        # with level flight.
        descending = np.ma.masked_greater(roc.array,
                                          RATE_OF_CLIMB_FOR_DESCENT_PHASE)
        desc_slices = np.ma.clump_unmasked(descending)
        self.create_phases(desc_slices)


class Descent(FlightPhaseNode):
    def derive(self, descending=Descending, roc=P('Rate Of Climb')):
        return NotImplemented


'''
class DescentToLanding(FlightPhaseNode):
'''

class DescentToBottomOfDescent(FlightPhaseNode):
    def derive(self, 
               tod = P('Top Of Descent'), 
               bod = P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod._kti_list:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod._kti_list:
            bod = this_bod.index
            # Scan the TODs
            closest_tod = None
            for this_tod in tod_list:
                if (bod > this_tod and
                    this_tod > closest_tod):
                    closest_tod = this_tod

            # Build the slice from what we have found.
            self.create_phase(slice(closest_tod, bod))        
        return 


class DescentLowClimb(FlightPhaseNode):
    def derive(self, alt=P('Altitude For Phases')):
        dlc = np.ma.masked_greater(alt.array, ALTITUDE_FOR_CLB_CRU_DSC)
        dlc_list = np.ma.clump_unmasked(dlc)
        for this_dlc in dlc_list:
            if this_dlc.start == 0:
                dlc_list.remove(this_dlc)
            if this_dlc.stop == len(alt.array):
                dlc_list.remove(this_dlc)
        self.create_phases(dlc_list)


class Fast(FlightPhaseNode):
    def derive(self, airspeed=P('Airspeed')):
        # Did the aircraft go fast enough to possibly become airborne?
        fast_where = np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD)
        fast_slices = np.ma.clump_unmasked(fast_where)
        self.create_phases(fast_slices)
 
class InGroundEffect(FlightPhaseNode):
    def derive(self, alt_rad=P('Altitude Radio')):
        low_where = np.ma.masked_greater(alt_rad.array, WING_SPAN)
        low_slices = np.ma.clump_unmasked(low_where)
        self.create_phases(low_slices)
 

class LevelFlight(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        level_flight = np.ma.masked_outside(roc.array, 
                                            -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                            RATE_OF_CLIMB_FOR_LEVEL_FLIGHT)
        level_slices = np.ma.clump_unmasked(level_flight)
        self.create_phases(level_slices)
        
'''
class LevelFlight(FlightPhaseNode):
    def derive(self, airspeed=P('Airspeed'),
               alt_std=P('Altitude Std')):
        # Rate of climb and descent limits of 800fpm gives good distinction with level flight.
        level_flight = np.ma.masked_where(
            np.ma.logical_or(np.ma.abs(RATE_OF_CLIMB_FOR_FLIGHT_PHASES) > 300.0, airspeed < 100.0), altitude_std)
        level_slices = np.ma.clump_unmasked(level_flight)
        self.create_phases(level_slices)
'''        

class OnGround(FlightPhaseNode):
    def derive(self, airspeed=P('Airspeed')):
        # Did the aircraft go fast enough to possibly become airborne?
        fast_where = np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD)
        a,b = np.ma.flatnotmasked_edges(fast_where)
        self.create_phases([slice(a, b, None)])
    
'''
RTO is like this :o)
class RejectedTakeoff(FlightPhaseNode):
    def derive(self, fast=Fast._section, level=LevelFlight._section):
        if len(fast) > 0 and len(level) = 0:
            # Went fast down the runway but never got airborne,
            # so must be a rejected takeoff.
            self.create_phases(somehow the same as the Fast slice !)
'''



    
class Turning(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES
    """
    def derive(self, rate_of_turn=P('Rate Of Turn')):
        turning = np.ma.masked_inside(rate_of_turn.array,
                                      RATE_OF_TURN_FOR_FLIGHT_PHASES * (-1.0),
                                      RATE_OF_TURN_FOR_FLIGHT_PHASES)
        
        turn_slices = np.ma.clump_unmasked(turning)
        self.create_phases(turn_slices)
        

  
    
#TODO: Move below into "Derived" structure!
def takeoff_and_landing(block, fp, ph, kpt, kpv):
    
    #==========================================================================
    # TAKEOFF 
    #==========================================================================

    # Record the heading at airspeed_threshold, part way down the runway:
    head_takeoff = head_mag[kpt['TakeoffStartEstimate']]
    
    # Track back to find where the aircraft turned onto the runway
    '''
    # Version 1 using Numpy - fails when there is no suitable data to trigger a mask edge.
    countback,dummy = np.ma.flatnotmasked_edges(
                      np.ma.masked_where(
                          np.ma.abs(head_takeoff - head_mag[start_toff:0:-1]) < 15.0,
                          head_mag[start_toff:0:-1] ))
    '''
    
    # Version 2 using loop - preferred as it deals with lack of turn data.
    turn_onto_runway = kpt['TakeoffStartEstimate']
    
    while abs(head_takeoff - head_mag[turn_onto_runway]) < 15.0:
        turn_onto_runway -= 1
        if turn_onto_runway == 0:
            logging.info ('\Data did not contain turn onto runway')
            break

    if turn_onto_runway < kpt['TakeoffStartEstimate']: # Only record the runway turnoff if it was recorded.
        kpt['TakeoffTurnOntoRunway'] = turn_onto_runway # A better indication of the start of the takeoff
        
    '''    
    # Version 3 using new Seek method:
    seek (self, start, end, value):
    Not used as it would mean computing a new parameter (abs(heading change)) for little benefit.
    '''    
        
    # Over the takeoff phase, average the stable altimeter readings to get the airport altitude
    '''
    First version:
    altitude_of_takeoff_airfield_a = np.ma.mean(np.ma.masked_where
                                                (np.ma.abs(rate_of_climb[kpt['TakeoffStartEstimate']:kpt['TakeoffEndEstimate']]) > 100,
                                                 altitude_std[kpt['TakeoffStartEstimate']:kpt['TakeoffEndEstimate']]))
    '''
    
    '''
    Second version:
    #Simpler (and better) computation:
    takeoff_level_begin = max(kpt['TakeoffStartEstimate']-30,0) # Trap for running out of data
    takeoff_level_end = kpt['TakeoffStartEstimate']
    takeoff_level_midpoint = (takeoff_level_begin + takeoff_level_end)/2.0
    altitude_of_takeoff_airfield = np.ma.mean(altitude_std[takeoff_level_begin:takeoff_level_end])
    
    kpv['AltitudeTakeoff'] = [( block.start+takeoff_level_midpoint, 
                              altitude_of_takeoff_airfield,
                              altitude_std.param_name)]
    
    altitude_aal_takeoff = DerivedParameter('Altitude_AAL_Takeoff', altitude_std)
    altitude_aal_takeoff.data -= altitude_of_takeoff_airfield
    '''
    '''
    Third version:
    #TODO:
    #Overwrite altitude_aal_takeoff.data with radio altitudes below one span rad alt.
    #Compute smoothed takeoff data and hence point of liftoff
    '''
    
    # Find where we pass through 35ft in climb.
    kpt['TakeoffEnd'] = altitude_radio.seek(block, kpt['TakeoffStartEstimate'], kpt['TakeoffEndEstimate']+30, TAKEOFF_END_HEIGHT)
    kpt['TakeoffGroundEffectEnds'] = altitude_radio.seek(block, kpt['TakeoffStartEstimate'], kpt['TakeoffEndEstimate']+30, WINGSPAN)
    #kpt['InitialClimbEnd'] = altitude_aal_takeoff.seek(block, kpt['TakeoffEnd'], kpt['TopOfClimb'], INITIAL_CLIMB_END_HEIGHT)
    
    # Create a Takeoff phase
    #ph['Takeoff'] = create_phase_inside(altitude_std, kpt['TakeoffTurnOntoRunway'], kpt['TakeoffEnd'])
    #ph['Initial_Climb'] = create_phase_inside(altitude_std, kpt['TakeoffEnd'], kpt['InitialClimbEnd'])
    #ph['Climb'] = create_phase_inside(altitude_std, kpt['InitialClimbEnd'], kpt['TopOfClimb'])

    #===========================================================================
    # LANDING 
    #===========================================================================

    # Find where we descend through 50ft.
    kpt['LandingStart'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['LandingEndEstimate']-30, LANDING_START_HEIGHT)
    
    # Record the heading on the runway
    head_landing = head_mag[kpt['LandingEndEstimate']]
    
    
    # Track on to find where the aircraft turned off the runway
    turn_off_runway = kpt['LandingEndEstimate']
    while abs(head_mag[turn_off_runway] - head_landing) < 15.0:
        turn_off_runway += 1
        if turn_off_runway == block.stop - block.start:
            logging.info ('\Data did not contain turn off of runway')
            break
        
    if block.start + turn_off_runway < block.stop: # Only record the runway turnoff if it was recorded.
        kpt['LandingTurnOffRunway'] = turn_off_runway # A better indication of the end of the landing process.
        
    '''
    # Compute the landing runway altitude:
    landing_level_begin = kpt['LandingEndEstimate'] # Retain the estimate, as this is passing through 80kts
    landing_level_end = min(kpt['LandingEndEstimate']+30, len(altitude_std.data)) # Trap for running out of data
    landing_level_midpoint = (landing_level_begin + landing_level_end)/2.0
    altitude_of_landing_airfield = np.ma.mean(altitude_std[landing_level_begin:landing_level_end])
    
    kpv['AltitudeLanding'] = [( block.start+landing_level_midpoint, 
                              altitude_of_landing_airfield,
                              altitude_std.param_name)]
    
    altitude_aal_landing = DerivedParameter('Altitude_AAL_Landing', altitude_std)
    altitude_aal_landing.data -= altitude_of_landing_airfield
    
    #TODO:
    #Overwrite altitude_aal_takeoff.data with radio altitudes below one span rad alt.
    #Compute smoothed takeoff data and hence point of liftoff
    '''

    
    # Calculate the approach phase transition points
    # Computed backwards from landing to make sure we get the latter moments in case of unstable approaches.
    #kpt['ApproachStart'] = altitude_aal_landing.seek(block, kpt['LandingStart'], kpt['TopOfDescent'], APPROACH_START_HEIGHT)
    #kpt['FinalApproachStart'] = altitude_aal_landing.seek(block, kpt['LandingStart'], kpt['ApproachStart'], FINAL_APPROACH_START_HEIGHT)
    #kpt['LandingGroundEffectBegins'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['FinalApproachStart'], WINGSPAN)
    kpt['LandingGroundEffectBegins'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['LandingEndEstimate']-60, WINGSPAN)
    
    #ph['Descent'] = create_phase_inside(altitude_std, kpt['TopOfDescent'], kpt['ApproachStart'])
    #ph['Approach'] = create_phase_inside(altitude_std, kpt['ApproachStart'], kpt['FinalApproachStart'])
    #ph['FinalApproach'] = create_phase_inside(altitude_std, kpt['FinalApproachStart'], kpt['LandingStart'])

    # Create the Landing phase, and Ground (outside takeoff and landing scopes)
    ph['Landing'] = create_phase_inside(altitude_std, kpt['LandingStart'], kpt['LandingTurnOffRunway'])
    ph['Ground'] = create_phase_outside(altitude_std, kpt['TakeoffTurnOntoRunway'], kpt['LandingTurnOffRunway'])
        

