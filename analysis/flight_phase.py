import logging
import numpy as np

from analysis.library import repair_mask, time_at_value
from analysis.node import A, Attribute, FlightPhaseNode, KeyTimeInstance, P, S, KTI
from analysis.settings import (AIRSPEED_THRESHOLD,
                               ALTITUDE_FOR_CLB_CRU_DSC,
                               HEADING_TURN_OFF_RUNWAY,
                               HEADING_TURN_ONTO_RUNWAY,
                               HYSTERESIS_FP_RAD_ALT,
                               INITIAL_CLIMB_THRESHOLD,
                               LANDING_THRESHOLD_HEIGHT,
                               RATE_OF_CLIMB_FOR_CLIMB_PHASE,
                               RATE_OF_CLIMB_FOR_DESCENT_PHASE,
                               RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                               RATE_OF_TURN_FOR_FLIGHT_PHASES,
                               )


def shift_slices(slicelist,offset):
    """
    This function shifts a list of slices by offset. The need for this arises
    when a phase condition has been used to limit the scope of another phase
    calculation.
    """
    newlist = []
    for each_slice in slicelist:
        a = each_slice.start + offset
        b = each_slice.stop + offset
        newlist.append(slice(a,b))
    return newlist
    
    
class Airborne(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb'), airs=P('Fast')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        for air in airs:
            not_level_flight = np.ma.masked_inside(roc.array[air.slice], 
                                               -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                               RATE_OF_CLIMB_FOR_LEVEL_FLIGHT)
            in_air = np.ma.flatnotmasked_edges(not_level_flight)
            in_air_slice = [slice(in_air[0],in_air[1])]
            self.create_phases(shift_slices(in_air_slice, air.slice.start))
            
            #try:
                #a,b = np.ma.flatnotmasked_edges(shift_slices(level_flight, air.slice.start))
                #self.create_phases([slice(a,b,None)])
            #except:
                #pass # Just don't create a phase if none exists.
        

class ApproachAndLanding(FlightPhaseNode):
    """
    This phase is used to identify an approach which may or may not include
    in a landing. It includes go-arounds, touch-and-go's and of course
    successful landings.
    
    Each Approach And Landing is associated with an airfield and a runway
    where possible. These are identified using:
    
    1. the heading on the runway (only if the aircraft lands)
    
    2. the ILS frequency (only if the ILS is tuned and localizer data is valid)
    
    3. the aircraft position at the lowest point of approach
    """

    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        if 'Altitude AAL For Flight Phases' in available:
            return True
        else:
            return False
        
    # List the optimal parameter set here
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases')):
        if len(alt_rad.array)>0: #  Test "if alt_rad" doesn't work - it's not None. TODO ??
            # Start the phase if we pass over high ground, so the radio
            # altitude falls below 3000ft before the pressure altitude
            app = np.ma.masked_where(np.ma.minimum(alt_AAL.array,alt_rad.array)
                                 >3000,alt_AAL.array)
        else:
            # Just use airfield elevation clearance
            app = np.ma.masked_where(alt_AAL.array>3000,alt_AAL.array)
        phases = np.ma.clump_unmasked(app)
        for phase in phases:
            begin = phase.start
            pit = np.ma.argmin(app[phase]) + begin
            if app[pit] < app[begin] :
                self.create_phase(phase)


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt=P('Altitude For Climb Cruise Descent')):
        ccd = np.ma.masked_less(alt.array, ALTITUDE_FOR_CLB_CRU_DSC)
        self.create_phases(np.ma.clump_unmasked(ccd))


class ClimbFromBottomOfDescent(FlightPhaseNode):
    def derive(self, 
               toc=P('Top Of Climb'),
               eot=P('Climb Start'), # AKA End Of Takeoff
               bod=P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)
            
        # Now see which follows a takeoff
        for this_eot in eot:
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
        for this_bod in bod:
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
    def derive(self, roc=P('Rate Of Climb'), airs=S('Fast')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(roc.array[air.slice],
                                         RATE_OF_CLIMB_FOR_CLIMB_PHASE)
            climbing_slices = np.ma.clump_unmasked(climbing)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))

      
class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=P('Climb Cruise Descent'),
               tocs=P('Top Of Climb'),
               tods=P('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent at this time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.
        for ccd in ccds:
            # Scan the TOCs.
            for toc in tocs:
                if ccd.slice.start <= toc.index <= ccd.slice.stop:
                    break
            else:
                toc = None
            # Scan the TODs.
            for tod in tods:
                if ccd.slice.start <= tod.index <= ccd.slice.stop:
                    break
            else:
                tod = None
            # Build the slice from what we have found.
            if toc and tod:
                self.create_phase(slice(toc.index, tod.index))
            elif toc:
                self.create_phase(slice(toc.index, ccd.slice.stop))
            elif tod:
                self.create_phase(slice(ccd.slice.start, tod.index))
            else:
                pass


class Descending(FlightPhaseNode):
    """ Descending faster than 800fpm towards the ground
    """
    def derive(self, roc=P('Rate Of Climb'), airs=P('Fast')):
        # Rate of climb and descent limits of 800fpm gives good distinction
        # with level flight.
        for air in airs:
            descending = np.ma.masked_greater(roc.array[air.slice],
                                              RATE_OF_CLIMB_FOR_DESCENT_PHASE)
            desc_slices = np.ma.clump_unmasked(descending)
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self, descending=Descending, roc=P('Rate Of Climb')):
        return NotImplemented


class DescentToBottomOfDescent(FlightPhaseNode):
    def derive(self, 
               tod=P('Top Of Descent'), 
               bod=P('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod:
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
    def derive(self, alt=P('Altitude For Flight Phases')):
        dlc = np.ma.masked_greater(alt.array, ALTITUDE_FOR_CLB_CRU_DSC)
        dlc_list = np.ma.clump_unmasked(dlc)
        for this_dlc in dlc_list:
            if this_dlc.start == 0:
                dlc_list.remove(this_dlc)
            elif this_dlc.stop == len(alt.array):
                dlc_list.remove(this_dlc)
        self.create_phases(dlc_list)


class Fast(FlightPhaseNode):
    def derive(self, airspeed=P('Airspeed')):
        # Did the aircraft go fast enough to possibly become airborne?
        fast_where = np.ma.masked_less(repair_mask(airspeed.array),
                                       AIRSPEED_THRESHOLD)
        fast_slices = np.ma.clump_unmasked(fast_where)
        self.create_phases(fast_slices)
 

class FinalApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio For Flight Phases'),
               app_lands=S('Approach And Landing')):
        for app_land in app_lands:
            
            # Allow for the hysteresis applied to the radio altimeter signal 
            # for phase computations
            thold = LANDING_THRESHOLD_HEIGHT+HYSTERESIS_FP_RAD_ALT
            app = np.ma.masked_where(np.ma.logical_or(
                alt_AAL.array[app_land.slice]>1000,
                alt_rad.array[app_land.slice]<thold), 
                                     alt_AAL.array[app_land.slice])
            phases = np.ma.clump_unmasked(app)
            for phase in phases:
                begin = app_land.slice.start + phase.start
                pit = np.ma.argmin(app[phase]) + begin
                if app[pit] < app[begin] :
                    self.create_phase(slice(begin, pit))
    
    
class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach And Landing')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                   app_land.slice.start))


class LevelFlight(FlightPhaseNode):
    def derive(self, roc=P('Rate Of Climb'),airs=S('Airborne')):
        # Rate of climb limit set to identify both level flight and 
        # end of takeoff / start of landing.
        for air in airs:
            level_flight = np.ma.masked_outside(roc.array[air.slice], 
                                                -RATE_OF_CLIMB_FOR_LEVEL_FLIGHT,
                                                RATE_OF_CLIMB_FOR_LEVEL_FLIGHT)
            level_slices = np.ma.clump_unmasked(level_flight)
            self.create_phases(shift_slices(level_slices, air.slice.start))


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
    pass # added to remove syntax error, CJ

    
#===============================================================================
#         TAKEOFF 
#===============================================================================
class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.
    """
    def derive(self, fast=S('Fast'),
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Phases')
               ):
        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.
            # We know a takeoff comes at the start of the phase.
            
            # The aircraft is part way down it's takeoff run at the start of 
            # the section.
            takeoff_run = speedy.slice.start
            
            # Note: This algorithm works across the entire data array, and
            # not just inside the speedy slice, so the final indeces are
            # absolute and not relative references.

            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[takeoff_run]
            
            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = takeoff_run - 300*head.frequency
            takeoff_begin = time_at_value(np.ma.abs(head.array-datum),
                                          head.frequency, head.offset,
                                          first, takeoff_run,
                                          HEADING_TURN_ONTO_RUNWAY)

            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.
            
            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            last = takeoff_run + 300*head.frequency
            takeoff_end = time_at_value(alt_aal.array,
                                        head.frequency, head.offset,
                                        takeoff_run, last,
                                        INITIAL_CLIMB_THRESHOLD)
 
            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            self.create_phases([slice(takeoff_begin, takeoff_end)])
            
#===============================================================================
#         LANDING 
#===============================================================================
class Landing(FlightPhaseNode):
    """
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.
    """
    def derive(self, fast=S('Fast'),
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Phases')
               ):
        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            landing_run = speedy.slice.stop
            datum = head.array[landing_run]
            
            first = landing_run - 300*head.frequency
            landing_begin = time_at_value(alt_aal.array,
                                        head.frequency, head.offset,
                                        first, landing_run,
                                        LANDING_THRESHOLD_HEIGHT)
 
            last = landing_run + 300*head.frequency
            landing_end = time_at_value(np.ma.abs(head.array-datum),
                                          head.frequency, head.offset,
                                          landing_run, last,
                                          HEADING_TURN_OFF_RUNWAY)

            self.create_phases([slice(landing_begin, landing_end)])
#===============================================================================
            
            
            
            """ Commented out to remove syntax error! CJ
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

            # Now find when we go through 35 ft in climbout.
            for any_toff in end_toff:
                if end_toff.index this is horrible, trying to find the takeoff we are looking for.
                
                Alternatively I compute exactly the same thing as in the KTI again.
                
                Either way this is ugly.
                
                            
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            
        
        
        

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
        



Reminder about how to load test data.....


from hdfaccess.file import hdf_file
hdf = hdf_file('C:\POLARIS Development\Data files\HDF5 example/4_3377853_146-301.hdf5')
hdf.search('airspeed')
[u'Airspeed', u'INDICATED AIRSPEED FAULT']

# Get a chunk of data - in this case the whole airspeed array.
airspeed = hdf['Airspeed']

import numpy as np
# Save to the *.npy file this chunk of data, then close the hdf file.
np.save('AnalysisEngine/tests/test_data/4_3377853_146-301_airspeed.npy', airspeed.array.data)
hdf.close()


# For the test routing, load the npy array...
ias = np.load('AnalysisEngine/tests/test_data/4_3377853_146-301_airspeed.npy')
ias[100:110]
array([ 55.26800949,  55.26800949,  55.42561622,  55.11040358,
        55.26800949,  55.26800949,  55.26800949,  55.26800949,
        55.42561622,  55.42561622])


"""