import numpy as np

from analysis_engine.library import (
    average_value,
    index_at_value,
    slices_after,
    slices_before,
    slices_below,
    slices_and,
    slice_duration,
    slices_duration,
    slice_midpoint,
    shift_slices,
)
from analysis_engine.node import (FlightPhaseNode, KeyPointValueNode, KPV, KTI,
                                  P, S)
from analysis_engine.settings import NAME_VALUES_FLAP

FMIS_CRUISE_NAME_VALUES = {'cruise_number': [1, 2, 3]}


#############################################################################
# Superclasses


class FMISCruiseKPV(object):
    '''
    Superclass for KPVs using FMIS Cruise.
    '''
    def _create_cruise_kpvs(self, cruises, array, function):
        for cruise_number, cruise in enumerate(cruises, start=1):
            self.create_kpv(*function(array, _slice=cruise.slice),
                            cruise_number=cruise_number)


#############################################################################
# Flight Phases


class FMISCruise(FlightPhaseNode):
    '''
    Q: Does this meet the original specification?
    '''
    name = 'FMIS Cruise'
    
    @staticmethod
    def _mid_slice(_slice):
        midpoint = slice_midpoint(_slice)
        return slice(midpoint - 15, midpoint + 15)
    
    @staticmethod
    def _start_slice(_slice):
        start = _slice.start + 300
        return slice(start, start + 30)
    
    @staticmethod
    def _end_slice(_slice):
        stop = _slice.stop - 300
        return slice(stop - 30, stop)
    
    def derive(self, cruises=S('Cruise')):
        '''
        '''
        cruise = cruises.get_longest()
        
        if not cruise:
            self.warning('Could not create FMIS Cruise sections because no '
                         'Cruise sections were available.')
            return
        
        duration = slice_duration(cruise.slice, cruises.hz)
        
        if duration < 300:
            return
        
        if duration < 600:
            # Only one data capture, halfway into the cruise.
            self.create_phase(self._mid_slice(cruise.slice),
                              name='FMIS Cruise 1')
            return
        
        if duration < 900:
            self.create_phase(self._start_slice(cruise.slice),
                              name='FMIS Cruise 1')
            self.create_phase(self._end_slice(cruise.slice),
                              name='FMIS Cruise 2')
            return
        
        self.create_phase(self._start_slice(cruise.slice),
                          name='FMIS Cruise 1')
        self.create_phase(self._mid_slice(cruise.slice),
                          name='FMIS Cruise 2')
        self.create_phase(self._end_slice(cruise.slice),
                          name='FMIS Cruise 3')


##############################################################################
# FMIS Cruise KPVs


class FMISMachCruise(KeyPointValueNode, FMISCruiseKPV):
    NAME_FORMAT = 'FMIS MachCruise %(cruise_number)d'
    NAME_VALUES = FMIS_CRUISE_NAME_VALUES
    
    def derive(self, cruises=S('FMIS Cruise'), mach=P('Mach')):
        self._create_cruise_kpvs(cruises, mach.array, average_value)


class FMISWeightCruise(KeyPointValueNode, FMISCruiseKPV):
    NAME_FORMAT = 'FMIS WeightCruise %(cruise_number)d'
    NAME_VALUES = FMIS_CRUISE_NAME_VALUES
    
    def derive(self, cruises=S('FMIS Cruise'),
               weight=P('Gross Weight Smoothed')):
        self._create_cruise_kpvs(cruises, weight.array, average_value)


class FMISHWindCompCruise(KeyPointValueNode, FMISCruiseKPV):
    NAME_FORMAT = 'FMIS HWindCompCruise %(cruise_number)d'
    NAME_VALUES = FMIS_CRUISE_NAME_VALUES
    
    def derive(self, cruises=S('FMIS Cruise'), mach=P('Headwind')):
        self._create_cruise_kpvs(cruises, mach.array, average_value)


class FMISPaltCruise(KeyPointValueNode, FMISCruiseKPV):
    NAME_FORMAT = 'FMIS MachCruise %(cruise_number)d'
    NAME_VALUES = FMIS_CRUISE_NAME_VALUES
    
    def derive(self, cruises=S('FMIS Cruise'), alt_std=P('Altitude STD')):
        self._create_cruise_kpvs(cruises, alt_std.array, average_value)


#class FMISFuelConsumedFF(KeyPointValueNode):
    #name = 'FMIS FuelConsumedFF'
    
    #def derive(self, fuel_qty=P('Fuel Qty')):
        #unmasked_indices = np.ma.where(fuel_qty.array)[0]
        #value = unmasked_indices[-1] - unmasked_indices[0]
        #index = (len(fuel_qty.array) / 2) * self.frequency
        #self.create_kpv(index, value)


class FMISMinutesInCruise(KeyPointValueNode):
    NAME_FORMAT = 'FMIS MinutesInCruise %(cruise_number)d'
    NAME_VALUES = FMIS_CRUISE_NAME_VALUES
    
    def derive(self, cruises=S('Cruise'), fmis_cruises=S('FMIS Cruise')):
        
        if not fmis_cruises:
            # Cruise was under 5 minutes.
            return
        
        cruise = cruises.get_longest()
        cruise_duration = slice_duration(cruise.slice, cruises.hz) / 60.0
        cruise_half_duration = cruise_duration / 2.0
        
        if not cruise:
            self.warning('Could not create FMIS MinutesCruise KPVs because no '
                         'Cruise sections were created.')
            return
        
        if not fmis_cruises:
            self.warning('Could not create FMIS MinutesCruise KPVs because no '
                         'FMIS Cruise sections were created.')
            return
        
        if len(fmis_cruises) == 3:
            fmis_cruise_1, fmis_cruise_2, fmis_cruise_3 = fmis_cruises
            self.create_kpv(slice_midpoint(fmis_cruise_1.slice), 5,
                            cruise_number=1)
            self.create_kpv(slice_midpoint(fmis_cruise_2.slice),
                            cruise_half_duration,
                            cruise_number=2)
            self.create_kpv(slice_midpoint(fmis_cruise_3.slice),
                            cruise_duration - 5,
                            cruise_number=3)
        elif len(fmis_cruises) == 2:
            fmis_cruise_1, fmis_cruise_2 = fmis_cruises
            self.create_kpv(slice_midpoint(fmis_cruise_1.slice), 5,
                            cruise_number=1)
            self.create_kpv(slice_midpoint(fmis_cruise_2.slice),
                            cruise_duration - 5,
                            cruise_number=2)
        elif len(fmis_cruises) == 1:
            fmis_cruise_1 = fmis_cruises[0]
            self.create_kpv(slice_midpoint(fmis_cruise_1.slice),
                            cruise_half_duration,
                            cruise_number=1)


class FMISTimeLastEngStart2Takeoff(KeyPointValueNode):
    '''
    Time from last engine start to application of takeoff thrust
    '''
    name = 'FMIS TimeLastEngStart2Takeoff'
    
    def derive(self, eng_start=KTI('Eng Start'),
               accel_start=KTI('Acceleration Start')):
        accel_start_index = accel_start.get_first().index
        eng_start_index = eng_start.get_last().index
        self.create_kpv(
            slice_midpoint(slice(eng_start_index, accel_start_index)),
            accel_start_index - eng_start_index)


class FMISTimeTouchDown2FirstEngShutDown(KeyPointValueNode):
    name = 'FMIS TimeTouchDown2FirstEngShutDown'
    
    def derive(self, touchdowns=KTI('Touchdown'),
               eng_stop=KTI('Eng Stop')):
        touchdown_index = touchdowns.get_last().index
        eng_stop_index = eng_stop.get_first().index
        self.create_kpv(
            slice_midpoint(slice(touchdown_index, eng_stop_index)),
            eng_stop_index - touchdown_index)


class FMISTimeFirst2LastEngineStart(KeyPointValueNode):
    name = 'FMIS TimeFirst2LastEngineStart'
    
    def derive(self, eng_start=KTI('Eng Start')):
        first_eng_start_index = eng_start.get_first().index
        last_eng_start_index = eng_start.get_last().index
        self.create_kpv(
            slice_midpoint(slice(first_eng_start_index, last_eng_start_index)),
            last_eng_start_index - first_eng_start_index)


class FMISTimeFirst2LastEngineStop(KeyPointValueNode):
    name = 'FMIS TimeFirst2LastEngineStop'
    
    def derive(self, eng_stop=KTI('Eng Stop')):
        first_eng_stop_index = eng_stop.get_first().index
        last_eng_stop_index = eng_stop.get_last().index
        self.create_kpv(
            slice_midpoint(slice(first_eng_stop_index, last_eng_stop_index)),
            last_eng_stop_index - first_eng_stop_index)


class FMISTimeFlap(KeyPointValueNode):
    '''
    These KPVs should only be created for Boeing aircraft.
    '''
    NAME_FORMAT = 'FMIS TimeFlap%(flap)d'
    NAME_VALUES = NAME_VALUES_FLAP.copy()
    
    def derive(self, flap=P('Flap'), apps=S('Approach And Landing')):
        landing = apps.get_last()
        for flap_setting in [1, 2, 5, 10, 15, 25, 30, 40]:
            self.create_kpv(
                slice_midpoint(landing.slice),
                len(np.ma.where(flap.array[landing.slice] == flap_setting)[0]) * flap.hz,
                flap=flap_setting)


class FMISTimeFlap_Airbus(KeyPointValueNode):
    NAME_FORMAT = 'FMIS TimeFlap%(conf)sAirbus'
    NAME_VALUES = {'conf': ['1', '1F', '2', '3', '4']}
    
    def derive(self, conf=P('Configuration'), apps=S('Approach And Landing')):
        landing = apps.get_last()
        for conf_setting in self.NAME_VALUES['conf']:
            duration = len(np.ma.where(conf.array[landing.slice] == conf_setting.replace('F', '+F'))[0]) * conf.hz
            self.create_kpv(
                slice_midpoint(landing.slice), duration, conf=conf_setting)


class FMISTimeFL250ToTDown(KeyPointValueNode):
    name = 'FMIS TimeFL250ToTDown'
    
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               touchdowns=KTI('Touchdown')):
        alt_25000 = index_at_value(alt_std.array, 25000, slice(None, None, -1))
        
        if not alt_25000:
            return
        
        touchdown = touchdowns.get_last()
        
        alt_slice = slice(alt_25000, touchdown.index)
        
        self.create_kpv(slice_midpoint(alt_slice),
                        slice_duration(alt_slice, self.hz))


class FMISTimeFL100ToTDown(KeyPointValueNode):
    name = 'FMIS TimeFL100ToTDown'
    
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               touchdowns=KTI('Touchdown')):
        alt_10000 = index_at_value(alt_std.array, 10000, slice(None, None, -1))
        
        if not alt_10000:
            return
        
        touchdown = touchdowns.get_last()
        
        alt_slice = slice(alt_10000, touchdown.index)
        
        self.create_kpv(slice_midpoint(alt_slice),
                        slice_duration(alt_slice, self.hz))


class FMISApuTime(KeyPointValueNode):
    name = 'FMIS ApuTime'
    
    def derive(self, eng_stops=KTI('Eng Stop')):
        eng_stop = eng_stops.get_last()
        
        if not eng_stop:
            self.warning("'Eng Stop' KTI could not be found.")
            return
        
        self.create_kpv(eng_stop.index, eng_stop.index)


class FMISAPUFuelConsumed(KeyPointValueNode):
    name = 'FMIS APUFuelConsumed'
    
    def derive(self, fuel_burn=P('Eng (*) Fuel Burn'),
               eng_stops=KTI('Eng Stop')):
        eng_stop = eng_stops.get_last()
        
        if not eng_stop:
            self.warning("'Eng Stop' KTI could not be found.")
            return
        
        eng_slice = slice(0, eng_stop.index)
        self.create_kpv(eng_stop.index, np.ma.sum(fuel_burn.array[eng_slice]))


class FMISTimeSpdBrakeExt(KeyPointValueNode):
    name = 'FMIS TimeSpdBrakeExt'
    
    def derive(self, speedbrake=P('Speedbrake'),
               alt_std=P('Altitude STD Smoothed'),
               touchdowns=KTI('Touchdown')):
        alt_25000 = index_at_value(alt_std.array, 25000, slice(None, None, -1))
        
        if not alt_25000:
            return

        touchdown = touchdowns.get_last()
        duration = \
            len(np.ma.nonzero(speedbrake.array[alt_25000:touchdown.index])[0]) / self.frequency
        self.create_kpv(slice_midpoint(slice(alt_25000, touchdown.index)),
                        duration)


class FMISGearDownAALApp(KeyPointValueNode):
    name = 'FMIS GearDownAALapp'
    
    def derive(self, alt_gear_downs=KPV('Altitude At Gear Down Selection')):
        alt_gear_down = alt_gear_downs.get_last()
        
        if not alt_gear_down:
            self.warning("'Altitude At Gear Down Selection' KPV could not be found.")
            return
        
        value = alt_gear_down.value if alt_gear_down.value > 1000 else 1000
        self.create_kpv(alt_gear_down.index, value)


class FMISAALFirstSlatFlapSelApp(KeyPointValueNode):
    name = 'FMIS AALFirstSlatFlapSelApp'
    
    @classmethod
    def can_operate(cls, available):
        return ('Altitude AAL' in available and 'Descending' in available and 
                ('Flap Set' in available or 'Slat Set' in available))
    
    def derive(self, alt_aal=P('Altitude AAL'), desc=S('Descending'), flap_sets=KTI('Flap Set'),
               slat_sets=KTI('Slat Set')):
        indices = []
        for kti_node in (flap_sets, slat_sets):
            if not kti_node:
                continue
            kti = kti_node.get_first(within_slices=desc.get_slices())
            if kti:
                indices.append(kti.index)
        
        if not indices:
            return
        
        index = min(indices)
        
        self.create_kpv(index, alt_aal.array[index])


class FMISTimeFirst2LastEngShutDown(KeyPointValueNode):
    name = 'FMIS TimeFirst2LastEngShutDown'
    
    def derive(self, eng_stops=P('Eng Stop')):
        if not eng_stops:
            return
        
        if len(eng_stops) == 1:
            eng_stop = eng_stops[0]
            self.create_kpv(eng_stop.index, 0)
            return
        
        first_eng_stop = eng_stops.get_first()
        last_eng_stop = eng_stops.get_last()
        
        index = slice_midpoint(slice(first_eng_stop.index, last_eng_stop.index))
        value = last_eng_stop.index - first_eng_stop.index
        self.create_kpv(index, value)


class FMISLevelFlight(FlightPhaseNode):
    name = 'FMIS LevelFlight'
    
    def derive(self, vert_spd=P('Vertical Speed Inertial'), airs=S('Airborne')):
        for air in airs:
            level_flight_slices = slices_below(vert_spd.array[air.slice], 200)[1]
            level_flight_slices = [s for s in level_flight_slices
                                   if slice_duration(s, self.hz) > 10]
            self.create_phases(
                shift_slices(level_flight_slices, air.slice.start))


class FMISTimeLvlFltBelowFL250Climb(KeyPointValueNode):
    name = 'FMIS TimeLvlFltBelowFL250Climb'
    
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               level_flights=S('FMIS LevelFlight')):
        
        alt_25000 = index_at_value(alt_std.array, 25000)
        
        if not alt_25000:
            # Should we create a KPV for climb up to 25000 if that value isn't
            # reached?
            self.warning("Could not find 25000 Ft during a Climbing section.")
            return
        
        duration = slices_duration(slices_before(level_flights.get_slices(),
                                                 alt_25000), self.hz)
        
        self.create_kpv(alt_25000, duration)


class FMISTimeLvlFltFL250toFL70Desc(KeyPointValueNode):
    name = 'FMIS TimeLvlFltFL250toFL70Desc'
    
    def derive(self, alt_std=P('Altitude STD Smoothed'), descs=S('Descending'),
               level_flights=S('FMIS LevelFlight')):
        desc_level = slices_and(level_flights.get_slices(),
                                descs.get_slices())
        desc_level_25000_to_7000 = slices_and(desc_level,
                                              alt_std.slices_from_to(25000, 7000))
        
        if not desc_level_25000_to_7000:
            return
        
        index = desc_level_25000_to_7000[0].start
        self.create_kpv(index, slices_duration(desc_level_25000_to_7000,
                                               self.frequency))


class FMISTimeLvlFltBelowFL70Desc(KeyPointValueNode):
    name = 'FMIS TimeLvlFltBelowFL70Desc'
    
    def derive(self, alt_std=KTI('Altitude STD Smoothed'),
               level_flights=S('FMIS LevelFlight')):
        alt_7000 = index_at_value(alt_std.array, 7000, slice(None, None, -1))
        
        if not alt_7000:
            self.warning("7000 Ft Descending not found.")
            return
        
        self.create_kpv(alt_7000, slices_duration(slices_after(
            level_flights.get_slices(), alt_7000), self.hz))
        

class FMISFuelConsumedFirst2LastEngStart(KeyPointValueNode):
    name = 'FMIS FuelConsumedFirst2LastEngStart'
    
    def derive(self, fuel_burn=P('Eng (*) Fuel Burn'),
               eng_start=KTI('Eng Start')):
        first_eng_start = eng_start.get_first()
        last_eng_start = eng_start.get_last()
        
        if not first_eng_start or not last_eng_start:
            self.warning("'Eng Start' KTI could not be found.")
            return
        
        if first_eng_start == last_eng_start:
            self.create_kpv(first_eng_start.index, 0)
        
        eng_slice = slice(first_eng_start.index, last_eng_start.index)
        self.create_kpv(slice_midpoint(eng_slice),
                        np.ma.sum(fuel_burn.array[eng_slice]))


class FMISFuelConsumedTouchDown2FirstEngShutDown(KeyPointValueNode):
    name = 'FMIS FuelConsumedTouchDown2FirstEngShutDown'
    
    def derive(self, fuel_burn=P('Eng (*) Fuel Burn'),
               touchdowns=KTI('Touchdown'),
               eng_stops=KTI('Eng Stop')):
        touchdown = touchdowns.get_last()
        eng_stop = eng_stops.get_next(touchdown.index)
        
        if not eng_stop:
            self.warning("'Eng Stop' KTI could not be found.")
            return
        
        eng_slice = slice(touchdown.index, eng_stop.index)
        
        self.create_kpv(slice_midpoint(eng_slice),
                        np.ma.sum(fuel_burn.array[eng_slice]))


class FMISFuelConsumedFirst2LastEngShutDown(KeyPointValueNode):
    name = 'FMIS FuelConsumedFirst2LastEngShutDown'
    
    def derive(self, fuel_burn=P('Eng (*) Fuel Burn'),
               eng_stops=KTI('Eng Stop')):
        first_eng_stop = eng_stops.get_first()
        last_eng_stop = eng_stops.get_last()
        
        if not first_eng_stop or not first_eng_stop:
            self.warning("'Eng Stop' KTI could not be found.")
            return
        
        if first_eng_stop.index == last_eng_stop.index:
            self.create_kpv(first_eng_stop.index, 0)
            return
        
        eng_slice = slice(first_eng_stop.index, last_eng_stop.index)
        
        self.create_kpv(slice_midpoint(eng_slice),
                        np.ma.sum(fuel_burn.array[eng_slice]))


class FMISFuelConsumedTaxiOutAllEng(KeyPointValueNode):
    name = 'FMIS FuelConsumedTaxiOutAllEng'
    
    def derive(self, fuel_burn=P('Eng (*) Fuel Burn'),
               eng_starts=KTI('Eng Start'),
               takeoff_accels=KTI('Takeoff Acceleration Start')):
        eng_start = eng_starts.get_first()
        start_index = eng_start.index if eng_starts.get_first() else 0
        
        takeoff_accel = takeoff_accels.get_last()
        
        if not takeoff_accel:
            self.warning("'Takeoff Acceleration Start' KTI could not be found.")
            return
        
        eng_slice = slice(start_index, takeoff_accel.index)
        
        self.create_kpv(slice_midpoint(eng_slice),
                        np.ma.sum(fuel_burn.array[eng_slice]))

