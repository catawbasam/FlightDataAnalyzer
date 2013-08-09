import numpy as np

from analysis_engine.library import (
    average_value,
    slice_duration,
    slice_midpoint,
)
from analysis_engine.node import (FlightPhaseNode, KeyPointValueNode, KTI, P, S)
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


class FMISFuelConsumedFF(KeyPointValueNode):
    name = 'FMIS FuelConsumedFF'
    
    def derive(self, fuel_qty=P('Fuel Qty')):
        unmasked_indices = np.ma.where(fuel_qty.array)[0]
        value = unmasked_indices[-1] - unmasked_indices[0]
        index = (len(fuel_qty.array) / 2) * self.frequency
        self.create_kpv(index, value)


class FMISMinutesCruise(KeyPointValueNode):
    NAME_FORMAT = 'FMIS MinutesCruise %(cruise_number)d'
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
    NAME_FORMAT = 'FMIS TimeFlap%(flap)d'
    NAME_VALUES = NAME_VALUES_FLAP.copy()
    
    def derive(self, flap=P('Flap'), apps=S('Approach And Landing')):
        landing = apps.get_last()
        for flap_setting in [1, 2, 5, 10, 15, 25, 30, 40]:
            self.create_kpv(
                slice_midpoint(landing.slice),
                len(np.ma.where(flap.array[landing.slice] == flap_setting)[0]) * flap.hz,
                flap=flap_setting)