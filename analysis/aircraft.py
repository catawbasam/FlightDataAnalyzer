##from utilities.dict_helpers import filter_empty_values
filter_empty_values = lambda d: dict( [(k,v) for k,v in d.items() if isinstance(v,int) or len(v)>0])

def populate_aircraft_params(aircraft):
    """
    Populate a dictionary of Aircraft specific parameters.
    
    :param aircraft: Aircraft details
    :type aircraft: Struct
    :returns: Available Aircraft parameters
    :rtype: Dict
    """
    # Struct does not raise when value is not there, but returns an empty struct
    aircraft_params = {
        'Tail Number': aircraft.tail_number, # Aircraft Registration
        'Identifier': aircraft.identifier, # Aircraft Ident
        'Manufacturer': aircraft.manufacturer,
        'Manufacturer Serial Number': aircraft.manufacturer_serial_number, #MSN
        'Model': aircraft.model,
        'Frame': aircraft.frame.name,
        'Main Gear To Altitude Radio': aircraft.model.geometry.main_gear_to_alt_rad,
        'Wing Span': aircraft.model.geometry.wing_span,
    }
    
    
    # filter out empty structs
    return filter_empty_values(aircraft_params)