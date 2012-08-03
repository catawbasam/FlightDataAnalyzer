
""" 
2012.01.12
----------
Created referencing previous code and checked over by FDS engine room.

"""

#############################################################################

"FLAP SELECTIONS"

# Notes:
# - B757 ACMS dataframe uses FLAP_LEVER - create a test case with this data
# - B777 Records using many discrete positions, use that! (create a multi-part parameter which scales FLAP_15 discrete by 15!

series_flap_map = {
    'ATR72-200'  : ( 0, 15, 30, 45), # TODO: Review max for -200 (28deg) -210 (33deg)
    'ATR72-500'  : ( 0, 15, 28, 33), # Q: Confirm values
}
 
family_flap_map = {
    'A300'       : ( 0, 15, 20, 40),
    #'A300'       : ( 0,  8, 15, 25), #!!!!!! (AGK),
    'A310'       : ( 0, 15, 20, 40),
    'A319'       : ( 0, 10, 15, 20, 40),
    'A320'       : ( 0, 10, 15, 20, 35),
    'A321'       : ( 0, 10, 14, 21, 25),
    'A330'       : ( 0,  8, 14, 22, 32),
    'ATR42'      : ( 0, 15, 25, 35),
    'BAE 146'    : ( 0, 18, 24, 30, 33), 
    'B737 Classic': ( 0,  1,  2,  5, 10, 15, 25, 30, 40),
    'B737 NG'     : ( 0,  1,  2,  5, 10, 15, 25, 30, 40),
    'B747'        : ( 0,  1,  5, 10, 20, 25, 30),
    'B757'        : ( 0,  1,  5, 15, 20, 25, 30),
    'B767'        : ( 0,  1,  5, 15, 20, 25, 30),
    'CL604'      : ( 0,  8, 20, 30, 45),
    'CL850'      : ( 0,  8, 20, 30, 45),
    'CRJ200'     : ( 0,  8, 20, 30, 45),
    'CRJ700'     : ( 0,  1,  8, 20, 30, 45),
    'CRJ900'     : ( 0,  1,  8, 20, 30, 45),
    'DHC 8'      : ( 0,  5, 10, 15, 35),
    'E135'       : ( 0,  9, 18, 22, 45),
    'E145'       : ( 0,  9, 18, 22, 45),
    'E170'       : ( 0,  5, 10, 20, 35),
    'E190'       : ( 0,  7, 10, 20, 37),
    'ED48A200'   : ( 0, 10, 15, 20, 40),
    'F7X'        : ( 0,  9, 20, 40),
    'GIV'        : ( 0, 10, 20, 39),
    'GV'         : ( 0, 10, 20, 39),
    'G550'       : ( 0, 10, 20, 39),
    'GLOBAL'     : ( 0,  1,  8, 20, 30, 45),
    'L382'       : ( 0, 15, 30, 45),
    'MD11'       : ( 0, 15, 22, 28, 35, 50),
    'MD80'       : ( 0, 11, 15, 28, 40), #TODO: Confirm MD80 or MD82?
    'RJ85'       : ( 0, 18, 24, 30, 33),
}

def get_flap_map(series=None, family=None):
    """
    Accessor for fetching flap mapping parameters.
    
    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: list of detent values
    :rtype: list
    """
    if series in series_flap_map:
        return series_flap_map[series]
    elif family in family_flap_map:
        return family_flap_map[family]
    else:
        raise KeyError("No flap mapping for Series '%s' Family '%s'" % (
            series, family))

#############################################################################

"SLAT SELECTIONS"

series_slat_map = {

}

family_slat_map = {
    'A330'       : ( 0, 16, 20, 23),
}

def get_slat_map(series=None, family=None):
    """
    Accessor for fetching slat mapping parameters.
    
    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: list of detent values
    :rtype: list
    """
    if series in series_slat_map:
        return series_slat_map[series]
    elif family in family_slat_map:
        return family_slat_map[family]
    else:
        raise KeyError("No slat mapping for Series '%s' Family '%s'" % (
            series, family))


#############################################################################

"AILERON SELECTIONS"

series_aileron_map = {
    
}

family_aileron_map = {
    'A330'       : ( 0, 5, 10),
}

def get_aileron_map(series=None, family=None):
    """
    Accessor for fetching aileron mapping parameters.
    
    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: list of detent values
    :rtype: list
    """
    if series in series_aileron_map:
        return series_aileron_map[series]
    elif family in family_aileron_map:
        return family_aileron_map[family]
    else:
        raise KeyError("No aileron mapping for Series '%s' Family '%s'" % (
            series, family))

#############################################################################

"AIRBUS CONFIG SELECTIONS"

# Notes:
# - Series config will be used over Family config settings
# - If using flap and slat to determine config, only create a tuple of length 2

series_config_map = {
    # this will take precidence over series_config_map
}

family_config_map = {
    'A330' : {
        #state : (slat, flap, aileron)
        0 : ( 0, 0, 0),
        1 : (16, 0, 0),
        2 : (16, 8, 5),
        3 : (20, 8,10),
        4 : (20,14,10),
        5 : (23,14,10),
        6 : (23,22,10),
        7 : (23,32,10),
        },
    #'A300' : {
        ##state : (slat, flap) - no aileron used for calculation
        #0 : ( 0, 0),
        #1 : (16, 0),
    #}
}



def get_config_map(series=None, family=None):
    """
    Accessor for fetching config mapping parameters.
    
    Return is a dictionary of state : tuple where tuple can contain either 
    (slat, flap) or (slat, flap, aileron) depending on Aircraft requirements.
    
    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: config mapping
    :rtype: dict
    """
    if series in series_config_map:
        return series_config_map[series]
    elif family in family_config_map:
        return family_config_map[family]
    else:
        raise KeyError("No config mapping for Series '%s' Family '%s'" % (
            series, family))
