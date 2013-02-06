"""
2012.01.12
----------
Created referencing previous code and checked over by FDS engine room.
"""

#############################################################################

"FLAP SELECTIONS"

# Notes:
# - B757 ACMS dataframe uses FLAP_LEVER - create a test case with this data
# - B777 Records using many discrete positions, use that! (create a multi-part
# parameter which scales FLAP_15 discrete by 15!

# Reference numbers are from FAA specification documentation.

series_flap_map = {
    # TODO: Review max for -200 (28deg) -210 (33deg)
    'ATR72-200': (0, 15, 30, 45),
    'ATR72-500': (0, 15, 28, 33),  # Q: Confirm values
    'DHC-8 Series 100': (0, 5, 15, 35),
    'DHC-8 Series 200': (0, 5, 15, 35),  # A13NM TODO Check -200 flap settings
    'DHC-8 Series 300': (0, 5, 10, 15, 35),  # A13NM
    'DHC-8 Series 400': (0, 5, 10, 15, 35),  # A13NM
    'ATR42-200': (0, 15, 30, 45),  # A53EU -200, -300
    'ATR42-300': (0, 15, 30, 45),  # A53EU -200, -300
    'ATR42-500': (0, 15, 25, 35),  # A53EU, -500
}

family_flap_map = {
    'A300': (0, 15, 20, 40),  # A35EU
    #'A300': (0, 8, 15, 25), # !!!!!! (AGK),A35EU
    'A310': (0, 15, 20, 40),  # A35EU
    'A318': (0, 10, 15, 20, 40),  # A28NM
    'A319': (0, 10, 15, 20, 40),  # A28NM
    'A320': (0, 10, 15, 20, 35),  # A28NM
    'A321': (0, 10, 14, 21, 25),  # A28NM
    'A330': (0, 8, 14, 22, 32),  # A46NM
    'BAE 146': (0, 18, 24, 30, 33),  # A49EU
    'B737 Classic': (0, 1, 2, 5, 10, 15, 25, 30, 40),  # A16WE
    'B737 NG': (0, 1, 2, 5, 10, 15, 25, 30, 40),  # A16WE
    'B747': (0, 1, 5, 10, 20, 25, 30),  # A20WE
    'B757': (0, 1, 5, 15, 20, 25, 30),  # A2NM
    'B767': (0, 1, 5, 15, 20, 25, 30),  # A1NM
    'B777': (0, 1, 5, 15, 20, 25, 30),  # T00001SE
    'CL604': (0, 20, 30, 45),  # A21EA
    'CL850': (0, 8, 20, 30, 45),  # A21EA
    # A21EA  Some variants of the -200 do not have the flap 8 setting
    'CRJ 100/200': (0, 8, 20, 30, 45),
    'CRJ 700': (0, 1, 8, 20, 30, 45),  # A21EA
    'CRJ 900': (0, 1, 8, 20, 30, 45),  # A21EA
    # Flap 18 not available on all aircarft - T00011AT
    'ERJ-135/145': (0, 9, 18, 22, 45),
    # Flap lever = 1, 2, 3, 4, 5, full - A57NM
    'ERJ-170/175': (0, 5, 10, 20, 35),
    # Flap lever = 1, 2, 3, 4, 5, full - A57NM
    'ERJ-190/195': (0, 7, 10, 20, 37),
    # 'ED48A200': (0, 10, 15, 20, 40),  # ???
    # A59NM, Flap selections, SF0 (clean), SF1 (9 degs), SF2 (flap20), SF3 (40
    # degs)
    'F7X': (0, 9, 20, 40),
    'G-IV': (0, 10, 20, 39),  # A12EA
    'G-V': (0, 10, 20, 39),  # A12EA
    'G550': (0, 10, 20, 39),  # A12EA
    'GLOBAL': (0, 1, 8, 20, 30, 45),  # T00003NY
    'L382': (0, 50, 100),  # 100% = 36 degs A1SO
    'MD-11': (0, 15, 22, 25, 28, 35, 50),  # A22WE
    # These flap settings apply to MD 81,82,83,87 and 88. There are
    'DC-9': (0, 11, 15, 28, 40),
    # variable flap positions between 0 and 11 and again between 15 and 24.
    'RJ85': (0, 18, 24, 30, 33),  # A49EU
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
    
    
def get_flap_detents():
    '''
    Get all flap combinations from all supported aircraft types
    '''
    all_detents = set()
    for detents in series_flap_map.itervalues():
        all_detents.update(detents)
    for detents in family_flap_map.itervalues():
        all_detents.update(detents)
    return sorted(all_detents)
        

#############################################################################

"SLAT SELECTIONS"

series_slat_map = {

}

family_slat_map = {
    'A330': (0, 16, 20, 23),
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
    'A330': (0, 5, 10),
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

"AIRBUS CONF SELECTIONS"

# Notes:
# - Series conf will be used over Family conf settings
# - If using flap and slat to determine conf, only create a tuple of length 2

series_conf_map = {
    # this will take precidence over series_conf_map
}

family_conf_map = {
    'A330': {
        #state: (slat, flap, aileron)
        0: (0, 0, 0),
        1: (16, 0, 0),
        2: (16, 8, 5),
        3: (20, 8, 10),
        4: (20, 14, 10),
        5: (23, 14, 10),
        6: (23, 22, 10),
        7: (23, 32, 10),
        },
    #'A300': {
        ##state: (slat, flap) - no aileron used for calculation
        #0: (0, 0),
        #1: (16, 0),
    #}
}


def get_conf_map(series=None, family=None):
    """
    Accessor for fetching conf mapping parameters.

    Return is a dictionary of state: tuple where tuple can contain either
    (slat, flap) or (slat, flap, aileron) depending on Aircraft requirements.

    :param series: Aircraft series e.g. B737-300
    :type series: String
    :param family: Aircraft family e.g. B737
    :type family: String
    :raises: KeyError if no mapping found
    :returns: conf mapping
    :rtype: dict
    """
    if series in series_conf_map:
        return series_conf_map[series]
    elif family in family_conf_map:
        return family_conf_map[family]
    else:
        raise KeyError("No conf mapping for Series '%s' Family '%s'" % (
            series, family))
