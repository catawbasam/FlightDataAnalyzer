# Selections are to use Model Series (new field on database).
# "767-200" will have a series "767"
# "A321" will be split into it's subparts and belong to a series "A321"

# Conf will be configured seperately within the code.


#############################################################################

"FLAP SELECTIONS"

# Notes:
# - B757 ACMS dataframe uses FLAP_LEVER - create a test case with this data
# - B777 Records using many discrete positions, use that! (create a multi-part parameter which scales FLAP_15 discrete by 15!
 
model_series_flap_map = {
    'A300'       : ( 0, 15, 20, 40),
    'A300'       : ( 0,  8, 15, 25), #!!!!!! (AGK),
    'A310'       : ( 0, 15, 20, 40),
    'A319'       : ( 0, 10, 15, 20, 40),
    'A320'       : ( 0, 10, 15, 20, 35),
    'A321'       : ( 0, 10, 14, 21, 25),
    'A330'       : ( 0,  8, 14, 22, 32),
    'ATR72'      : ( 0, 15, 30, 45),
    'ATR72'      : ( 0, 15, 28, 33), #!!!!!!!!!!!!!!!!!
    'ATR42'      : ( 0, 15, 25, 35),
    '146'        : ( 0, 18, 24, 30, 33), # BAE
    '737'        : ( 0,  1,  2,  5, 10, 15, 25, 30, 40),
    '747'        : ( 0,  1,  5, 10, 20, 25, 30),
    '757'        : ( 0,  1,  5, 15, 20, 25, 30),
    '767'        : ( 0,  1,  5, 15, 20, 25, 30),
    'CL604'      : ( 0,  8, 20, 30, 45),
    'CL850'      : ( 0,  8, 20, 30, 45),
    'CRJ200'     : ( 0,  8, 20, 30, 45),
    'CRJ700'     : ( 0,  1,  8, 20, 30, 45),
    'CRJ900'     : ( 0,  1,  8, 20, 30, 45),
    'DHC 8'      : ( 0,  5, 10, 15, 35),
    'E135-145'   : ( 0,  9, 18, 22, 45),
    'E190'       : ( 0,  7, 10, 20, 37),
    'ED48A200'   : ( 0, 10, 15, 20, 40),
    'F7X'        : ( 0,  9, 20, 40),
    'G550'       : ( 0, 10, 20, 39),
    'GLOBAL'     : ( 0,  1,  8, 20, 30, 45),
    'L382'       : ( 0, 15, 30, 45),
    'MD11'       : ( 0, 15, 22, 28, 35, 50),
    'MD82'       : ( 0, 11, 15, 28, 40),
    'RJ85'       : ( 0, 18, 24, 30, 33),
}


#############################################################################

"SLAT SELECTIONS"

# Notes:
# - 
 
model_series_slat_map = {
    'A330'       : ( 0, 16, 20, 23),
	
}



#############################################################################

"AILERON SELECTIONS"

# Notes:
# - 
 
model_series_aileron_map = {
    'A330'       : ( 0, 5, 10),
	
}



