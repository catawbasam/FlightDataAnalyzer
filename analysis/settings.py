# use local_settings.py to override settings for your local environment.

# Modules to import all derived Nodes from. Additional modules can be
# appended to this list in local_settings.py
NODE_MODULES = ['analysis.derived_parameters',
                'analysis.key_point_values', 
                'analysis.key_time_instances',
                'analysis.sections',
                'analysis.flight_phase']

##########################
## Parameter Analysis
##########################

# Minimum duration for a flight to analyse
MIN_FLIGHT_TO_ANALYSE = 300  # (sec)

# Less than 5 mins you can't do a circuit, so we'll presume this is a data snippet
##FLIGHT_WORTH_ANALYSING_SEC = 300

# Minimum duration of flight in seconds
##DURATION_THRESHOLD = 60  # (sec)

# Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
DESCENT_MIN_DURATION = 10  # (sec)

# Level flight minimum duration
LEVEL_FLIGHT_MIN_DURATION = 60  # (sec)

# An airspeed below which you just can't possibly be flying.
AIRSPEED_THRESHOLD = 80  # (kts)

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 10 # (kts)

# Threshold for flight phase rate of climb hysteresis.
# We're going to ignore changes smaller than this to avoid repeatedly changing
# phase if the aircraft is climbing/descending close to a threshold level.
HYSTERESIS_FPROC = 100 # (fpm)

# Rate of climb and descent limits of 800fpm gives good distinction with
# level flight.
RATE_OF_CLIMB_FOR_FLIGHT_PHASES = 800 # (fpm) rate per second

##########################
## Data Analysis Hooks
##########################

# Perform some analysis before analysing the file
PRE_FILE_ANALYSIS = None
"""
# create processing function
def fn(hdf):
    # do something
    return # no return args required
PRE_FILE_ANALYSIS = fn
"""

PRE_FLIGHT_ANALYSIS = None
"""
def fn(hdf, aircraft, params):
    return # no return args required
PRE_FLIGHT_ANALYSIS = fn
"""

# Function for post process analysis of parameters - see example below
POST_DERIVED_PARAM_PROCESS = None
POST_LFL_PARAM_PROCESS = None
"""
# create post processing function
def fn(hdf, param):
    "Do something to param. Return Param if changes are to be saved, else None"
    return f(param)
# set as post process
POST_DERIVED_PARAM_PROCESS = fn
POST_LFL_PARAM_PROCESS = fn
"""


# Import from local_settings if exists
try:
    from analysis.local_settings import *
except ImportError:
    pass

