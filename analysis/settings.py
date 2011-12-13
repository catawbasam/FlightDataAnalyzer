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


# An airspeed below which you just can't possibly be flying.
AIRSPEED_THRESHOLD = 80  # (kts)

# Altitude to break flights into separate climb/cruise/descent segments.
# This is applied to altitude with hysteresis, so break will happen when
# climbing above 15000 ft and below 10000 ft.
ALTITUDE_FOR_CLB_CRU_DSC = 12500

# Resolved vertical acceleration washout time constant.
# This long period function removes any standing offset to the resolved 
# acceleration signal and is essential in the vertical velocity complementary filter.
AZ_WASHOUT_TC = 30.0

# Less than 5 mins you can't do a circuit, so we'll presume this is a data snippet
##FLIGHT_WORTH_ANALYSING_SEC = 300

# Minimum duration of flight in seconds
##DURATION_THRESHOLD = 60  # (sec)

# Threshold for start of climb phase
CLIMB_THRESHOLD = 1000 # ft AAL

# Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
DESCENT_MIN_DURATION = 10  # (sec)

# Acceleration due to gravity
GRAVITY = 32.2 # (ft/sec^2)

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 10 # (kts)

# Threshold for flight phase altitude hysteresis.
HYSTERESIS_FPALT = 200 # (ft)

# Threshold for flight phase radio altitude hysteresis.
HYSTERESIS_FP_RAD_ALT = 5 # (ft)

# Threshold for flight phase altitude hysteresis specifically for deparating 
# Climb Cruise Descent phases.
HYSTERESIS_FPALT_CCD = 2500 # (ft)

# Threshold for flight phase rate of climb hysteresis.
# We're going to ignore changes smaller than this to avoid repeatedly changing
# phase if the aircraft is climbing/descending close to a threshold level.
HYSTERESIS_FPROC = 100 # (fpm)

# Threshold for start of initial climb phase
INITIAL_CLIMB_THRESHOLD = 35 # ft (Radio, where available)

# Threshold for start of landing phase
LANDING_THRESHOLD_HEIGHT = 50 # (Radio, where available)

# Level flight minimum duration
LEVEL_FLIGHT_MIN_DURATION = 60  # (sec)


# Rate of climb and descent limits of 800fpm gives good distinction with
# level flight. Separately defined to allow for future adjustment.
RATE_OF_CLIMB_FOR_CLIMB_PHASE = 800 # (fpm)
RATE_OF_CLIMB_FOR_DESCENT_PHASE = -800 # (fpm)

# Rate of climb and descent limits of 300 fpm to identify airborne after takeoff
# and end of descent, when relying solely upon pressure altitude data.
RATE_OF_CLIMB_FOR_LEVEL_FLIGHT = 300 # (fpm)

# Rate of turn limits of +/- 90 deg/minute work well in flight and on ground.
RATE_OF_TURN_FOR_FLIGHT_PHASES = 1.5 # deg per second

# Duration of masked data to repair by interpolation for flight phase analysis
REPAIR_DURATION = 10 # seconds 

# Rate of Climb complementary filter timeconstant
RATE_OF_CLIMB_LAG_TC = 10.0 # sec



"""  Top of Climb / Top of Descent Threshold.
This threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
This was often OK, but one test data sample had a 4000ft climb 20 mins
after level off. This led to reducing the threshold to 600 fpm in 3
minutes which has been found to give good qualitative segregation
between climb, cruise and descent phases."""
SLOPE_FOR_TOC_TOD = (600/float(180))

WING_SPAN = 14 # TODO: Replace with aircraft parameter

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

