# use local_settings.py to override settings for your local environment.

# Modules to import all derived Nodes from. Additional modules can be
# appended to this list in local_settings.py
NODE_MODULES = ['analysis.derived_parameters',
                'analysis.key_point_values', 
                'analysis.key_time_instances',
                'analysis.sections',
                'analysis.flight_phase',
                'analysis.flight_attribute']

# Handler
HANDLER = 'analysis.api_handler_http.APIHandlerHTTP'
BASE_URL = 'http://127.0.0.1'

##########################
## Parameter Analysis
##########################

# An airspeed below which you just can't possibly be flying.
AIRSPEED_THRESHOLD = 80  # kts

# Altitude to break flights into separate climb/cruise/descent segments.
# This is applied to altitude with hysteresis, so break will happen when
# climbing above 15000 ft and below 10000 ft.
ALTITUDE_FOR_CLB_CRU_DSC = 12500

# Resolved vertical acceleration washout time constant. This long period
# function removes any standing offset to the resolved acceleration signal
# and is essential in the vertical velocity complementary filter.
AZ_WASHOUT_TC = 60.0

#Less than 5 mins you can't do a circuit, so we'll presume this is a data
#snippet 
FLIGHT_WORTH_ANALYSING_SEC = 300

# Minimum duration of flight in seconds
##DURATION_THRESHOLD = 60  # sec

# Threshold for start of climb phase
CLIMB_THRESHOLD = 1000 # ft AAL

# Minimum period of a climb or descent for testing against thresholds
# (reduces number of KPVs computed in turbulence)
CLIMB_OR_DESCENT_MIN_DURATION = 10  # sec

# Tolerance of controls (Pitch/Roll (Captain/FO)) when in use in degrees.
# Used when trying determine which pilot is actively using the controls.
CONTROLS_IN_USE_TOLERANCE = 1

# Acceleration due to gravity
GRAVITY = 32.2 # ft/sec^2 - used for combining acceleration and height terms

# Acceleration due to gravity
GRAVITY_METRIC = 9.81 # m/sec^2 - used for comibining acceleration and groundspeed terms

# Threshold for turn onto runway at start of takeoff.
# This will usually be overwritten by the peak curvature test.
HEADING_TURN_ONTO_RUNWAY = 15.0 # deg

# Threshold for turn off runway at end of takeoff.
# This will usually be overwritten by the peak curvature test.
HEADING_TURN_OFF_RUNWAY = 15.0 # deg

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 10 # kts

# Threshold for flight phase altitude hysteresis.
HYSTERESIS_FPALT = 200 # ft

# Threshold for flight phase radio altitude hysteresis.
HYSTERESIS_FP_RAD_ALT = 5 # ft

# Threshold for flight phase altitude hysteresis specifically for separating 
# Climb Cruise Descent phases.
HYSTERESIS_FPALT_CCD = 2500 # ft

# Threshold for flight phase rate of climb hysteresis.
# We're going to ignore changes smaller than this to avoid repeatedly changing
# phase if the aircraft is climbing/descending close to a threshold level.
HYSTERESIS_FPROC = 400 # fpm
# The 400 fpm value has been selected from inspection of Hercules test data
# which is notoriously noisy. This may need to be revised to suit a wider
# range of aircraft.

# Threshold for rate of turn hysteresis.
HYSTERESIS_FPROT = 2 # deg/sec

# Full scale reading on the ILS
ILS_MAX_SCALE = 2.5 # dots

# Threshold for start of initial climb phase
INITIAL_CLIMB_THRESHOLD = 35 # ft (Radio, where available)

# Conversion from knots to ft/sec (used in airspeed rate of change)
KTS_TO_FPS = 1.68781 #  ft/sec

# Conversion from knots to m/sec (used in groundspeed computation)
KTS_TO_MPS = 0.514444 #  m/sec

# Threshold for start of braking / reverse thrust on landing.
LANDING_ACCELERATION_THRESHOLD = -0.1 # g
# TODO: Was -0.2g set to -0.1 for Herc testing - revert or not???

# Threshold for start of landing phase
LANDING_THRESHOLD_HEIGHT = 50 # (Radio, where available)

# Level flight minimum duration
LEVEL_FLIGHT_MIN_DURATION = 60  # sec

# Conversion from degrees of latitude to metres. I know it's approximate, but
# good enough for the uses we have here. To convert deg longitude, allow for
# the cos(latitude) reduction in distance as we get away from the equator.
METRES_PER_DEG_LATITUDE = 111120 # metres/deg

# Rate of climb and descent limits of 800fpm gives good distinction with
# level flight. Separately defined to allow for future adjustment.
RATE_OF_CLIMB_FOR_CLIMB_PHASE = 800 # fpm
RATE_OF_CLIMB_FOR_DESCENT_PHASE = -800 # fpm

# Rate of climb and descent limits of 300 fpm to identify airborne after takeoff
# and end of descent, when relying solely upon pressure altitude data.
RATE_OF_CLIMB_FOR_LEVEL_FLIGHT = 300 # fpm

# Rate of climb for liftoff. This builds upon the intertially smoothed rate of
# climb computation to identify accurately the point of liftoff.
RATE_OF_CLIMB_FOR_LIFTOFF = 5 # fpm

# Rate of climb for touchdown.
RATE_OF_CLIMB_FOR_TOUCHDOWN = -10 # fpm

# Rate of turn limits of +/- 90 deg/minute work well in flight and on ground.
RATE_OF_TURN_FOR_FLIGHT_PHASES = 1.5 # deg per second

# Duration of masked data to repair by interpolation for flight phase analysis
REPAIR_DURATION = 10 # seconds 

# Rate of Climb complementary filter timeconstant
RATE_OF_CLIMB_LAG_TC = 6.0 # sec

# Acceleration forwards at the start of the takeoff roll.
TAKEOFF_ACCELERATION_THRESHOLD = 0.1 # g

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_INTERVAL = 3 # samples: should be odd.

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_PERIOD = 7 # samples


"""  Top of Climb / Top of Descent Threshold.
This threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
This was often OK, but one test data sample had a 4000ft climb 20 mins
after level off. This led to reducing the threshold to 600 fpm in 3
minutes which has been found to give good qualitative segregation
between climb, cruise and descent phases."""
SLOPE_FOR_TOC_TOD = (600/float(180))

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

