#####################################
##                                 ##
##    ANALYSIS ENGINE SETTINGS     ##
##                                 ##
#####################################

# Note: Create a custom_settings.py module to override settings for your local
# environment and append customised modules.



###################
## Configuration ##
###################

# Modules to import all derived Nodes from. Additional modules can be
# appended to this list in custom_settings.py by creating a similar list of
# modules with the variable name ending with "_MODULES"
NODE_MODULES = ['analysis_engine.derived_parameters',
                'analysis_engine.key_point_values', 
                'analysis_engine.key_time_instances',
                'analysis_engine.sections',
                'analysis_engine.flight_phase',
                'analysis_engine.flight_attribute']

# API Handler
API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP'
##API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerDUMMY'

# Base URL for the API for determining nearest airport/runway, etc:
BASE_URL = 'https://polaris.flightdataservices.com'

# Location of the CA certificates to be used by the HTTP API handler:
# Note: This is the system-wide default location on Ubuntu.
CA_CERTIFICATE_FILE = '/etc/ssl/certs/ca-certificates.crt'

# Cache parameters which are used more than n times in HDF
CACHE_PARAMETER_MIN_USAGE = 4

#############################
## Splitting into Segments ##
#############################

# Minimum duration of slow airspeed in seconds to split flights inbetween.
# TODO: Find sensible value.
MINIMUM_SPLIT_DURATION = 120

# When the average normalised value of selected parameters drops below this
# value, a flight split can be made.
MINIMUM_SPLIT_PARAM_VALUE = 0.2

# Threshold for splitting based upon rate of turn. This threshold dictates
# when the aircraft is not considered to be turning.
RATE_OF_TURN_SPLITTING_THRESHOLD = 0.1

# Parameter names to be normalised for splitting flights.
SPLIT_PARAMETERS = ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',
                    'Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',
                    'Eng (1) NP', 'Eng (2) NP', 'Eng (3) NP', 'Eng (4) NP')





########################
## Parameter Analysis ##
########################

# An airspeed below which you just can't possibly be flying.
AIRSPEED_THRESHOLD = 80  # kts

# Altitude AAL complementary filter timeconstant
ALTITUDE_AAL_LAG_TC = 3.0

# Altitude to break flights into separate climb/cruise/descent segments.
# This is applied to altitude with hysteresis, so break will happen when
# climbing above 15000 ft and below 10000 ft.
ALTITUDE_FOR_CLB_CRU_DSC = 12500

# Minimum descent height range for an approach and landing phase.
APPROACH_MIN_DESCENT = 500

# Resolved vertical acceleration washout time constant. This long period
# function removes any standing offset to the resolved acceleration signal
# and is essential in the vertical velocity complementary filter.
AZ_WASHOUT_TC = 60.0

# As above for the along-track resolved acceleration term.
AT_WASHOUT_TC = 60.0

# Force to start checking control stiffness. Intended to be the same setting
# for all three flying controls.
CONTROL_FORCE_THRESHOLD = 3.0 # lb

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

# This strange conversion is for the tail clearance calculation
FEET_PER_NM = 6076

# Acceleration due to gravity
GRAVITY_IMPERIAL = 32.2 # ft/sec^2 - used for combining acceleration and height terms

# Acceleration due to gravity
GRAVITY_METRIC = 9.81 # m/sec^2 - used for comibining acceleration and groundspeed terms

# Groundspeed complementary filter time constant.
GROUNDSPEED_LAG_TC = 6.0 # seconds

# Threshold for turn onto runway at start of takeoff.
# This will usually be overwritten by the peak curvature test.
HEADING_TURN_ONTO_RUNWAY = 15.0 # deg

# Threshold for turn off runway at end of takeoff.
# This will usually be overwritten by the peak curvature test.
HEADING_TURN_OFF_RUNWAY = 15.0 # deg

# Threshold for flight phase altitude hysteresis.
HYSTERESIS_FPALT = 200 # ft

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 5 #kts

# Threshold for flight phase altitude hysteresis specifically for separating 
# Climb Cruise Descent phases.
HYSTERESIS_FPALT_CCD = 2500 # ft

# Threshold for radio altimeter hysteresis 
# (used for flight phase calculations only)
HYSTERESIS_FP_RAD_ALT = 5 # ft

# Threshold for flight phase rate of climb hysteresis.
# We're going to ignore changes smaller than this to avoid repeatedly changing
# phase if the aircraft is climbing/descending close to a threshold level.
HYSTERESIS_FPROC = 40 # fpm / RMS altitude noise
# The threshold used is scaled in proportion to the altitude noise level, so
# that for the Hercules we can get up to 400 fpm or more, a value which has
# been selected from inspection of test data which is notoriously noisy. By
# measuring the noise, we don't burden "quieter" aircraft unnecessarily.

# Threshold for rate of turn hysteresis.
HYSTERESIS_FPROT = 2 # deg/sec

# Full scale reading on the ILS
ILS_MAX_SCALE = 2.5 # dots

# Initial approach threshold height
INITIAL_APPROACH_THRESHOLD = 3000 # ft

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

# Conversion of length units
METRES_TO_FEET = 1000/25.4/12

# Many flap KPVs require the same naming convention
NAME_VALUES_FLAP = {'flap': range(1,101,1)}
NAME_VALUES_CONF = {'conf': range(1,6,1)} # Needs fixing for Conf vales.
NAME_VALUES_CLIMB = {'altitude': [10000,9000,8000,7000,6000,5000,4000,3500,\
                                   3000,2500,2000,1500,1000,750,500,400,300,\
                                   200,150,100,75,50,35,25]}
NAME_VALUES_DESCENT = {'altitude':[10000,9000,8000,7000,6000,5000,4000,3500,\
                                   3000,2500,2000,1500,1000,750,500,400,300,\
                                   200,150,100,75,50,35,25]}

# Rate of climb and descent limits of 800fpm gives good distinction with
# level flight. Separately defined to allow for future adjustment.
RATE_OF_CLIMB_FOR_CLIMB_PHASE = 800 # fpm
RATE_OF_CLIMB_FOR_DESCENT_PHASE = -500 # fpm

# Rate of climb and descent limits of 300 fpm to identify airborne after takeoff
# and end of descent, when relying solely upon pressure altitude data.
RATE_OF_CLIMB_FOR_LEVEL_FLIGHT = 300 # fpm

# Rate of climb for liftoff. This builds upon the intertially smoothed rate of
# climb computation to identify accurately the point of liftoff.
RATE_OF_CLIMB_FOR_LIFTOFF = 200 # fpm

# Rate of climb for touchdown.
RATE_OF_CLIMB_FOR_TOUCHDOWN = -100 # fpm

# Rate of turn limits for flight. 
# (Also used for validation of accelerometers on ground).
RATE_OF_TURN_FOR_FLIGHT_PHASES = 2.5 # deg per second

# Rate of turn limit for taxi event.
RATE_OF_TURN_FOR_TAXI_TURNS = 8.0 # deg per second

# Duration of masked data to repair by interpolation for flight phase analysis
REPAIR_DURATION = 10 # seconds 

# Rate of Climb complementary filter timeconstant
RATE_OF_CLIMB_LAG_TC = 3.0 # sec

# Acceleration forwards at the start of the takeoff roll.
TAKEOFF_ACCELERATION_THRESHOLD = 0.1 # g

# Height in ft where Altitude AAL switches between Radio and STD sources.
TRANSITION_ALT_RAD_TO_STD = 100

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_INTERVAL = 3 # samples: should be odd.

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_PERIOD = 7 # samples



# Top of Climb / Top of Descent Threshold.
"""This threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
This was often OK, but one test data sample had a 4000ft climb 20 mins
after level off. This led to reducing the threshold to 600 fpm in 3
minutes which has been found to give good qualitative segregation
between climb, cruise and descent phases."""
SLOPE_FOR_TOC_TOD = 600 / float(3*60) # 600fpm in 3 mins





# Import from custom_settings if exists
try:
    from analyser_custom_settings import *
    # add any new modules to the list of modules
    from copy import copy
    [NODE_MODULES.extend(v) for k, v in copy(locals()).iteritems() \
                            if k.endswith('_MODULES') and k!= 'NODE_MODULES']
    NODE_MODULES = list(set(NODE_MODULES))
except ImportError as err:
    import logging
    logging.info("Unable to import analysis_engine custom_settings.py")
    pass
