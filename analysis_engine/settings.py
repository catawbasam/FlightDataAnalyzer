# -*- coding: utf-8 -*-
##############################################################################
# Flight Data Analyzer Settings
##############################################################################


# Note: Create a custom_settings.py module to override settings for your local
# environment and append customised modules.


##############################################################################
# Configure Logging


import logging
logger = logging.getLogger(name=__name__)


##############################################################################
# General Configuration


# Modules to import all derived Nodes from. Additional modules can be
# appended to this list in analyzer_custom_settings.py by creating a similar list of
# modules with the variable name ending with "_MODULES"
# e.g. MY_EXTRA_MODULES = ['my_package.extra_attributes', 'my_package.extra_params']
NODE_MODULES = ['analysis_engine.derived_parameters',
                'analysis_engine.key_point_values',
                'analysis_engine.key_time_instances',
                'analysis_engine.sections',
                'analysis_engine.flight_phase',
                'analysis_engine.flight_attribute']

# API Handler
API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerHTTP'
##API_HANDLER = 'analysis_engine.api_handler_analysis_engine.AnalysisEngineAPIHandlerDummy'

# Base URL for the API for determining nearest airport/runway, etc:
BASE_URL = 'https://polaris-test.flightdataservices.com'

# Location of the CA certificates to be used by the HTTP API handler:
# Note: This is the system-wide default location on Ubuntu.
CA_CERTIFICATE_FILE = '/etc/ssl/certs/ca-certificates.crt'

# Cache parameters which are used more than n times in HDF
CACHE_PARAMETER_MIN_USAGE = 0

# Calculate the current year as 'YYYY' here once to avoid repitition
from datetime import datetime
CURRENT_YEAR = str(datetime.now().year)


##############################################################################
# Segment Splitting


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


##############################################################################
# Parameter Analysis


# The limit of compensation for normal accelerometer errors. For example, to
# allow for an accelerometer to lie in the range 0.8g to 1.2g, enter a value
# of 0.2. An accelerometer with an average reading of 1.205g during the taxi
# phases (in and out) will not be corrected, and all the acceleration KPVs
# will carry that offset.
ACCEL_NORM_OFFSET_LIMIT = 0.3 #g

# The limit of compensation for normal accelerometer errors. For example, to
# allow for an accelerometer to lie in the range 0.8g to 1.2g, enter a value
# of 0.2. An accelerometer with an average reading of 1.205g during the taxi
# phases (in and out) will not be corrected, and all the acceleration KPVs
# will carry that offset.
ACCEL_LAT_OFFSET_LIMIT = 0.1 #g

# The minimum sensible duration for being airborne, used to reject skips and bounced landings.
AIRBORNE_THRESHOLD_TIME = 60 # secs

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

# Minimum threshold for detecting a bounced landing. Bounced landings lower
# than this will not be identified or held in a database. Note: The event
# threshold is higher than this.
BOUNCED_LANDING_THRESHOLD = 2.0

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

# Change in altitude to create a descent low climb phase, from which
# approaches, go-around and touch-and-go phases and instances derive.
DESCENT_LOW_CLIMB_THRESHOLD = 500 #ft

# This strange conversion is for the tail clearance calculation
FEET_PER_NM = 6076

# Acceleration due to gravity
GRAVITY_IMPERIAL = 32.2 # ft/sec^2 - used for combining acceleration and height terms

# Acceleration due to gravity
GRAVITY_METRIC = 9.81 # m/sec^2 - used for comibining acceleration and groundspeed terms

# Groundspeed complementary filter time constant.
GROUNDSPEED_LAG_TC = 6.0 # seconds

# Threshold for start and end of Mobile phase when groundspeed is available.
GROUNDSPEED_FOR_MOBILE = 5.0 # kts

# Threshold for start and end of Mobile phase
HEADING_RATE_FOR_MOBILE = 2.0 # deg/sec

# Threshold for turn onto runway at start of takeoff.
# This will usually be overwritten by the peak curvature test.
HEADING_TURN_ONTO_RUNWAY = 15.0 # deg

#Threshold for turn off runway at end of takeoff. This allows for turning
#onto a rapid exit turnoff, and so we are treating deceleration down the RET
#as part of the landing phase. Notice that the KTI "Landing Turn Off Runway"
#will determine the point of turning off the runway centreline in either
#case, using the peak curvature technique.
HEADING_TURN_OFF_RUNWAY = 60.0 # deg

# Holding pattern criteria.
# Minimum time is 4 minutes, corresponding to one racetrack pattern.
HOLDING_MIN_TIME = 4*60 #sec
# Maximum groundspeed over the period in the hold. This segregates true
# holds, where the effective speed is significantly reduced (that's the point
# of the hold), from curving departures or approaches.
HOLDING_MAX_GSPD = 60.0 # kts

# Threshold for flight phase altitude hysteresis.
HYSTERESIS_FPALT = 200 # ft

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 5 #kts

# Threshold for flight phase altitude hysteresis specifically for separating
# Climb Cruise Descent phases.
HYSTERESIS_FPALT_CCD = 500 # ft
# Note: Original value was 2,500ft, based upon normal operations, but
# circuits flown below 2,000ft agl were being processed incorrectly.

# Threshold for radio altimeter hysteresis
# (used for flight phase calculations only)
HYSTERESIS_FP_RAD_ALT = 5 # ft

# Threshold for flight phase vertical speed hysteresis.
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

# Conversion from kg to lb.
# Thanks to David A. Forbes of Aero Tech Research for the conversion figure.
KG_TO_LB = 2.2046226218487757 #lb/kg

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

# Maximum age of a Segment's timebase in days. If a calculated timebase is
# older or in the future, fallback_dt will be used instead. A value of None
# allows any
MAX_TIMEBASE_AGE = 365 * 10

# Conversion from degrees of latitude to metres. I know it's approximate, but
# good enough for the uses we have here. To convert deg longitude, allow for
# the cos(latitude) reduction in distance as we get away from the equator.
METRES_PER_DEG_LATITUDE = 111120 # metres/deg

# Conversion of length units
METRES_TO_FEET = 1000/25.4/12

# Conversion from metres to nautical miles.
METRES_TO_NM = 1852.0

'''
See experimental KTP LandingStopLimitPointPoorBraking et seq.
# Mu values for good, medium and poor braking action (Boeing definition).
MU_GOOD = 0.2
MU_MEDIUM = 0.1
MU_POOR = 0.05 # dimensionless.
'''
# Vertical speed limits of 800 fpm and -500 fpm gives good distinction with
# level flight. Separately defined to allow for future adjustment.
VERTICAL_SPEED_FOR_CLIMB_PHASE = 800  # fpm
VERTICAL_SPEED_FOR_DESCENT_PHASE = -500  # fpm

# Vertical speed limits of 300 fpm to identify airborne after takeoff and end
# of descent, when relying solely upon pressure altitude data.
VERTICAL_SPEED_FOR_LEVEL_FLIGHT = 300  # fpm

# Vertical speed for liftoff. This builds upon the intertially smoothed
# vertical speed computation to identify accurately the point of liftoff.
VERTICAL_SPEED_FOR_LIFTOFF = 200  # fpm

# Vertical speed for touchdown.
VERTICAL_SPEED_FOR_TOUCHDOWN = -100  # fpm

# Vertical speed complementary filter timeconstant
VERTICAL_SPEED_LAG_TC = 3.0  # sec

# Rate of turn limits for flight.
# (Also used for validation of accelerometers on ground).
RATE_OF_TURN_FOR_FLIGHT_PHASES = 2.0 # deg per second

# Rate of turn limit for taxi event.
RATE_OF_TURN_FOR_TAXI_TURNS = 5.0 # deg per second

# Duration of masked data to repair by interpolation for flight phase analysis
REPAIR_DURATION = 10 # seconds

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


##############################################################################
# KPV Name Values

# These are some common frequently used name values defined here to be used in
# multiple key point values or key time instances for consistency.


NAME_VALUES_ENGINE = {
    'number': [1, 2, 3, 4],
}


NAME_VALUES_FLAP = {
    'flap': range(0, 46) + [50, 100],  # NOTE: 1-45° and 50%/100% for C-130
}


NAME_VALUES_CONF = {
    'conf': range(1, 6),  # FIXME: Proper conf values needed.
}


NAME_VALUES_CLIMB = {
    'altitude': [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000,
        1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000,
        10000],
}


NAME_VALUES_DESCENT = {
    'altitude': NAME_VALUES_CLIMB['altitude'][::-1],
}


##############################################################################
# Custom Settings


# Import from custom_settings if exists
try:
    from analyser_custom_settings import *  # NOQA
    # add any new modules to the list of modules
    from copy import copy
    [NODE_MODULES.extend(v) for k, v in copy(locals()).iteritems() \
                            if k.endswith('_MODULES') and k!= 'NODE_MODULES']
    NODE_MODULES = list(set(NODE_MODULES))
except ImportError as err:
    # logger.info preferred, but stack trace is important when trying to
    # determine an unexpected ImportError lower down the line.
    logger.exception("Unable to import analysis_engine custom_settings.py")
    pass


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
