# Verbose output for text output file
VERBOSE = True

# Minimum duration for a flight to analyse (sec)
MIN_FLIGHT_TO_ANALYSE = 300

# Starting point for data file walk
DATA_FILE_ROOT = '..\Data files\FDR Replay Test Data'

# Text output filename
TEXT_OUTPUT_FILENAME = 'Multiple files test output.txt'

# An airspeed below which you just can't possibly be flying.
AIRSPEED_FOR_FLIGHT = 80

# Less than 5 mins you can't do a circuit, so we'll presume this is a data snippet
FLIGHT_WORTH_ANALYSING_SEC = 300

# List of the three key parameters we must have for analysis.
ESSENTIAL_PARAMETERS = ['Head Mag','Airspeed','Altitude STD']

# Lowest level of parameter validity acceptable for essential parameter set
ESSENTIAL_PARAM_MIN_ACCEPTABLE_VALIDITY = 80.0

### Minimum period of a descent for testing against thresholds (reduces number of KPVs computed in turbulence)
##DESCENT_MIN_DURATION = 10

# Minimum duration of a flight (above AIRSPEED_THRESHOLD) that is worth analysis
FLIGHT_WORTH_ANALYSING_SEC = 200 # planned to use 300 but 200 for test

# Minimum airspeed for flight
AIRSPEED_THRESHOLD = 80
