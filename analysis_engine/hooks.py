
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

### Function for post process analysis of parameters - see example below
##POST_DERIVED_PARAM_PROCESS = None
##"""
### create post processing function
##def fn(hdf, param):
    ##"Do something to param. Return Param if changes are to be saved, else None"
    ##return f(param)
### set as post process
##POST_DERIVED_PARAM_PROCESS = fn
##"""


try:
    from analyser_custom_hooks import *
except ImportError as err:
    import logging
    logger = logging.getLogger(name=__name__)
    logger.info("Unable to import custom_hooks.py")
    pass

