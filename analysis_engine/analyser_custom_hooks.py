######################################
# Flight Data Analyser Custom Hooks
################################################################################


import logging
logger = logging.getLogger(name='analysis_engine.analyser_custom_hooks')

logger.info('Importing analyser custom hooks...')


# Attempt to load in flight data cleanser analysis hooks:
try:
    from data_validation.analysis_hooks import *  # NOQA
except ImportError:
    logger.exception('Unable to import flight data cleanser analysis hooks!')
    raise


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4