#!/usr/bin/env bash

set

# Overlay the additional Analysis Egnine tests
rm -rf tests/AnalysisEngine_tests
bzr branch http://vindictive.flightdataservices.com/Bazaar/AnalysisEngine_tests tests/AnalysisEngine_tests
rm -rf test/AnalysisEngine_tests/.bzr
