#!/usr/bin/env bash

PACKAGE="analysis_engine"
VIRTVER="2.7"

# 0 = Keep existing virtualenv. 1 = Create a new virtualenv.
VIRTNEW=0

# 0 = Run Pyflakes. 1 = Run Pyflakes & Pyint
PYLINT=0

VIRTENV=${WORKSPACE}/.pyenv

# Delete previously built virtualenv
if [ ${VIRTNEW} -eq 1 ] && [ -d ${VIRTENV} ]; then
    rm -rf ${VIRTENV}
fi

# Create virtualenv
virtualenv --python=python${VIRTVER} --no-site-packages --distribute ${VIRTENV}

# Enter the virtualenv
. ${VIRTENV}/bin/activate
cd ${WORKSPACE}

# Install testing and code metric tools
pip install clonedigger
pip install nosexcover
pip install pep8
pip install pyflakes
pip install sphinx
if [ ${PYLINT} -eq 1 ]; then
  pip install pylint
fi

# Overlay the additional Analysis Egnine tests
rm -rf tests/AnalysisEngine_tests
bzr branch http://vindictive.flightdataservices.com/Bazaar/AnalysisEngine_tests tests/AnalysisEngine_tests

# Install requirements
if [ -f requirements.txt ]; then
    pip install --upgrade -r requirements.txt
fi

# Install runtime requirements.
if [ -f setup.py ]; then
    python setup.py develop
fi

# Remove existing output files
rm nosetests.xml coverage.xml pylint.log pep8.log cpd.xml sloccount.log || :

# Run the tests and calculate coverage
nosetests --with-xcoverage --with-xunit --cover-package=${PACKAGE} --cover-erase

# Pyflakes code quality metric, in Pylint format
pyflakes ${PACKAGE} | awk -F\: '{printf "%s:%s: [E]%s\n", $1, $2, $3}' > pylint.log

# Pylint code quality tests
if [ ${PYLINT} -eq 1 ]; then
    pylint --output-format parseable --reports=y \
    --disable W0142,W0403,R0201,W0212,W0613,W0232,R0903,C0301,R0913,C0103,F0401,W0402,W0614,C0111,W0611 \
    ${PACKAGE} | tee --append pylint.log
fi

# PEP8 code quality metric
pep8 ${PACKAGE} > pep8.log || :

# Copy and Paste Detector code quality metric
clonedigger --fast --cpd-output --output=cpd.xml ${PACKAGE}

# Count lines of code
sloccount --duplicates --wide --details ${PACKAGE} > sloccount.log
