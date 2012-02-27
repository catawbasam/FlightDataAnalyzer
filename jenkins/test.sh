#!/usr/bin/env bash

function usage() {
    local MODE=${1}
    echo "Usage"
    echo
    echo "  ${0} -p package_name -v python_version -l"
    echo
    echo "Required parameters"
    echo "  -p package_name   : The package to test."
    echo "  -v python version : The virtualenv Python version to use."
    echo "Optional parameters"
    echo "  -l                : Enable 'pylint'. By default only 'pyflakes' is used."
    echo "  -h                : This help"
    exit 1
}

# Make sure we are running from within Jenkins
if [ -z "${WORKSPACE}" ]; then
    echo "ERROR! This script is designed to run from within a Jenkins build."
    exit1
fi

# Init the variables
VIRTVER=""
PACKAGE=""
PYLINT=0    # 0 = Run Pyflakes. 1 = Run Pyflakes & Pyint

# Parse the options
OPTSTRING=hlp:v:
while getopts ${OPTSTRING} OPT
do
    case ${OPT} in
        h) usage;;
        l) PYLINT=1;;
        p) PACKAGE=${OPTARG};;
        v) VIRTVER=${OPTARG};;
        *) usage;;
    esac
done
shift "$(( $OPTIND - 1 ))"

# Check that the package dir exists.
if [ ! -d ${WORKSPACE}/${PACKAGE} ]; then
    echo "ERROR! Can't find directory ${WORKSPACE}/${PACKAGE}"
    exit 1
fi

# Remove old style virtualenv
if [ -d ${WORKSPACE}/.pyenv ]; then
    rm -rf ${WORKSPACE}/.pyenv
fi

VIRTENV=${WORKSPACE}/.py${VIRTVER}

# Create virtualenv, but only if doesn't exist
if [ ! -f ${VIRTENV}/bin/python${VIRTVER} ]; then
    virtualenv --python=python${VIRTVER} --no-site-packages --distribute ${VIRTENV}
fi

# Enter the virtualenv
. ${VIRTENV}/bin/activate

# Enter the Jenkins workspace
cd ${WORKSPACE}

# Update pip to the latest version and use the interna PyPI server
export PIP_INDEX_URL=http://pypi.flightdataservices.com/simple/
pip install --upgrade pip

# Install Jenkins requirements
if [ -f requirements-jenkins.txt ]; then
    pip install --upgrade -r requirements-jenkins.txt
fi

# Install Sphinx requirements
if [ -f requirements-sphinx.txt ]; then
    pip install --upgrade -r requirements-sphinx.txt
fi

#eval pip install --upgrade file:///.#egg=${PACKAGE}[jenkins,sphinx]

# Run any additional setup steps
if [ -x ${WORKSPACE}/jenkins/setup-extra.sh ]; then
    ${WORKSPACE}/jenkins/setup-extra.sh
fi

# Install runtime requirements.
if [ -f setup.py ]; then
    python setup.py develop
fi

# Remove existing output files
rm coverage.xml nosetests.xml pylint.log pep8.log cpd.xml sloccount.log

# Run the tests and coverage
if [ -f setup.py ]; then
    python setup.py jenkins
fi

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
