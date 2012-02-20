#!/usr/bin/env bash

# Enter the virtualenv
VIRTENV=${WORKSPACE}/.pyenv
. ${VIRTENV}/bin/activate
cd ${WORKSPACE}

# Remove previous builds
rm -rfv ${WORKSPACE}/dist/* || :

# Make a source distribution
if [ -f setup.py ] && [ -f setup.cfg ]; then

    # Get the tag.
    TAG_BUILD=`grep tag_build ${WORKSPACE}/setup.cfg | cut -d'=' -f2 | sed 's/ //g'`

    # If the build is tagged, in any way, then append the Jenkins build number.
    # If the build is not tagged, it is assumed to be a release.
    if [ -n "${TAG_BUILD}" ]; then
        python setup.py egg_info -b ${TAG_BUILD}.${BUILD_NUMBER} sdist
    else
        python setup.py sdist
    fi

    # Grab the last commit log.
    if [ -d ${WORKSPACE}/.bzr ]; then
        LAST_LOG=`bzr log -l 1`
    elif [ -d ${WORKSPACE}/.git ]; then
        LAST_LOG=`git log -n 1`
    elif [ -d ${WORKSPACE}/.hg ]; then
        LAST_LOG=`hg log -l 1`
    else
        LAST_LOG="None"
    fi

    # Create a build record
    BUILD_RECORD="`ls -1tr ${WORKSPACE}/dist/*.zip | tail -n1`.html"
    echo "<html><head><title>${BUILD_TAG}</title></head><body><h2>${BUILD_ID}</h2><ul><li><a href=\"${BUILD_URL}\" target=\"_blank\">${BUILD_TAG}</a></li></ul><h3>Last Commit Log</h3><pre>${LAST_LOG}</pre></body></html>" > ${BUILD_RECORD}

    # Build sphinx documentation
    if [ -f ${WORKSPACE}/doc/Makefile ]; then
        cd ${WORKSPACE}
        python setup.py build_sphinx
    fi
fi