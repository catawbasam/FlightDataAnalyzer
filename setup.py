#!/usr/bin/env python
# -*- coding: utf-8 -*-

# A tool parsing HDF5 files containing engineering unit representation of flight
# data.
# Copyright (c) 2011-2012 Flight Data Services Ltd
# http://www.flightdataservices.com

try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

from analysis_engine import ___version___ as VERSION
from requirements import RequirementsParser
requirements = RequirementsParser()

setup(
    name='AnalysisEngine',
    version=VERSION,   
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',
    description='A tool parsing HDF5 files containing engineering unit \
    representation of flight data.',
    long_description=open('README').read() + open('CHANGES').read(),
    license='Open Software License (OSL-3.0)',
    url='http://www.flightdatacommunity.com/',
    download_url='',    
    packages=find_packages(exclude=("tests",)),
    # The 'include_package_data' keyword tells setuptools to install any 
    # data files it finds specified in the MANIFEST.in file.    
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements.install_requires,
    setup_requires=requirements.setup_requires,
    tests_require=requirements.tests_require,
    extras_require=requirements.extras_require,
    dependency_links=requirements.dependency_links,
    test_suite='nose.collector',
    platforms=[
        'OS Independent',
    ],        
    keywords=['flight', 'data', 'analyser', 'analysis', 'monitoring', 'foqa'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: OS Independent',
        'Topic :: Software Development',
    ],
)
