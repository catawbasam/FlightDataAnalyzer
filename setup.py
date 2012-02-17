#!/usr/bin/env python

import re

try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

# http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
def parse_requirements(file_name):
    requirements = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'(\s*#)|(\s*$)', line):
            continue
        if re.match(r'\s*-e\s+', line):
            # TODO support version numbers
            requirements.append(re.sub(r'\s*-e\s+.*#egg=(.*)$', r'\1', line))
        elif re.match(r'\s*-f\s+', line):
            pass
        else:
            requirements.append(line)

    requirements.reverse()
    return requirements

def parse_dependency_links(file_name):
    dependency_links = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'\s*-[ef]\s+', line):
            dependency_links.append(re.sub(r'\s*-[ef]\s+', '', line))

    dependency_links.reverse()
    return dependency_links

from analysis_engine import ___version___ as VERSION

setup(
    name='AnalysisEngine',
    version = VERSION,
    url='http://www.flightdataservices.com/',    
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',            
    description='Flight Data Analysis Engine',    
    long_description = open('README').read() + open('CHANGES').read(),
    download_url='http://www.flightdataservices.com/',
    platforms='',
    license='OSL-3.0',
    packages = find_packages(),                      
    include_package_data = True, 
    test_suite = 'nose.collector',        
    install_requires = parse_requirements('requirements.txt'),    
    setup_requires = ['nose>=1.0'],
    dependency_links = parse_dependency_links('requirements.txt'),
    zip_safe = False,    
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "License :: OSI Approved :: Open Software License (OSL-3.0)"
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )
