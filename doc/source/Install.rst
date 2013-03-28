.. _Install:

======================================
Install FlightDataAnalyzer from Source
======================================

------
Python
------

Python 2.7 is required. Python 3 is not supported.

Windows
-------

Install python(x,y) 2.7.3.1

Download the Python(x,y) installer.

* http://www.mirrorservice.org/sites/pythonxy.com/pythonxy/Python(x,y)-2.7.3.1.exe

Make a directory in the same location as ``Python(x,y)-2.7.3.1.exe`` called ``plugins``

Download the following and save them in the ``plugins`` directory.

* http://pythonxy.googlecode.com/files/h5py-2.0.1_py27.exe
* http://pythonxy.googlecode.com/files/ply-3.4_py27.exe
* http://pythonxy.googlecode.com/files/lxml-2.3.4_py27.exe
* http://pythonxy.googlecode.com/files/xlrd-0.8.0_py27.exe
* http://pythonxy.googlecode.com/files/swig-2.0.8.exe

FlightDataAnalyzer currently requires h5py 2.0.1 but python(x,y) provides 2.1.0. Therefore
the h5py 2.0.1 python(x,y) plugin is installed to forcibly downgrade to a compatible
version.

To install Python(x,y) do the following.

* Double click the install 'Python(x,y)-2.7.3.1.exe'
* Click 'Run' when the 'The publisher could not be verified.' dialog is displayed.
* Click 'I Agree' when the 'License Agreement' is displayed.

When 'Choose Components' is displayed make the following selections.

    Select the type of install: Recommended
    Install for
        All Users (default)
    Directories
        Default
    Python (in addition to what is already selected, enable the following)
        Pip
        Cython
        simplejson
        rst2pdf
        NetworkX
        xlrd
        winpdb
        GDAL
    Other
        WinMerge 2.12.4.2
        SWIG 2.0.8
        gettext 0.14.4.3
        Plugins - unticked
    Click 'Next'

From the options presents, select the following.

* 'Choose Install Location' click 'Next'.
* 'Choose Start Menu Folder' click 'Next'.

Wait for the install to complete.

* At the 'Installation Complete' menu click 'Next'.
* Click 'Finish'.

Create ``distutils.cfg``. Open a Commpand Prompt an execute the following commands.

.. code-block:: shell

    echo [build]            >  C:\Python27\Lib\distutils\distutils.cfg
    echo compiler = mingw32 >> C:\Python27\Lib\distutils\distutils.cfg

Install graphviz 2.30.1

Download graphviz 2.30.1

* http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.30.1.msi

When the download is complete double-click ``graphviz-2.30.1.msi``.

* Click 'Next'.
* Click 'Next'.
* Click 'Next'.
* Click 'Close'.

Install pygraphviz

PyGraphviz is awkward to build on Windows, fortunately a ready made package is available.

* http://www.lfd.uci.edu/~gohlke/pythonlibs/

To install PyGraphviz

* Download ``pygraphviz-1.1.win32-py2.7.‌exe`` from http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz
* Double-click ``pygraphviz-1.1.win32-py2.7.‌exe``.
* Click 'Next'.
* Click 'Next'.
* Click 'Next'.
* Click 'Finish'.

Linux
-----

Ubuntu 10.04 and 12.04 are recommended.

Enable the "Old and New Python Versions" PPA.

.. code-block:: shell

    sudo apt-add-repository ppa:fkrull/deadsnakes

Update the system and install Python 2.7.

.. code-block:: shell

    sudo apt-get update
    sudo apt-get install libpython2.7 python2.7 python2.7-dev python2.7-minimal

Remove any packaging tools installed that might have been installed via `apt`.
The versions of these packages in the Ubuntu repositories and PPAs are too old.

.. code-block:: shell

    sudo apt-get purge python-setuptools python-virtualenv python-pip python-profiler

Install distribute.

.. code-block:: shell

    curl -O http://python-distribute.org/distribute_setup.py
    sudo python2.7 distribute_setup.py

Install pip.

.. code-block:: shell

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    sudo python2.7 get-pip.py

Use pip to install virtualenv and virtualenv wrapper.

.. code-block:: shell

    sudo pip-2.7 install virtualenv --upgrade
    sudo pip install virtualenvwrapper

Install the development tools and headers required to build Numpy and SciPy.

.. code-block:: shell

    sudo apt-get install build-essential libpng12-dev libfreetype6-dev pkg-config
    sudo apt-get install gfortran libatlas-base-dev
    sudo apt-get install libxslt-dev
    sudo apt-get install libhdf5-serial-dev
    sudo apt-get install graphviz libgraphviz-dev
    sudo apt-get install swig

------------------
FlightDataAnalyzer
------------------

The FightDataAnalyzer requires Numpy and a number of other libraries developed by Flight
Data Services.

Requirements
------------

Open a shell (Linux) or Command Prompt (Windows) and execute the following
commands.

.. code-block:: shell

    pip install numpy
    pip install --upgrade git+https://github.com/FlightDataServices/FlightDataUtilities.git
    pip install --upgrade git+https://github.com/FlightDataServices/FlightDataAccessor.git

Now ``clone`` the FlightDataAnalyzer repository change to the directory where
FlightDataAnalyzer was cloned to and execute the following.

.. code-block:: shell

    pip install requirements.txt

All the FlightDataAnalyzer requirements are now installed and you can run the
following tools from source.

* ``python split_hdf_to_segments.py``
* ``python process_flight.py``
