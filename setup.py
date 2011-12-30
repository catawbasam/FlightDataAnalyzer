#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

#from analysis import __version__ as VERSION

setup(
    # === Meta data ===
    
    # Required meta data
    name='AnalysisEngine',
    version = '0.0.1',
    url='http://www.flightdataservices.com/',
    
    # Optional meta data   
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',            
    description='Flight Data Analysis Engine',    
    long_description='''
    Flight Data Analysis Engine.
    ''',    
    download_url='http://www.flightdataservices.com/',
    classifiers='',
    platforms='',
    license='',

    # === Include and Exclude ===
   
    # For simple projects, it's usually easy enough to manually add packages to 
    # the packages argument of setup(). However, for very large projects it can 
    # be a big burden to keep the package list updated. That's what 
    # setuptools.find_packages() is for.

    # find_packages() takes a source directory, and a list of package names or 
    # patterns to exclude. If omitted, the source directory defaults to the 
    # same directory as the setup script. Some projects use a src or lib 
    # directory as the root of their source tree, and those projects would of 
    # course use "src" or "lib" as the first argument to find_packages(). 
    # (And such projects also need something like package_dir = {'':'src'} in 
    # their setup() arguments, but that's just a normal distutils thing.)

    # Anyway, find_packages() walks the target directory, and finds Python 
    # packages by looking for __init__.py files. It then filters the list of 
    # packages using the exclusion patterns.

    # Exclusion patterns are package names, optionally including wildcards. For 
    # example, find_packages(exclude=["*.tests"]) will exclude all packages 
    # whose last name part is tests. Or, find_packages(exclude=["*.tests", 
    # "*.tests.*"]) will also exclude any subpackages of packages named tests, 
    # but it still won't exclude a top-level tests package or the children 
    # thereof. The exclusion patterns are intended to cover simpler use cases 
    # than this, like excluding a single, specified package and its subpackages.

    # Regardless of the target directory or exclusions, the find_packages() 
    # function returns a list of package names suitable for use as the packages 
    # argument to setup(), and so is usually the easiest way to set that 
    # argument in your setup script. Especially since it frees you from having 
    # to remember to modify your setup script whenever your project grows 
    # additional top-level packages or subpackages.

    packages = find_packages(),      
                
    # Often, additional files need to be installed into a package. These files 
    # are often data that's closely related to the package's implementation, or 
    # text files containing documentation that might be of interest to 
    # programmers using the package. These files are called package data.

    # Setuptools offers three ways to specify data files to be included in your 
    # packages. First, you can simply use the include_package_data keyword. This 
    # tells setuptools to install any data files it finds in your packages. The 
    # data files must be under CVS or Subversion control, or else they must be 
    # specified via the distutils' MANIFEST.in file. (They can also be tracked 
    # by another revision control system, using an appropriate plugin. 

    #include_package_data = True, 

    # If you want finer-grained control over what files are included (for 
    # example, if you have documentation files in your package directories and 
    # want to exclude them from installation), then you can also use the 
    # package_data keyword, e.g.:

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.dat files found in the 'data' directory of the 
        # 'skeleton' package amd all the files in the 'scripts' directory
        # of the 'skeleton' package too.
        #'skeleton': ['scripts/*'],
    },

    # === Test Suite ===

    # A string naming a unittest.TestCase subclass (or a package or module 
    # containing one or more of them, or a method of such a subclass), or naming 
    # a function that can be called with no arguments and returns a 
    # unittest.TestSuite. If the named suite is a module, and the module has an 
    # additional_tests() function, it is called and the results are added to the 
    # tests to be run. If the named suite is a package, any submodules and 
    # subpackages are recursively added to the overall test suite.

    # Specifying this argument enables use of the test command to run the 
    # specified test suite, e.g. via setup.py test. See the section on the test 
    # command below for more details.

    test_suite = 'nose.collector',
        
    # === Dependancies ===        
        
    # Dependencies on other Python modules and packages can be specified by 
    # supplying the requires keyword argument to setup(). The value must be a 
    # list of strings. Each string specifies a package that is required, 
    # and optionally what versions are sufficient.

    # To specify that any version of a module or package is required, the string 
    # should consist entirely of the module or package name. 

    # If specific versions are required, a sequence of qualifiers can be supplied 
    # in parentheses. Each qualifier may consist of a comparison operator and a 
    # version number. The accepted comparison operators are:
    
    #  <    >    ==
    #  <=   >=   !=

    # A string or list of strings specifying what other distributions need to be 
    # installed when this one is.
    install_requires = ['distribute', 'Utilties', 'HDFAccess', 'numpy'
                        'matplotlib', 'networkx', 'pygraphviz', 'scipy', 
                        'mock'],
         
    # Sometimes a project has "recommended" dependencies, that are not required 
    # for all uses of the project. For example, a project might offer optional 
    # reStructuredText support if docutils is installed. These optional features 
    # are called "extras", and setuptools allows you to define their requirements 
    # as well. In this way, other projects that require these optional features 
    # can force the additional requirements to be installed, by naming the 
    # desired extras in their install_requires.
      
    # A dictionary mapping names of "extras" (optional features of your project) 
    # to strings or lists of strings specifying what other distributions must be 
    # installed to support those features.    
    #extras_require = {
    #    'reST': ["docutils>=0.3"],
    #},

    
    # A string or list of strings specifying what other distributions need to be 
    # present in order for the setup script to run. setuptools will attempt to 
    # obtain these (even going so far as to download them using EasyInstall) 
    # before processing the rest of the setup script or commands. 
    # This argument is needed if you are using distutils extensions as part of 
    # your build process; for example, extensions that process setup() arguments 
    # and turn them into EGG-INFO metadata files.

    # (Note: projects listed in setup_requires will NOT be automatically 
    # installed on the system where the setup script is being run. They are 
    # simply downloaded to the setup directory if they're not locally available 
    # already. If you want them to be installed, as well as being available when 
    # the setup script is run, you should add them to install_requires and 
    # setup_requires.)
    setup_requires = ['nose>=1.0'],


    # If your project's tests need one or more additional packages besides those 
    # needed to install it, you can use this option to specify them. It should 
    # be a string or list of strings specifying what other distributions need to 
    # be present for the package's tests to run.     
    tests_require = [],


    # If your project depends on packages that aren't registered in PyPI, you 
    # may still be able to depend on them, as long as they are available for 
    # download as an egg, in the standard distutils sdist format, or as a single 
    # .py file. You just need to add some URLs to the dependency_links argument 
    # to setup().

    # The URLs must be either:
    #  - direct download URLs, or
    #  - the URLs of web pages that contain direct download links

    # In general, it's better to link to web pages, because it is usually less 
    # complex to update a web page than to release a new version of your 
    # project. You can also use a SourceForge showfiles.php link in the case 
    # where a package you depend on is distributed via SourceForge.

    # The dependency_links option takes the form of a list of URL strings. 
    # For example, the below will cause EasyInstall to search the specified page 
    # for eggs or source distributions, if the package's dependencies aren't 
    # already installed:

    dependency_links = [
        'http://vindictive.flightdataservices.com/Nest/dist/'
    ],

    # === Script Creation ===
    
    # So far we have been dealing with pure and non-pure Python modules, which 
    # are usually not run by themselves but imported by scripts.
    
    # Scripts are files containing Python source code, intended to be started 
    # from the command line. Scripts don't require Distutils to do anything very 
    # complicated. The only clever feature is that if the first line of the 
    # script starts with #! and contains the word "python", the Distutils will 
    # adjust the first line to refer to the current interpreter location. By 
    # default, it is replaced with the current interpreter location. 

    # The scripts option simply is a list of files to be handled in this way. 
    #scripts=['skeleton/scripts/skull', 'skeleton/scripts/cross_bones'],

    # === Entry Points (are better than 'scripts') ===

    # Packaging and installing scripts can be a bit awkward using the method 
    # above For one thing, there's no easy way to have a script's filename match 
    # local conventions on both Windows and POSIX platforms. For another, you 
    # often have to create a separate file just for the "main" script, when your 
    # actual "main" is a function in a module somewhere. 

    # setuptools fixes all of these problems by automatically generating scripts 
    # for you with the correct extension, and on Windows it will even create an 
    # .exe file so that users don't have to change their PATHEXT settings. The 
    # way to use this feature is to define "entry points" in your setup script 
    # that indicate what function the generated script should import and run. 
    # It is possible to create console scripts and GUI scripts.        
        
    # Two notations are popular, the first is easy to read and maintain.
    
    #entry_points = """
        #[console_scripts]
        #cross_bones_too = skeleton.cross_bones_too:run
        #skull_too = skeleton.skull_too:run
        # 
        #[gui_scripts]
        #skull_gui = skeleton.gui.skull_gui:run
     #   """,
        

       
    # A boolean flag specifying whether the project can be safely installed and 
    # run from a zip file. If this argument is not supplied, the bdist_egg 
    # command will have to analyze all of your project's contents for possible 
    # problems each time it buids an egg. 
    
    # There are potential compatibility issue with zipped Eggs. So default to 
    # False unless you really know why you want a zipped Egg.
    zip_safe = False,
    
    )
