.. _Profiling:

=========================
Profiling Analysis Engine
=========================


------
Timeit
------
Make sure your code isn't going to run crazy slow by putting in a broad time limit

Create a test case like the following

.. code-block:: python
    :linenos:
    
    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.test_flap_using_md82_settings)
        time_taken = min(timer.repeat(2, 100))
        print "Time taken %s secs" % time_taken
        self.assertLess(time_taken, 1.0, msg="Took too long")


--------
cProfile
--------

This sample has been used to profile a process_flight using a sample unittest case.

Add a few lines to the bottom of the test module

.. code-block:: python
    :linenos:
    
    if __name__ == '__main__':
        suite = unittest.TestSuite()
        suite.addTest(TestProcessFlight('test_time_taken'))
        unittest.TextTestRunner(verbosity=2).run(suite)

Create the profile stats::
    
    python -m cProfile -o profile_hyst5.pstats process_flight_test.py &


View cProfile with Run Snake Run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install runsnake::
    
    pip install RunSnakeRun

Execute runsnake on the profile::
    
    runsnake profile.pstats
    

View cProfile stats with Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the Gprof2Dot python module from <http://code.google.com/p/jrfonseca/wiki/Gprof2Dot#Download>

Execute it against the pstats output::
    
    python gprof2dot.py -f pstats profile.pstats | dot -Tpng -o output.png



--------------
Line Profiling 
--------------

Install line-profiler into your python environment::
    
    pip install line-profiler

Wrap method of interest with @profile decorator. Remember to remove when finished!

.. code-block:: python
    :linenos:

    @profile
    def derive(self, flap=P('Flap'), flap_steps=A('Flap Settings')):
        pass

Use kernprof to profile your code then line_profiler to output stats. (I'm using EPD at the moment,hence the path)::

    /opt/epd/bin/kernprof.py -l derived_parameters_test.py
    /opt/epd/bin/python -m line_profiler derived_parameters_test.py.lprof 
    
Sample line_profiler output::

    Timer unit: 1e-06 s
    
    File: /home/chris/src/nelson/AnalysisEngine/analysis/derived_parameters.py
    Function: derive at line 711
    Total time: 0.876006 s
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       711                                               @profile
       712                                               def derive(self, flap=P('Flap'), flap_steps=A('Flap Settings')):
       714                                                   # for the moment, round off to the nearest 5 degrees
       715       200        46322    231.6      5.3          steps = np.ediff1d(flap_steps.value, to_end=[0])/2.0 + flap_steps.value
       716       200         4224     21.1      0.5          flap_stepped = np.zeros_like(flap.array.data)
       717       200          906      4.5      0.1          low = None
       718      1200         5738      4.8      0.7          for level, high in zip(flap_steps.value, steps):
       719      1000       722946    722.9     82.5              i = (low < flap.array) & (flap.array <= high)
       720      1000        10948     10.9      1.2              flap_stepped[i] = level
       721      1000         4910      4.9      0.6              low = high
       722                                                   else:
       723                                                       # all flap values above the last
       724       200        51710    258.6      5.9              flap_stepped[low < flap.array] = level    
       725       200        27019    135.1      3.1          self.array = np.ma.array(flap_stepped, mask=flap.array.mask)
    
    
