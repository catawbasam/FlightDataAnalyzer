try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest

from analysis.derived import Node
from analysis.dependencies import (ordered_set, dependencies, dependencies2, dependencies3,
                                   dependency_tree)

# mock function
f = lambda x: x

@unittest.skip("Dependency is now superceeded by Dependency2!")
class TestDependency(unittest.TestCase):
    def test_dependency(self):
        P0 = {'name':'Raw0', 'parents':[]}
        P1 = {'name':'Raw1', 'parents':[]}
        P4 = {'name':'Raw4', 'parents':[]}
        P7 = {'name':'Raw7', 'parents':[]}
        P2 = {'name':'P2', 'parents':[P0, P1, P4]}
        P3 = {'name':'P3', 'parents':[P2, P1]}
        P5 = {'name':'P5', 'parents':[P1, P2, P4]}
        P6 = {'name':'P6', 'parents':[P5, P7]}
        app = {'name':'app', 'parents':[P6, P2, P3, P5]}
        
        nodes = dependencies(app)
        process_order = ordered_set(reversed(nodes))
        print process_order 
        self.assertEqual(process_order,
                         ['Raw4', 'Raw1', 'Raw0', 'P2', 'Raw7', 'P5', 'P3', 'P6'])



class TestDependency2(unittest.TestCase):

    def test_basic_dependency_old(self):
        R1 = {'name':'Raw1', 'parents':[]}
        R2 = {'name':'Raw2', 'parents':[]}
        R3 = {'name':'Raw3', 'parents':[]}
        P4 = {'name':'P4', 'parents':[R1, R2]}
        P5 = {'name':'P5', 'parents':[R3]}
        P6 = {'name':'P6', 'parents':[]}
        P7 = {'name':'P7', 'parents':[P4, P5, P6]}
        app = {'name':'small', 'parents':[P7]}
        self.assertEqual(dependencies2(app),
                         ['Raw1', 'Raw2', 'P4', 'Raw3', 'P5', 'P6', 'P7', 'small'])
        
    def test_missing_dependency_breaks_tree(self):
        """ Inactive param traverses up tree to root
        """
        R1  = {'name':'Raw1', 'parents':[]}
        R2  = {'name':'Raw2', 'parents':[], 'inactive':True} # should break app
        P4  = {'name':'P4',   'parents':[R1, R2]}
        app = {'name':'broken',  'parents':[P4]}
        self.assertEqual(dependencies2(app), [])

    def test_missing_optional_accepted(self):
        """ Inactive param is optional so doesn't break the tree
        """
        R1 = {'name':'Raw1', 'parents':[],}
        R2 = {'name':'Raw2', 'parents':[],       'inactive':True} 
        P4 = {'name':'P4',   'parents':[R1, R2], 'optional':[R2,]} # missing one was optional!
        app = {'name':'optional',  'parents':[P4]}
        process_order = dependencies2(app) # need test cases for this one!!
        #print process_order
        self.assertEqual(process_order, ['Raw1', 'P4', 'optional'])


    def test_sample_tree_from_dj(self):

        # Example from DJ's diagram
        P0 = {'name':'TAT', 'parents':[]}
        P1 = {'name':'Indicated Airspeed', 'parents':[]}
        P4 = {'name':'Pressure Altitude', 'parents':[]}
        P2 = {'name':'SAT', 'parents':[P0, P1, P4]}
        P3 = {'name':'MACH', 'parents':[P2, P1]}
        P5 = {'name':'True Airspeed', 'parents':[P1, P2, P4]}
        P7 = {'name':'Heading','parents':[]}
        P8 = {'name':'Latitude', 'parents':[]}
        P9 = {'name':'Longitude', 'parents':[]}
        P6 = {'name':'Smoothed Track', 'parents':[P5, P7, P8, P9]}
        P10 = {'name':'Longitudinal g', 'parents':[]}
        P11 = {'name':'Lateral g','parents':[]}
        P12 = {'name':'Normal g', 'parents':[]}
        P13 = {'name':'Pitch', 'parents':[]}
        P14 = {'name':'Roll', 'parents':[]}
        P15 = {'name':'Vertical g', 'parents':[P10, P11, P12, P13, P14]}
        P16 = {'name':'Horizontal g along track','parents':[P10, P11, P12, P13, P14]} 
        P17 = {'name':'Horizontal g across track', 'parents':[P10, P11, P12, P13, P14]}
        P18 = {'name':'Heading Rate', 'parents':[P7]}
        P19 = {'name':'Radio Altimeter', 'parents':[]}
        P20 = {'name':'Height above Ground', 'parents':[P15, P19]}
        P21 = {'name':'Moment of Takeoff', 'parents':[P20]}
        P23 = {'name':'Groundspeed', 'parents':[]}
        P24 = {'name':'Smoothed Groundspeed', 'parents':[P16, P23]}
        P22 = {'name':'Slip on Runway', 'parents':[P17, P18, P24]}
        P25 = {'name':'Vertical Speed', 'parents':[P4, P15]}
        
        app = {'name':'dj_example', 'parents':[P6, P21, P25, P22]}
        nodes = dependencies2(app)
        pos = nodes.index
        print nodes
        self.assertTrue(pos('Vertical Speed') > pos('Pressure Altitude'))
        self.assertTrue(pos('Slip on Runway') > pos('Groundspeed'))
        self.assertTrue(pos('Slip on Runway') > pos('Horizontal g across track'))
        self.assertTrue(pos('Horizontal g across track') > pos('Roll'))

        # Results from dependency:
        #new_dep2 = ['Indicated Airspeed', 'TAT', 'Raw1', 'P4', 'SAT', 'Pressure Altitude', 'True Airspeed', 'Heading', 'Latitude', 'Longitude', 'Smoothed Track', 'Longitudinal g', 'Lateral g', 'Normal g', 'Pitch', 'Roll', 'Vertical g', 'Radio Altimeter', 'Height above Ground', 'Moment of Takeoff', 'Vertical Speed', 'Horizontal g across track', 'Heading Rate', 'Horizontal g along track', 'Groundspeed', 'Smoothed Groundspeed', 'Slip on Runway', 'dj_example']
        #old_dep = ['Roll', 'Pitch', 'Normal g', 'Lateral g', 'Longitudinal g', 'Groundspeed', 'Horizontal g along track', 'Heading', 'Smoothed Groundspeed', 'Heading Rate', 'Horizontal g across track', 'Vertical g', 'Pressure Altitude', 'Radio Altimeter', 'Height above Ground', 'Raw2', 'Raw1', 'P4', 'Indicated Airspeed', 'TAT', 'SAT', 'Longitude', 'Latitude', 'True Airspeed', 'Slip on Runway', 'Vertical Speed', 'Moment of Takeoff', 'Smoothed Track']
        
        
class TestDependency3(unittest.TestCase):

    def test_raw_params_not_available_on_lfl(self):
        self.assertTrue(False)
    
    def test_basic_dependency(self):
        ##R1 = type('Raw1', (Node,), {})
        ##R2 = type('Raw2', (Node,), {})
        ##R3 = type('Raw3', (Node,), {})
        P4 = type('P4', (Node,), dict(derive=f, dependencies=['Raw1', 'Raw2']))
        P5 = type('P5', (Node,), dict(derive=f, dependencies=['Raw3']))
        P6 = type('P6', (Node,), dict(derive=f))
        P7 = type('P7', (Node,), dict(derive=f, dependencies=[P4, P5, P6]))
        
        # don't instantiate them - it's not required until later!
        app = type('SmallApp', (Node,), dict(derive=f, dependencies=[P7]))
        lfl_param_names = ['Raw1', 'Raw2', 'Raw3']
        
        process_order = dependencies3(app, lfl_param_names)
        self.assertEqual(process_order,
                         ['Raw1', 'Raw2', 'P4', 'Raw3', 'P5', 'P6', 'P7', 'Small App'])
                
    def test_missing_dependency_breaks_tree(self):
        """ Inactive param traverses up tree to root
        """
        P4 = type('P4', (Node,), dict(derive=f, dependencies=['Raw1', 'Raw2']))
        app = type('BrokenApp', (Node,), dict(derive=f, dependencies=[P4]))
        
        lfl_params = ['Raw1']
        self.assertEqual(dependencies3(app, lfl_params), [])

    def test_missing_optional_accepted(self):
        """ Inactive param is optional so doesn't break the tree
        """
        P4 = type('P4', (Node,), dict(derive=f, dependencies=['Raw1', 'Raw2']))
        any_available = lambda s, avail: any([y in ['Raw1', 'Raw2'] for y in avail])
        app = type('OptionalApp', (Node,), dict(derive=f, dependencies=[P4], 
                                                can_operate=any_available))
        # only one dep available
        lfl_params = ['Raw1']
        process_order = dependencies3(app, lfl_params)
        self.assertEqual(process_order, ['Raw1', 'P4', 'Optional App'])
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# Parameter Names
ALTITUDE_STD = "Pressure Altitude"
AIRSPEED = "Indicated Airspeed"
MACH = "MACH"
SAT = "SAT"
TAT = "TAT"
# KPV Names
MAX_MACH_CRUISE = "Max Mach Cruise"
class Derived(object):
    def can_operate(self, available):
        if sorted(available) == sorted(self.dependencies):
            return True

class Sat(Derived):
    dependencies = [TAT, ALTITUDE_STD]
    def derive(self, params):
        return sum([params.TAT.value,])

class Mach(Derived):
    dependencies = [AIRSPEED, SAT, TAT, ALTITUDE_STD]
    def can_operate(self, available):
        if AIRSPEED in available and (SAT in available or TAT in available):
            return True
        else:
            return False
    def derive(self, params):
        return 12
    
class MaxMachCruise(Derived):
    dependencies = [MACH, ALTITUDE_STD]
    def derive(self, params):
        return max(params[MACH][PHASE_CRUISE])
    
class TestDependecyTree(unittest.TestCase):
    
    def test_dependency_tree_process_order(self):
        # Instantiate
        # ===========    
        nodes = {
            SAT : Sat(),
            MACH : Mach(),
            MAX_MACH_CRUISE : MaxMachCruise(),
            }

        # raw parameters recorded on this frame
        recorded_params = (TAT, AIRSPEED, ALTITUDE_STD)
        for param_name in recorded_params:
            nodes[param_name] = param_name ##type(param_name, (object,), {})
            
        # what we need at the end
        app = Derived()
        app.dependencies = (MAX_MACH_CRUISE, )
        # result
        process_order = dependency_tree(nodes, app)
        
        self.assertEqual(len(process_order), 6)
        # assert dependencies are met
        self.assertTrue(process_order.index(TAT) < process_order.index(SAT))
        self.assertTrue(process_order.index(ALTITUDE_STD) < process_order.index(SAT))
        self.assertEqual(process_order, 
                         [ALTITUDE_STD, TAT, SAT, AIRSPEED, MACH, MAX_MACH_CRUISE])
        
        # check we can still process MACH without SAT (uses TAT only)
        ##del nodes[SAT]
        ##process_order = dependency_tree(nodes, app)
        ##assert process_order == [ALTITUDE_STD, AIRSPEED, TAT, MACH, MAX_MACH_CRUISE]
        
        # without IAS, nothing can work
        del nodes[AIRSPEED]
        process_order = dependency_tree(nodes, app)
        self.assertEqual(process_order, [])  #fails but why?
        