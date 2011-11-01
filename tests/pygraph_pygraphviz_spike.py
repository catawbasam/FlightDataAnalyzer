import pygraphviz as pgv  # sudo apt-get install graphviz libgraphviz-dev

from copy import deepcopy

#TODO: Move to networkx for graphs - it's a little more pythonic in behaviour
# and manages assigning of attributes (such as accessing the original object)
# easier #from pygraph.classes.graph import graph # pip install
# python-graph-dot
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write


from analysis.derived import Node
import inspect
def isclassandsubclass(value, classinfo):
    return inspect.isclass(value) and issubclass(value, classinfo)

def get_nodes(module_names):
    """ Get all nodes into a dictionary
    """
    nodes = {}
    for name in module_names:
        module = __import__(name)
        for c in vars(module).values():
            if isclassandsubclass(c, Node):
                nodes[c.name()] = c
    return nodes

MODULES = ('analysis.key_point_values', 
           'analysis.key_time_instanes', 
           'analysis.flight_phase')
derived_nodes = get_nodes(MODULES)




def breadth_first_search_all_nodes(di_graph, root):
    """
    only useful for us with a di_graph
    
    Returns all nodes traversed, not just new ones.
    
    Removed filter (as not required) 
    """
    def bfs():
        """
        Breadth-first search subfunction.
        """
        while (queue != []):
            node = queue.pop(0)
            for other in di_graph[node]:
                #if other not in spanning_tree:
                queue.append(other)
                ordering.append(other)
                spanning_tree[other] = node
    
    queue = [root]            # Visiting queue
    spanning_tree = dict(root=None)    # Spanning tree
    ordering = [root]
    bfs()
    return spanning_tree, ordering





##########################################################
##                SETUP TEST 
##########################################################
class MockParam(object):
    def __init__(self, dependencies=[], operational=True):
        self.dependencies = dependencies
        self.operational = operational
        
    def can_operate(self, avail):
        return self.operational
    
# nodes found on this aircraft's LFL
lfl_nodes = {
    'Raw1' : object,
    'Raw2' : object,
    'Raw3' : object,
    'Raw4' : object,
    'Raw5' : object,
    }

# nodes found from all the derived params code (top level, not their dependencies)
#NOTE: For picturing it, it should show ALL raw params required.
derived_nodes = {
    'P4' : MockParam(['Raw1', 'Raw2'], False),
    'P5' : MockParam(['Raw3', 'Raw4']),
    'P6' : MockParam(['Raw3']),
    'P7' : MockParam(['P4', 'P5', 'P6']),
    }
##########################################################


# Display entire dependency graph, not taking into account which are active for a frame
def draw_graph(graph, name):
    """
    Draws a graph to file with label and filename taken from name argument.
    """
    dot = write(graph)
    G = pgv.AGraph(dot, directed=True)
    G.layout(prog='dot')
    G.graph_attr['label'] = name
    G.draw('graph_%s.png' % name.lower().replace(' ', '_'))


    

def node_operational(name, available):
    """
    Looks up the node and tells you whether it can operate.
    
    :returns: Result of Operational test on parameter.
    :rtype: Boolean
    """
    if name in derived_nodes:
        return derived_nodes[name].can_operate(available)
    elif name in lfl_nodes:
        return True
    else:  #elif name in unavailable_deps:
        return False  #TODO: TEST!!!


def graph_nodes(lfl_nodes, derived_nodes):
    """
    :param lfl_nodes: Raw parameter nodes from the Logical Frame Layout
    :type lfl_nodes: Dict
    :param derived_nodes: Derived nodes from KPI / KTI
    :type derived_nodes: Dict
    """
    # gr_all will contain all nodes
    gr_all = digraph()
    # create nodes without attributes now as you can only add attributes once
    # (limitation of add_node_attribute())
    gr_all.add_nodes(lfl_nodes.keys() + derived_nodes.keys())
    
    # Note: All nodes need to be added before referenced by add_edge
    
    ##gr_all.add_edge(('P4', 'Raw1'))
    ##gr_all.add_edge(('P4', 'Raw2'))
    ##gr_all.add_edge(('P5', 'Raw3'))
    ##gr_all.add_edge(('P5', 'Raw4'))
    ##gr_all.add_edge(('P6', 'Raw3'))
    ##gr_all.add_edge(('P7', 'P4'))
    ##gr_all.add_edge(('P7', 'P5'))
    ##gr_all.add_edge(('P7', 'P6'))
    
    derived_deps = set()  # list of derived dependencies
    for node_name, node_obj in derived_nodes.iteritems():
        derived_deps.update(node_obj.dependencies)
        # Create edges between node and its dependencies
        for dep in node_obj.dependencies:
            gr_all.add_edge((node_name, dep))
    
    unavailable_deps = derived_deps - set(derived_nodes) - set(lfl_nodes)
    print "These dependencies are not available: %s" % unavailable_deps
    # add nodes to graph so it shows everything
    gr_all.add_nodes(unavailable_deps)  #these should all be RAW parameters not in LFL unless something has gone wrong with the derived_nodes dict!



    #TODO: List all the nodes:
    # * LFL used
    # * LFL unused
    # * Derived used
    # * Derived not operational
    # * Derived not used -- coz not referenced by a dependency kpv etc therefore not part of the spanning tree




    # Then, draw the breadth first search spanning tree rooted at top of application
    st, order = breadth_first_search_all_nodes(gr_all, root="P7")
    print "Complete processing order:", order[::-1]  #process in reverse order of dependencies




    # gr_st will be a copy of gr_all which we'll delete inactive nodes from
    gr_st = deepcopy(gr_all)
    
    # Determine whether nodes are operational
    process_order = []
    for n, node in enumerate(reversed(order)):
        if node_operational(node, process_order):
            process_order.append(node)
        else:
            gr_st.del_node(node)
            gr_all.add_node_attribute(node, ('color', 'grey'))
            #TODO: Grey edges
            ##for edge in node.edges...
            ##gr_all.add_edge_attribute(('P5', 'Raw3'), ('color', 'grey'))
    
    draw_graph(gr_all, 'All Nodes')
    

    # Breadth First Search Spanning Tree
    st, order = breadth_first_search(gr_st, root="P7")
    print "Final processing order:", order[::-1]  #process in reverse order of dependencies
    
    #gr_st.add_spanning_tree(st)  #Q: Is this required?
    draw_graph(gr_st, 'Operational Nodes BFS')
    
    
    test_gr = digraph()
    test_gr.add_spanning_tree(st)  #Q: Is this required?
    draw_graph(test_gr, 'TEST DIFF')
    
    
    #TODO: Label edges the order in which they'll be analysed / traversed?





import unittest

class TestBFS(unittest.TestCase):
    def test_bfs(self):
        # build dummy graph
        pass





'''
Validate the derived parameters to ensure that all dependencies exist as
classes OR are referenced in one of the LFL documents!
'''

# test validation for ALL algorithm dependencies across ALL LFLs
from compass.dataframe_parser import get_all_parameter_names
raw_param_list = get_all_parameter_names() # Don't restrict to any particular LFL unless requested 
#build_dependencies(raw_param_list, all_kpv)

# test validation for an aircraft's required algorithm dependencies across it's LFL

raw_param_list = get_all_parameter_names(lfl_name)
'determine whether some of the events required cannot be detected as the raw parameters does not exist in the LFL'


class TestValidation(unittest.TestCase):
    # continusouly test that the dependency structure works
    
    # 

