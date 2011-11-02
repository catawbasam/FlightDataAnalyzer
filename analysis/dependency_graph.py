##import pygraphviz as pgv  # sudo apt-get install graphviz libgraphviz-dev

import inspect
import logging 
import networkx as nx

from analysis.node import Node


def get_derived_nodes(module_names):
    """ Get all nodes into a dictionary
    """
    def isclassandsubclass(value, classinfo):
        return inspect.isclass(value) and issubclass(value, classinfo)

    nodes = {}
    for name in module_names:
        #Ref:
        #http://code.activestate.com/recipes/223972-import-package-modules-at-runtime/
        # You may notice something odd about the call to __import__(): why is
        # the last parameter a list whose only member is an empty string? This
        # hack stems from a quirk about __import__(): if the last parameter is
        # empty, loading class "A.B.C.D" actually only loads "A". If the last
        # parameter is defined, regardless of what its value is, we end up
        # loading "A.B.C"
        module = __import__(name, globals(), locals(), [''])
        for c in vars(module).values():
            if isclassandsubclass(c, Node):
                try:
                    nodes[c.get_name()] = c()
                except TypeError:
                    #TODO: Handle the expected error of top level classes
                    # Can't instantiate abstract class DerivedParameterNode
                    # - but don't know how to detect if we're at that level without resorting to 'if c.get_name() in 'derived parameter node',..
                    logging.exception('Failed to import class: %s' % c.get_name())
    return nodes




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
    ##return spanning_tree, ordering
    return ordering

# Display entire dependency graph, not taking into account which are active for a frame
def draw_graph(graph, name):
    """
    Draws a graph to file with label and filename taken from name argument.
    """
    G = nx.to_agraph(graph)
    G.layout(prog='dot')
    G.graph_attr['label'] = name
    G.draw('graph_%s.png' % name.lower().replace(' ', '_'))


    
def graph_nodes(lfl_params, required_params, derived_nodes):
    """
    :param lfl_params: Raw parameter nodes from the Logical Frame Layout
    :type lfl_params: List
    :param required_params: Required for event detection, graphing and any other exports such as APM/EHM
    :type required_params: list of node names
    :param derived_nodes: Derived nodes from KPI / KTI
    :type derived_nodes: Dict
    """
    # gr_all will contain all nodes
    gr_all = nx.DiGraph()
    # create nodes without attributes now as you can only add attributes once
    # (limitation of add_node_attribute())
    gr_all.add_nodes_from(lfl_params, color='forestgreen')
    gr_all.add_nodes_from(derived_nodes.keys())
    
    # assert that all required_params are in derived or lfl
    ##gr_all.add_nodes_from(required_params, color='blue')
    
    # build list of dependencies
    derived_deps = set()  # list of derived dependencies
    for node_name, node_obj in derived_nodes.iteritems():
        derived_deps.update(node_obj.get_dependency_names())
        # Create edges between node and its dependencies
        edges = [(node_name, dep) for dep in node_obj.get_dependency_names()]
        gr_all.add_edges_from(edges)
            
    # add root - the top level application dependency structure based on required nodes
    gr_all.add_node('root', color='red')
    root_edges = [('root', node_name) for node_name in required_params]
    gr_all.add_edges_from(root_edges, color='red')
    
    #TODO: Split this up into the following lists of nodes
    # * LFL used
    # * LFL unused
    # * Derived used
    # * Derived not operational
    # * Derived not used -- coz not referenced by a dependency kpv etc therefore not part of the spanning tree
    
    # Note: It's hard to tell whether a missing dependency is a mistyped
    # reference to another derived parameter or a parameter not available on
    # this LFL
    available_nodes = set(derived_nodes).union(set(lfl_params))
    missing_derived_dep = list(derived_deps - available_nodes)
    missing_required = list(set(required_params) - available_nodes)
    
    if missing_derived_dep:
        logging.warning("Dependencies referenced are not in LFL nor Node modules: %s",
                        missing_derived_dep)
    if missing_required:
        raise ValueError("Missing required parameters: %s" % missing_required)

    # add nodes to graph so it shows everything
    gr_all.add_nodes_from(missing_derived_dep)  #these should all be RAW parameters not in LFL unless something has gone wrong with the derived_nodes dict!    
    return gr_all



def node_operational(name, available, derived_nodes, lfl_nodes):
    """
    Looks up the node and tells you whether it can operate.
    
    :returns: Result of Operational test on parameter.
    :rtype: Boolean
    """
    if name in derived_nodes:
        return derived_nodes[name].can_operate(available)
    elif name in lfl_nodes or name == 'root':
        return True
    else:  #elif name in unavailable_deps:
        logging.warning("Confirm - node unavailable: %s", name)
        return False  #TODO: TEST!!!
    
    
def process_order(gr_all, lfl_params, derived_nodes):
    """
    :param gr_all:
    :type gr_all: nx.DiGraph
    :param derived_nodes: 
    :type derived_nodes: dict
    :param lfl_params:
    :type lfl_nodes: list of strings
    :returns:
    :rtype: 
    """
    # Then, draw the breadth first search spanning tree rooted at top of application
    order = breadth_first_search_all_nodes(gr_all, root="root")
    
    #Q: Should we delete nodes or make the edges weak?
    # gr_st will be a copy of gr_all which we'll delete inactive nodes from
    gr_st = gr_all.copy() 
    
    # Determine whether nodes are operational
    process_order = []
    for node in reversed(order):
        if node_operational(node, process_order, derived_nodes, lfl_params): #TODO = make a class method
            if node not in lfl_params + ['root']:
                gr_all.node[node]['color'] = 'blue'
            process_order.append(node)
        else:
            gr_st.remove_node(node)
            gr_all.node[node]['color'] = 'grey'
            inactive_edges = gr_all.in_edges(node)
            gr_all.add_edges_from(inactive_edges, color='grey')
    
    # Breadth First Search Spanning Tree
    #st, order = breadth_first_search(gr_st, root="root")
    order = list(nx.breadth_first_search.bfs_edges(gr_st, 'root')) #Q: Is there a method like in pygraph for retrieving the order of nodes traversed?
    if not order:
        raise ValueError("No relationship between any nodes - no process order can be defined!")
    
    # reduce edges to node list and assign process order labels to the edges
    # Note: this will skip last node (as it doesn't have an edge), which should 
    # always be 'root' - this is desirable!
    node_order = []
    for n, edge in enumerate(reversed(order)):
        node_order.append(edge[1]) #Q: is there a neater way to get the nodes?
        gr_all.edge[edge[0]][edge[1]]['label'] = n
    
    logging.debug("Node processing order: %s", node_order)
        
    return gr_all, gr_st, node_order 

     
MODULES = ('analysis.key_point_values', 
           'analysis.key_time_instances', 
           'analysis.flight_phase')
def dependency_order(lfl_params, required_params, modules=MODULES, draw=True):
    """
    Main method for retrieving processing order of nodes.
    
    :param lfl_params: Raw parameter names available within the LFL
    :type lfl_params: list of strings
    :param required_params: Derived node names required for Graphical representation or Event detection. Note that no LFL Params are "required" as they already stored in HDF file. List of names of any Node type (KPV, KTI, FlightPhase or DerivedParameters)
    :type required_params: list of strings
    :param modules: Modules to import Derived nodes from.
    :type modules: list of strings
    :param draw: Will draw the graph. Green nodes are available LFL params, Blue are operational derived, Black are not required derived, Red are active top level requested params, Grey are inactive params. Edges are labelled with processing order.
    :type draw: boolean
    :returns: Processing order for nodes
    :rtype: list of strings
    """
    # go through modules to get derived nodes
    derived_nodes = get_derived_nodes(modules)
    _graph = graph_nodes(lfl_params, required_params, derived_nodes)
    gr_all, gr_st, order = process_order(_graph, lfl_params, derived_nodes)
    
    inoperable_required = list(set(required_params) - set(order))
    if inoperable_required:
        logging.warning("Required parameters are inoperable: %s", inoperable_required)
    if draw:
        draw_graph(gr_all, 'Dependency Tree')
    return order

'''
Validate the derived parameters to ensure that all dependencies exist as
classes OR are referenced in one of the LFL documents!


# test validation for ALL algorithm dependencies across ALL LFLs
from compass.dataframe_parser import get_all_parameter_names
raw_param_list = get_all_parameter_names() # Don't restrict to any particular LFL unless requested 
#build_dependencies(raw_param_list, all_kpv)

# test validation for an aircraft's required algorithm dependencies across it's LFL


# Should probably also assert that there are no duplicate Node names (copy and paste error!)

raw_param_list = get_all_parameter_names(lfl_name)
'determine whether some of the events required cannot be detected as the raw parameters does not exist in the LFL'

class TestValidation(unittest.TestCase):
    # continusouly test that the dependency structure works
    
    # 

    pass
'''

