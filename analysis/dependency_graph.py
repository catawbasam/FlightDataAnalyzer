import inspect
import logging 
import networkx as nx # pip install networkx or /opt/epd/bin/easy_install networkx

from analysis import settings
from analysis.node import Node, NodeManager


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
        ##abstract_nodes = ['Node', 'Derived Parameter Node', 'Key Point Value Node', 'Flight Phase Node'
        module = __import__(name, globals(), locals(), [''])
        for c in vars(module).values():
            if isclassandsubclass(c, Node) and c.__module__ != 'analysis.node':
                try:
                    nodes[c.get_name()] = c
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
    if filter(lambda e: e[0] == e[1], di_graph.edges()):
        # If there is a recursive loop, raise an exception rather than looping
        # until a MemoryError is eventually raised.
        raise ValueError("Traversal with fail with recursive dependencies in "
                         "the digraph.")
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
    
    Note: Graphviz binaries cannot be easily installed on Windows (you must
    build it from source), therefore you shouldn't bother trying to
    draw_graph unless you've done so!
    """
    file_path = 'graph_%s.png' % name.lower().replace(' ', '_')

    # Trying to get matplotlib to install nicely
    # Warning: pyplot does not render the graphs well!
    ##import matplotlib.pyplot as plt
    ##nx.draw(graph)
    ##plt.show()
    ##plt.savefig(file_path)
    try:
        ##import pygraphviz as pgv 
        # sudo apt-get install graphviz libgraphviz-dev
        # pip install pygraphviz
        #Note: nx.to_agraph performs pygraphviz import
        G = nx.to_agraph(graph)
    except ImportError:
        logging.exception("Unable to import pygraphviz to draw graph '%s'", name)
        return
    G.layout(prog='dot')
    G.graph_attr['label'] = name
    G.draw(file_path)
    
def graph_nodes(node_mgr): ##lfl_params, required_params, derived_nodes):
    """
    
    """
    # gr_all will contain all nodes
    gr_all = nx.DiGraph()
    # create nodes without attributes now as you can only add attributes once
    # (limitation of add_node_attribute())
    gr_all.add_nodes_from(node_mgr.lfl, color='forestgreen')
    gr_all.add_nodes_from(node_mgr.derived_nodes.keys())
    
    # build list of dependencies
    derived_deps = set()  # list of derived dependencies
    for node_name, node_obj in node_mgr.derived_nodes.iteritems():
        derived_deps.update(node_obj.get_dependency_names())
        # Create edges between node and its dependencies
        edges = [(node_name, dep) for dep in node_obj.get_dependency_names()]
        gr_all.add_edges_from(edges)
            
    # add root - the top level application dependency structure based on required nodes
    gr_all.add_node('root', color='red')
    root_edges = [('root', node_name) for node_name in node_mgr.requested]
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
    # Set of all derived and LFL Nodes.
    available_nodes = set(node_mgr.derived_nodes.keys()).union(set(node_mgr.lfl))
    # Missing dependencies.
    missing_derived_dep = list(derived_deps - available_nodes)
    # Missing dependencies which are required.
    missing_required = list(set(node_mgr.requested) - available_nodes)
    
    if missing_derived_dep:
        logging.warning("Dependencies referenced are not in LFL nor Node modules: %s",
                        missing_derived_dep)
    if missing_required:
        raise ValueError("Missing required parameters: %s" % missing_required)

    # Add missing nodes to graph so it shows everything. These should all be
    # RAW parameters missing from the LFL unless something has gone wrong with
    # the derived_nodes dict!    
    gr_all.add_nodes_from(missing_derived_dep)  
    return gr_all

    
def process_order(gr_all, node_mgr): ##lfl_params, derived_nodes):
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
    
    # Determine whether nodes are operational, this will repeatedly ask some 
    # nodes as they may only become operational later on.
    process_order = []
    for node in reversed(order):
        if node_mgr.operational(node, process_order):
            if node not in node_mgr.lfl + ['root']:
                gr_all.node[node]['color'] = 'blue'
            # un-grey edges that were previously inactive
            active_edges = gr_all.in_edges(node)
            gr_all.add_edges_from(active_edges, color='black')
            process_order.append(node)
        else:
            gr_all.node[node]['color'] = 'grey'
            inactive_edges = gr_all.in_edges(node)
            gr_all.add_edges_from(inactive_edges, color='grey')

    # remove nodes from gr_st that aren't in the process order
    gr_st.remove_nodes_from(set(gr_st.nodes()) - set(process_order))
    
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


def remove_nodes_without_edges(graph):
    nodes = list(graph)
    for node in nodes:
        if not graph.edges(node):
            print node
            graph.remove_node(node)
    return graph
     
def dependency_order(lfl_params, required_params, aircraft_info, 
                     achieved_flight_record, modules=settings.NODE_MODULES, draw=True):
    """
    Main method for retrieving processing order of nodes.
    
    :param lfl_params: Raw parameter names available within the Logical Frame Layout (LFL)
    :type lfl_params: list of strings
    :param required_params: Derived node names required for Graphical representation, data exports or Event detection. Note that no LFL Params are "required" as they already stored in HDF file. List of names of any Node type (KPV, KTI, FlightPhase or DerivedParameters)
    :type required_params: list of strings
    :param modules: Modules to import Derived nodes from.
    :type modules: list of strings
    :param draw: Will draw the graph. Green nodes are available LFL params, Blue are operational derived, Black are not required derived, Red are active top level requested params, Grey are inactive params. Edges are labelled with processing order.
    :type draw: boolean
    :returns: Tuple of NodeManager and a list determining the order for processing the nodes.
    :rtype: (NodeManager, list of strings)
    """    
    # go through modules to get derived nodes
    derived_nodes = get_derived_nodes(modules)
    # if required_params isn't set, try using ALL derived_nodes!
    if not required_params:
        logging.warning("No required_params declared, using all derived nodes")
        required_params = derived_nodes.keys()
        
    #TODO: Review whether adding all Flight Attributes as required params here is a sensible location
    required_params += get_derived_nodes(['analysis.flight_attribute']).keys()
    
    # keep track of all the node types
    node_mgr = NodeManager(lfl_params, required_params, derived_nodes,
                           aircraft_info, achieved_flight_record)
    _graph = graph_nodes(node_mgr)
    
    ##_graph = remove_nodes_without_edges(_graph)
    ##draw_graph(_graph, 'Dependency Tree')
    gr_all, gr_st, order = process_order(_graph, node_mgr)
    
    inoperable_required = list(set(required_params) - set(order))
    if inoperable_required:
        logging.warning("Required parameters are inoperable: %s", inoperable_required)
    if draw:
        draw_graph(gr_all, 'Dependency Tree')
        draw_graph(gr_st, 'Active Nodes in Spanning Tree')
    return node_mgr, order


