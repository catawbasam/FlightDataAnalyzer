import os
import sys
import logging 
import networkx as nx # pip install networkx or /opt/epd/bin/easy_install networkx

from collections import defaultdict

from utilities.dict_helpers import dict_filter

from analysis_engine.node import (
    DerivedParameterNode,
    FlightAttributeNode,
    FlightPhaseNode,
    KeyPointValueNode,
    KeyTimeInstanceNode,
)

logger = logging.getLogger(__name__)
not_windows = sys.platform not in ('win32', 'win64') # False for Windows :-(

"""
TODO:
=====
* Colour nodes by derived parameter type
* reverse digraph to get arrows poitning towards the root - use pre's rather than successors in tree traversal


"""


##def breadth_first_search_all_nodes(di_graph, root):
    ##"""
    ##only useful for us with a di_graph
    
    ##Returns all nodes traversed, not just new ones.
    
    ##Removed filter (as not required) 
    ##"""
    ##def bfs():
        ##"""
        ##Breadth-first search subfunction.
        ##"""
        ##while (queue != []):
            ##node = queue.pop(0)
            ##for other in di_graph[node]:
                ###if other not in spanning_tree:
                ##queue.append(other)
                ##ordering.append(other)
                ##spanning_tree[other] = node
    ##if filter(lambda e: e[0] == e[1], di_graph.edges()):
        ### If there is a recursive loop, raise an exception rather than looping
        ### until a MemoryError is eventually raised.
        ##raise ValueError("Traversal with fail with recursive dependencies in "
                         ##"the digraph.")
    ##queue = [root]            # Visiting queue
    ##spanning_tree = dict(root=None)    # Spanning tree
    ##ordering = [root]
    ##bfs()
    ####return spanning_tree, ordering
    ##return ordering


def dependencies3(di_graph, root, node_mgr):
    
    def traverse_tree(node):
        # check this first to improve performance
        if node in active_nodes:
            # node already discovered operational
            return True
        
        layer = []
        for dependency in di_graph.successors(node):
            # traverse again
            if traverse_tree(dependency):
                layer.append(dependency)
            
        if node_mgr.operational(node, layer):
            # node will work at this level
            active_nodes.add(node)
            ordering.append(node)
            return True # layer below works
        else:
            # node does not work
            return False
        
    ordering = [] # reverse
    active_nodes = set() # operational nodes visited for fast lookup
    traverse_tree(root) # start recursion
    return ordering


# Display entire dependency graph, not taking into account which are active for a frame
def draw_graph(graph, name, horizontal=False):
    """
    Draws a graph to file with label and filename taken from name argument.
    
    Note: Graphviz binaries cannot be easily installed on Windows (you must
    build it from source), therefore you shouldn't bother trying to
    draw_graph unless you've done so!
    """
    # hint: change filename extension to change type (.png .pdf .ps)
    file_path = 'graph_%s.ps' % name.lower().replace(' ', '_')
    # TODO: Set the number of pages #page="8.5,11";
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
        if horizontal:
            # set layout left to right before converting all nodes to new format
            graph.graph['graph'] = {'rankdir' : 'LR'}
        G = nx.to_agraph(graph)
    except ImportError:
        logger.exception("Unable to import pygraphviz to draw graph '%s'", name)
        return
    G.layout(prog='dot')
    G.graph_attr['label'] = name
    G.draw(file_path)
    logger.info("Dependency tree drawn: %s", os.path.abspath(file_path))
    
def graph_adjacencies(G):
    data = []
    for n,nbrdict in G.adjacency_iter():
        # build the dict for this node
        d = dict(id=n, name=G.node[n].get('label', n), data=G.node[n])
        adj = []
        for nbr, nbrd in nbrdict.items():
            adj.append(dict(nodeTo=nbr, data=nbrd))
        d['adjacencies'] = adj
        data.append(d)
    return data

def graph_nodes(node_mgr):
    """
    :param node_mgr:
    :type node_mgr: NodeManager
    """
    # gr_all will contain all nodes
    gr_all = nx.DiGraph()
    # create nodes without attributes now as you can only add attributes once
    # (limitation of add_node_attribute())
    gr_all.add_nodes_from(node_mgr.lfl, color='forestgreen')
    derived_minus_lfl = dict_filter(node_mgr.derived_nodes, remove=node_mgr.lfl)
    # Group into node types to apply colour. TODO: Make colours less garish.
    colors = {
        DerivedParameterNode: 'yellow',
        FlightAttributeNode: 'blue',
        FlightPhaseNode: 'brown',
        KeyPointValueNode: 'purple',
        KeyTimeInstanceNode: 'orange',
    }
    gr_all.add_nodes_from(
        [(name, {'color': colors[node.__base__]}) for name, node in derived_minus_lfl.items()])
    
    # build list of dependencies
    derived_deps = set()  # list of derived dependencies
    for node_name, node_obj in derived_minus_lfl.iteritems():
        derived_deps.update(node_obj.get_dependency_names())
        # Create edges between node and its dependencies
        edges = [(node_name, dep, {'color': 'Gray'}) for dep in node_obj.get_dependency_names()]
        gr_all.add_edges_from(edges)
            
    # add root - the top level application dependency structure based on required nodes
    # filter only nodes which are at the top of the tree (no predecessors)
    gr_all.add_node('root', color='red')
    root_edges = [('root', node_name) for node_name in node_mgr.requested \
                  if not gr_all.predecessors(node_name)]
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
    ##available_nodes = set(node_mgr.derived_nodes.keys()).union(set(node_mgr.lfl))
    available_nodes = set(node_mgr.keys())
    # Missing dependencies.
    missing_derived_dep = list(derived_deps - available_nodes)
    # Missing dependencies which are required.
    missing_required = list(set(node_mgr.requested) - available_nodes)
    
    if missing_derived_dep:
        logger.warning("Found %s dependencies which don't exist in LFL "
                        "nor Node modules.", len(missing_derived_dep))
        logger.info("The missing dependencies: %s", missing_derived_dep)
    if missing_required:
        raise ValueError("Missing required parameters: %s" % missing_required)

    # Add missing nodes to graph so it shows everything. These should all be
    # RAW parameters missing from the LFL unless something has gone wrong with
    # the derived_nodes dict!    
    gr_all.add_nodes_from(missing_derived_dep, color='fushcia')  
    return gr_all

    
def process_order(gr_all, node_mgr):
    """
    :param gr_all:
    :type gr_all: nx.DiGraph
    :param node_mgr: 
    :type node_mgr: NodeManager
    :returns:
    :rtype: 
    """
    process_order = dependencies3(gr_all, 'root', node_mgr)
    logger.info("Processing order of %d nodes is: %s", len(process_order), process_order)
    
    for n, node in enumerate(process_order):
        gr_all.node[node]['label'] = '%d: %s' % (n, node)
        
    inactive_nodes = set(gr_all.nodes()) - set(process_order)
    logger.info("Inactive nodes: %s", list(sorted(inactive_nodes)))
    gr_st = gr_all.copy()
    gr_st.remove_nodes_from(inactive_nodes)
    
    for node in inactive_nodes:
        gr_all.node[node]['color'] = 'Silver'
        inactive_edges = gr_all.in_edges(node)
        gr_all.add_edges_from(inactive_edges, color='Silver')
    
    return gr_all, gr_st, process_order[:-1] # exclude 'root'


def remove_floating_nodes(graph):
    """
    Remove all nodes which aren't referenced within the dependency tree
    """
    nodes = list(graph)
    for node in nodes:
        if not graph.predecessors(node) and not graph.successors(node):
            graph.remove_node(node)
    return graph
     
     
def dependency_order(node_mgr, draw=not_windows):
    """
    Main method for retrieving processing order of nodes.
    
    :param node_mgr: 
    :type node_mgr: NodeManager
    :param draw: Will draw the graph. Green nodes are available LFL params, Blue are operational derived, Black are not required derived, Red are active top level requested params, Grey are inactive params. Edges are labelled with processing order.
    :type draw: boolean
    :returns: List of Nodes determining the order for processing.
    :rtype: list of strings
    """
    _graph = graph_nodes(node_mgr)
    gr_all, gr_st, order = process_order(_graph, node_mgr)
    
    if draw:
        from json import dumps
        logger.info("JSON Graph Representation:\n%s", dumps( graph_adjacencies(gr_st), indent=2))
    inoperable_required = list(set(node_mgr.requested) - set(order))
    if inoperable_required:
        logger.warning("Found %s inoperable required parameters.",
                        len(inoperable_required))
        logger.info("Inoperable required parameters: %s",
                     inoperable_required)
    if draw:
        draw_graph(gr_st, 'Active Nodes in Spanning Tree')
        # reduce number of nodes by removing floating ones
        gr_all = remove_floating_nodes(gr_all)
        draw_graph(gr_all, 'Dependency Tree')
    return order, gr_st


