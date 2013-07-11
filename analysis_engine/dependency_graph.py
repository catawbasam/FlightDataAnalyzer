import os
import sys
import logging 
import networkx as nx # pip install networkx or /opt/epd/bin/easy_install networkx

from collections import deque
from pprint import pformat
from flightdatautilities.dict_helpers import dict_filter

from analysis_engine.node import (
    ApproachNode,
    DerivedParameterNode,
    MultistateDerivedParameterNode,
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

* Nice to have: reverse digraph to get arrows poitning towards the root - use
  pre's rather than successors in tree traversal
"""


class InoperableDependencies(KeyError):
    ##def __init__(self, inoperable):
        ##self.inoperable = inoperable
    pass


def indent_tree(graph, node, level=0, space='  ', delim='- ', label=True, 
                recurse_active=True):
    '''
    Small tool to assist representing a tree on the console.
    
      print '\n'.join(indent_tree(gr_all, 'root'))
      
      - root
        - sub1
          - sub2
        - sub3
      
    :param graph: Entire graph
    :type graph: nx.DiGraph
    :param node: Node to start recursing successors from
    :type node: String/object
    :param level: Current indent level down the tree
    :type level: Int
    :param space: Multiplied by indent level
    :type space: String
    :param delim: Delimiter between space and name
    :type delim: String
    :param label: Whether to add labels about the node type
    :type label: Boolean
    :param recurse_active: Whether to show the tree for active params
    :type recurse_active: Boolean
    '''
    
    def recurse_tree(node, level):
        if node in path:
            # circular dependency started!
            path.append(node)
            return ['<<Circular Depenency to: %s>>' % node]
        path.append(node)
        if graph.node[node].get('active', True):
            if recurse_active:
                node_repr = node
            else:
                return []
        else:
            node_repr = '[%s]' % node
        node_type = graph.node[node].get('node_type')
        if node_type and label:
            node_repr = '%s (%s)' % (node_repr, node_type)
        row = '%s%s%s' % (space*level, delim, node_repr)
        level_rows = [row]
        for succ in sorted(graph.successors(node)):
            sub_level = recurse_tree(succ, level=level+1)
            path.pop()
            level_rows.extend(sub_level)
        return level_rows
    
    path = deque()  # current branch path
    return recurse_tree(node, level)


def print_tree(graph, node='root', **kwargs):
    '''
    Helper to shortcut printing of indent_tree.
    
    See indent_tree for help with args/kwargs.
    '''
    print '\n'.join(indent_tree(graph, node, **kwargs))


def dependencies3(di_graph, root, node_mgr):
    '''
    Performs a Depth First Search down each dependency node in the tree
    (di_graph) until each branch's dependencies are best satisfied.
    
    Avoids circular dependencies within the DiGraph by building up a path of
    the nodes visited down the current depth search and if encountering a
    node already visited (in the path) this node is declared unavailable.
    
    This allows nodes to depend upon a possible circular dependency which may
    or may not exist depending on the recorded parameters within the frame.
    
    e.g.
    Heading -> Heading True + Magnetic Variation
    Heading True -> Heading - Magnetic Variation
    
    :param di_graph: Directed graph of all nodes and their dependencies.
    :type di_graph: nx.DiGraph
    :param root: Root node to start traversing from, usually named 'root'
    :type root: String
    :param node_mgr: Node manager which can assess whether nodes are operational with the available dependencies at each layer of the tree.
    :type node_mgr: analysis_engine.node.NodeManager
    '''
    def traverse_tree(node):
        "Begin the recursion at this node's position in the dependency tree"
        if node in path:
            # add node for it to be removed (pop'd) in a moment
            path.append(node)
            # we've met this node before; start of circular dependency?
            logger.info("Circular dependency avoided at node '%s'. "
                        "Branch path: %s", node, path)
            return False  # establishing if available; cannot yet be available
        # we're recursing down
        path.append(node)
        if node in active_nodes:
            # node already discovered operational
            return True
        
        layer = []  # layer of current node's available dependencies
        for dependency in di_graph.successors(node):
            # traverse again, 'like we did last summer'
            if traverse_tree(dependency):
                layer.append(dependency)
            # each time traverse_tree returns, remove node from visited path
            path.pop()
            
        if node_mgr.operational(node, layer):
            # node will work at this level with the available dependencies
            active_nodes.add(node)
            ordering.append(node)
            return True  # layer below works
        else:
            # node will not work with available dependencies
            return False
        
    ordering = []
    path = deque()  # current branch path
    active_nodes = set()  # operational nodes visited for fast lookup
    traverse_tree(root)  # start recursion
    return ordering


def draw_graph(graph, name, horizontal=False):
    """
    Draws a graph to file with label and filename taken from name argument.
    
    Note: Graphviz binaries cannot be easily installed on Windows (you must
    build it from source), therefore you shouldn't bother trying to
    draw_graph unless you've done so!
    
    :param graph: Dependency graph to draw
    :type graph: nx.DiGraph
    :param name: Name of graph being drawn. Added to filename: graph_[name].ps
    :type name: String
    :param horizontal: Draw graph from left to right. Default: False (top to bottom)
    :type horizontal: Boolean
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


def any_predecessors_in_requested(node_name, requested, graph):
    '''
    Recursively move towards the start of the tree (those which depend upon
    this node) searching for a node predecessor that's within the list of
    requested nodes.
    
    If the node itself has been requested, and none of its predecessors are
    requested, the result is "False".
    
    :param node_name: Name of the node to recurse down the graph
    :type node_name: String / object
    :param requested: List of nodes requested
    :type requested: List of Strings/objects
    :param graph: Directed graph to recurse across links
    :type graph: nx.DiGraph
    '''
    for predecessor in graph.predecessors(node_name):
        if predecessor in requested:
            return predecessor
        else:
            # recurse this predecessor's path
            return any_predecessors_in_requested(predecessor, requested, graph)
    else:
        return False


def graph_adjacencies(graph):
    '''
    Create a dictionary of each nodes adjacencies within the graph. Useful for
    JIT javascript presentation of the tree.
    
    :param graph: Dependency tree graph
    :type graph: nx.Graph
    :returns: Restructured graph
    '''
    data = []
    for n,nbrdict in graph.adjacency_iter():
        # build the dict for this node
        d = dict(id=n, name=graph.node[n].get('label', n), data=graph.node[n])
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
    gr_all.add_nodes_from(node_mgr.hdf_keys, color='#72f4eb', # turquoise
                          node_type='HDFNode')
    derived_minus_lfl = dict_filter(node_mgr.derived_nodes,
                                    remove=node_mgr.hdf_keys)
    # Group into node types to apply colour. TODO: Make colours less garish.
    colors = {
        ApproachNode: '#663399', # purple
        MultistateDerivedParameterNode: '#2aa52a', # dark green
        DerivedParameterNode: '#72cdf4',  # fds-blue
        FlightAttributeNode: '#b88a00',  # brown
        FlightPhaseNode: '#d93737',  # red
        KeyPointValueNode: '#bed630',  # fds-green
        KeyTimeInstanceNode: '#fdbb30',  # fds-orange
    }
    derived_nodes = []
    for name, node in derived_minus_lfl.items():
        node_info = (name, {'color': colors[node.__base__],
                            'node_type': node.__base__.__name__})
        derived_nodes.append(node_info)
    gr_all.add_nodes_from(derived_nodes)
    
    # build list of dependencies
    derived_deps = set()  # list of derived dependencies
    for node_name, node_obj in derived_minus_lfl.iteritems():
        derived_deps.update(node_obj.get_dependency_names())
        # Create edges between node and its dependencies
        edges = [(node_name, dep, {}) for dep in node_obj.get_dependency_names()]
        gr_all.add_edges_from(edges)
            
    # add root - the top level application dependency structure based on required nodes
    # filter only nodes which are at the top of the tree (no predecessors)
    # TODO: Ask Chris about this causing problems with the trimmer.
    gr_all.add_node('root', color='#ffffff')
    root_edges = []
    for node_req in node_mgr.requested:
        if any_predecessors_in_requested(node_req, node_mgr.requested, gr_all):
            # no need to link root to this requested node as one of it's
            # predecessors will have the link therefore the tree will be
            # built inclusive of this node.
            continue
        else:
            # This node is required to build the tree
            root_edges.append(('root', node_req))
    gr_all.add_edges_from(root_edges) ##, color='red')
    
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
    ##available_nodes = set(node_mgr.derived_nodes.keys()).union(set(node_mgr.hdf_keys))
    available_nodes = set(node_mgr.keys())
    # Missing dependencies.
    missing_derived_dep = list(derived_deps - available_nodes)
    # Missing dependencies which are requested.
    missing_required = list(set(node_mgr.requested) - available_nodes)
    
    if missing_derived_dep:
        logger.warning("Found %s dependencies which don't exist in LFL "
                        "nor Node modules.", len(missing_derived_dep))
        logger.debug("The missing dependencies: %s", missing_derived_dep)
    if missing_required:
        raise ValueError("Missing required parameters: %s" % missing_required)

    # Add missing nodes to graph so it shows everything. These should all be
    # RAW parameters missing from the LFL unless something has gone wrong with
    # the derived_nodes dict!    
    gr_all.add_nodes_from(missing_derived_dep, color='#6a6e70')  # fds-grey
    return gr_all

    
def process_order(gr_all, node_mgr, raise_inoperable_required=False):
    """
    :param gr_all:
    :type gr_all: nx.DiGraph
    :param node_mgr: 
    :type node_mgr: NodeManager
    :returns:
    :rtype: 
    """
    process_order = dependencies3(gr_all, 'root', node_mgr)
    logger.debug("Processing order of %d nodes is: %s", len(process_order), process_order)
    
    for n, node in enumerate(process_order):
        gr_all.node[node]['label'] = '%d: %s' % (n, node)
        gr_all.node[node]['active'] = True
        
    inactive_nodes = set(gr_all.nodes()) - set(process_order)
    logger.debug("Inactive nodes: %s", list(sorted(inactive_nodes)))
    gr_st = gr_all.copy()
    gr_st.remove_nodes_from(inactive_nodes)
    
    for node in inactive_nodes:
        # add attributes to the node to reflect it's inactivity
        gr_all.node[node]['color'] = '#c0c0c0'  # silver
        gr_all.node[node]['active'] = False
        inactive_edges = gr_all.in_edges(node)
        gr_all.add_edges_from(inactive_edges, color='#c0c0c0')  # silver
        
    inoperable_required = list(set(node_mgr.requested) - set(process_order))
    if inoperable_required:
        logger.warning("Found %s inoperable required parameters.",
                        len(inoperable_required))
        items = []
        for n in sorted(inoperable_required):
            tree = indent_tree(gr_all, n, recurse_active=False)
            if tree:
                items.append('------- INOPERABLE -------')
                items.extend(tree)
        logger.warn('\n'+'\n'.join(items))        
        ##for p in sorted(inoperable_required):
            ##available = []
            ##unavailable = []
            ##for d in sorted(gr_all[p].keys()):
                ##if d in inactive_nodes:
                    ##unavailable.append(d)
                ##else:
                    ##available.append(d)
            ##deps_avail = "'%s'\n - Available: %s\n - Unavailable: %s" % (p,
                ##', '.join(available),
                ##', '.join(unavailable))
            ##items.append(deps_avail)
        ##logger.info("Inoperable required parameters: \n%s",
                    ##'\n'.join(items))
        if raise_inoperable_required:
            raise InoperableDependencies(inoperable_required)
    
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
     
     
def dependency_order(node_mgr, draw=not_windows, raise_inoperable_required=False):
    """
    Main method for retrieving processing order of nodes.
    
    :param node_mgr: 
    :type node_mgr: NodeManager
    :param draw: Will draw the graph. Green nodes are available LFL params, Blue are operational derived, Black are not required derived, Red are active top level requested params, Grey are inactive params. Edges are labelled with processing order.
    :type draw: boolean
    :returns: List of Nodes determining the order for processing and the spanning tree graph.
    :rtype: (list of strings, dict)
    """
    _graph = graph_nodes(node_mgr)
    gr_all, gr_st, order = process_order(_graph, node_mgr, raise_inoperable_required)
    if draw:
        from json import dumps
        logger.info("JSON Graph Representation:\n%s", dumps( graph_adjacencies(gr_st), indent=2))
    
    if draw:
        draw_graph(gr_st, 'Active Nodes in Spanning Tree')
        # reduce number of nodes by removing floating ones
        gr_all = remove_floating_nodes(gr_all)
        draw_graph(gr_all, 'Dependency Tree')
    return order, gr_st


