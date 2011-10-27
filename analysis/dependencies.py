import pygraphviz as pgv  # sudo apt-get install graphviz libgraphviz-dev

#-------------------------------------------------------------------------------
# Dependency Tree
# ===============
def ordered_set(alist):
    """
    Creates an ordered set from a list of tuples or other hashable items
    TODO: Move to library
    """
    mmap = {} # implements hashed lookup
    oset = [] # storage for set
    for item in alist:
        #Save unique items in input order
        if item not in mmap:
            mmap[item] = 1
            oset.append(item)
    return oset

###########################################################################
###########################################################################

#--- Superceeded ---
def dependencies(app):
    """ Returns a Graph Breadth First Search across a tree of dependencies
    ref: http://en.wikipedia.org/wiki/Breadth-first_search
    
    @param app: {'name':str, 'parents':list}
    """
    node_list = []
    def traverse_tree(app):
        if not app['parents']:
            return # end of this branch
        #print [node['name'] for node in app['parents']]
        for node in app['parents']:
            node_list.append(node['name'])    
        for node in app['parents']:
            traverse_tree(node)
    traverse_tree(app)
    return node_list

#--- Superceeded ---
def dependency_tree(nodes, app):
    """ Returns a Graph Breadth First Search across a tree of dependencies
    ref: http://en.wikipedia.org/wiki/Breadth-first_search
    
    @param nodes: [obj, obj]
    @param app: []
    """
    node_list = []
    def traverse_tree(app):
        if isinstance(app, str):
            # recorded raw parameter
            return True #end of this branch
        elif not app.dependencies:
            # derived param without children. end of this branch
            if app.recorded():
                return True
            else:
                return False
        #print [node['name'] for node in app['parents']]
        for node_name in app.dependencies:
            node_list.append(node_name)
        dependencies_available = []
        for node_name in app.dependencies:
            try:
                node = nodes[node_name]
            except KeyError:
                # node unavailable
                continue
            active = traverse_tree(node)
            if active:
                dependencies_available.append(node_name)
            else:
                continue
        if app.can_operate(dependencies_available):
            return True
        else:
            return False
    traverse_tree(app)
    return ordered_set(reversed(node_list)) #REVERSE?

###########################################################################
###########################################################################

#-------------------------------------------------------------------------------













###########################################################################
###########################################################################
###########################################################################








def dependencies2(app):
    """ Technique: return dependencies and self when at end of recursion branch
    This works well as it's easier to invalidate a branch
    
    TODO: Replace "optional" check with "can operate" check
    """
    
    G = pgv.AGraph()
    G.graph_attr['label']='Tree Traversal'
    G.node_attr['shape']='circle'
    G.edge_attr['color']='black'
    
    def traverse_tree(app):
        print 'recursed to:', app['name']
        if app.get('inactive'):
            n = G.get_node(app['name'])
            n.attr['color'] = 'red'
            return []
        if not app['parents']:
            n = G.get_node(app['name'])
            n.attr['color'] = 'green'
            return [app['name']]
        layer = []
        parent_names = [x['name'] for x in app['parents']]
        print 'traversing parents:', parent_names
        ##G.add_node(app['name'])
        for node in app['parents']:
            G.add_edge(app['name'], node['name'])
            branch = traverse_tree(node)
            if branch: #optional as .extend([]) does nothing!
                layer.extend(branch)
            else:
                # replace edge color to show it's not there
                e = G.get_edge(app['name'], node['name'])
                e.attr['color'] = 'grey'
        #>>>>>>>>> Replace with can_operate() test
        if app.get('optional'):
            optional = [x['name'] for x in app['optional']]
        else:
            optional = []
        missing = set(parent_names) - set(layer) - set(optional)
        if missing:
            n = G.get_node(app['name'])
            n.attr['color'] = 'grey'
            return [] # cannot operate when some required params are missing
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        layer.append(app['name'])
        return layer
    
    print '-'*10 + ' Traversing tree with %d trunk(s) '%len(app['parents']) + '-'*10
    nodes = traverse_tree(app)
    
    G.layout(prog='dot')
    G.draw('graph_%s.png'%app['name'])
    return ordered_set(nodes)


