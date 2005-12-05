Help on module bayesnet:

NAME
    bayesnet

FILE
    /home/dickrp/svn/support/python/bayesnet/bayesnet.py

DESCRIPTION
    Bayesian network implementation.  Influenced by Cecil Huang's and Adnan
    Darwiche's "Inference in Belief Networks: A Procedural Guide," International
    Journal of Approximate Reasoning, 1994.
    
    Copyright 2005, Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
    license file for legal information.

CLASSES
    delegate.Delegate(__builtin__.object)
        RawCPT
            CPT
                BVertex(graph.Vertex, CPT)
            JoinTreePotential
                Cluster(graph.Vertex, JoinTreePotential)
                SepSet(graph.UndirEdge, JoinTreePotential)
            Likelihood
    graph.Graph(delegate.Delegate)
        BNet
        JoinTree
        MoralGraph
    graph.UndirEdge(graph.RawEdge)
        SepSet(graph.UndirEdge, JoinTreePotential)
    graph.Vertex(delegate.Delegate)
        BVertex(graph.Vertex, CPT)
        Cluster(graph.Vertex, JoinTreePotential)
    
    class BNet(graph.Graph)
     |  Method resolution order:
     |      BNet
     |      graph.Graph
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  InitCPTs(self)
     |  
     |  Moralize(self)
     |  
     |  __init__(self, name=None)
     |  
     |  add_e(self, e)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.Graph:
     |  
     |  __str__(self)
     |  
     |  add_v(self, v)
     |      Add and return a vertex.
     |  
     |  all_pairs_sp(self, weight_func=None)
     |      Return a dictionary of shortest path lists for all vertex pairs.
     |      
     |      Keys are (source, destination) tuples.
     |      'weight_func' is a function taking (edge, v1, v2) that returns a weight.
     |      Defaults to e.weight()
     |  
     |  connected_components(self)
     |      Return a list of lists.  Each holds transitively-connected vertices.
     |  
     |  greedy_paths(self, start, goal, weight_func=None)
     |      Return a dict of greedy paths with (start vertex, end vertex) keys.
     |      
     |      Always makes the highest-gain decision.  Will find a path if one exists.
     |      Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
     |      returning a weight.  Defaults to e.weight()
     |  
     |  minimal_span_tree(self, **kargs)
     |      Return minimal spanning 'Tree'.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  shortest_tree(self, start, **kargs)
     |      Return a 'Tree' of shortest paths to all nodes.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from graph.Graph:
     |  
     |  breadth_first_search(start_v)
     |      Return a breadth-first search list of vertices.
     |  
     |  depth_first_search(start_v)
     |      Return a depth-first search list of vertices.
     |  
     |  path_weight(path, weight_func=None)
     |      Return the weight of the path, which is a list of vertices.
     |      
     |      'weight_func' is a function taking (edge, v1, v2) and returning a weight.
     |  
     |  topological_sort(start_v)
     |      Return a topological sort list of vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.Graph:
     |  
     |  intermed_v
     |      List of all vertices with both incoming and outgoing edges.
     |  
     |      <get> = intermed_v(self)
     |  
     |  sink_v
     |      List of all vertices without outgoing edges.
     |  
     |      <get> = sink_v(self)
     |  
     |  src_v
     |      List of all vertices without incoming edges.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class BVertex(graph.Vertex, CPT)
     |  Method resolution order:
     |      BVertex
     |      graph.Vertex
     |      CPT
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  InitCPT(self)
     |  
     |  __cmp__(a, b)
     |      sort by name, any other criterion can be used
     |  
     |  __init__(self, name, nvalues=2)
     |      Name neen't be a string but must be hashable and immutable.
     |      nvalues = number of possible values for variable contained in Vertex
     |      CPT = Conditional Probability Table = Pr(V|Pa(V))
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.Vertex:
     |  
     |  __getstate__(self)
     |      Need to break cycles to prevent recursion blowup in pickle.
     |      
     |      Dump everything except for edges.
     |  
     |  __str__(self)
     |  
     |  attach_e(self, e)
     |      Attach an edge.
     |  
     |  connecting_e(self, v)
     |      List of edges connecting self and other vertex.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.Vertex:
     |  
     |  adjacent_v
     |      Set of adjacent vertices.  Edge direction ignored.
     |  
     |      <get> = adjacent_v(self)
     |  
     |  all_e
     |      All edges.
     |  
     |      <get> = all_e(self)
     |  
     |  in_e
     |      Incoming edges.
     |  
     |      <get> = in_e(self)
     |  
     |  in_v
     |      Set of vertices connected by incoming edges.
     |  
     |      <get> = in_v(self)
     |  
     |  is_intermed
     |      True if vertex has incoming and outgoing edges.
     |  
     |      <get> = is_intermed(self)
     |  
     |  is_sink
     |      True if vertex has no outgoing edges.
     |  
     |      <get> = is_sink(self)
     |  
     |  is_src
     |      True if vertex has no incoming edges.
     |  
     |      <get> = is_src(self)
     |  
     |  out_e
     |      Outgoing edges.
     |  
     |      <get> = out_e(self)
     |  
     |  out_v
     |      Set of vertices connected by outgoing edges.
     |  
     |      <get> = out_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from CPT:
     |  
     |  makecpt(self)
     |      makes a consistent conditional probability distribution
     |      sum(parents)=1
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class CPT(RawCPT)
     |  Method resolution order:
     |      CPT
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, nvalues, parents)
     |  
     |  makecpt(self)
     |      makes a consistent conditional probability distribution
     |      sum(parents)=1
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  __str__(self)
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class Cluster(graph.Vertex, JoinTreePotential)
     |  A Cluster/Clique node for the Join Tree structure
     |  
     |  Method resolution order:
     |      Cluster
     |      graph.Vertex
     |      JoinTreePotential
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  CollectEvidence(self, X=None)
     |      Recursive Collect Evidence,
     |      X is the cluster that invoked CollectEvidence
     |  
     |  ContainsVar(self, v)
     |      v = list of variable name
     |      returns True if cluster contains them all
     |  
     |  DistributeEvidence(self)
     |      Recursive Distribute Evidence,
     |  
     |  MessagePass(self, c)
     |      Message pass from self to cluster c
     |  
     |  NotSetSepOf(self, clusters)
     |      returns True if this cluster is a sepset of any of the clusters
     |  
     |  __init__(self, *Bvertices)
     |  
     |  not_in_s(self, sepset)
     |      set of variables in cluster but not not in sepset, X\S
     |  
     |  other(self, v)
     |      set of all variables contained in cluster except v
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.Vertex:
     |  
     |  __getstate__(self)
     |      Need to break cycles to prevent recursion blowup in pickle.
     |      
     |      Dump everything except for edges.
     |  
     |  __str__(self)
     |  
     |  attach_e(self, e)
     |      Attach an edge.
     |  
     |  connecting_e(self, v)
     |      List of edges connecting self and other vertex.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.Vertex:
     |  
     |  adjacent_v
     |      Set of adjacent vertices.  Edge direction ignored.
     |  
     |      <get> = adjacent_v(self)
     |  
     |  all_e
     |      All edges.
     |  
     |      <get> = all_e(self)
     |  
     |  in_e
     |      Incoming edges.
     |  
     |      <get> = in_e(self)
     |  
     |  in_v
     |      Set of vertices connected by incoming edges.
     |  
     |      <get> = in_v(self)
     |  
     |  is_intermed
     |      True if vertex has incoming and outgoing edges.
     |  
     |      <get> = is_intermed(self)
     |  
     |  is_sink
     |      True if vertex has no outgoing edges.
     |  
     |      <get> = is_sink(self)
     |  
     |  is_src
     |      True if vertex has no incoming edges.
     |  
     |      <get> = is_src(self)
     |  
     |  out_e
     |      Outgoing edges.
     |  
     |      <get> = out_e(self)
     |  
     |  out_v
     |      Set of vertices connected by outgoing edges.
     |  
     |      <get> = out_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from JoinTreePotential:
     |  
     |  Normalise(self)
     |  
     |  __add__(c, s)
     |      sum(X\S)phiX
     |      
     |      marginalise the variables contained in BOTH SepSet AND in Cluster
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class JoinTree(graph.Graph)
     |  Join Tree
     |  
     |  Method resolution order:
     |      JoinTree
     |      graph.Graph
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  ConstructOptimalJTree(self)
     |  
     |  GlobalPropagation(self, start=None)
     |  
     |  GlobalRetraction(self, d)
     |  
     |  GlobalUpdate(self, d)
     |  
     |  Initialization(self)
     |  
     |  Marginalise(self, v)
     |      returns Pr(v), v is a variable name
     |  
     |  ObservationEntry(self, v, val)
     |  
     |  SetObs(self, v, val)
     |      Incorporate new evidence
     |  
     |  UnmarkAllClusters(self)
     |  
     |  __init__(self, name, BNet)
     |      Creates an 'Optimal' JoinTree from a BNet
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.Graph:
     |  
     |  __str__(self)
     |  
     |  add_e(self, e)
     |      Add and return an edge.
     |  
     |  add_v(self, v)
     |      Add and return a vertex.
     |  
     |  all_pairs_sp(self, weight_func=None)
     |      Return a dictionary of shortest path lists for all vertex pairs.
     |      
     |      Keys are (source, destination) tuples.
     |      'weight_func' is a function taking (edge, v1, v2) that returns a weight.
     |      Defaults to e.weight()
     |  
     |  connected_components(self)
     |      Return a list of lists.  Each holds transitively-connected vertices.
     |  
     |  greedy_paths(self, start, goal, weight_func=None)
     |      Return a dict of greedy paths with (start vertex, end vertex) keys.
     |      
     |      Always makes the highest-gain decision.  Will find a path if one exists.
     |      Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
     |      returning a weight.  Defaults to e.weight()
     |  
     |  minimal_span_tree(self, **kargs)
     |      Return minimal spanning 'Tree'.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  shortest_tree(self, start, **kargs)
     |      Return a 'Tree' of shortest paths to all nodes.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from graph.Graph:
     |  
     |  breadth_first_search(start_v)
     |      Return a breadth-first search list of vertices.
     |  
     |  depth_first_search(start_v)
     |      Return a depth-first search list of vertices.
     |  
     |  path_weight(path, weight_func=None)
     |      Return the weight of the path, which is a list of vertices.
     |      
     |      'weight_func' is a function taking (edge, v1, v2) and returning a weight.
     |  
     |  topological_sort(start_v)
     |      Return a topological sort list of vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.Graph:
     |  
     |  intermed_v
     |      List of all vertices with both incoming and outgoing edges.
     |  
     |      <get> = intermed_v(self)
     |  
     |  sink_v
     |      List of all vertices without outgoing edges.
     |  
     |      <get> = sink_v(self)
     |  
     |  src_v
     |      List of all vertices without incoming edges.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class JoinTreePotential(RawCPT)
     |  The potential of each node/Cluster and edge/SepSet in a
     |  Join Tree Structure
     |  
     |  self.cpt = Pr(X)
     |  
     |  where X is the set of variables contained in Cluster or SepSet
     |  self.vertices contains the graph.vertices instances where the variables
     |  come from
     |  
     |  Method resolution order:
     |      JoinTreePotential
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  Normalise(self)
     |  
     |  __add__(c, s)
     |      sum(X\S)phiX
     |      
     |      marginalise the variables contained in BOTH SepSet AND in Cluster
     |  
     |  __init__(self)
     |      self. vertices must be set
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  __str__(self)
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class Likelihood(RawCPT)
     |  Likelihood class
     |  
     |  Method resolution order:
     |      Likelihood
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  AllOnes(self)
     |  
     |  IsRetracted(self, val)
     |      returns True if likelihood is retracted.
     |      
     |      V=v1 in e1. In e2 V is either unobserved, or V=v2
     |  
     |  IsUnchanged(self, val)
     |  
     |  IsUpdated(self, val)
     |  
     |  SetObs(self, i)
     |  
     |  __init__(self, BVertex)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  __str__(self)
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class MoralGraph(graph.Graph)
     |  Method resolution order:
     |      MoralGraph
     |      graph.Graph
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  ChooseVertex(g)
     |      Chooses a vertex from the list according to criterion :
     |      
     |      Selection Criterion :
     |      Choose the node that causes the least number of edges to be added in
     |      step 2b, breaking ties by choosing the nodes that induces the cluster with
     |      the smallest weight 
     |      Implementation in Graph.ChooseVertex()
     |      
     |      The WEIGHT of a node V is the nmber of values V can take (BVertex.nvalues)
     |      The WEIGHT of a CLUSTER is the product of the weights of its
     |          constituent nodes
     |          
     |      Only works with graphs composed of BVertex instances
     |  
     |  Triangulate(G)
     |      Returns a Triangulated graph and its clusters.
     |      
     |      POST :  Graph, list of clusters
     |      
     |      An undirected graph is TRIANGULATED iff every cycle of length
     |      four or greater contains an edge that connects two
     |      nonadjacent nodes in the cycle.
     |      
     |      Procedure for triangulating a graph :
     |      
     |      1. Make a copy of G, call it Gt
     |      2. while there are still nodes left in Gt:
     |          a) Select a node V from Gt according to the criterion 
     |             described below
     |          b) The node V and its neighbours in Gt form a cluster.
     |             Connect of the nodes in the cluster. For each edge added
     |             to Gt, add the same corresponding edge t G
     |          c) Remove V from Gt
     |      3. G, modified by the additional arcs introduces in previous
     |         steps is now triangulated.
     |      
     |      The WEIGHT of a node V is the nmber of values V can take (BVertex.nvalues)
     |      The WEIGHT of a CLUSTER is the product of the weights of its
     |          constituent nodes
     |      
     |      Selection Criterion :
     |      Choose the node that causes the least number of edges to be added in
     |      step 2b, breaking ties by choosing the nodes that induces the cluster with
     |      the smallest weight
     |      Implementation in Graph.ChooseVertex()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.Graph:
     |  
     |  __init__(self, name=None)
     |  
     |  __str__(self)
     |  
     |  add_e(self, e)
     |      Add and return an edge.
     |  
     |  add_v(self, v)
     |      Add and return a vertex.
     |  
     |  all_pairs_sp(self, weight_func=None)
     |      Return a dictionary of shortest path lists for all vertex pairs.
     |      
     |      Keys are (source, destination) tuples.
     |      'weight_func' is a function taking (edge, v1, v2) that returns a weight.
     |      Defaults to e.weight()
     |  
     |  connected_components(self)
     |      Return a list of lists.  Each holds transitively-connected vertices.
     |  
     |  greedy_paths(self, start, goal, weight_func=None)
     |      Return a dict of greedy paths with (start vertex, end vertex) keys.
     |      
     |      Always makes the highest-gain decision.  Will find a path if one exists.
     |      Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
     |      returning a weight.  Defaults to e.weight()
     |  
     |  minimal_span_tree(self, **kargs)
     |      Return minimal spanning 'Tree'.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  shortest_tree(self, start, **kargs)
     |      Return a 'Tree' of shortest paths to all nodes.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from graph.Graph:
     |  
     |  breadth_first_search(start_v)
     |      Return a breadth-first search list of vertices.
     |  
     |  depth_first_search(start_v)
     |      Return a depth-first search list of vertices.
     |  
     |  path_weight(path, weight_func=None)
     |      Return the weight of the path, which is a list of vertices.
     |      
     |      'weight_func' is a function taking (edge, v1, v2) and returning a weight.
     |  
     |  topological_sort(start_v)
     |      Return a topological sort list of vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.Graph:
     |  
     |  intermed_v
     |      List of all vertices with both incoming and outgoing edges.
     |  
     |      <get> = intermed_v(self)
     |  
     |  sink_v
     |      List of all vertices without outgoing edges.
     |  
     |      <get> = sink_v(self)
     |  
     |  src_v
     |      List of all vertices without incoming edges.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class RawCPT(delegate.Delegate)
     |  Method resolution order:
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __init__(self, names, shape)
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  __str__(self)
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class SepSet(graph.UndirEdge, JoinTreePotential)
     |  A Separation Set
     |  
     |  Method resolution order:
     |      SepSet
     |      graph.UndirEdge
     |      graph.RawEdge
     |      JoinTreePotential
     |      RawCPT
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __cmp__(self, other)
     |      first = sepset with largest mass and smallest cost
     |  
     |  __init__(self, name, c1, c2)
     |      SepSet between c1, c2
     |      
     |      c1, c2 are Cluster instances
     |  
     |  __str__(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.UndirEdge:
     |  
     |  enters(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  leaves(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  weight(self, v1, v2)
     |      1 if this edge connects the vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.UndirEdge:
     |  
     |  dest_v
     |      Destination vertices.
     |  
     |      <get> = dest_v(self)
     |  
     |  src_v
     |      Source vertices.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from graph.RawEdge:
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from graph.RawEdge:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from JoinTreePotential:
     |  
     |  Normalise(self)
     |  
     |  __add__(c, s)
     |      sum(X\S)phiX
     |      
     |      marginalise the variables contained in BOTH SepSet AND in Cluster
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawCPT:
     |  
     |  AllOnes(self)
     |  
     |  FindCorrespond(a, b)
     |  
     |  Marginalise(self, varnames)
     |      sum(varnames) self.cpt
     |  
     |  __mul__(a, b)
     |      a keeps the order of its dimensions
     |      
     |      always use a = a*b or b=b*a, not b=a*b
     |  
     |  rand(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)

FUNCTIONS
    MultiplyElements(d)
        multiplies the elements of d between them

DATA
    __author__ = 'Kosta Gaitanis'
    __author_email__ = 'gaitanis@tele.ucl.ac.be'
    __version__ = '0.1'

VERSION
    0.1

AUTHOR
    Kosta Gaitanis


