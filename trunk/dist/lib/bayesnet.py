'''Bayesian network implementation.  Influenced by Cecil Huang's and Adnan
Darwiche's "Inference in Belief Networks: A Procedural Guide," International
Journal of Approximate Reasoning, 1994.

Copyright 2005, Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
license file for legal information.'''

__version__ = '0.1'
__author__ = 'Kosta Gaitanis'
__author_email__ = 'gaitanis@tele.ucl.ac.be'
import types
import graph.graph as graph
import delegate
import numarray as na
import numarray.mlab
from numarray.random_array import randint, seed
from numarray.ieeespecial import setnan, getnan
import copy
from timeit import Timer, time
import profile
import bisect       # for appending elements to a sorted list
import logging

seed()
na.Error.setMode(invalid='ignore')
#logging.basicConfig(level= logging.INFO)

class RawCPT(delegate.Delegate):
    def __init__(self, names, shape):
        # keys = variable names, values = index
        self.p = dict((k,v) for k,v in zip(names, range(len(names))))
        self.Fv = self.p.items()
        self.Fv.sort(cmp=lambda x,y: cmp(x[1],y[1]))    # sort family by index
        self.Fv = [f[0] for f in self.Fv]   # list of names of vars ordered by index
        # self.Fv contains the names of the Family of V
        # only use self.Fv for iterating over dimensions... not self.p.items()
        
        
        self.cpt = na.ones(shape, type='Float32')
        
    def setCPT(self, cpt):
        ''' put values into self.cpt'''
        self.cpt = na.array(cpt, shape=self.cpt.shape, type='Float32')
        
    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = na.mlab.rand(*self.cpt.shape)

    def Marginalise(self, varnames):
        """ sum(varnames) self.cpt """
        temp = self.cpt
        ax = [self.p[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        return temp

    def AllOnes(self):
        self.cpt = na.ones(self.cpt.shape, type='Float32')

    def Uniform(self):
        ' Uniform distribution '
        N = len(self.cpt.flat)
        self.cpt = na.array([1.0/N for n in range(N)], shape = self.cpt.shape, type='Float32')

    def __mul__(a,b):
        """
        a keeps the order of its dimensions
        
        always use a = a*b or b=b*a, not b=a*b
        """
        
        aa,bb = a.cpt, b.cpt
        
        correspondab = a.FindCorrespond(b)
        
        while aa.rank < len(correspondab): aa = aa[..., na.NewAxis]
        while bb.rank < len(correspondab): bb = bb[..., na.NewAxis]

        bb = na.transpose(bb, correspondab)

        return aa*bb

    def __str__(self): return str(self.cpt)

    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.  Index can be a string as in PBNT ('1,:,1'), a dictionary of var name:value pairs, or pure numbers as in the standard way of accessing a numarray array array[1,:,1]
        """
        if isinstance(index, types.DictType):
            strIndex = self._strIndexFromDict(index)
            return self._getStrIndex(strIndex)
        if isinstance(index, types.StringType):
            return self._getStrIndex(index)
        return self._getNumItem(index)
    
    def _getStrIndex(self, index):
        """ Helper function for __getitem__, takes a string as index.
        """
        return eval("self.cpt["+index+"]")
    
    def _getNumItem(self, index):
        """ Helper function for __getitem__, index are numbers as in array[1,:,1]
        """
        return self.cpt[index]
    
    def __setitem__(self, index, value):
        """ Overload array-style indexing and setting behaviour, as in __getitem__ this will take a dictionary, string, or normal set of numbers as index
        """
        if isinstance(index, types.DictType):
            strIndex = self._strIndexFromDict(index)
            return self._setStrIndex(strIndex, value)
        if isinstance(index, types.StringType):
            return self._setStrIndex(index, value)
        return self._setNumItem(index, value)
    
    def _setStrIndex(self, index, value):
        exec "self.cpt["+index+"]=" + repr(value)
    
    def _setNumItem(self, index, value):
        self.cpt[index] = value
        return
    
    def _strIndexFromDict(self, d):
        index = '';
        for vname in self.Fv:
            if d.has_key(vname):
                index += repr(d[vname]) + ','
            else:
                index += ':,'
        return index[:-1]

    def FindCorrespond(a,b):
        correspond = []
        k = len(b.p)
        for p in a.Fv:   #p=var name
            if b.p.has_key(p): correspond.append(b.p[p])
            else:
                correspond.append(k)
                k += 1

        for p in b.Fv:
            if not a.p.has_key(p):
                correspond.append(b.p[p])
                
        return correspond

    def Printcpt(self):
        string =  str(self.cpt) + '\nshape:'+str(self.cpt.shape)+'\nFv:'+str(self.Fv)+'\nsum : ' +str(na.sum(self.cpt.flat))
        print string



class CPT(RawCPT):
    def __init__(self, nvalues, parents):
        names = [self.name]
        names.extend([p.name for p in parents])
        shape = [p.nvalues for p in parents]
        shape.insert(0, nvalues)
        
        RawCPT.__init__(self, names, shape)

    def makecpt(self):
        """
        makes a consistent conditional probability distribution
        sum(parents)=1
        """
        shape = self.cpt.shape
        self.cpt.shape = (shape[0], MultiplyElements(shape[1:]))
        self.cpt /= na.sum(self.cpt, axis=0)
        self.cpt.shape = shape

def MultiplyElements(d):
    "multiplies the elements of d between them"
    #this one is the fastest
    nel = 1
    for x in d: nel = nel * x

    return nel

class BVertex(graph.Vertex, CPT, delegate.Delegate):
    def __init__(self, name, nvalues = 2, observed = False):
        '''
        Name neen't be a string but must be hashable and immutable.
        nvalues = number of possible values for variable contained in Vertex
        CPT = Conditional Probability Table = Pr(V|Pa(V))
        '''
        graph.Vertex.__init__(self, name)
        self.nvalues = nvalues
        
        self.observed = observed

    def InitCPT(self):
        ''' Initialise CPT, all edges must be added '''
        CPT.__init__(self, self.nvalues, sorted(self.in_v))


    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function
    def __cmp__(a,b):
        ''' sort by name, any other criterion can be used '''
        return cmp(a.name, b.name)

class JoinTreePotential(RawCPT):
    """
    The potential of each node/Cluster and edge/SepSet in a
    Join Tree Structure
    
    self.cpt = Pr(X)
    
    where X is the set of variables contained in Cluster or SepSet
    self.vertices contains the graph.vertices instances where the variables
    come from
    """
    def __init__(self):
        """ self. vertices must be set """
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        RawCPT.__init__(self, names, shape)


    def __add__(c,s):
        """
        sum(X\S)phiX

        marginalise the variables contained in BOTH SepSet AND in Cluster
        """
        var = set(v.name for v in c.vertices) - set(v.name for v in s.vertices)
        return c.Marginalise(var)

    # result has the same variable order as c (cluster) (without some variables)
    # result has also the same variable order as s (SepSet)
    # this is because variables are sorted at initialisation
    
    def Normalise(self):
        self.cpt = na.divide(self.cpt, na.sum(self.cpt.flat), self.cpt)


class Cluster(graph.Vertex, JoinTreePotential):
    """
    A Cluster/Clique node for the Join Tree structure
    """
    def __init__(self, *Bvertices):
        
        self.vertices = [v for v in Bvertices]    # list of vertices contained in this cluster
        self.vertices.sort()    # sort list, much better for math operations
        
        name = ''
        for v in self.vertices: name += v.name
        
        JoinTreePotential.__init__(self)        # self.psi = ones
        graph.Vertex.__init__(self, name)
        
        # weight
        self.weight = reduce(lambda x,y: x*y, [v.nvalues for v in self.vertices])
        
    def NotSetSepOf(self, clusters):
        """
        returns True if this cluster is not a sepset of any of the clusters
        """
        for c in clusters:
            count = 0
            for v in self.vertices:
                if v.name in [cv.name for cv in c.vertices]: count += 1
            if count == len(self.vertices): return False

            return True

    def ContainsVar(self, v):
        """
        v = list of variable name
        returns True if cluster contains them all
        """
        for vv in v:
            if self.name.find(vv)==-1: return False

        return True

    def not_in_s(self, sepset):
        """ set of variables in cluster but not not in sepset, X\S"""
        return set(v.name for v in self.vertices) - set(v.name for v in sepset.vertices)

    def other(self,v):
        """ set of all variables contained in cluster except v, only one at a time... """
        return set(vv.name for vv in self.vertices) - set((v,))

    def MessagePass(self, c):
        """ Message pass from self to cluster c """
        logging.debug('Message Pass from '+ str(self)+' to '+str(c))
        # c must be connected to self by a sepset
        e = self.connecting_e(c)    # sepset that connect the two clusters
        if not e: raise 'Clusters '+str(self)+' and '+ str(c)+ ' are not connected'
        e = e[0]    # only one edge should connect 2 clusters
        
        # Projection
        oldphiR = copy.copy(e.cpt)  # oldphiR = phiR
        newphiR = self+e            # phiR = sum(X/R)phiX
        
        # Absorption
        e.cpt = newphiR/oldphiR         ## WARNING, division by zero, avoided using na.Error.setMode(invalid='ignore')
        e.cpt[getnan(e.cpt)] = 0        # replace -1.#IND by 0
        
        c.cpt = c*e
        e.cpt = newphiR

    def CollectEvidence(self, X=None):
        """
        Recursive Collect Evidence,
        X is the cluster that invoked CollectEvidence
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: v.CollectEvidence(self)

        if not X == None: self.MessagePass(X)

    def DistributeEvidence(self):
        """
        Recursive Distribute Evidence,
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: self.MessagePass(v)
            
        for v in self.in_v:
            if not v.marked: v.DistributeEvidence()


class SepSet(graph.UndirEdge, JoinTreePotential):
    """
    A Separation Set
    """
    def __init__(self, name, c1, c2):
        """
        SepSet between c1, c2
        
        c1, c2 are Cluster instances
        """
        # find intersection between c1 and c2
        self.vertices = list(set(c1.vertices) & set(c2.vertices))
        self.vertices.sort()
        
        self.label = ''
        for v in self.vertices: self.label += v.name
        
        JoinTreePotential.__init__(self)        # self.psi = ones
        graph.UndirEdge.__init__(self, name, c1, c2)
        
        # mass and cost
        self.mass = len(self.vertices)
        self.cost = self._v[0].weight + self._v[1].weight
        
        
    def __str__(self):
        # this also prints mass and cost
        #return '%s: %s -- %s -- %s, mass: %s, cost: %s' % (str(self.name), str(self._v[0]),
        #       str(self.label), str(self._v[1]), str(self.mass), str(self.cost))
        return '%s: %s -- %s -- %s' % (str(self.name), str(self._v[0]),
                                       str(self.label), str(self._v[1]))
    
    def __cmp__(self, other):
        """ first = sepset with largest mass and smallest cost """
        comp = cmp(other.mass, self.mass ) # largest mass
        if comp == 0:
            return cmp(self.cost, other.cost ) # smallest cost
        else: return comp


#=======================================================================

class MoralGraph(graph.Graph):
    def ChooseVertex(self):
        """
        Chooses a vertex from the list according to criterion :
        
        Selection Criterion :
        Choose the node that causes the least number of edges to be added in
        step 2b, breaking ties by choosing the nodes that induces the cluster with
        the smallest weight
        Implementation in Graph.ChooseVertex()
        
        The WEIGHT of a node V is the nmber of values V can take (BVertex.nvalues)
        The WEIGHT of a CLUSTER is the product of the weights of its
        constituent nodes
        
        Only works with graphs composed of BVertex instances
        """
        vertices = self.all_v
        # for each vertex, check how many edges will be added
        edgestoadd = [0 for v in vertices]
        clusterweight = [1 for v in vertices]
        
        for v,i in zip(vertices,range(len(vertices))):
            cluster = [a.name for a in v.adjacent_v]
            cluster.append(v.name)
            clusterleft = copy.copy(cluster)
            
            # calculate clusterweight
            for c in cluster:
                clusterweight[i] *= self.v[c].nvalues
                
            for v1 in cluster:
                clusterleft.pop(0)
                for v2 in clusterleft:
                    if not v1 in [a.name for a in self.v[v2].adjacent_v]:
                        edgestoadd[i] += 1

        # keep only the smallest ones, the index
        minedges = min(edgestoadd)
        mini = [vertices[i] for e,i in zip(edgestoadd,range(len(edgestoadd))) if e == minedges]
        
        # from this list, pick the one that has the smallest clusterweight = nvalues
        # this only works with BVertex instances
        v = mini[na.argmin([clusterweight[vertices.index(v)] for v in mini])]
        
        return v

    def Triangulate(G):
        """
        Returns a Triangulated graph and its clusters.
        
        POST :  Graph, list of clusters
        
        An undirected graph is TRIANGULATED iff every cycle of length
        four or greater contains an edge that connects two
        nonadjacent nodes in the cycle.
        
        Procedure for triangulating a graph :
        
        1. Make a copy of G, call it Gt
        2. while there are still nodes left in Gt:
        a) Select a node V from Gt according to the criterion
        described below
        b) The node V and its neighbours in Gt form a cluster.
        Connect of the nodes in the cluster. For each edge added
        to Gt, add the same corresponding edge t G
        c) Remove V from Gt
        3. G, modified by the additional arcs introduces in previous
        steps is now triangulated.
        
        The WEIGHT of a node V is the nmber of values V can take (BVertex.nvalues)
        The WEIGHT of a CLUSTER is the product of the weights of its
        constituent nodes
        
        Selection Criterion :
        Choose the node that causes the least number of edges to be added in
        step 2b, breaking ties by choosing the nodes that induces the cluster with
        the smallest weight
        Implementation in Graph.ChooseVertex()
        """
        logging.info('Triangulating Tree and extracting Clusters')
        # don't touch this graph, create a copy of it
        G = copy.deepcopy(G)
        G.name = 'Triangulised ' + str(G.name)
        
        # make a copy of G
        G2 = copy.deepcopy(G)
        G2.name = 'Copy of '+ G.name
        
        clusters = []
        
        while len(G2.v):
            v = G2.ChooseVertex()
            #logging.debug('Triangulating: chosen '+str(v))
            cluster = list(v.adjacent_v)
            cluster.append(v)
            
            #logging.debug('Cluster: '+str([str(c) for c in cluster]))

            c = Cluster(*cluster)
            if c.NotSetSepOf(clusters):
                #logging.debug('Appending cluster')
                clusters.append(c)

            clusterleft = copy.copy(cluster)

            for v1 in cluster:
                clusterleft.pop(0)
                for v2 in clusterleft:
                    if not (v1 in v2.adjacent_v):
                        v1g = G.v[v1.name]
                        v2g = G.v[v2.name]
                        G.add_e(graph.UndirEdge(max(G.e.keys())+1,v1g,v2g))
                        G2.add_e(graph.UndirEdge(max(G2.e.keys())+1,v1,v2))
                        
            # remove from G2
            del G2.v[v.name]


            return G, clusters

#=======================================================================

class BNet(graph.Graph):
    log = logging.getLogger('BNet')
    log.setLevel(logging.ERROR)
    def __init__(self, name = None):
        graph.Graph.__init__(self, name)

    def add_e(self, e):
        if e.__class__.__name__ == 'DirEdge':
            graph.Graph.add_e(self, e)
        else:
            raise "All edges should be directed"

    def Moralize(self):
        logging.info('Moralising Tree')
        G = MoralGraph(name = 'Moralized ' + str(self.name))
        
        # for each vertice, create a corresponding vertice
        for v in self.v.values():
            G.add_v(BVertex(v.name, v.nvalues))

        # create an UndirEdge for each DirEdge in current graph
        for e in self.e.values():
            # get corresponding vertices in G (and not in self!)
            v1 = G.v[e._v[0].name]
            v2 = G.v[e._v[1].name]
            G.add_e(graph.UndirEdge(len(G.e), v1, v2))

        # add moral edges
        # connect all pairs of parents for each node
        for v in self.v.values():
            # get parents for each vertex
            self.log.debug('Node : ' + str(v))
            parents = [G.v[p.name] for p in list(v.in_v)]
            self.log.debug('parents: ' + str([p.name for p in list(v.in_v)]))
            
            for p1,i in zip(parents, range(len(parents))):
                for p2 in parents[i+1:]:
                    if not p1.connecting_e(p2):
                        self.log.debug('adding edge '+ str(p1) + ' -- ' + str(p2))
                        G.add_e(graph.UndirEdge(len(G.e), p1, p2))

        return G
    @graph._roprop('List of observed vertices.')
    def observed(self):
        return [v for v in self.v.values() if v.observed]
    
    def InitCPTs(self):
        for v in self.v.values(): v.InitCPT()

    def RandomizeCPTs(self):
        for v in self.v.values():
            v.rand()
            v.makecpt()

#========================================================================
class Likelihood(RawCPT):
    """ Likelihood class """
    def __init__(self, BVertex):
        RawCPT.__init__(self, (BVertex.name,), BVertex.nvalues)
        self.v = BVertex
        self.AllOnes()      # -1 = unobserved
        
    def AllOnes(self):
        self.val = -1
        self.cpt = na.ones(self.cpt.shape, type='Float32')
        
    def SetObs(self, i):
        if i == -1: self.AllOnes()
        else:
            self.cpt = na.zeros(self.cpt.shape, type='Float32')
            self.cpt[i] = 1
            self.val = i

    def IsRetracted(self, val):
        """
        returns True if likelihood is retracted.
        
        V=v1 in e1. In e2 V is either unobserved, or V=v2
        """
        return (self.val != -1 and self.val != val)
    
    def IsUnchanged(self, val):
        return self.val == val
    
    def IsUpdated(self, val):
        return (self.val == -1 and val != -1)

#========================================================================

class JoinTree(graph.Graph):
    """ Join Tree """
    
    def __init__(self, BNet):
        """Creates an 'Optimal' JoinTree from a BNet """
        logging.info('Creating JoinTree for '+str(BNet.name))
        graph.Graph.__init__(self, 'JT: ' + str(BNet.name))
        self.BNet = BNet
        
        # key = variable name, value = cluster instance containing variable
        self.clusterdict = dict()
        
        # likelihood dictionary, key = var name, value = likelihood instance
        self.likelihoods = [Likelihood(v) for v in self.BNet.observed]
        self.likedict = dict((v.name, l) for v,l in zip(self.BNet.observed, self.likelihoods))
        
        logging.info('Constructing Optimal Tree')
        self.ConstructOptimalJTree()
        self.Initialization()
        self.GlobalPropagation()
        
    def ConstructOptimalJTree(self):
        # Moralize Graph
        Gm = self.BNet.Moralize()
        
        
        # triangulate graph and extract clusters
        Gt, clusters = Gm.Triangulate()
        
        # Create Clusters for this JoinTree
        for c in clusters: self.add_v(c)
        
        logging.info('Connecting Clusters Optimally')
        # Create candidate SepSets
        # one candidate sepset for each pair of clusters
        candsepsets = []
        clustersleft = copy.copy(clusters)
        for c1 in clusters:
            clustersleft.pop(0)
            for c2 in clustersleft:
                candsepsets.append(SepSet(len(candsepsets),c1,c2))

        # remove all edges added to clusters by creating candidate sepsets
        for c in clusters:  c._e=[]
        
        # sort sepsets, first = largest mass, smallest cost
        candsepsets = sorted(candsepsets)
        
        # Create trees
        # initialise = one tree for each cluster
        # key = cluster name, value = tree index
        trees = dict([(c.name, i) for c,i in zip(clusters,range(len(clusters)))])

        # add SepSets according to criterion, iff the two clusters connected
        # are on different trees
        for s in candsepsets:
            # if on different trees
            if trees[s._v[0].name] != trees[s._v[1].name]:
                # add SepSet
                self.add_e(SepSet(len(self.e),s._v[0],s._v[1]))
                
                # merge trees
                oldtree = trees[s._v[1].name]
                for t in trees.items():
                    if t[1] == oldtree: trees[t[0]] = trees[s._v[0].name]

            del s
            # end if n-1 sepsets have been added
            if len(self.e) == len(clusters) - 1: break

    def Initialization(self):
        logging.info('Initialising Potentials for clusters and SepSets')
        # for each cluster and sepset X, set phiX = 1
        for c in self.v.values():   c.AllOnes()         # PhiX = 1
        for s in self.e.values():   s.AllOnes()
        
        # assign a cluster to each variable
        # multiply cluster potential by v.cpt,
        for v in self.BNet.v.values():
            for c in self.v.values():
                if c.ContainsVar(v.Fv):
                    self.clusterdict[v.name] = c
                    v.parentcluster = c
                    c.cpt = c*v         # phiX = phiX*Pr(V|Pa(V))
                    break   # stop c loop, continue with next v

        # set all likelihoods to ones
        for l in self.likelihoods: l.AllOnes()


    def UnmarkAllClusters(self):
        for v in self.v.values(): v.marked = False

    def GlobalPropagation(self, start = None):
        if start == None: start = self.v.values()[0]    # first cluster found
        
        logging.info('Global Propagation, starting at :'+ str(start))
        logging.info('      Collect Evidence')
        
        self.UnmarkAllClusters()
        start.CollectEvidence()
        
        logging.info('      Distribute Evidence')
        self.UnmarkAllClusters()
        start.DistributeEvidence()
        
    def Marginalise(self, v):
        """ returns Pr(v), v is a variable name"""
        
        # find a cluster containing v
        # v.parentcluster is a convenient choice, can make better...
        c = self.clusterdict[v]
        res = c.Marginalise(c.other(v))
        
        # normalize
        return na.divide(res,na.sum(res.flat))
    
    def SetObs(self, v,val):
        """ Incorporate new evidence """
        logging.info('Incorporating Observations')
        temp = dict((vi,vali) for vi,vali in zip(v,val))
        # add any missing vales:
        for vv in self.BNet.v.values():
            if not temp.has_key(vv.name): temp[vv.name] = -1
                    
        # Check for Global Retraction, or Global Update
        retraction = False
        for vv in self.BNet.v.values():
            if self.likedict[vv.name].IsRetracted(temp[vv.name]):
                retraction = True
            elif self.likedict[vv.name].IsUnchanged(temp[vv.name]):
                del temp[vv.name]
                # remove any unobserved variables
            elif temp[vv.name] == -1: del temp[vv.name]
            
            # initialise
            if retraction: self.GlobalRetraction(temp)
            else: self.GlobalUpdate(temp)
            
    def SetFinding(self, v):
        ''' v becomes True (v=1), all other observed variables are false '''
        logging.info('Set finding, '+ str(v))
        temp = dict((vi.name,0) for vi in self.BNet.observed)
        if temp.has_key(v): temp[v] = 1
        else: raise str(v)+''' is not observable or doesn't exist'''
        
        
        self.Initialization()
        self.ObservationEntry(temp.keys(),temp.values())
        self.GlobalPropagation()
        
    def GlobalUpdate(self, d):
        logging.info('Global Update')
        self.ObservationEntry(d.keys(),d.values())
        
        # check if only one Cluster is updated. If true, only DistributeEvidence
        startcluster = set()
        for v in d.keys():
            startcluster.add(self.BNet.v[v].parentcluster)
            
            if len(startcluster) == 1:
                # all variables that have changed are in the same cluster
                # perform DistributeEvidence only
                logging.info('distribute only')
                self.UnmarkAllClusters()
                startcluster.pop().DistributeEvidence()
            else:
                # perform global propagation
                self.GlobalPropagation()
    
    def GlobalRetraction(self,d ):
        logging.info('Global Retraction')
        self.Initialization()
        self.ObservationEntry(d.keys(),d.values())
        self.GlobalPropagation()
        
    def ObservationEntry(self, v, val):
        logging.info('Observation Entry')
        for vv,vval in zip(v,val):
            c = self.clusterdict[vv]     # cluster containing likelihood, same as v
            l = self.likedict[vv]    
            l.SetObs(vval)
            c.cpt = c*l

    def MargAll(self):
        for v in self.BNet.v.values():
            if not v.observed: print v, self.Marginalise(v.name)
        for v in self.BNet.observed:
            print v, self.Marginalise(v.name)
                
    def Print(self):
        for c in self.v.values():
            print c
            print c.cpt
            print c.cpt.shape
            print na.sum(c.cpt.flat)
            
        for c in self.e.values():
            print c
            print c.cpt
            print c.cpt.shape
            print na.sum(c.cpt.flat)
            

class MCMCEngine(graph.Graph):
        """ Implementation of MCMC (aka Gibbs Sampler), as described on p.517 of Russell and Norvig
        """
        def __init__(self, name, BNet, cut=100):
            graph.Graph.__init__(name)
            self.BNet = Bnet
            self.cut = cut
            self.evidence = {}
        
        def SetObs(self, v, val):
            """ Incorporate new evidence 
            """
            self.evidence = dict((vi,vali) for vi,vali in zip(v,val))
                
        def Marginalise(self, v, N):
            """ Compute the Pr(v) where v is a variable name, N is the number of iterations of MCMC to perform.
            """
            # the return distribution
            vDist = RawCPT(v, (self.BNet.v[v].nvalues,1))
            nonEvidence = []
            # find out which values are not specified in evidence
            for vv in self.BNet.v.values():
                if not self.evidence.has_key(vv.name): nonEvidence.append(vv.name)
            # CHECK: this copy is deep enough
            state = copy.copy(self.evidence)
            for vname in nonEvidence:
                # CHECK: legal values are 0 - nvalues-1
                state[vname] = random.randint(0, self.BNet.v[vname].nvalues-1)
            for i in range(N):
                if i > self.cut:
                        # RESEARCH: how to index into CPT
                        vDist[state[v]] += 1
                for vname in nonEvidence:
                        state[vname] = self.sampleGivenMB(self.BNet.v[vname], state)
            # CHECK: how do i normalize? for now assume that RawCPT now has a makecpt() method
            vDist.makecpt()
            return vDist
        
        def sampleGivenMB(self, v, state):
            MBval = RawCPT(v.name, (v.nvalues,1))
            children = v.out_v()
            index = {}
            for vert in [v,v.in_v()]:
                    index[vert.name] = state[vert.name]
            childrenAndIndex = []
            for child in children:
                cindex = {}
                for cvert in [child,child.in_v()]:
                    cindex[cvert.name] = state[cvert.name]       
                childrenAndIndex.append((child,cindex))
            #OPTIMIZE: could vectorize this code
            for value in range(v.nvalues):
                index[v.name] = value
                cindex[v.name] = value
                MBval[value] = v[index]
                for child,cindex in childrenAndIndex:
                    MBval[value] *= child[cindex]
            MBval.makecpt()                
            val = MBval.sample()
            return val

if __name__=='__main__':
    ''' Water Sprinkler example '''
    
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(nm,2,True)) for nm in 'c s r w'.split()]
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))
        
    G.InitCPTs()
    
    c.setCPT([0.5, 0.5])
    s.setCPT([0.5, 0.9, 0.5, 0.1])
    r.setCPT([0.8, 0.2, 0.2, 0.8])
    w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
    
    
    print G
    
    JT = JoinTree(G)
    
    JT.SetObs(['w','r'],[1,1])
    JT.MargAll()

if __name__=='__mains__':
    G = BNet('Bnet')
    
    a, b, c, d, e, f, g, h = [G.add_v(BVertex(nm)) for nm in 'a b c d e f g h'.split()]
    a.nvalues = 3
    e.nvalues = 4
    c.nvalues = 5
    g.nvalues = 6
    for ep in [(a, b), (a,c), (b,d), (d,f), (c,e), (e,f), (c,g), (e,h), (g,h)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))
        
    G.InitCPTs()
    G.RandomizeCPTs()
    
    
    JT = JoinTree(G)
    
    print JT

    
    print JT.Marginalise('c')
    
    JT.SetObs(['b'],[1])
    print JT.Marginalise('c')
    
    #JT.SetObs(['b','a'],[1,2])
    #print JT.Marginalise('c')
    
    #JT.SetObs(['b'],[1])
    #print JT.Marginalise('c')
    
    logging.basicConfig(level=logging.CRITICAL)
    
    def RandomObs(JT, G):
        for N in range(100):
            n = randint(len(G.v))
            
            obsn = []
            obs = []
            for i in range(n):
                v = randint(len(G.v))
                vn = G.v.values()[v].name
                if vn not in obsn:
                    obsn.append(vn)
                    val = randint(G.v[vn].nvalues)
                    obs.append(val)
                    
            JT.SetObs(obsn,obs)
            
    t = time.time()
    RandomObs(JT,G)
    t = time.time() - t
    print t
    
    #profile.run('''JT.GlobalPropagation()''')
                