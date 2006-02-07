import numarray as na
from numarray.ieeespecial import getnan
import logging
import copy
import unittest
import types

#Library Specific Modules
import graph
import bayesnet
import distributions
from potentials import DiscretePotential

logging.basicConfig(level= logging.DEBUG)
class InferenceEngine(graph.Graph):
    def __init__(self, BNet):
        graph.Graph.__init__(self, self.name)
        self.BNet = BNet
        self.evidence = {}
    
    def SetObs(self, v, val):
            """ Incorporate new evidence 
            """
            self.evidence = dict((vi,vali) for vi,vali in zip(v,val))
    
    def Marginalise(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at Child level'
    
    def MarinaliseFamily(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at appropriate level'
    
    def LearnMLParams(self, cases):
        """ Learn and set the parameters of the network to the ML estimate contained in cases.  Warning: this is destructive, it does not take any prior parameters into account. Assumes that all evidence is specified.
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.initializeCounts()
        for case in cases:
            #CHECK: all vertices in case are set
            assert(set(case.keys()) == set(self.BNet.v.keys())), "Not all values of 'case' are set"
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.incrCounts(case)
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.Normalize()
    
    def LearnEMParams(self, cases, iterations=10):
        isConverged = False
        iter = 0
        while not isConverged and iter < iterations:
            LL = self.EMStep(cases)
            iter += 1
            isConverged = EMConverged(LL, prevLL, thresh)
    
    def EMStep(self, cases):
        for v in self.BNet.v.values():
            v.distribution.initializeCounts()
        for case in cases:
            self.SetObs(case)
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.counts += self.MarginaliseFamily(v.name)
        for v in self.BNet.v.values():
            v.ResetParameters()
        
class Cluster(graph.Vertex):
    """
    A Cluster/Clique node for the Join Tree structure
    """
    def __init__(self, Bvertices):
        
        self.vertices = [v for v in Bvertices]    # list of vertices contained in this cluster
        #self.vertices.sort()    # sort list, much better for math operations
        
        name = ''
        for v in self.vertices: name += v.name
        graph.Vertex.__init__(self, name)
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)
        #### Debug ####
        if not len(names) == len(self.vertices):
            lkj = 1
        
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
        success = True
        for vv in v:
            if not vv.name in self.potential.names: 
                success = False
                break
        return success

    def not_in_s(self, sepset):
        """ set of variables in cluster but not not in sepset, X\S"""
        return set(self.potential.names) - set(sepset.potential.names)
        #return set(v.name for v in self.vertices) - set(v.name for v in sepset.vertices)

    def other(self,v):
        """ set of all variables contained in cluster except v, only one at a time... """
        allVertices = set(vv.name for vv in self.vertices)
        if isinstance(v, (list, set, tuple)):
            setV = set(v)
        else:
            setV = set((v,))
        return allVertices - setV

    def MessagePass(self, c):
        """ Message pass from self to cluster c """
        ####################################################
        ### This part must be revisioned !!!!!!!!!
        ####################################################
        logging.debug('Message Pass from '+ str(self)+' to '+str(c))
        # c must be connected to self by a sepset
        e = self.connecting_e(c)    # sepset that connects the two clusters
        if not e: raise 'Clusters '+str(self)+' and '+ str(c)+ ' are not connected'
        e = e[0]    # only one edge should connect 2 clusters
        
        # Projection
        
        oldphiR = e.potential                           # oldphiR = phiR
        newphiR = self.potential.Marginalise(e.potential.names)            # phiR = sum(X/R)phiX
        e.potential = copy.copy(newphiR)

        # Absorption
        newphiR/oldphiR
        
        c.potential * newphiR


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


class SepSet(graph.UndirEdge):
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
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)        # self.psi = ones
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

    def Triangulate(self):
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
        Gt = copy.deepcopy(self)
        Gt.name = 'Triangulised ' + str(Gt.name)
        
        # make a copy of Gt
        G2 = copy.deepcopy(Gt)
        G2.name = 'Copy of '+ Gt.name
    
        clusters = []
    
        while len(G2.v):
            v = G2.ChooseVertex()
            #logging.debug('Triangulating: chosen '+str(v))
            cluster = list(v.adjacent_v)
            cluster.append(v)
        
            #logging.debug('Cluster: '+str([str(c) for c in cluster]))
        
            c = Cluster(cluster)
            if c.NotSetSepOf(clusters):
                #logging.debug('Appending cluster')
                clusters.append(c)
            
            clusterleft = copy.copy(cluster)
            
            for v1 in cluster:
                clusterleft.pop(0)
            for v2 in clusterleft:
                if not (v1 in v2.adjacent_v):
                    v1g = Gt.v[v1.name]
                    v2g = Gt.v[v2.name]
                    Gt.add_e(graph.UndirEdge(max(Gt.e.keys())+1,v1g,v2g))
                    G2.add_e(graph.UndirEdge(max(G2.e.keys())+1,v1,v2))
                    
            # remove from G2
            del G2.v[v.name]
        return Gt, clusters
       
#=======================================================================
#========================================================================
class Likelihood(distributions.MultinomialDistribution):
    """ Likelihood class """
    def __init__(self, BVertex):
        distributions.MultinomialDistribution.__init__(self, BVertex)
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

class JoinTree(InferenceEngine):
    """ Join Tree """
    
    def __init__(self, BNet):
        """Creates an 'Optimal' JoinTree from a BNet """
        logging.info('Creating JoinTree for '+str(BNet.name))
        self.name = 'JT: ' + str(BNet.name)
        InferenceEngine.__init__(self, BNet)
        
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
        for c in self.v.values():   c.potential.AllOnes()         # PhiX = 1
        for s in self.e.values():   s.potential.AllOnes()
        
        # assign a cluster to each variable
        # multiply cluster potential by v.cpt,
        for v in self.BNet.all_v:
            for c in self.all_v:
                if c.ContainsVar(v.family):
                    self.clusterdict[v.name] = c
                    v.parentcluster = c
                    # in place multiplication!
                    c.potential * v.distribution         # phiX = phiX*Pr(V|Pa(V)) (special in-place op)

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
        res = c.potential.Marginalise(v)
        res.Normalise()
        return res
    
    def MarginaliseFamily(self, v):
        """ returns Pr(fam(v)), v is a variable name
        """
        c = self.clusterdict[v]
        res = c.Marginalise(c.other(self.BNet.v[v].family))
        return res.Normalise()
    
    def SetObs(self, v,val):
        """ Incorporate new evidence """
        logging.info('Incorporating Observations')
        temp = dict((vi,vali) for vi,vali in zip(v,val))
        # add any missing vales:
        for vv in self.BNet.v.values():
            if not temp.has_key(vv.name): temp[vv.name] = -1
            
        self.EnterEvidence(temp)
    
    def EnterEvidence(self, ev):
        # Check for Global Retraction, or Global Update
        retraction = False
        for vv in self.BNet.v.values():
            if self.likedict[vv.name].IsRetracted(ev[vv.name]):
                retraction = True
            elif self.likedict[vv.name].IsUnchanged(ev[vv.name]):
                del ev[vv.name]
                # remove any unobserved variables
            elif ev[vv.name] == -1: del ev[vv.name]
            
            # initialise
            if retraction: self.GlobalRetraction(ev)
            else: self.GlobalUpdate(ev)
            
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
            c.potential *= l

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
        def __init__(self, BNet, cut=100):
            self.name = 'MCMC: ' + str(BNet.name)
            InferenceEngine.__init__(self, BNet)
            self.cut = cut
        
        def SetObs(self, v, val):
            pass
                
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

        
        

class InferenceEngineTestCase(unittest.TestCase):
    """ An abstract set of inference test cases.  Basically anything that is similar between the different inference engines can be implemented here and automatically applied to lower engines.  For example, we can define the learning tests here and they shouldn't have to be redefined for different engines.
    """
    def setUp(self):
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setCPT([0.5, 0.5])
        s.setCPT([0.5, 0.9, 0.5, 0.1])
        r.setCPT([0.8, 0.2, 0.2, 0.8])
        w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G
    
    def testLearning(self):
        """ Sample network and then learn parameters and check that they are relatively close to original.
        """
        ev = self.engine.BNet.Sample(n=1000)
        #Remember what the old CPTs looked like and keep track of original dimension order
        cCPT = self.c.distribution.cpt.copy()
        cdims = self.c.distribution.names_list
        sCPT = self.s.distribution.cpt.copy()
        sdims = self.s.distribution.names_list
        rCPT = self.r.distribution.cpt.copy()
        rdims = self.r.distribution.names_list
        wCPT = self.w.distribution.cpt.copy()
        wdims = self.w.distribution.names_list
        self.engine.LearnMLParams(ev)
        #reorder dims to match original
        self.c.distribution.transpose(cdims)
        self.s.distribution.transpose(sdims)
        self.r.distribution.transpose(rdims)
        self.w.distribution.transpose(wdims)
        # Check that they match original parameters
        assert(na.allclose(cCPT,self.c.distribution.cpt,atol=.1) and \
               na.allclose(sCPT,self.s.distribution.cpt,atol=.1) and \
               na.allclose(rCPT,self.r.distribution.cpt,atol=.1) and \
               na.allclose(wCPT,self.w.distribution.cpt,atol=.1)),\
              "CPTs were more than atol=.1 apart"
    
class MCMCTestCase(InferenceEngineTestCase):
    """ MCMC unit tests.
    """
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = MCMCEngine(self.BNet,cut=100)
        
    def testUnobserved(self):
        """ Compute and check the probability of c=true and r=true given no evidence
        """
        N=1000
        cprob = self.engine.Marginalise(self.c.name, N)
        rprob = self.engine.Marginalise(self.r.name, N)
        #FIX: fill in actual value
        assert(cprob[True] == 'value' and \
               rprob[True] == 'value'), \
              "Incorrect probability of Cloudy or Rain being true"
    
    def testObserved(self):
        """ Compute and check the probability of w=true|r=false,c=true and s=false|w=true,c=false
        """
        N=1000
        self.engine.SetObs([self.c.name,self.r.name],[True,False])
        wprob = self.engine.Marginalise(self.w.name,N)
        #Violates abstraction
        self.engine.evidence.clear()
        self.engine.SetObs([self.c.name,self.w.name],[False,True])
        sprob = self.engine.Marginalise(self.s.name,N)
        #FIX: fill in actual value
        assert(wprob[True] == 'value' and \
               sprob[False] == 'value'), \
              "Either P(w=true|c=true,r=false) or P(s=false|c=false,w=true) was incorrect"
    
    def testLearning(self):
        InferenceEngineTestCase.testLearning()

class JTreeTestCase(InferenceEngineTestCase):
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = JoinTree(self.BNet)
    
    
if __name__=='__main__':
    suite = unittest.makeSuite(MCMCTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
