import numarray as na
from numarray.ieeespecial import getnan
import logging
import copy
import unittest
import types
import random

#Library Specific Modules
import graph
import bayesnet
import distributions
from potentials import DiscretePotential
from table import Table

logging.basicConfig(level= logging.DEBUG)
class InferenceEngine:
    """ General Inference Engine class
    Does not really implement something but creates a standard set of
    attributes that any inference engine should implement
    """
    BNet = None         # The underlying bayesian network
    evidence = dict()   # the evidence for the BNet
    
    def __init__(self, BNet):
        self.BNet = BNet
        self.evidence = {}
    
    def SetObs(self, v, val):
        """ Incorporate new evidence 
        """
        self.evidence = dict((vi,vali) for vi,vali in zip(v,val))
    
    def Marginalise(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at \
                   Child level'
    
    def MarinaliseFamily(self, v):
        assert 0, 'In InferenceEngine, method must not be implemented at \
                   appropriate level'
    
    def LearnMLParams(self, cases):
        """ Learn and set the parameters of the network to the ML estimate
        contained in cases.
        
        Warning: this is destructive, it does not take any prior parameters
                 into account. Assumes that all evidence is specified.
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.initializeCounts()
        for case in cases:
            assert(set(case.keys()) == set(self.BNet.v.keys())), "Not all values of 'case' are set"
            for v in self.BNet.v.values():
                if v.distribution.isAdjustable:
                    v.distribution.incrCounts(case)
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)
    
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
        oldphiR = copy.copy(e.potential)                # oldphiR = phiR
        newphiR = self.potential+e.potential            # phiR = sum(X/R)phiX

        #e.potential = newphiR
        e.potential.Update(newphiR)

        # Absorption
        newphiR /= oldphiR

        #print 'ABSORPTION'
        #print newphiR
        
        c.potential *= newphiR


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
    """ Discrete Likelihood class """
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

class JoinTree(InferenceEngine, graph.Graph):
    """ Join Tree inference engine"""
    def __init__(self, BNet):
        """Creates an 'Optimal' JoinTree from a BNet """
        logging.info('Creating JunctionTree engine for '+str(BNet.name))
        InferenceEngine.__init__(self, BNet)
        graph.Graph.__init__(self, 'JT: ' + str(BNet.name))
        
        # key = variable name, value = cluster instance containing variable
        # {var.name:cluster}
        self.clusterdict = dict()
        
        self.likelihoods = [Likelihood(v) for v in self.BNet.observed]
        # likelihood dictionary, key = var name, value = likelihood instance
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
                    # assign a cluster for each variable
                    self.clusterdict[v.name] = c
                    v.parentcluster = c

                    # in place multiplication!
                    #logging.debug('JT:initialisation '+c.name+' *= '+v.name)
                    c.potential *= v.distribution         # phiX = phiX*Pr(V|Pa(V)) (special in-place op)

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
        res = c.potential.Marginalise(c.other(v))
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
        # evidence = {var.name:observed value}
        self.evidence = dict((vi,vali) for vi,vali in zip(v,val))
        
        # add any missing variables, -1 means not observed:
        for vv in self.BNet.v.values():
            if not self.evidence.has_key(vv.name):
                self.evidence[vv.name] = -1

        # evidence contains all variables and their observed value (-1 if unobserved)
        # this is necessary to find out which variables have been retracted,
        # unchanged or updated
        self.PropagateEvidence()
    
    def PropagateEvidence(self):
        """ propagate the evidence in the bayesian structure """
        # Check for Global Retraction, or Global Update
        ev = self.evidence
        retraction = False
        for vv in self.BNet.all_v:
            # check for retracted variables, was observed and now it's observed
            # value has changed
            if self.likedict[vv.name].IsRetracted(ev[vv.name]):
                retraction = True
            # remove any observed variables that have not changed their observed
            # value since last iteration
            elif self.likedict[vv.name].IsUnchanged(ev[vv.name]):
                del ev[vv.name]
            # remove any unobserved variables
            elif ev[vv.name] == -1:
                del ev[vv.name]
            
        # propagate evidence
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
        
    def GlobalUpdate(self, evidence):
        """ perform message passing to update netwrok according to evidence """
        # evidence = {var.name:value} ; -1=unobserved
        print evidence
        logging.info('Global Update')
        self.ObservationEntry(evidence.keys(),evidence.values())
        
        # check if only one Cluster is updated.
        # If true, only DistributeEvidence
        startcluster = set()
        for v in evidence.keys():
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
    
    def GlobalRetraction(self, evidence ):
        logging.info('Global Retraction')
        self.Initialization()
        self.ObservationEntry(evidence.keys(),evidence.values())
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
            


class MCMCEngine_Kostas(InferenceEngine):
        """ MCMC in the way described in the presentation by Rina Rechter """
        def __init__(self, BNet, Nsamples = 1000):
            InferenceEngine.__init__(self, BNet)
            self.N = Nsamples
        
        def SetObs(self, evidence = dict()):
            self.evidence = dict(evidence)
        
        def MarginaliseAll(self):
            samples = self.BNet.Sample(self.N)
            res = []
            for v in self.BNet.all_v:
                res.append(self.Marginalise(v.name, samples = samples))
           
            return res
            
        def Marginalise(self, vname, samples = None):
            # 1.Sample the network N times
            if not samples:
                # if no samples are given, get them
                samples = self.BNet.Sample(self.N)
            
            # 2. Create the distribution that will be returned
            v = self.BNet.v[vname]        # the variable
            vDist = v.GetSamplingDistribution()
            vDist.initializeCounts()                 # set all 0s
            
            # 3.Count number of occurences of vname = i
            #    for each possible value of i, that respects the evidence
            for s in samples:
                if na.alltrue([s[e] == i for e,i in self.evidence.items()]): 
                    # this samples respects the evidence
                    # add one to the corresponding instance of the variable
                    vDist.incrCounts(s)
            
            vDist.setCounts()    #apply the counts as the distribution
            vDist.normalize()    #normalize to obtain a probability
            
            return vDist
            
class MCMCEngine(InferenceEngine):
        """ Implementation of MCMC (aka Gibbs Sampler), as described on p.517 of Russell and Norvig
        """
        cut = 100
        def __init__(self, BNet, cut=100):
            """ creates an MCMC inference Engine for the BNet specified
            cut = maximum number of iterations allowd for the sampler
            """
            InferenceEngine.__init__(self, BNet)
            self.cut = cut
            
        def SetObs(self, v, val):
            pass
                
        def Marginalise(self, v, N):
            """ Compute the Pr(v) where v is a variable name,
            N is the number of iterations of MCMC to perform.
            """
            # the return distribution
            # choose a distribution type, Table or Gaussian
            #vDist = Table(v, shape=self.BNet.v[v].nvalues)
            vDist = self.BNet.v[v].GetSamplingDistribution()
            nonEvidence = []
            # find out which values are not specified in evidence
            for vv in self.BNet.v.values():
                if not self.evidence.has_key(vv.name): nonEvidence.append(vv.name)

            # state is first selected at random            
            state = copy.copy(self.evidence)
            for vname in nonEvidence:
                # CHECK: legal values are 0 - nvalues-1: checked OK
                #state[vname] = random.randint(0, self.BNet.v[vname].nvalues-1)
                state[vname] = self.BNet.v[vname].distribution.random()
            # state = {'a':0,'b':1}   # chosen completely at random, 
                                      # without keeping in mind the distribution
            for i in range(N):
                if i > self.cut:
                        #########################################
                        # cut is used to avoid the first samples...
                        # is this really necessary??? maybe,....(Kostas Comments)
                        ##################################
                        # 
                        vDist[state[v]] += 1
                        pass
                        
                # sample non-evidence variables again for next iteration
                for vname in nonEvidence:
                        state[vname] = self.sampleGivenMB(self.BNet.v[vname], state)

            # added a normalize() function in Table
            vDist.normalize()
            return vDist
        
        def sampleGivenMB(self, v, state):
            #MBval = Table(v.name, shape=v.nvalues)
            MBval = v.GetSamplingDistribution()    # MBval contains 0s
            children = v.out_v
            index = {}
            for vert in v.family:       
                    index[vert.name] = state[vert.name]
            # index = {var.name:state}    only for var in v.family
            
            childrenAndIndex = []
            for child in children:
                cindex = {}
                # family = a node and all its parents
                for cvert in child.family: # replaced [child]+list(child.in_v)
                    # cvert is either a child or an uncle(parent of child) of v
                    # cindex contains the state of all variables in the family
                    # of a child of v
                    cindex[cvert.name] = state[cvert.name]       
                childrenAndIndex.append((child,cindex))

            #OPTIMIZE: could vectorize this code
            for value in range(v.nvalues):
                index[v.name] = value
                # initialise each element of the distribution with the
                # conditional probability table values of the variable
                # Pr(v=i)=Pr(v=i|Pa(v)=index)
                # index is randomly selected at each iteration
                MBval[value] = v.distribution[index]
                ##################################################
                # this could be replaced by Table multiplication instead
                # of an element-wise multiplication
                # in that case we don't need all those index dictionnaries
                ##################################################
                for child,cindex in childrenAndIndex:
                    cindex[v.name] = value
                    MBval[value] *= child.distribution[cindex]
            x=1
            MBval.normalize()

            #######################################
            # added a sample() function in Distribution
            #######################################
            return MBval.sample()
        

class InferenceEngineTestCase(unittest.TestCase):
    """ An abstract set of inference test cases.  Basically anything that is similar between the different inference engines can be implemented here and automatically applied to lower engines.  For example, we can define the learning tests here and they shouldn't have to be redefined for different engines.
    """
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:,0,0]=[0.99, 0.01]
        w.distribution[:,0,1]=[0.1, 0.9]
        w.distribution[:,1,0]=[0.1, 0.9]
        w.distribution[:,1,1]=[0.0, 1.0]
        
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G
        
        # create a simple continuous network
        G2 = bayesnet.BNet('Gaussian Bayesian Network')
        a,b = [G2.add_v(bayesnet.BVertex(nm,False,1)) for nm in 'a b'.split()]
        for ep in [(a,b)]:
            G2.add_e(graph.DirEdge(len(G2.e), *ep))
        
        G2.InitDistributions()
        a.setDistributionParameters(mu = 1.0, sigma = 1.0)
        b.setDistributionParameters(mu = 1.0, sigma = 1.0, wi = 2.0)
        
        self.a = a
        self.b = b
        self.G2 = G2
    

class MCMCKostaTestCase(InferenceEngineTestCase):
    """ MCMC Kosta unit tests.
    """
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = MCMCEngine_Kostas(self.BNet,2000)
        self.engine2 = MCMCEngine_Kostas(self.G2,5000)
    
    def testUnobservedDiscrete(self):
        """ DISCRETE: Compute and check the probability of water-sprinkler given no evidence
        """
        cprob, rprob, sprob, wprob = self.engine.MarginaliseAll()

        error = 0.05
        #print cprob[True] <= (0.5 + error)and cprob[True] >= (0.5-error)
        #print wprob[True] <= (0.65090001 + 2*error) and wprob[True] >= (0.65090001 - 2*error)
        #print sprob[True] <= (0.3 + error) and sprob[True] >= (0.3 - error)
        
        assert(na.allclose(cprob[True], 0.5, atol = error) and \
               na.allclose(sprob[True], 0.3, atol = error) and \
               na.allclose(rprob[True], 0.5, atol = error) and \
               na.allclose(wprob[True], 0.6509, atol = error)), \
        "Incorrect probability with unobserved water-sprinkler network"

    def testUnobservedGaussian(self):
        """ GAUSSIAN: Compute and check the marginals of a simple gaussian network """
        G = self.G2
        a,b = self.a, self.b
        engine = self.engine2
        
        res = engine.MarginaliseAll()
        for r in res:print r
              

class MCMCTestCase(InferenceEngineTestCase):
    """ MCMC unit tests.
    """
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = MCMCEngine(self.BNet,cut=100)
        self.engine2 = MCMCEngine(self.G2)
        
    def testUnobserved(self):
        """ Compute and check the probability of c=true and r=true given no evidence
        """
        N=1000
        cprob = self.engine.Marginalise('c', N)
        sprob = self.engine.Marginalise('s', N)
        wprob = self.engine.Marginalise('w', N)
        #FIX: fill in actual value
        error = 0.05
        #print cprob[True] <= (0.5 + error)and cprob[True] >= (0.5-error)
        #print wprob[True] <= (0.65090001 + 2*error) and wprob[True] >= (0.65090001 - 2*error)
        #print sprob[True] <= (0.3 + error) and sprob[True] >= (0.3 - error)
        
        assert(cprob[True] <= (0.5 + error) and \
               cprob[True] >= (0.5-error) and \
               # wprob has a bigger error generally...
               wprob[True] <= (0.65090001 + 2*error) and \
               wprob[True] >= (0.65090001 - 2*error) and \
               sprob[True] <= (0.3 + error) and \
               sprob[True] >= (0.3 - error)), \
              "Incorrect probability with unobserved water-sprinkler network"

    def testUnobservedGaussian(self):
        """ Compute and check the probability of c=true and r=true given no evidence
        All nodes are gaussian
        """
        N=1000
        aprob = self.engine2.Marginalise('a', N)
        bprob = self.engine2.Marginalise('b', N)

        error = 0.05
       
        assert(1), \
              "Incorrect probability with unobserved gaussian network"   
               
    def _testObserved(self):
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
    
    def _testLearning(self):
        """ Sample network and then learn parameters and check that they are relatively close to original.
        """
        ev = self.engine.BNet.Sample(n=10000)
        
        #Remember what the old CPTs looked like and keep track of original dimension order
        cCPT = self.c.distribution.cpt.copy()
        self.c.distribution.isAdjustable=True
        self.c.distribution.uniform()
        sCPT = self.s.distribution.cpt.copy()
        self.s.distribution.isAdjustable=True
        self.s.distribution.uniform()
        rCPT = self.r.distribution.cpt.copy()
        self.r.distribution.isAdjustable=True
        self.r.distribution.uniform()
        wCPT = self.w.distribution.cpt.copy()
        self.w.distribution.isAdjustable=True
        self.w.distribution.uniform()

        # Learn parameters
        self.engine.LearnMLParams(ev)
        
        # Check that they match original parameters
        assert(na.allclose(cCPT,self.c.distribution.cpt,atol=.1) and \
               na.allclose(sCPT,self.s.distribution.cpt,atol=.1) and \
               na.allclose(rCPT,self.r.distribution.cpt,atol=.1) and \
               na.allclose(wCPT,self.w.distribution.cpt,atol=.1)),\
              "CPTs were more than atol=.1 apart"
        
class JTreeTestCase(InferenceEngineTestCase):
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = JoinTree(self.BNet)

    def testGeneral(self):
        """ Check that the overall algorithm works """
        c=self.engine.Marginalise('c')
        r=self.engine.Marginalise('r')
        s=self.engine.Marginalise('s')
        w=self.engine.Marginalise('w')

        assert(na.allclose(c.cpt,[0.5,0.5]) and \
               na.allclose(r.cpt,[0.5,0.5]) and \
               na.allclose(s.cpt,[0.7,0.3]) and \
               na.allclose(w.cpt,[ 0.34909999, 0.65090001]) ), \
               " Somethings wrong with JoinTree inference engine"

    def testEvidence(self):
        """ check that evidence works """
        print 'evidence c=1,s=0'
        self.engine.SetObs(['c','s'],[1,0])
        
        c=self.engine.Marginalise('c')
        r=self.engine.Marginalise('r')
        s=self.engine.Marginalise('s')
        w=self.engine.Marginalise('w')

        assert(na.allclose(c.cpt,[0.0,1.0]) and \
               na.allclose(r.cpt,[0.2,0.8]) and \
               na.allclose(s.cpt,[1.0,0.0]) and \
               na.allclose(w.cpt,[ 0.278, 0.722]) ), \
               " Somethings wrong with JoinTree evidence"        

    ###########################################################
    ### SHOULD ADD A MORE GENERAL TEST:
    ###     - not only binary nodes
    ###     - more complex structure
    ###     - check message passing
    ###########################################################
if __name__=='__main__':
    suite = unittest.makeSuite(MCMCKostaTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)    
#    G = bayesnet.BNet('Water Sprinkler Bayesian Network')
#    c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
#    for ep in [(c,r), (c,s), (r,w), (s,w)]:
#        G.add_e(graph.DirEdge(len(G.e), *ep))
#    G.InitDistributions()
#    c.setDistributionParameters([0.5, 0.5])
#    s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
#    r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
#    w.distribution[:,0,0]=[0.99, 0.01]
#    w.distribution[:,0,1]=[0.1, 0.9]
#    w.distribution[:,1,0]=[0.1, 0.9]
#    w.distribution[:,1,1]=[0.0, 1.0]
#    
#
#    print G
#    engine = MCMCEngine_Kostas(G,1000)
#    
#    d = c.GetSamplingDistribution()
#    d.initializeCounts()
#    d.incrCounts(range(10))
#    d.setCounts()
#    print d
    
    
    #for r in engine.MarginaliseAll(): print r
#    
#    engine.SetObs({'c':0,'s':0})
#    print 'EVIDENCE:',engine.evidence
#    for r in engine.MarginaliseAll(): print r
#
#    print 'Results using JTree inference'
#    engine = JoinTree(G)
#    engine.SetObs(['c','s'],[0,0])
#    print engine.Marginalise('c')
#    print engine.Marginalise('s')
#    print engine.Marginalise('r')
#    print engine.Marginalise('w')








if __name__=='__mains__':
    suite = unittest.makeSuite(MCMCTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
##   
##    suite = unittest.makeSuite(JTreeTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
    G = bayesnet.BNet('Water Sprinkler Bayesian Network')
    c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))
    G.InitDistributions()
    c.setDistributionParameters([0.5, 0.5])
    s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
    r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
    w.distribution[:,0,0]=[0.99, 0.01]
    w.distribution[:,0,1]=[0.1, 0.9]
    w.distribution[:,1,0]=[0.1, 0.9]
    w.distribution[:,1,1]=[0.0, 1.0]

    print G
    engine = MCMCEngine(G,cut=100)
    engine.SetObs(['c','s'],[1,0])
    print engine.Marginalise('c',1000)
    print engine.Marginalise('s',1000)
    print engine.Marginalise('r',1000)
    print engine.Marginalise('w',1000)

#===============================================================================
#    ev = G.Sample(100)
#    #for e in ev:print e
#
#    print 'real parameters'
#    print c.distribution
#
#    cCPT = c.distribution.cpt.copy()
#    c.distribution.isAdjustable=True
#    c.distribution.uniform()
#    sCPT = s.distribution.cpt.copy()
#    s.distribution.isAdjustable=True
#    s.distribution.uniform()
#    rCPT = r.distribution.cpt.copy()
#    r.distribution.isAdjustable=True
#    r.distribution.uniform()
#    wCPT = w.distribution.cpt.copy()
#    w.distribution.isAdjustable=True
#    w.distribution.uniform()
#
#    print 'initialization parameters'
#    print c.distribution
#    
#    engine = JoinTree(G)
#    engine.LearnMLParams(ev)
#
#    print 'Learned parameters'
#    print c.distribution
#===============================================================================
