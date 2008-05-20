"""
The inference module of the OpenBayes package
"""
import logging
import copy

import numpy

#Library Specific Modules
import openbayes.graph as graph
import openbayes.distributions as distributions
from openbayes.potentials import DiscretePotential


__all__ = ['JoinTree', 'MCMCEngine']

# show INFO messages
#logging.basicConfig(level= logging.INFO)
#uncomment the following to remove all messages
logging.basicConfig(level = logging.NOTSET)


class InferenceEngine:
    """ General Inference Engine class
    Does not really implement something but creates a standard set of
    attributes that any inference engine should implement
    """
    BNet = None         # The underlying bayesian network
    evidence = dict()   # the evidence for the BNet
    
    def __init__(self, network):
        self.network = network
        self.evidence = {}
    
    def set_obs(self, ev = None):
        """ Incorporate new evidence """
        if ev == None:
            ev = {}
        logging.info('Incorporating Observations')
        # evidence = {var.name:observed value}
        self.evidence = dict(ev)
           
    def marginalise_all(self):
        raise NotImplementedError
                   
    def marginalise(self, vertex):
        raise NotImplementedError
    
    def marginalise_family(self, vertex):
        raise NotImplementedError
    
    def learn_ml_params(self, cases):
        """ Learn and set the parameters of the network to the ML estimate
        contained in cases.
        
        Warning: this is destructive, it does not take any prior parameters
                 into account. Assumes that all evidence is specified.
        """
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                v.distribution.initialize_counts()
        for case in cases:
            assert(set(case.keys()) == set(self.network.v.keys())), \
                   "Not all values of 'case' are set"
            for v in self.network.v.values():
                if v.distribution.is_adjustable:
                    v.distribution.incr_counts(case)
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                v.distribution.set_counts()
                v.distribution.normalize(dim=v.name)

        
class Cluster(graph.Vertex):
    """
    A Cluster/Clique node for the Join Tree structure
    """
    def __init__(self, vertices):
        # list of vertices contained in this cluster
        self.vertices = [v for v in vertices]    
        #self.vertices.sort()    # sort list, much better for math operations 
        name = ''
        for v in self.vertices: 
            name += v.name
        graph.Vertex.__init__(self, name)
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)
        # weight
        self.weight = reduce(lambda x, y:x * y, 
                             [v.nvalues for v in self.vertices])
        self.marked = None
        
    def not_set_sep_of(self, clusters):
        """
        returns True if this cluster is not a sepset of any of the clusters
        """
        for c in clusters:
            count = 0
            for v in self.vertices:
                if v.name in [cv.name for cv in c.vertices]: 
                    count += 1
            if count == len(self.vertices): 
                return False
        return True

    def contains_var(self, var_names):
        """
        var_names = list of variable name
        returns True if cluster contains them all
        """
        success = True
        for vv in var_names:
            if not vv.name in self.potential.names: 
                success = False
                break
        return success

    def not_in_s(self, sepset):
        """ set of variables in cluster but not not in sepset, X\S"""
        return set(self.potential.names) - set(sepset.potential.names)
        #return set(v.name for v in self.vertices) - \
        # set(v.name for v in sepset.vertices)

    def other(self, var):
        """ set of all variables contained in cluster except v, only one at
        a time... """
        all_vertices = set(vv.name for vv in self.vertices)
        if isinstance(var, (list, set, tuple)):
            set_v = set(var)
        else:
            set_v = set((var,))
        return all_vertices - set_v

    def message_pass(self, cluster):
        """ Message pass from self to cluster c """
        ####################################################
        ### This part must be revisioned !!!!!!!!!
        ####################################################
        logging.debug('Message Pass from '+ str(self)+' to '+str(cluster))
        # c must be connected to self by a sepset
        # sepset that connects the two clusters
        e = self.connecting_e(cluster)    
        if not e: 
            raise ValueError('Clusters ' + str(self) + ' and '
                             + str(cluster) + ' are not connected')
        e = e[0]    # only one edge should connect 2 clusters
        # Projection
        oldphi_r = copy.copy(e.potential)            # oldphi_r = phiR
        newphi_r = self.potential + e.potential      # phiR = sum(X/R)phiX
        #e.potential = newphiR
        e.potential.update(newphi_r)
        # Absorption
        newphi_r /= oldphi_r
        #print 'ABSORPTION'
        #print newphi_r
        cluster.potential *= newphi_r

    def collect_evidence(self, cluster=None):
        """
        Recursive Collect Evidence,
        X is the cluster that invoked collect_evidence
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: 
                v.collect_evidence(self)
        if cluster is not None: 
            self.message_pass(cluster)

    def distribute_evidence(self):
        """
        Recursive Distribute Evidence,
        """
        self.marked = True
        for v in self.in_v:
            if not v.marked: 
                self.message_pass(v)
        for v in self.in_v:
            if not v.marked: 
                v.distribute_evidence()


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
        for v in self.vertices: 
            self.label += v.name
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        #---TODO: Continuous....
        self.potential = DiscretePotential(names, shape)        
        # self.psi = ones
        graph.UndirEdge.__init__(self, name, c1, c2)
        
        # mass and cost
        self.mass = len(self.vertices)
        self.cost = self._v[0].weight + self._v[1].weight
        
        
    def __str__(self):
        # this also prints mass and cost
        # return '%s: %s -- %s -- %s, mass: %s, cost: %s' % 
        # (str(self.name), str(self._v[0]), str(self.label), 
        # str(self._v[1]), str(self.mass), str(self.cost))
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
    def choose_vertex(self):
        """
        Chooses a vertex from the list according to criterion :
        
        Selection Criterion :
        Choose the node that causes the least number of edges to be added in
        step 2b, breaking ties by choosing the nodes that induces the 
        cluster with the smallest weight
        Implementation in Graph.choose_vertex()
        
        The WEIGHT of a node V is the nmber of values V can take 
        (BVertex.nvalues)
        The WEIGHT of a CLUSTER is the product of the weights of its
        constituent nodes
        
        Only works with graphs composed of BVertex instances
        """
        vertices = self.all_v
        # for each vertex, check how many edges will be added
        edgestoadd = [0 for v in vertices]
        clusterweight = [1 for v in vertices]
        
        for v, i in zip(vertices, range(len(vertices))):
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
        mini = [vertices[i] for e, i in zip(edgestoadd, \
                range(len(edgestoadd))) if e == minedges]
        
        # from this list, pick the one that has the smallest 
        # clusterweight = nvalues this only works with BVertex instances
        v = mini[numpy.argmin([clusterweight[vertices.index(v)] for v in mini])]
        
        return v

    def triangulate(self):
        """
        Returns a triangulated graph and its clusters.
        
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
        Implementation in Graph.choose_vertex()
        """
        logging.info('Triangulating Tree and extracting Clusters')
        # don't touch this graph, create a copy of it
        gt = copy.deepcopy(self)
        gt.name = 'Triangulised ' + str(gt.name)
        
        # make a copy of Gt
        graph_copy = copy.deepcopy(gt)
        graph_copy.name = 'Copy of '+ gt.name
    
        clusters = []
    
        while len(graph_copy.v):
            v = graph_copy.choose_vertex()
            #logging.debug('Triangulating: chosen '+str(v))
            cluster = list(v.adjacent_v)
            cluster.append(v)
        
            #logging.debug('Cluster: '+str([str(c) for c in cluster]))
        
            c = Cluster(cluster)
            if c.not_set_sep_of(clusters):
                #logging.debug('Appending cluster')
                clusters.append(c)
            
            clusterleft = copy.copy(cluster)
            
            for v1 in cluster:
                clusterleft.pop(0)
            for v2 in clusterleft:
                if not (v1 in v2.adjacent_v):
                    v1g = gt.v[v1.name]
                    v2g = gt.v[v2.name]
                    gt.add_e(graph.UndirEdge(max(gt.e.keys())+1, v1g, v2g))
                    graph_copy.add_e(graph.UndirEdge(
                                        max(graph_copy.e.keys()) + 1, 
                                            v1, v2))
                    
            # remove from G2
            del graph_copy.v[v.name]
        return gt, clusters
       
#=======================================================================
#=======================================================================
class Likelihood(distributions.MultinomialDistribution):
    """ Discrete Likelihood class """
    def __init__(self, vertex):
        distributions.MultinomialDistribution.__init__(self, vertex)
        self.vertex = vertex
        self.val = None
        self.all_ones()      # -1 = unobserved
        
    def all_ones(self):
        self.val = -1
        self.cpt = numpy.ones(self.cpt.shape, dtype='Float32')
        
    def set_obs(self, i):
        if i == -1: 
            self.all_ones()
        else:
            self.cpt = numpy.zeros(self.cpt.shape, dtype='Float32')
            self.cpt[i] = 1
            self.val = i

    def is_retracted(self, val):
        """
        returns True if likelihood is retracted.
        
        V=v1 in e1. In e2 V is either unobserved, or V=v2
        """
        return (self.val != -1 and self.val != val)
    
    def is_unchanged(self, val):
        return self.val == val
    
    def is_updated(self, val):
        return (self.val == -1 and val != -1)

#========================================================================

class JoinTree(InferenceEngine, graph.Graph):
    """ Join Tree inference engine"""
    def __init__(self, network):
        """Creates an 'Optimal' JoinTree from a BNet """
        logging.info('Creating JunctionTree engine for ' + str(network.name))
        InferenceEngine.__init__(self, network)
        graph.Graph.__init__(self, 'JT: ' + str(network.name))
        
        # key = variable name, value = cluster instance containing variable
        # {var.name:cluster}
        self.clusterdict = dict()
        
        self.likelihoods = [Likelihood(v) for v in self.network.observed]
        # likelihood dictionary, key = var name, value = likelihood instance
        self.likedict = dict((v.name, like) for v, like in 
                                                zip(self.network.observed, 
                                                    self.likelihoods))
        
        logging.info('Constructing Optimal Tree')
        self.construct_optimal_Jtree()

        JoinTree.initialization(self)

        self.global_propagation()
        
    def construct_optimal_Jtree(self):
        # Moralize Graph
        graph_moral = self.network.moralize()

        # triangulate graph and extract clusters
        graph_tri, clusters = graph_moral.triangulate()
        
        # Create Clusters for this JoinTree
        for c in clusters: 
            self.add_v(c)
        logging.info('Connecting Clusters Optimally')
        # Create candidate SepSets
        # one candidate sepset for each pair of clusters
        candsepsets = []
        clustersleft = copy.copy(clusters)
        for c1 in clusters:
            clustersleft.pop(0)
            for c2 in clustersleft:
                candsepsets.append(SepSet(len(candsepsets), c1, c2))

        # remove all edges added to clusters by creating candidate sepsets
        for c in clusters:
            c._e = []
        
        # sort sepsets, first = largest mass, smallest cost
        candsepsets = sorted(candsepsets)
        
        # Create trees
        # initialise = one tree for each cluster
        # key = cluster name, value = tree index
        trees = dict([(c.name, i) for c, i in zip(clusters, 
                     range(len(clusters)))])

        # add SepSets according to criterion, iff the two clusters connected
        # are on different trees
        for s in candsepsets:
            # if on different trees
            if trees[s._v[0].name] != trees[s._v[1].name]:
                # add SepSet
                self.add_e(SepSet(len(self.e), s._v[0], s._v[1]))
                
                # merge trees
                oldtree = trees[s._v[1].name]
                for t in trees.items():
                    if t[1] == oldtree: 
                        trees[t[0]] = trees[s._v[0].name]

            del s
            # end if n-1 sepsets have been added
            if len(self.e) == len(clusters) - 1: 
                break

    def initialization(self):
        logging.info('Initialising Potentials for clusters and SepSets')
        # for each cluster and sepset X, set phiX = 1
        for c in self.v.values():
            c.potential.all_ones()         # PhiX = 1
        for s in self.e.values():
            s.potential.all_ones()
        # assign a cluster to each variable
        # multiply cluster potential by v.cpt,
        for v in self.network.all_v:
            for c in self.all_v:
                if c.contains_var(v.family):
                    # assign a cluster for each variable
                    self.clusterdict[v.name] = c
                    v.parentcluster = c

                    # in place multiplication!
                    #logging.debug('JT:initialisation '+c.name+' *= '+v.name)
                    c.potential *= v.distribution   
                    # phiX = phiX*Pr(V|Pa(V)) (special in-place op)

                    # stop here for this node otherwise we count it
                    # more than once, bug reported by Michael Munie
                    break
        # set all likelihoods to ones
        for like in self.likelihoods: 
            like.all_ones()

    def unmark_all_clusters(self):
        for v in self.v.values(): 
            v.marked = False

    def global_propagation(self, start = None):
        if start == None: 
            start = self.v.values()[0]    # first cluster found
        
        logging.info('Global Propagation, starting at :'+ str(start))
        logging.info('      Collect Evidence')
        
        self.unmark_all_clusters()
        start.collect_evidence()
        
        logging.info('      Distribute Evidence')
        self.unmark_all_clusters()
        start.distribute_evidence()
        
    def marginalise(self, variable):
        """ returns Pr(v), v is a variable name"""
        
        # find a cluster containing v
        # v.parentcluster is a convenient choice, can make better...
        c = self.clusterdict[variable]
        res = c.potential.marginalise(c.other(variable))
        res.normalise()
        
        v_dist = self.network.v[variable].get_sampling_distribution()
        v_dist.set_parameters(res)
        return v_dist
    
    def marginalise_family(self, variable):
        """ returns Pr(fam(v)), v is a variable name
        """
        c = self.clusterdict[variable]
        res = c.marginalise(c.other(self.network.v[variable].family))
        return res.normalise()
    
    def set_obs(self, ev = dict()):
        """ Incorporate new evidence """
        InferenceEngine.set_obs(self, ev)

        
        # add any missing variables, -1 means not observed:
        for vv in self.network.v.values():
            if not self.evidence.has_key(vv.name):
                self.evidence[vv.name] = -1

        # evidence contains all variables and their observed value 
        # (-1 if unobserved) this is necessary to find out which variables 
        # have been retracted, unchanged or updated
        self.propagate_evidence()
    
    def propagate_evidence(self):
        """ propagate the evidence in the bayesian structure """
        # Check for Global Retraction, or Global update
        ev = self.evidence
        retraction = False
        for vv in self.network.all_v:
            # check for retracted variables, was observed and now it's observed
            # value has changed
            if self.likedict[vv.name].is_retracted(ev[vv.name]):
                retraction = True
            # remove any observed variables that have not changed their 
            # observed value since last iteration
            elif self.likedict[vv.name].is_unchanged(ev[vv.name]):
                del ev[vv.name]
            # remove any unobserved variables
            elif ev[vv.name] == -1:
                del ev[vv.name]
        # propagate evidence
        if retraction: 
            self.global_retraction(ev)
        else: self.global_update(ev)
            
    def set_finding(self, variable):
        ''' v becomes True (v=1), all other observed variables are false '''
        logging.info('Set finding, '+ str(variable))
        temp = dict((vi.name, 0) for vi in self.network.observed)
        if temp.has_key(variable): 
            temp[variable] = 1
        else: 
            raise ValueError(str(variable) + 
                             "is not observable or doesn't exist")
        self.initialization()
        self.observation_entry(temp.keys(), temp.values())
        self.global_propagation()
        
    def global_update(self, evidence):
        """ perform message passing to update netwrok according to evidence """
        # evidence = {var.name:value} ; -1=unobserved
        #print evidence
        logging.info('Global update')
        self.observation_entry(evidence.keys(), evidence.values())
        
        # check if only one Cluster is updated.
        # If true, only distribute_evidence
        startcluster = set()
        for v in evidence.keys():
            startcluster.add(self.network.v[v].parentcluster)
            
        if len(startcluster) == 1:
            # all variables that have changed are in the same cluster
            # perform distribute_evidence only
            logging.info('distribute only')
            self.unmark_all_clusters()
            startcluster.pop().distribute_evidence()
        else:
            # perform global propagation
            self.global_propagation()
    
    def global_retraction(self, evidence ):
        logging.info('Global Retraction')
        self.initialization()
        self.observation_entry(evidence.keys(), evidence.values())
        self.global_propagation()
        
    def observation_entry(self, variable, val):
        logging.info('Observation Entry')
        for vv, vval in zip(variable, val):
            # cluster containing likelihood, same as v 
            c = self.clusterdict[vv]     
            like = self.likedict[vv]    
            like.set_obs(vval)
            c.potential *= like

    def marginalise_all(self):
        """ returns a dict with all the marginals """
        res = dict()
        for v in self.network.v.values():
            if not v.observed: 
                res[v.name] = self.marginalise(v.name)
        for v in self.network.observed:
            res[v.name] = self.marginalise(v.name)
        
        return res
 
    def learn_ml_params(self, cases):
        InferenceEngine.learn_ml_params(self, cases)
        # reinitialize the JunctionTree to take effect of new parameters 
        # learned
        self.initialization()
        self.global_propagation()
               
    def print_(self):
        for c in self.v.values():
            print c
            print c.cpt
            print c.cpt.shape
            print numpy.sum(c.cpt.flat)
            
        for c in self.e.values():
            print c
            print c.cpt
            print c.cpt.shape
            print numpy.sum(c.cpt.flat)
            
    def extract_cpt (self, variable):
        return self.marginalise(variable).cpt
        
class ConnexeInferenceJTree(JoinTree):
    """ Accepts a non connexe BNet as entry.
        Creates an JoinTree Inference engine for each component of the BNet
        and acts transparently to the user
    """
    def __init__(self, network):
        #JoinTree.__init__(self, BNet)
        self.networks = network.split_into_components()
        self.engines = {}
        for nets in self.networks:
            JoinTree.__init__(self, nets)
            self.engines[nets] = JoinTree(nets)

    def marginalise(self, vname):
        """ trouver dans quel reseau appartient le noeud et faire l'inference 
        sur celui-ci"""
        for nets in self.networks:
            for v in nets.all_v:
                if v.name == vname:
                    #engine = JoinTree(nets)
                    return self.engines[nets].marginalise(vname)

##    def set_obs(self, ev = dict()):
##        """ Find the cluster of the vertex and do inference
##        on it """
##        for vert in ev:
##            for G in self.networks:
##                for v in G.all_v:
##                    if v.name == vert:
##                        evidence = {vert:ev[vert]}
##                        #engine = JoinTree(G)
##                        self.engines[G].set_obs(evidence)

    def set_obs(self, ev=dict()):
        """ trouver dans quel reseau appartient le noeud et faire l'inference 
        sur celui-ci"""
        for nets in self.networks:
            evidence = {}
            for vert in ev:
                for v in nets.all_v:
                    if v.name == vert:
                        evidence[vert] = ev[vert]
            self.engines[nets].set_obs(evidence)



class MCMCEngine(InferenceEngine):
    """ MCMC in the way described in the presentation by Rina Rechter """
    def __init__(self, network, nbr_samples = 1000):
        InferenceEngine.__init__(self, network)
        self.nbr_samples = nbr_samples
    
    def marginalise_all(self):
        samples = self.network.sample(self.nbr_samples)
        res = dict()
        for v in self.network.all_v:
            res[v.name] = self.marginalise(v.name, samples = samples)
       
        return res
        
    def marginalise(self, vname, samples = None):
        # 1.Sample the network N times
        if not samples:
            # if no samples are given, get them
            samples = self.network.sample(self.nbr_samples)
        
        # 2. Create the distribution that will be returned
        v = self.network.v[vname]        # the variable
        v_dist = v.get_sampling_distribution()
        v_dist.initialize_counts()                 # set all 0s
        
        # 3.Count number of occurences of vname = i
        #    for each possible value of i, that respects the evidence
        for s in samples:
            if numpy.alltrue([s[e] == i for e, i in self.evidence.items()]): 
                # this samples respects the evidence
                # add one to the corresponding instance of the variable
                v_dist.incr_counts(s)
        
        v_dist.set_counts()    #apply the counts as the distribution
        v_dist.normalize()    #normalize to obtain a probability
        
        return v_dist
