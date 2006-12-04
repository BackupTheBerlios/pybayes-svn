"""Bayesian network implementation.  Influenced by Cecil Huang's and Adnan
Darwiche's "Inference in Belief Networks: A Procedural Guide," International
Journal of Approximate Reasoning, 1994.

Copyright 2005, Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
license file for legal information.
"""

__all__ = ['BVertex','BNet']
__version__ = '0.1'
__author__ = 'Kosta Gaitanis  & Elliot Cohen'
__author_email__ = 'gaitanis@tele.ucl.ac.be; elliot.cohen@gmail.com'
#Python Standard Distribution Packages
import sys
import unittest
import types
import copy
#from timeit import Timer, time
#import profile
import bisect       # for appending elements to a sorted list
import logging
#Major Packages
import numarray as na
import numarray.mlab
from numarray.random_array import randint, seed
from numarray.ieeespecial import setnan, getnan
#Library Specific Modules
import graph
import delegate

import distributions
import potentials
import inference

seed()
#logging.basicConfig(level= logging.INFO)

class BVertex(graph.Vertex):
    def __init__(self, name, discrete = True, nvalues = 2, observed = True):
        '''
        Name neen't be a string but must be hashable and immutable.
        if discrete = True:
                nvalues = number of possible values for variable contained in Vertex
        if discrete = False:
                nvalues is not relevant = 0
        observed = True means that this node CAN be observed
        '''
        graph.Vertex.__init__(self, name)
        self.distribution = None
        self.nvalues = int(nvalues)
        
        self.discrete = discrete
            # a continuous node can be scalar (self.nvalues=1)
            # or vectorial (self.nvalues=n)
            # n=2 is equivalent to 2D gaussian for example

        # True if variable can be observed
        self.observed = observed
        self.family = [self] + list(self.in_v)

    def InitDistribution(self, *args, **kwargs):
        """ Initialise the distribution, all edges must be added"""
        #first decide which type of Distribution
        #if all nodes are discrete, then Multinomial)
        if na.alltrue([v.discrete for v in self.family]):
            #print self.name,'Multinomial'
            #FIX: should be able to pass through 'isAdjustable=True' and it work
            self.distribution = distributions.MultinomialDistribution(self, *args, **kwargs) 
            return

        #gaussian distribution
        if not self.discrete:
            #print self.name,'Gaussian'
            self.distribution = distributions.Gaussian_Distribution(self, *args, **kwargs)
            return
        
        #other cases go here
    
    def setDistributionParameters(self, *args, **kwargs):
        # sets any parameters for the distribution of this node
        self.distribution.setParameters(*args, **kwargs)
        
    def __str__(self):
        if self.discrete:
            return graph.Vertex.__str__(self)+'    (discrete, %d)' %self.nvalues
        else:
            return graph.Vertex.__str__(self)+'    (continuous)'

    #============================================================
    # This is used for the MCMC engine
    # returns a new distributions of the correct type, containing only
    # the current without its family
    def GetSamplingDistribution(self):
        if self.discrete:
            d = distributions.MultinomialDistribution(self, ignoreFamily = True)
        else:
            d = distributions.Gaussian_Distribution(self, ignoreFamily = True)
        
        return d
            
    
    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function
    def __cmp__(a,b):
        ''' sort by name, any other criterion can be used '''
        return cmp(a.name, b.name)


class BNet(graph.Graph):
    log = logging.getLogger('BNet')
    log.setLevel(logging.ERROR)
    def __init__(self, name = ''):
        graph.Graph.__init__(self, name)

    def add_e(self, e):
        if e.__class__.__name__ == 'DirEdge':
            graph.Graph.add_e(self, e)
            for v in e._v:
                v.family = [v] + list(v.in_v)
        else:
            raise "All edges should be directed"

    def Moralize(self):
        logging.info('Moralising Tree')
        G = inference.MoralGraph(name = 'Moralized ' + str(self.name))
        
        # for each vertice, create a corresponding vertice
        for v in self.v.values():
            G.add_v(BVertex(v.name, v.discrete, v.nvalues))

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

    def InitDistributions(self):
        """ Finalizes the network, all edges must be added. A distribution (unknown)
        is added to each node of the network"""
        # this replaces the InitCPTs() function
        for v in self.v.values(): v.InitDistribution()
    
##    def InitCPTs(self):
##        for v in self.v.values(): v.InitCPT()

    def RandomizeCPTs(self):
        for v in self.v.values():
            v.rand()
            v.makecpt()
    
    def Sample(self,n=1):
        """ Generate a sample of the network, n is the number of samples to generate
        """
        assert(len(self.v) > 0)
        samples = []

        # find a node without parents and start from there.
        # There is always at least one node without parents
        # because a BNet is a Directed Acyclic Graph
        # this is critical in small networks:
        # e.g. A--> B
        #      starting at B will produce an empty output...
        for v in self.v.values():
            if len(v.in_v) == 0:
                start_node = v
                break

        topological = self.topological_sort(start_node)
        
        for i in range(n):
            sample = {}
            for v in topological:
                assert(not v.distribution == None), "vertex's distribution is not initialized"
                sample[v.name] = v.distribution.sample(sample)
            samples.append(sample)

        return samples
    

class BNetTestCase(unittest.TestCase):
    """ Basic Test Case suite for BNet
    """
    def setUp(self):
        G = BNet('Water Sprinkler Bayesian Network')
        
        c,s,r,w = [G.add_v(BVertex(name,True,2)) for name in 'c s r w'.split()]
        
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
            
        G.InitDistributions()

        c.setDistributionParameters([0.5, 0.5])
        # the following 2 lines are equivalent
        c.distribution.cpt = na.array([0.5,0.5])
        c.distribution[:]  = na.array([0.5,0.5])
        
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        
        w.setDistributionParameters([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
        # this is equivalent, we access directly the cpt property of the distribution
        w.distribution[:,0,0]=[0.99, 0.01]
        w.distribution[:,0,1]=[0.1, 0.9]
        w.distribution[:,1,0]=[0.1, 0.9]
        w.distribution[:,1,1]=[0.0, 1.0]
        
        # same thing using dictionnaries
        w.distribution[{'s':0,'r':0}]=[0.99, 0.01]
        w.distribution[{'s':0,'r':1}]=[0.1, 0.9]
        w.distribution[{'s':1,'r':0}]=[0.1, 0.9]
        w.distribution[{'s':1,'r':1}]=[0.0, 1.0]
        
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

    def testTopoSort(self):
        sorted = self.BNet.topological_sort(self.c)

        assert(sorted[0] == self.c and \
               sorted[1] == self.s and \
               sorted[2] == self.r and \
               sorted[3] == self.w), \
               "Sorted was not in proper topological order"

    
    def testFamily(self):
        cFamily = set(self.BNet.v['c'].distribution.names_list[1:])
        sFamily = set(self.BNet.v['s'].distribution.names_list[1:])
        rFamily = set(self.BNet.v['r'].distribution.names_list[1:])
        wFamily = set(self.BNet.v['w'].distribution.names_list[1:])

        assert(cFamily == set([]) and sFamily == set(['c']) and \
               rFamily == set(['c']) and wFamily == set(['r','s'])),\
              "Families are not set correctly"
    
if __name__=='__main__':
    suite = unittest.makeSuite(BNetTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
