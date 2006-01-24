"""Bayesian network implementation.  Influenced by Cecil Huang's and Adnan
Darwiche's "Inference in Belief Networks: A Procedural Guide," International
Journal of Approximate Reasoning, 1994.

Copyright 2005, Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
license file for legal information.
"""

__version__ = '0.1'
__author__ = 'Kosta Gaitanis & Elliot Cohen'
__author_email__ = 'gaitanis@tele.ucl.ac.be; elliot.cohen@gmail.com'
#Python Standard Distribution Packages
import sys
import unittest
import types
import copy
from timeit import Timer, time
import profile
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
na.Error.setMode(invalid='ignore')
#logging.basicConfig(level= logging.INFO)

# removed CPT and placed Distriution
# also removed delegate.Delegate, delaegation is now performed by distributions
# we can put it back if we really need it, but for the moment i think it's ok
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

        # self.distribution contains the type of distribution for this node
        self.distribution = None
        
        if discrete:
            # discrete node
            self.discrete = True
            self.nvalues =  int(nvalues)
        else:
            # continuous node
            self.discrete = False
            self.nvalues = 0    # irrelevant when node is continuous

        # True if variable can be observed
        self.observed = observed

        # family is created by the distribution
        #self.family = [self] + list(self.in_v)

    def InitDistribution(self, cpt=None):
        """ Initialise the distribution, all edges must be added"""
        #first decide which type of Distribution (if all nodes are discrete, then Multinomial)
        if na.alltrue([v.discrete for v in self.in_v]):
            self.distribution = distributions.MultinomialDistribution(self, cpt=cpt) 
            return

        #---TODO: other cases go here

    def __getattr__(self, name):
        """ any attributes not found on this instance are delegated to
        self.distribution, if they exist """
        # delegate only if self.distribution contains that attribute
        try:
            return getattr(self.distribution, name)
        except:
            raise 'Could not find attribute',name

        
    # also delegate get and set to the distribution
    def __getitem__(self, index):
        #print 'BVertex.__getitem__'
        return self.distribution.__getitem__(index)

    def __setitem__(self,index,value):
        return self.distribution.__setitem__(index,value)
        
    def __str__(self):
        string = ''
        if self.discrete:
            string += graph.Vertex.__str__(self)+'    (discrete)'
        else:
            string += graph.Vertex.__str__(self)+'    (continuous)'

        #if self.distribution != None:
        #    string += self.distribution.__str__()

        return string
            
    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function
    def __cmp__(a,b):
        ''' sort by name, any other criterion can be used '''
        return cmp(a.name, b.name)


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
        G = inference.MoralGraph(name = 'Moralized ' + str(self.name))
        
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

    def InitDistributions(self):
        """ Finalizes the network, all edges must be added. A distribution (unknown)
        is added to each node of the network"""
        # this replaces the InitCPTs() function
        for v in self.all_v: v.InitDistribution()
    
##    def InitCPTs(self):
##        for v in self.v.values(): v.InitCPT()

    def RandomizeCPTs(self):
        for v in self.v.values():
            v.rand()
            v.makecpt()
    
    def Sample(self):
        """ Generate a sample of the network
        """
        sample = {}
        #OPTIMIZE: could make this faster
        vertices = self.v.values()
        lastN = len(vertices) - 1
        while len(vertices) > 0:
            assert(lastN < len(self.v.values())), 'No nodes have no parents'
            for v in vertices:
                allSet = True
                for parent in v.in_v():
                    if not sample.has_key(parent.name):
                        allSet = False
                        break
                if allSet:
                    sample[v.name] = v.distribution.Sample(sample)
                    vertices -= v
                    lastN -= 1
        return sample

class BNetTestCase(unittest.TestCase):
    """ Basic Test Case suite for BNet
    """
    def setUp(self):
        G = BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(BVertex(name,2,True)) for name in 'c s r w'.split()]
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitCPTs()
        c.setCPT([0.5, 0.5])
        s.setCPT([0.5, 0.9, 0.5, 0.1])
        r.setCPT([0.8, 0.2, 0.2, 0.8])
        w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
        
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G

    def testTopoSort(self):
        sorted = self.BNet.topological_sort(self.s)
        assert(sorted[0] == self.c and \
               sorted[1] == self.s and \
               sorted[2] == self.r and \
               sorted[3] == self.w), \
               "Sorted was not in proper topological order"

    def testSample(self):
        cCPT = distributions.MultinomialDistribution(self.c)
        sCPT = distributions.MultinomialDistribution(self.s)
        rCPT = distributions.MultinomialDistribution(self.r)
        wCPT = distributions.MultinomialDistribution(self.w)
        for i in range(1000):
            sample = self.BNet.Sample
            # Can use sample in all of these, it will ignore extra variables
            cCPT[sample] += 1
            sCPT[sample] += 1
            rCPT[sample] += 1
            wCPT[sample] += 1
        assert(na.allclose(cCPT,self.c.cpt,atol=.1) and \
               na.allclose(sCPT,self.s.cpt,atol-.1) and \
               na.allclose(rCPT,self.r.cpt,atol-.1) and \
               na.allclose(wCPT,self.w.cpt,atol-.1)), \
               "Samples did not generate appropriate CPTs"
            
        
if __name__=='__main__':
    ''' Water Sprinkler example '''
    #suite = unittest.makeSuite(CPTIndexTestCase, 'test')
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(name,True,2)) for name in 'c s r w'.split()]
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))
        
    G.InitDistributions()
    
    c.setCPT([0.5, 0.5])
    s.setCPT([0.5, 0.9, 0.5, 0.1])
    r.setCPT([0.8, 0.2, 0.2, 0.8])
    w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
    
    
    print G
    
    JT = inference.JoinTree(G)
    
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
        
    G.InitDistributions()
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
                
