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
import graph.graph as graph
import delegate

from distributions import *
from potentials import *

from JunctionTree import *
from MCMC import *
seed()
na.Error.setMode(invalid='ignore')
#logging.basicConfig(level= logging.INFO)

# removed CPT and placed Distriution
class BVertex(graph.Vertex, delegate.Delegate):
    def __init__(self, name, discrete = True, nvalues = 2, observed = False):
        '''
        Name neen't be a string but must be hashable and immutable.
        if discrete = True:
                nvalues = number of possible values for variable contained in Vertex
        if discrete = False:
                nvalues is not relevant = 0
        observed = True means that this node CAN be observed
        '''
        graph.Vertex.__init__(self, name)
        if discrete:
            # discrete node
            self.discrete = True
            self.nvalues =  nvalues
        else:
            # continuous node
            self.discrete = False
            self.nvalues = 0

        # True if variable can be observed
        self.observed = observed

        self.distribution = None # to be set using SetDistribution

    def SetDistribution(self, distribution, *args, **kwargs):
        ''' sets and returns the distribution for this node
            distribution = a CLASS, not an instance

            POST : self.distribution = a distribution instance

            # with no arguments, uses defaults            
            >>> bvertex.SetDistribution(Multinomial_Distribution)

            # with some arguments, uses the order defined in the distribution.__init__
            >>> bvertex.SetDistribution(Multinomial_Distribution, cpt_array)

            # with some keyword arguments
            >>> bvertex.SetDistribution(Multinomial_Distribution, cpt = cpt_array)
            
        '''

        # this would be much better if BVertex was a Distribution class too like
        # it was before a CPT class
        # the problem is that you must specify the class name into the __init__
        # and you don't know which type of distribution is going to be used
        self.distribution = distribution.__new__(distribution)
        self.distribution.__init__(self, *args, **kwargs)
        
        return self.distribution
    
    def InitCPT(self):
        ''' Initialise CPT, all edges must be added '''
        CPT.__init__(self, self)  #second self is for BVertex

    def __str__(self):
        if self.discrete:
            return graph.Vertex.__str__(self)+'    (discrete)'
        else:
            return graph.Vertex.__str__(self)+'    (continuous)'

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

        
if __name__=='__main__':
    ''' Water Sprinkler example '''
    suite = unittest.makeSuite(CPTIndexTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
    
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(name,2,True)) for name in 'c s r w'.split()]
    
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
                
