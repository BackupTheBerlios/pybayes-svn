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
    def __init__(self, name = None):
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
        topological = self.topological_sort(self.v.values()[0])
        
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
    
    def testFamily(self):
        cFamily = self.BNet.v.values['c'].family
        sFamily = self.BNet.v.values['s'].family
        rFamily = self.BNet.v.values['r'].family
        wFamily = self.BNet.v.values['w'].family
        
        assert(cFamily == set([]) and sFamily == set(['c']) and \
               rFamily == set(['c']) and wFamily == set(['r','s'])),\
              "Families are not set correctly"
    
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
    
    c.setDistributionParameters([0.5, 0.5])
    s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
    r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
    w.distribution[:,0,0]=[0.99, 0.01]
    w.distribution[:,0,1]=[0.1, 0.9]
    w.distribution[:,1,0]=[0.1, 0.9]
    w.distribution[:,1,1]=[0.0, 1.0]
    
    print G
    
    JT = inference.JoinTree(G)
    clusters = JT.all_v
    c1,c2 = clusters[:]
    
    #print c1.potential
    #print c2.potential
    #JT.SetObs(['c'],[1])
    #print JT.Marginalise('w')
    #print 'RESULT, after Junction Tree:'
    JT.MargAll()

    # verification
    print 'VERIFICATION, Results should look like this :'
    cp = potentials.DiscretePotential(['c'],[2],[0.5,0.5])
    sp = potentials.DiscretePotential(['s','c'],[2,2],[0.5, 0.9, 0.5, 0.1])
    rp = potentials.DiscretePotential(['r','c'],[2,2],[0.8,0.2,0.2,0.8])
    wp = potentials.DiscretePotential(['w','s','r'],[2,2,2])
    wp[:,0,0]=[0.99, 0.01]
    wp[:,0,1]=[0.1, 0.9]
    wp[:,1,0]=[0.1, 0.9]
    wp[:,1,1]=[0.0, 1.0]

    cr = cp*rp
    crs = cr*sp
    crsw = crs*wp

    print 'c:', crsw.Marginalise('s r w'.split())
    print 's:', crsw.Marginalise('c r w'.split())
    print 'r:', crsw.Marginalise('c s w'.split())
    print 'w:', crsw.Marginalise('c s r'.split())

    #add some evidence
    #JT.SetObs(['c'],[1])
    #JT.MargAll()
    G.Sample(1)
##    print 'DEBUGGING: performing same calculations as JunctionTree:'
##    #create the clusters and set them to allOnes
##    c1 = inference.Cluster([r,s,c])
##    c2 = inference.Cluster([w,s,r])
##    e  = inference.SepSet('c1-c2',c1,c2)
##
##    #assign each vertex to a cluster and initialise potentials
##    #c,r,s -> c1, w -> c2
##    c1.potential *= c.distribution
##    c1.potential *= s.distribution
##    c1.potential *= r.distribution
##    c2.potential *= w.distribution
##
##    print c1.potential,c1.potential.names_list
##    print "JT after initialisation"
##    jtp0 = JT.all_v[1].potential
##    print jtp0, jtp0.names_list
##    
##    print c2.potential,c2.potential.names_list
##    print "JT after initialisation"
##    jtp1 = JT.all_v[0].potential
##    print jtp1, jtp1.names_list
##    #up to here all is OK
##    
##    #start message passing, root = c1
##    print 'collect evidence: c2-->c1'
##    c2.MessagePass(c1)
##    #normally c1 should contain the correct values for c,s and r
##    print 'c:', c1.potential.Marginalise('s r'.split())
##    print 's:', c1.potential.Marginalise('c r'.split())
##    print 'r:', c1.potential.Marginalise('c s'.split())
##
##    #do the same thing for the JT
##    jt0 = JT.all_v[1]
##    jt1 = JT.all_v[0]
##    jt1.MessagePass(jt0)
##    #verify the results
##    print 'Results for JTree:'
##    print 'c:', jt0.potential.Marginalise('s r'.split())
##    print 's:', jt0.potential.Marginalise('c r'.split())
##    print 'r:', jt0.potential.Marginalise('c s'.split())
##    
##    print 'distribute evidence: c1-->c2'
##    c1.MessagePass(c2)
##    #normally c2 should contain the correct values for w
##    print 'w:', c2.potential.Marginalise('s r'.split())
##
##    jt0.MessagePass(jt1)
##    #verify the results
##    print 'Results for JTree:'
##    print 'w:', jt1.potential.Marginalise('s r'.split())

##    #start message passing, root = c2
##    print 'collect evidence: c1-->c2'
##    c1.MessagePass(c2)
##    #normally c2 should contain the correct values for w
##    print 'w:', c2.potential.Marginalise('s r'.split())
##
##    print 'distribute evidence: c2-->c1'
##    c2.MessagePass(c1)
##    #normally c1 should contain the correct values for c,s and r
##    print 'c:', c1.potential.Marginalise('s r'.split())
##    print 's:', c1.potential.Marginalise('c r'.split())
##    print 'r:', c1.potential.Marginalise('c s'.split())


if __name__=='__mains__':
    ''' Water Sprinkler example with more than binary nodes'''
    #suite = unittest.makeSuite(CPTIndexTestCase, 'test')
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    
    G = BNet('Water Sprinkler Bayesian Network')

    print 'c:5,s:4,r:3,w:2'
    c,s,r,w = [G.add_v(BVertex(name,True,s)) for s,name in zip([5,4,3,2],'c s r w'.split())]
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))

    G.InitDistributions()
    
    c.setCPT([1.0/c.nvalues]*c.nvalues)
    s.distribution.cpt = na.arange(5*4,shape=s.distribution.cpt.shape,type='Float32')
    s.distribution.normalize()
    r.distribution.cpt = na.arange(5*3,shape=r.distribution.cpt.shape,type='Float32')
    r.distribution.normalize()
    w.distribution.cpt = na.arange(2*3*4,shape=w.distribution.cpt.shape,type='Float32')
    w.distribution.normalize()    
    
    print G
    
    JT = inference.JoinTree(G)
    clusters = JT.all_v
    c1,c2 = clusters[:]
    
    #print c1.potential
    #print c2.potential
    #JT.SetObs(['c'],[1])
    #print JT.Marginalise('w')
    #print 'RESULT, after Junction Tree:'
    JT.MargAll()
    
    G.Sample(1)

##    # verification
##    print 'VERIFICATION, Results should look like this :'
##    cp = potentials.DiscretePotential(['c'],[5],c.distribution.cpt)
##    sp = potentials.DiscretePotential(['s','c'],[4,5],s.distribution.cpt)
##    rp = potentials.DiscretePotential(['r','c'],[3,5],r.distribution.cpt)
##    wp = potentials.DiscretePotential(['w','s','r'],[2,4,3],w.distribution.cpt)
##
##
##    cr = cp*rp
##    crs = cr*sp
##    crsw = crs*wp
##
##    print 'c:', crsw.Marginalise('s r w'.split())
##    print 's:', crsw.Marginalise('c r w'.split())
##    print 'r:', crsw.Marginalise('c s w'.split())
##    print 'w:', crsw.Marginalise('c s r'.split())

if __name__=='__mains__':
    G = BNet('Bnet')
    
    a, b, c, d, e, f, g, h = [G.add_v(BVertex(nm,True,s+2)) for s,nm in enumerate('a b c d e f g h'.split())]

    for ep in [(a, b), (a,c), (b,d), (d,f), (c,e), (e,f), (c,g), (e,h), (g,h)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))
        
    G.InitDistributions()
    #G.RandomizeCPTs()
    
    
    JT = inference.JoinTree(G)
    JT2 = inference.MCMCEngine(G,1000)
    
    print JT

    
    print JT.Marginalise('c')
    
    JT.SetObs(['b'],[1])
    print JT.Marginalise('c')

    print JT2.Marginalise('c')
    
    #JT.SetObs(['b','a'],[1,2])
    #print JT.Marginalise('c')
    
    #JT.SetObs(['b'],[1])
    #print JT.Marginalise('c')
    
##    logging.basicConfig(level=logging.CRITICAL)
##    
##    def RandomObs(JT, G):
##        for N in range(100):
##            n = randint(len(G.v))
##            
##            obsn = []
##            obs = []
##            for i in range(n):
##                v = randint(len(G.v))
##                vn = G.v.values()[v].name
##                if vn not in obsn:
##                    obsn.append(vn)
##                    val = randint(G.v[vn].nvalues)
##                    obs.append(val)
##                    
##            JT.SetObs(obsn,obs)
##            
##    t = time.time()
##    RandomObs(JT,G)
##    t = time.time() - t
##    print t
    
    #profile.run('''JT.GlobalPropagation()''')
                
