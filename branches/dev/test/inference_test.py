#!/usr/bin/env python

import unittest

import numpy as na

from OpenBayes.inference import *
from OpenBayes import bayesnet, graph


class InferenceEngineTestCase(unittest.TestCase):
    """ An abstract set of inference test cases.  Basically anything 
    that is similar between the different inference engines can be 
    implemented here and automatically applied to lower engines.  
    For example, we can define the learning tests here and they 
    shouldn't have to be redefined for different engines.
    """
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [G.add_v(bayesnet.BVertex(nm, True, 2)) \
                      for nm in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.init_distributions()
        c.set_distribution_parameters([0.5, 0.5])
        s.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
        r.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
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
        a, b = [G2.add_v(bayesnet.BVertex(nm, False, 1)) \
                for nm in 'a b'.split()]
        for ep in [(a, b)]:
            G2.add_e(graph.DirEdge(len(G2.e), *ep))
        
        G2.init_distributions()
        a.set_distribution_parameters(mu = 1.0, sigma = 1.0)
        b.set_distribution_parameters(mu = 1.0, sigma = 1.0, wi = 2.0)
        
        self.a = a
        self.b = b
        self.G2 = G2

####class LearningTestCase(InferenceEngineTestCase):
####    """ Learning Test case """
####    def setUp(self):
####        InferenceEngineTestCase.setUp(self)   
####    
####    def testML(self):
####        # sample the network 2000 times
####        cases = self.BNet.Sample(2000)
####        
####        # create a new BNet with same nodes as self.BNet but all parameters
####        # set to 1s
####        G = copy.deepcopy(self.BNet)
####        
####        G.init_distributions()
####        
####        # create an infeence engine
####        engine = JoinTree(G)
####        
####        # learn according to the test cases
####        engine.LearnMLParams(cases)
####        
####        tol = 0.05
####        assert(na.alltrue([na.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
####               for v in G.all_v])), \
####                " Learning does not converge to true values "

class MCMCTestCase(InferenceEngineTestCase):
    """ MCMC unit tests.
    """
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = MCMCEngine(self.BNet, 2000)
        self.engine2 = MCMCEngine(self.G2, 5000)
    
    def testUnobservedDiscrete(self):
        """ DISCRETE: Compute and check the probability of 
        water-sprinkler given no evidence
        """
        res = self.engine.marginalise_all()
        
        cprob, sprob, rprob, wprob = res['c'], res['s'], res['r'], res['w']

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
        """ GAUSSIAN: Compute and check the marginals of a simple 
        gaussian network 
        """
        G = self.G2
        a, b = self.a, self.b
        engine = self.engine2
        
        res = engine.marginalise_all()
        
        #---TODO: find the true results and compare them...
    
    def testObservedDiscrete(self):
        """ DISCRETE: Compute and check the probability of 
        water-sprinkler given some evidence
        """
        self.engine.set_obs({'c':1,'s':0})
        res = self.engine.marginalise_all()
        
        cprob, sprob, rprob, wprob = res['c'], res['s'], res['r'], res['w']

        error = 0.05        
        assert(na.allclose(cprob.cpt, [0.0,1.0], atol=error) and \
               na.allclose(rprob.cpt, [0.2,0.8], atol=error) and \
               na.allclose(sprob.cpt, [1.0,0.0], atol=error) and \
               na.allclose(wprob.cpt, [ 0.278, 0.722], atol=error) ), \
               " Somethings wrong with MCMC inference with evidence "        

 
        
class JTreeTestCase(InferenceEngineTestCase):
    def setUp(self):
        InferenceEngineTestCase.setUp(self)
        self.engine = JoinTree(self.BNet)

    def testGeneral(self):
        """ Check that the overall algorithm works """
        c = self.engine.marginalise('c')
        r = self.engine.marginalise('r')
        s = self.engine.marginalise('s')
        w = self.engine.marginalise('w')

        assert(na.allclose(c.cpt, [0.5, 0.5]) and \
                na.allclose(r.cpt, [0.5, 0.5]) and \
                na.allclose(s.cpt, [0.7, 0.3]) and \
                na.allclose(w.cpt, [0.34909999, 0.65090001])), \
                " Somethings wrong with JoinTree inference engine"

    def testEvidence(self):
        """ check that evidence works """
        self.engine.set_obs({'c':1,'s':0})
        
        c = self.engine.marginalise('c')
        r = self.engine.marginalise('r')
        s = self.engine.marginalise('s')
        w = self.engine.marginalise('w')

        assert(na.allclose(c.cpt,[0.0,1.0]) and \
                na.allclose(r.cpt,[0.2,0.8]) and \
                na.allclose(s.cpt,[1.0,0.0]) and \
                na.allclose(w.cpt,[ 0.278, 0.722]) ), \
                " Somethings wrong with JoinTree evidence"        

    def testMarginaliseAll(self):
        res = self.engine.marginalise_all()
        
        assert(res.__class__.__name__ == 'dict' and \
               set(res.keys()) == set(self.BNet.v)), \
               " MarginaliseAll is not a correct dictionary "
               

    ###########################################################
    ### SHOULD ADD A MORE GENERAL TEST:
    ###     - not only binary nodes
    ###     - more complex structure
    ###     - check message passing
    ###########################################################


if __name__ == "__main__":
    unittest.main() 
