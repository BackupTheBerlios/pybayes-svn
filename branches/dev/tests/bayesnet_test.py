#!/usr/bin/env python

import unittest

import numpy

from openbayes.bayesnet import BNet, BVertex
from openbayes import graph

class BNetTestCase(unittest.TestCase):
    """ Basic Test Case suite for BNet
    """
    def setUp(self):
        g = BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [g.add_v(BVertex(name, True, 2)) 
                      for name in 'c s r w'.split()]
        for start, end in [(c, r), (c, s), (r, w), (s, w)]:
            g.add_e(graph.DirEdge(len(g.e), start, end))
        g.init_distributions()
        c.set_distribution_parameters([0.5, 0.5])
        s.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
        r.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
        w.set_distribution_parameters([1, 0.1, 0.1, 0.01, 0.0, 0.9, 
                                     0.9, 0.99])    
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.network = g

    def testTopoSort(self):
        sorted_ = self.network.topological_sort()
        assert(sorted_[0] == self.c and \
               sorted_[1] == self.s and \
               sorted_[2] == self.r and \
               sorted_[3] == self.w), \
               "Sorted was not in proper topological order"

    def test_sample(self):
        c_cpt = self.c.distribution
        s_cpt = self.s.distribution
        r_cpt = self.r.distribution
        w_cpt = self.w.distribution
        
        c_cpt.initialize_counts() 
        s_cpt.initialize_counts()
        r_cpt.initialize_counts()
        w_cpt.initialize_counts()

        for _ in range(1000):
            sample = self.network.sample()[0]
            # Can use sample in all of these, it will ignore extra variables
            c_cpt.incr_counts(sample) 
            s_cpt.incr_counts(sample) 
            r_cpt.incr_counts(sample) 
            w_cpt.incr_counts(sample)        
##            cCPT[sample] += 1
##            sCPT[sample] += 1
##            rCPT[sample] += 1
##            wCPT[sample] += 1
        assert(numpy.allclose(c_cpt,self.c.distribution.cpt,atol=.1) and \
               numpy.allclose(s_cpt,self.s.distribution.cpt,atol=.1) and \
               numpy.allclose(r_cpt,self.r.distribution.cpt,atol=.1) and \
               numpy.allclose(w_cpt,self.w.distribution.cpt,atol=.1)), \
               "Samples did not generate appropriate CPTs"
    
    def test_damily(self):
        c_family = self.network.v['c'].family
        s_family = self.network.v['s'].family
        r_family = self.network.v['r'].family
        w_family = self.network.v['w'].family
        
        assert(set(c_family) == set([self.c]) and \
               set(s_family) == set([self.s, self.c]) and \
               set(r_family) == set([self.r, self.c]) and \
               set(w_family) == set([self.w, self.r, self.s])), \
              "Families are not set correctly"
    
if __name__ == '__main__':
    unittest.main()
