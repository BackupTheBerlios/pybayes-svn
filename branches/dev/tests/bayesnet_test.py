#!/usr/bin/env python

# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

import unittest

import numpy

from openbayes.bayesnet import BNet, BVertex
from openbayes.vertex import DiscreteVertex
from openbayes import graph
from openbayes.tests.utils import ExtendedTestCase

class BNetTestCase(ExtendedTestCase):
    """ Basic Test Case suite for BNet
    """
    def setUp(self):
        g = BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [g.add_vertex(DiscreteVertex(name, 2)) 
                      for name in 'c s r w'.split()]
        for start, end in [(c, r), (c, s), (r, w), (s, w)]:
            g.add_edge( (start, end))
        g.finalize()
        c.cpt.set_values([0.5, 0.5])
        s.cpt.set_values([0.5, 0.9, 0.5, 0.1])
        r.cpt.set_values([0.8, 0.2, 0.2, 0.8])
        w.cpt.set_values([1, 0.1, 0.1, 0.01, 0.0, 0.9, 
                                     0.9, 0.99])    
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.network = g

    def testTopoSort(self):
        sorted_ = self.network.topological_order()
        self.assertEqual(sorted_[0], self.c)
        self.assertEqual(sorted_[3], self.w)
        self.assertEqual(set(sorted_[1:3]), set([self.s, self.r]))

    def test_sample(self):
        """
        This test sampling a network,. Because of the inhernet
        randomness of sampling, this test may fail ...
        """
        c_cpt = self.c.cpt.copy()
        s_cpt = self.s.cpt.copy()
        r_cpt = self.r.cpt.copy()
        w_cpt = self.w.cpt.copy()
        
        c_cpt.zeros() 
        s_cpt.zeros()
        r_cpt.zeros()
        w_cpt.zeros()

        for _ in range(10000):
            sample = self.network.sample()[0]
            # Can use sample in all of these, it will ignore extra variables
            c_cpt[sample] +=  1
            s_cpt[sample] += 1 
            r_cpt[sample] += 1 
            w_cpt[sample] += 1

        c_cpt.normalize()
        self.assertAllClose(c_cpt, self.c.cpt, .1)  
        s_cpt.normalize('s')
        self.assertAllClose(s_cpt, self.s.cpt, .1)
        r_cpt.normalize('r')
        self.assertAllClose(r_cpt, self.r.cpt, .1)
        w_cpt.normalize('w')
        self.assertAllClose(w_cpt, self.w.cpt, .20)
    
    def test_family(self):
        c_family = self.network.family(self.c)
        s_family = self.network.family(self.s)
        r_family = self.network.family(self.r)
        w_family = self.network.family(self.w)
        
        self.assertEqual(set(c_family), set([self.c]))
        self.assertEqual(set(s_family), set([self.s, self.c]))
        self.assertEqual(set(r_family), set([self.r, self.c]))
        self.assertEqual(set(w_family), set([self.w, self.r, self.s]))
    
if __name__ == '__main__':
    unittest.main()
