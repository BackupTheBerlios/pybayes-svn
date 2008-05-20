#!/usr/bin/env python
"""
This module is used to test the different type of vertex
"""
# Copyright (C) 2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

import unittest

from openbayes.vertex import *

class DiscreteTest(unittest.TestCase):
    """
    This class test for the simple DiscreteVertex
    """
    def setUp(self):
        self.n1 = DiscreteVertex(1)
        self.n2 = DiscreteVertex(2, 3)

    def test_discrete(self):
        self.assert_(self.n1.discrete)
        self.assert_(self.n2.discrete)

    def test_neq(self):
        self.assertNotEqual(self.n1, self.n2)
        self.assertNotEqual(self.n1, 2)
        self.assertNotEqual(self.n1, '1')

    def test_eq(self):
        self.assertEqual(self.n1, 1)
        self.assertEqual(self.n2, 2)

    def test_set_parents(self):
        self.n1.set_parents([self.n2])
        self.assertEqual(self.n1.cpt.names_list, [1, 2])
        self.assertEqual(self.n1.cpt.shape, (2, 3))

    def test_sample(self):
        """
        This test is randomize and can therefor while sometime
        """
        self.n1.set_parents([self.n2])
        self.n1.cpt[{2:2}] = [0.5, 0.5]
        samples = 0
        for _ in xrange(30000):
            samples += self.n1.sample({2:2})
        self.assertAlmostEqual(samples / 30000.0, 0.5, 2)

if __name__ == '__main__':
    unittest.main()
