#!/usr/bin/env python
"""
This test unit is responsible for covering distribution
"""
# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

import unittest

import numpy

from openbayes.distributions import *
from openbayes.distributions import Distribution
from openbayes import graph, BVertex, BNet

#=================================================================
#	Test case for Distribution class
#=================================================================
class DistributionTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        # create a small BayesNet
        g = BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [(BVertex(nm, discrete=True, nvalues=nv)) for nm, nv
                      in zip('c s r w'.split(), [2, 3, 4, 2])]
        for edge in [(c, r), (c, s), (r, w), (s, w)]:
            g.add_edge(edge)
        g.init_distributions()
        self.network = g

    def test_str(self):
        c = Distribution(BVertex("a", 2))        
        self.assertEqual(str(c), "Distribution for node : a\n"
                                 "Type : None")

    def test_default(self):
        """ We test the defaul value of __init__"""
        c = Distribution(BVertex("a", 2))
        self.assertEqual(c.names_list, ["a"])
        self.assertEqual(c.is_adjustable, False)
        self.assertEquals(c.family, ["a"])

    def test_family(self):
        """ test parents, family, etc... """
        g = self.network
        c, s, r, w = g.v['c'], g.v['s'], g.v['r'], g.v['w']
        self.assertEqual(c.distribution.parents, [])
        self.assertEqual(set(w.distribution.parents), set([r, s]))
        self.assertEqual(r.distribution.parents, [c])
        self.assertEqual(s.distribution.parents, [c])
        self.assertEqual(c.distribution.family, [c])
        self.assertEqual(set(s.distribution.family), set([c, s]))
        self.assertEqual(set(r.distribution.family), set([r, c]))
        self.assertEqual(set(w.distribution.family), set([w, r, s]))
        self.assertEqual(c.distribution.nvalues, c.nvalues)

#=================================================================
#	Test case for GaussianDistribution class
#=================================================================
class GaussianTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        # create a small BayesNet
        self.G = G = BNet('Test')

        self.a = a = BVertex('a', discrete=False, nvalues=1)
        self.b = b = BVertex('b', discrete=False, nvalues=2)
        self.c = c = BVertex('c', discrete=False, nvalues=1)
        self.d = d = BVertex('d', discrete=True, nvalues=2)
        self.e = e = BVertex('e', discrete=True, nvalues=3)
        self.f = f = BVertex('f', discrete=False, nvalues=1)
        self.g = g = BVertex('g', discrete=False, nvalues=1)

        for edge in [(a, c), (b, c), (d, f), (e, f), 
                   (a, g), (b, g), (d, g), (e, g)]:
            G.add_edge(edge)

        #a,b : continuous(1,2), no parents
        #c	 : continuous(3), 2 continuous parents (a,b)
        #d,e : discrete(2,3), no parents
        #f	 : continuous(1), 2 discrete parents (d,e)
        #g	 : continuous(1), 2 discrete parents (d,e) & 
        #      2 continuous parents (a,b)

        G.init_distributions()

        self.ad = a.distribution
        self.bd = b.distribution
        self.cd = c.distribution
        self.fd = f.distribution
        self.gd = g.distribution

    def test_random(self):
        """ tests that the random number generator is correct """
        self.ad.set_parameters(sigma=0.1, mu = 1.0)
        for _ in range(1000):
            r = self.ad.random()
            assert(r[0] >= float(self.ad.mean[0] - 5 * self.ad.sigma[0]) and \
                   r[0] <= float(self.ad.mean[0] + 5 * self.ad.sigma[0])), \
                   """ random generation is out of borders """

        self.bd.set_parameters(sigma = [0.1, 0.0, 0, 1], mu = [1, -1])

        for _ in range(1000):
            r = self.bd.random()
            assert(r[0] >= float(self.bd.mean[0] - 5 * self.bd.sigma.flat[0]) and
                   r[0] <= float(self.bd.mean[0] + 5 * self.bd.sigma.flat[0]) and
                   r[1] >= float(self.bd.mean[1] - 5 * self.bd.sigma.flat[-1]) and
                   r[1] <= float(self.bd.mean[1] + 5 * self.bd.sigma.flat[-1])), \
                   """ random generation is out of borders """
    
    def test_no_parents(self):
        ad = self.ad
        bd = self.bd
        # a and b have no parents, test basic parameters
        self.assertEqual(ad.mean.shape, (1,))
        self.assertEqual(bd.mean.shape, (2,))
        self.assertEqual(ad.sigma.shape, (1, 1))
        self.assertEqual(bd.sigma.shape, (2, 2))
        self.assertEqual(ad.weights.shape, (1, 0))
        self.assertEqual(bd.weights.shape, (2, 0))
        
    def test_continuous_parents(self):
        """ test a gaussian with continuous parents """
        cd = self.cd
        #c has two continuous parents
        self.assertEqual(cd.mean.shape, (cd.nvalues, ))
        self.assertEquals(cd.sigma.shape, (cd.nvalues, cd.nvalues))
        self.assertEqual(cd.weights.shape, tuple([cd.nvalues]+cd.parents_shape))

    def test_discrete_parents(self):
        """ test a gaussian with discrete parents """
        fd = self.fd
        self.assertEqual(fd.mean.shape,
                         tuple([fd.nvalues] + fd.discrete_parents_shape))
        self.assertEqual(fd.sigma.shape, 
                         tuple([fd.nvalues, fd.nvalues] + 
                               fd.discrete_parents_shape))
        self.assertEqual(fd.weights.shape,
                        tuple([fd.nvalues] + fd.discrete_parents_shape))

    def test_discrete_and_Continuous_Parents(self):
        gd = self.gd
        self.assertEqual(gd.mean.shape, 
                         tuple([gd.nvalues] + gd.discrete_parents_shape))
        self.assertEqual(gd.sigma.shape,
                         tuple([gd.nvalues,gd.nvalues] + 
                               gd.discrete_parents_shape))
        self.assertEqual(gd.weights.shape,
                         tuple([gd.nvalues] + gd.parents_shape))

    def test_counting(self):
        a = self.a.distribution
        a.initialize_counts()

        self.assertEqual(a.samples, list(),
                          "Samples list not initialized correctly")

        a.incr_counts(range(10))
        a.incr_counts(5)

        self.assertEqual(a.samples, range(10) + [5],
               "Elements are not added correctly to Samples list")

        a.set_counts()
        self.assertAlmostEquals(a.mean,4.545454, 5)
        self.assertAlmostEquals(a.sigma, 2.876234, 4)

    def testIndexing(self):
        fd = self.f.distribution
        fd.set_parameters(mu=range(6), sigma=range(6))
        #self.assert_(numpy.allclose(fd[0][0].flat, numpy.arange(3)))
        #self.assert_(numpy.allclose(fd[0][1].flat, numpy.arange(3)))
        #self.assert_(numpy.allclose(fd[1,2][0].flat, numpy.array(5)))
        #self.assert_(numpy.allclose(fd[1,2][1].flat, numpa.array(5)))
        # test dict indexing
        print fd[{'d':0}][0]
        print fd

        self.assert_(numpy.all(fd[{'d':0}][0].flat == numpy.arange(3)))
        self.assert_(numpy.allclose(fd[{'d':0}][1].flat, numpy.array(range(3), dtype='Float32')))
        self.assert_(numpy.allclose(fd[{'d':1,'e':2}][0].flat, numpy.array(5, dtype='Float32')))
        self.assert_(numpy.allclose(fd[{'d':1,'e':2}][1].flat, numpy.array(5, dtype='Float32')))

        # now test setting of parameters
        fd[{'d':1, 'e':2}] = {'mean':0.5, 'sigma':0.6}
        fd[{'d':0}] = {'mean':[0, 1.2, 2.4], 'sigma':[0, 0.8, 0.9]}
        numpy.allclose(fd[{'d':0}][0].flat, numpy.array([0,1.2,2.4],dtype='Float32'))
        self.assert_(numpy.allclose(fd[{'d':0}][0].flat, numpy.array([0,1.2,2.4],dtype='Float32')) and
               numpy.allclose(fd[{'d':0}][1].flat,numpy.array([0,0.8,0.9],dtype='Float32')) and
               numpy.allclose(fd[{'d':1,'e':2}][0].flat, numpy.array(0.5, dtype='Float32')) and
               numpy.allclose(fd[{'d':1,'e':2}][1].flat, numpy.array(0.6, dtype='Float32')),
        "Setting of values using dict does not seem to work...")

        # now run tests for continuous parents
        cd = self.c.distribution	# 2 continuous parents a(1),b(2)

        cd[{'a':0, 'b':1}] = {'weights':69}
        self.assert_(numpy.allclose(cd[{'a':0, 'b':1}][2], 69.0))

    def testSampling(self):
        " Test the sample() function "
        a = self.a.distribution
        a.set_parameters(mu=5, sigma=1)
        samples = [a.sample() for _ in range(5000)]
        # verify values
        b = self.a.get_sampling_distribution()
        b.initialize_counts()
        b.incr_counts(samples)
        b.set_counts()
        self.assertAlmostEqual(b.mean, a.mean, 1)
        self.assertAlmostEqual(b.sigma, a.sigma, 1)
        

#=================================================================
#	Test case for Multinomial_Distribution class
#=================================================================
class MultinomialTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        # create a small BayesNet, Water-Sprinkler
        g = BNet('Test')

        a, b, c, d = [BVertex(nm, discrete=True, nvalues=nv) for 
                      nm, nv in zip('a b c d'.split(), [2, 3, 4, 2])]
        # sizes = (2,3,4,2)
        # a has 3 parents, b,c and d
        for edge in [(b, a), (c, a), (d, a)]:
            g.add_edge(edge)

        g.init_distributions()
        self.network = g
        self.a, self.b, self.c, self.d = a, b, c, d

        
    def test_normalize(self):
        a = MultinomialDistribution(self.network.v['a'])
        a.set_parameters(range(48))
        a.normalize()

    # the test below fails
    # >>> a.distribution.sizes
    # [2, 4, 3, 2]
    def testSizes(self):
        self.assertEqual(self.a.distribution.sizes, [2, 3, 4, 2])

    # test the indexing of the cpt
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked 
        correctly, by getting things out of the matrix
        """
        self.assert_((self.a.distribution[0, 0, 0, :] == self.a.distribution.cpt[0, 0, 0, :]).all()) 
        self.assert_((self.a.distribution[1, 0, 0, :] == self.a.distribution.cpt[1, 0, 0, :]).all())

    def testSetCPT(self):
        """ Violate abstraction and check that we can actually set elements.
        """
        self.a.distribution.cpt[0, 1, 0, :] = numpy.array([4, 5])
        self.assert_((self.a.distribution[0, 1, 0, :] == numpy.array([4, 5])).all())

    def testDictIndex(self):
        """ test that an index using a dictionary works correctly
        """
        index = {'a':0, 'b':0, 'c':0}
        index2 = {'a':1, 'b':0, 'c':0}
        self.assert_((self.a.distribution[0, 0, 0, :] == self.a.distribution[index]).all())
        self.assert_(numpy.all((self.a.distribution[1, 0, 0, :] == self.a.distribution[index2])))

    # the test below fails
    # the case of index3 fails because the order of the nodes
    # is not a, b, c, d, but a, c, b, d
    # >>> print G.v
    # {'a': <OpenBayes.bayesnet.BVertex object at 0x879e10c>, 
    #  'c': <OpenBayes.bayesnet.BVertex object at 0x87a628c>, 
    #  'b': <OpenBayes.bayesnet.BVertex object at 0x87a648c>, 
    #  'd': <OpenBayes.bayesnet.BVertex object at 0x87a64cc>}
    def testDictSet(self):
        """ test that an index using a dictionary can set a value 
        within the cpt 
        """
        index = {'a':0, 'b':0, 'c':0}
        index2 = {'a':1, 'b':0, 'c':0}
        index3 = {'a':1, 'b':1, 'c':0}
        self.a.distribution[index] = -1
        self.a.distribution[index2] = 100
        self.a.distribution[index3] = numpy.array([-2, -3])
        self.assert_(numpy.all(self.a.distribution[0, 0, 0, :] == numpy.array([-1, -1])))
        self.assert_(numpy.all(self.a.distribution[1, 0, 0, :] == numpy.array([100, 100])))
        self.assert_(numpy.all(self.a.distribution[1, 1, 0, :] == numpy.array([-2, -3])))      
    
    def testNumIndex(self):
        """ test that a raw index of numbers works correctly
        """
        assert(numpy.all(self.a.distribution[0, :, 0, :] == self.a.distribution[0, :, 0, :]) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == self.a.distribution[1, 0, 0, :])), \
               "Error getting item with num indices"

    def testNumSet(self):
        """ test that a raw index of numbers can access and set a position in the 
        """
        self.a.distribution[0, 0, 0, :] = -1
        self.a.distribution[1, 0, 0, :] = 100
        self.a.distribution[1, 1, 0, :] = numpy.array([-2, -3])
        assert(numpy.all(self.a.distribution[0, 0, 0, :] == \
               numpy.array([-1, -1])) and \
               numpy.all(self.a.distribution[1, 0, 0, :] == \
               numpy.array([100, 100])) and \
               numpy.all(self.a.distribution[1, 1, 0, :] == \
               numpy.array([-2, -3]))), \
               "Error Setting cpt with num indices"


if __name__ == "__main__":
    unittest.main()
