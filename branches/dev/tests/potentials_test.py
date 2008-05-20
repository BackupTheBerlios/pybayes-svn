#!/usr/bin/env python
"""
This is the test module for potential.py
"""

import unittest
import numpy

from openbayes.potentials import GaussianPotential, DiscretePotential

class GaussianPotentialTestCase(unittest.TestCase):
    """
    This test the Gaussian potential
    """
    def setUp(self):
        """
        This function creates all the needed test variables
        """
        names = ('a', 'b')
        shape = (1, 2)
        self.potential_a = GaussianPotential(names, shape)  
        g = 2
        h = [1, 2, 3]
        k = range(9)
        self.potential_b = GaussianPotential(names, shape, g, h, k)
    
    def test_init(self):
        a = self.potential_a
        b = self.potential_b
        self.assertEquals(a.g, 0.0)
        self.assert_(numpy.allclose(a.h, numpy.zeros(3)))
        self.assert_(numpy.allclose(a.K, numpy.zeros(shape=(3, 3))))
        self.assert_(b.g, 2.0)
        self.assert_(numpy.allclose(b.h, numpy.arange(1, 4)))
        self.assert_(numpy.allclose(b.K, numpy.arange(9).reshape((3, 3))))   

class DiscretePotentialTestCase(unittest.TestCase):
    """
    This test the discrete potential 
    """
    def setUp(self):
        names = ('a', 'b', 'c')
        shape = (2, 3, 4)
        self.potential = DiscretePotential(names, shape, numpy.arange(24))
        self.names = names
        self.shape = shape
   
    def test_marginalise(self):
        var = set('c')
        b = self.potential.marginalise(var)
        var2 = set(['c', 'a'])
        c = self.potential.marginalise(var2)
        # extended test
        a = DiscretePotential('a b c d e f'.split(), [2, 3, 4, 5, 6, 7], \
                              numpy.arange(7*6*5*4*3*2))
        aa = a.marginalise('c f a'.split())
      

        self.assertEqual(b.names, self.potential.names - var)
        self.assertEqual(b[0, 1], numpy.sum(self.potential[0, 1]))
        self.assertEqual(c.names, self.potential.names - var2)
        self.assert_(numpy.alltrue(c.cpt.flat == 
                                   numpy.sum(numpy.sum(self.potential.cpt,
                                                       axis=2), 
                                             axis=0)))
        self.assertEqual(aa.shape, (3, 5, 6)) 
        self.assertEqual(aa.names_list, 'b d e'.split()) 
        self.assertEqual(aa[2, 4, 3], numpy.sum(a[:, 2, :, 4, 3, :].flat))

    def test_add(self):
        d = DiscretePotential(['b', 'c'], [3, 4], numpy.arange(12))
        self.assertEqual(self.potential + d, 
                         self.potential.marginalise(['a']))
    
    def test_int_eq_index(self):
        self.potential[1, 1, 1] = -2
        self.potential[self.potential == -2] = -3
        self.assertEquals(self.potential[1, 1, 1], -3)

    def test_all(self):
        """ this is actually the Water-sprinkler example """
        c = DiscretePotential(['c'], [2], [0.5, 0.5])
        s = DiscretePotential(['s', 'c'], [2, 2], [0.5, 0.9, 0.5, 0.1])
        r = DiscretePotential(['r', 'c'], [2, 2], [0.8, 0.2, 0.2, 0.8])
        w = DiscretePotential(['w', 's', 'r'], [2, 2, 2])
        w[:, 0, 0] = [0.99, 0.01]
        w[:, 0, 1] = [0.1, 0.9]
        w[:, 1, 0] = [0.1, 0.9]
        w[:, 1, 1] = [0.0, 1.0]

        cr = c * r        # Pr(C,R)     = Pr(R|C) * Pr(C)
        crs = cr * s      # Pr(C,S,R)   = Pr(S|C) * Pr(C,R)
        crsw = crs * w    # Pr(C,S,R,W) = Pr(W|S,R) * Pr(C,R,S)

        # this can be verified using any bayesian network software

        # check the result for the multiplication and marginalisation
        self.assert_(numpy.allclose(crsw.marginalise('s r w'.split()).cpt,
                                    [0.5, 0.5]))
        self.assert_(numpy.allclose(crsw.marginalise('c r w'.split()).cpt, 
                                    [0.7, 0.3]))
        self.assert_(numpy.allclose(crsw.marginalise('c s w'.split()).cpt, 
                                    [0.5, 0.5]))
        self.assert_(numpy.allclose(crsw.marginalise('c s r'.split()).cpt, 
                                    [0.349099, 0.6509]))

if __name__ == '__main__':
    unittest.main()

#    names = ('a','b','c')
#    shape = (2,3,4)
#    a = DiscretePotential(names,shape,na.arange(24))
#
#    names = ('a','d','b')
#    shape = (2,5,3)
#    b = DiscretePotential(names,shape,na.arange(2*5*3))
#
#    c = DiscretePotential(['c'],[2],[0.5,0.5])
#    s = DiscretePotential(['s','c'],[2,2],[0.5, 0.9, 0.5, 0.1])
#    r = DiscretePotential(['r','c'],[2,2],[0.8,0.2,0.2,0.8])
#    w = DiscretePotential(['w','s','r'],[2,2,2])
#    w[:,0,0]=[0.99, 0.01]
#    w[:,0,1]=[0.1, 0.9]
#    w[:,1,0]=[0.1, 0.9]
#    w[:,1,1]=[0.0, 1.0]
#
#    cr = c*r
#    crs = cr*s
#    crsw = crs*w
#
#    print 'c:', crsw.marginalise('s r w'.split())
#    print 's:', crsw.marginalise('c r w'.split())
#    print 'r:', crsw.marginalise('c s w'.split())
#    print 'w:', crsw.marginalise('c s r'.split())
