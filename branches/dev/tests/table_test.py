#!/usr/bin/env python
"""
The test for table.py
"""

import unittest
from copy import copy

import numpy
from openbayes.table import Table
from openbayes.tests.utils import ExtendedTestCase

class TableTestCase(ExtendedTestCase):
    """
    The unique test case in this file
    """
    def setUp(self):
        names = ('a', 'b', 'c')
        shape = (2, 3, 4)
        self.table_a = Table(names, shape, dtype='Float32')
        self.names = names
        self.shape = shape
   
    def test_eq(self):
        a = Table(['a', 'b'], [2, 3], range(6), 'Float32')
        b = Table(['a', 'b'], [2, 3], range(6), 'Float32')
        c = Table(['a'], [6], range(6), 'Float32')
        d = numpy.arange(6).reshape(2, 3)
        e = Table(['b', 'a'], [3, 2], numpy.transpose(a))
        self.assertAllEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertAllEqual(a, d)
        self.assertAllEqual(a, e)
        self.assertAllEqual(e, a)
    """
    def test_imul(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], \
                 range(2*3*4*5*6))
   
        bcpt = b[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose([3, 1, 0, 4, 2])
        res = bcpt*c
        c *= b
        self.assertAllEqual(c, res)

    def test_idiv(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
        bcpt = b[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose([3, 1, 0, 4, 2])
        res = c/bcpt
        res[numpy.isnan(res)] = 0.0
        c /= b
        self.assertAllEqual(c, res)
              
    def test_mul(self):
        a = Table(['a', 'b', 'c', 'd'], [2, 3, 4, 5], range(2*3*4*5))
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
   
        acpt = a[..., numpy.newaxis]
        bcpt = b[..., numpy.newaxis, numpy.newaxis]
        bcpt = numpy.transpose(bcpt, [3, 1, 0, 4, 2])
        resab = acpt * bcpt
        ab = a * b
        cc = c * c
        bb = b * b
        self.assertEqual(ab, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], 
                                   resab))
        self.assertEqual(cc, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6],
                                    numpy.arange(2*3*4*5*6)**2))
        self.assertEqual(bb, Table(['c','b','e'], [4,3,6], numpy.arange(12*6)**2))

    def test_div(self):
        a = Table(['a', 'b', 'c', 'd'], [2, 3, 4, 5], range(2*3*4*5))
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(4*3*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
        acpt = copy(a)[..., numpy.newaxis]
        bcpt = copy(b)[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose(3, 1, 0, 4, 2)
        ab = a/b
        cc = c/c
        bb = b/b
        cres = numpy.ones(2*3*4*5*6)
        cres[0] = 0
        bres = numpy.ones(12*6)
        bres[0] = 0
        ares = acpt/bcpt
        ares[numpy.isnan(ares)] = 0.0
        self.assertEqual(ab, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], 
                                   ares))
        self.assertEqual(cc, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], 
                                   cres))
        self.assertEqual(bb, Table(['c', 'b', 'e'], [4, 3, 6], bres))
    """    

    def test_basic_index(self):
        self.assertEqual(self.table_a[1,1,1], 1.0)
   
    def test_dict_index(self):
        index = dict(zip(self.names, (1,1,1)))
        self.assertEqual(self.table_a[index], self.table_a[1,1,1])
   
    def test_basic_set(self):
        self.table_a[1, 1, 1] = 2.0
        self.assertEqual(self.table_a[1, 1, 1], 2)
   
    def tes_dict_set(self):
        index = dict(zip(self.names,(1,1,1)))
        self.table_a[index] = 3.0
        self.assertEqual(self.table_a[index], self.table_a[1,1,1])
        self.assertEqual(self.table_a[index] == 3.0)

    def test_copy(self):
        a = Table('a b c'.split())
        b = a.copy()
        self.assertEqual(b.names_list, ['a', 'b', 'c'])

    def test_add_dim(self):
        a = Table('a b c'.split())
        a.add_dim('d')
        self.assertEqual(a.names, set('a b c d'.split()))
        self.assertEqual(a.names_list, 'a b c d'.split()) 
        self.assert_(a.assocdim.has_key('d'))
        self.assert_(a.assocname.has_key(3))
        self.assertEqual(a.shape ,(2,2,2,1))

    def test_transpose(self):
        a = Table("a b".split(), None, range(4))
        b = a.transpose()
        # only the copy must be changed
        self.assertEqual(b.names_list, ['b', 'a'])
        self.assertEqual(a.names_list, ['a', 'b'])


    def test_union(self):
        """ test Union between two Tables """
        a = Table(['a','b','c','d'], [2,3,4,5], range(2*3*4*5))
        b = Table(['c','b','e'], [4,3,6], range(12*6))
        ab, bb = a.union(b)
        self.assertEqual(ab.names_list, ['a','b','c','d','e'])
        self.assertEqual(ab.shape, tuple([2,3,4,5,1]))
        b.add_dim("a")
        b.add_dim("d")
        self.assertAllEqual(b, bb)

    def test_prepare_other(self):
        c = Table(['e','b'], [2,3], range(6))
        d = Table(['a','b','c','d','e'], [2,3,2,2,2], range(3*2**4))
        e = Table(['e','b','f'], [2,3,4], range(6*4))
        src = Table(['s','r','c'], [2,3,4], range(24))
        cr = Table(['c','r'], [4,3], range(12))
        
        dc = d.prepare_other(c)
        self.assertRaises(ValueError, d.prepare_other, e)
        cr_ = src.prepare_other(cr)
        self.assertEqual(dc.shape, (1, 3, 1, 1, 2))
        assert((dc[0, :, 0, 0, :] == numpy.transpose(c, axes=[1, 0])).all())
        assert(cr_.shape == (1,3,4))       
      
if __name__ == '__main__':
    unittest.main()


