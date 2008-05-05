#!/usr/bin/env python
"""
The test for table.py
"""

import unittest
from copy import copy

import numpy
from OpenBayes.table import Table

class TableTestCase(unittest.TestCase):
    """
    The unique test case in this file
    """
    def setUp(self):
        names = ('a', 'b', 'c')
        shape = (2, 3, 4)
        self.table_a = Table(names, shape, dtype='Float32')
        self.table_b = Table(names, shape[1:], dtype='Float32')
        self.names = names
        self.shape = shape
   
    def test_eq(self):
        a = Table(['a', 'b'], [2, 3], range(6), 'Float32')
        b = Table(['a', 'b'], [2, 3], range(6), 'Float32')
        c = Table(['a'], [6], range(6), 'Float32')
        d = numpy.arange(6).reshape(2, 3)
        e = Table(['b', 'a'], [3, 2], numpy.transpose(a.cpt))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, d)
        self.assertEqual(a, e)
        self.assertEqual(e, a)

    def test_imul(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], \
                 range(2*3*4*5*6))
   
        bcpt = b.cpt[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose([3, 1, 0, 4, 2])
        res = bcpt*c.cpt
        c *= b
        self.assert_(numpy.all(c.cpt == res))

    def test_idiv(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
        bcpt = b.cpt[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose([3, 1, 0, 4, 2])
        res = c.cpt/bcpt
        res[numpy.isnan(res)] = 0.0
        c /= b
        self.assert_(numpy.all(c.cpt == res))
              
    def test_mul(self):
        a = Table(['a', 'b', 'c', 'd'], [2, 3, 4, 5], range(2*3*4*5))
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
   
        acpt = a.cpt[..., numpy.newaxis]
        bcpt = b.cpt[..., numpy.newaxis, numpy.newaxis]
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
        acpt = copy(a.cpt)[..., numpy.newaxis]
        bcpt = copy(b.cpt)[..., numpy.newaxis, numpy.newaxis]
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
             
    def test_delegate(self):
        self.assert_(numpy.alltrue(self.table_a.flat == self.table_a.cpt.flat))

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

    def test_add_dim(self):
        a = Table('a b c'.split())
        a.add_dim('d')

        assert(a.names == set('a b c d'.split()) and \
               a.names_list == 'a b c d'.split() and \
               a.assocdim.has_key('d') and \
               a.assocname.has_key(3)), \
               "add Dim does not work correctly..."
 
    def test_union(self):
        """ test Union between two Tables """
        a = Table(['a','b','c','d'], [2,3,4,5], range(2*3*4*5))
        b = Table(['c','b','e'], [4,3,6], range(12*6))
        ab, bb = a.union(b)
        assert(ab.names_list == ['a','b','c','d','e'] and \
               ab.shape == tuple([2,3,4,5,1]) and \
               numpy.all(bb == numpy.transpose(b.cpt[..., numpy.newaxis,numpy.newaxis], axes=[3,1,0,4,2]))), \
               """ union does not work ..."""
 
    def testPrepareOther(self):
        c = Table(['e','b'], [2,3], range(6))
        d = Table(['a','b','c','d','e'], [2,3,2,2,2], range(3*2**4))
        e = Table(['e','b','f'], [2,3,4], range(6*4))
        src = Table(['s','r','c'], [2,3,4], range(24))
        cr = Table(['c','r'], [4,3], range(12))
        
        dc = d.prepare_other(c)
        self.assertRaises(ValueError, d.prepare_other, e)
        cr_ = src.prepare_other(cr)
        assert(dc.shape == (1, 3, 1, 1, 2))
        assert((dc[0, :, 0, 0, :] == numpy.transpose(c.cpt, axes=[1, 0])).all())
        assert(cr_.shape == (1,3,4))       
      
if __name__ == '__main__':
    unittest.main()
##    a*c
##    print a
##    print c

    #a*b
    #print 'mul'
    #print a


