#!/usr/bin/env python
"""
The test for table.py
"""
# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

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
        self.table_a = Table(names, shape, range(2*3*4), dtype='Float32')
        self.names = names
        self.shape = shape
  
    def test_getitem(self):
        self.failIf(isinstance(self.table_a[1,2,3],Table))
        self.assert_(isinstance(self.table_a[0:1,2,3], numpy.ndarray))
        self.assertEqual(self.table_a[0:1,0,0], 0)

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

    def test_normalize(self):
        a = Table(['a'], [2], range(2))
        a.normalize()
        self.assertAllEqual(a, numpy.array([0,1]))
        b = Table(['a','b','c'], [3,3,3], range(27))
        b.normalize('a')
   
    def test_num_index_from_dict(self):
        a = Table(['a','b'])
        index = {'a':1,'b':1}
        self.assertEqual(a._num_index_from_dict(index),([], (1,1)))
        index['b'] = 0
        self.assertEqual(a._num_index_from_dict(index),([], (1,0)))
        self.assertEqual(a[index] , 1)
        index['c'] = 123
        self.assertEqual(a[index], 1)
        b = Table(['b'])
        index = {'a':1,'b':1}

    def test_increment(self):
        a = Table(['a','b'])
        a[{'a':1,'b':1}] += 1
        self.assertAllEqual(a, numpy.array([[1,1],[1,2]]))
        a[1,0] += 1
        self.assertAllEqual(a, numpy.array([[1,1],[2,2]]))
        a[0,0] += 1
        self.assertAllEqual(a, numpy.array([[2,1],[2,2]]))
    
   
    def test_imul(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], \
                 range(2*3*4*5*6))
        d = c.copy()
        c *= b
        self.assertAllEqual(c, d*b)
    
    def test_idiv(self):
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(12*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))
        bcpt = b[..., numpy.newaxis, numpy.newaxis]
        bcpt = bcpt.transpose([3, 1, 0, 4, 2])
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
        self.assertAllEqual(ab, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], 
                                   resab))
        self.assertAllEqual(cc, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6],
                                    numpy.arange(2*3*4*5*6)**2))
        self.assertAllEqual(bb, Table(['c','b','e'], [4,3,6], numpy.arange(12*6)**2))

    def test_div(self):
        """
        We need to test the division function
        """
        a = Table(['a', 'b'], [2,3], range(6))
        self.assertAllEqual( a/1, a)
        self.assertAllClose(a/2, [[0, 0.5, 1],[1.5, 2, 2.5]])
        b = Table(['b'], [3], range(1,4))
        self.assertAllClose(a/b, [[0, 0.5 , 2/3.0],[3, 2 , 5/3.0]])
        b = Table(['a'], [2], range(1,3))
        self.assertAllClose(a/b, [[0,1,2],[3,2,5/3.0]])
        
        a = Table(['a', 'b', 'c'], [2,2,2], [[[1,1] ,[2,2]],[[3,3],[4,4]]])
        self.assertAllClose(a/a.sum("a"), 
                            [[[0.25, 0.25], [1/3.0]*2], 
                             [[0.75, 0.75],[2/3.0]*2] ])
        a = Table(['a', 'b', 'c', 'd'], [2, 3, 4, 5], range(2*3*4*5))
        b = Table(['c', 'b', 'e'], [4, 3, 6], range(4*3*6))
        c = Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], range(2*3*4*5*6))

        # this simply test division by itself yield 1 and that incompatible dimension
        # get raised as error
        bcpt = copy(b)[..., numpy.newaxis, numpy.newaxis]
        bcpt.transpose(3, 1, 0, 4, 2)
        self.assertRaises(ValueError, Table.__div__, a, b)
        cc = c/c
        bb = b/b
        cres = numpy.ones(2*3*4*5*6)
        cres[0] = 0
        bres = numpy.ones(12*6)
        bres[0] = 0
        self.assertAllEqual(cc, Table(['a', 'b', 'c', 'd', 'e'], [2, 3, 4, 5, 6], 
                                   cres))
        self.assertAllEqual(bb, Table(['c', 'b', 'e'], [4, 3, 6], bres))
      

    def test_basic_index(self):
        self.assertEqual(self.table_a[1,1,1], 17.0)
        self.assertEqual(self.table_a[0,0,0], 0)
   
    def test_dict_index(self):
        index = dict(zip(self.names, (1,1,1)))
        self.assertAllEqual(self.table_a[index], self.table_a[1,1,1])
   
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
      
    def test_str(self):
        a = Table(['a', 'b'] , [2,3])
        self.assert_(str(a))
        a = Table([1,2],[2,3])
        self.assert_(str(a))

if __name__ == '__main__':
    unittest.main()


