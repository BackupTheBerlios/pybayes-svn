#!/usr/bin/env python
""" This is a set of code for subclassing numarray.  
It creates a new table class which is similar to numarray's basic array
except that each dimension of the array is associated with a name.
This allows indexing via a dictionary and transposing dimensions 
according to an ordered list of dimension names.

Copyright 2005 Elliot Cohen and Kosta Gaitanis, please see the license
file for further legal information.
"""

__version__ = '0.1'
__author__ = 'Kosta Gaitanis & Elliot Cohen'
__author_email__ = 'gaitanis@tele.ucl.ac.be; elliot.cohen@gmail.com'
import unittest
import types
import numarray as na

class Table(na.NumArray):
   def __init__(self, names, shape, elements, type):
      ''' n provides the length of each dimension,
      a is the constant value to be plugged.
      '''
      arr=na.array(sequence=elements, shape=shape, type=type)
      self.__setstate__(arr.__getstate__())
      self.names = set(names)
      #dict of name:dim number pairs
      self.assocdim = dict(zip(names,range(len(names))))
      #dict of dim:name pairs
      self.assocname = dict(enumerate(names))

   def __getitem__(self, index):
      """ Overload array-style indexing behaviour.
      Index can be a dictionary of var name:value pairs, 
      or pure numbers as in the standard way
      of accessing a numarray array array[1,:,1]
      """
      if isinstance(index, types.DictType):
         numIndex = self._numIndexFromDict(index)
      else:
         numIndex = index
      return na.NumArray.__getitem__(self, numIndex)
   
   def __setitem__(self, index, value):
      """ Overload array-style indexing behaviour.
      Index can be a dictionary of var name:value pairs, 
      or pure numbers as in the standard way
      of accessing a numarray array array[1,:,1]
      """
      if isinstance(index, types.DictType):
         numIndex = self._numIndexFromDict(index)
      else:
         numIndex = index
      return na.NumArray.__setitem__(self, numIndex, value)

   def _numIndexFromDict(self, d):
      index = []
      for dim in range(len(self.shape)):
         if d.has_key(self.assocname[dim]):
            index.append(d[self.assocname[dim]])
         else:
            index.append(slice(None,None,None))
      return tuple(index) # must convert to tuple in order to work, bug fix
     
   def __repr__(self):
      " Return printable representation of instance."
      className= self.__class__.__name__
      className= className.zfill(5).replace('0', ' ')
      arr= self.copy()
      arr.__class__= na.NumArray
      rep= className + na.NumArray.__repr__(arr)[5:]
      return rep

   def __str__(self):
     " Return a pretty printed string of the instance."
     stri= self.copy()
     stri.__class__= na.NumArray
     return na.NumArray.__str__(stri)


def ones(names, shape, type='Int32'):
   return Table(names,shape,na.product(shape)*[1],type)

def zeros(names, shape, type='Int32'):
   return Table(names,shape,na.product(shape)*[0],type)      

def array(names, shape, type='Int32'):
   return Table(names,shape,range(na.product(shape)),type)

class TableTestCase(unittest.TestCase):
   def setUp(self):
      names = ('a','b','c')
      shape = (2,3,4)
      self.a = ones(names,shape,type='Float32')
      self.names = names
      self.shape = shape
   
   def testBasic(self):
      a = na.ones(self.shape, type='Float32')
      c = na.ones(self.shape, type='Float32')
      # This fails, need to reimplement __eq__, but others seem to work
      b = self.a == a
      assert(na.all(b)), \
            "Ones array no longer comparable to a normal na ones array"
   
   def testBasicIndex(self):
      assert(self.a[1,1,1] == 1.0),\
            "Could not execute basic index 1,1,1 properly"
   
   def testDictIndex(self):
      index = dict(zip(self.names,(1,1,1)))
      assert(self.a[index] == self.a[1,1,1]),\
            "Dictionary Index is not equivalent to standard index"
   
   def testBasicSet(self):
      self.a[1,1,1] = 2.0
      assert(self.a[1,1,1] == 2),\
            "Could not set execute basic set 1,1,1 = 2"
   
   def testDictSet(self):
      index = dict(zip(self.names,(1,1,1)))
      self.a[index] = 3.0
      assert(self.a[index] == self.a[1,1,1] and \
             self.a[index] == 3.0), \
            "Dictionary Index not equivalent to normal index or could not set properly"
      
      
if __name__ == '__main__':
   suite = unittest.makeSuite(TableTestCase, 'test')
   runner = unittest.TextTestRunner()
   runner.run(suite)