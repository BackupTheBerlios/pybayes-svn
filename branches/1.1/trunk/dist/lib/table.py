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

# same behaviour as your class, but this one is not a children of numarray
# it simply contains an array. Operations are delegated to this array
class Table:
    def __init__(self, names, shape, elements = None, type = 'Float32'):
      ''' names = ['a','b',...]
          shape = (2,3,...)
          elements = [0,1,2,....] (a list)
          type = 'Float32' or 'Float64' or 'UInt8', etc...
      '''
      if elements == None:
          elements = na.ones(shape = shape)
          
      self.cpt = na.array(sequence=elements, shape=shape, type=type)
      
      self.shape = tuple(shape)
      self.names = set(names)
      self.names_list = list(names) # just to keep the order in an easy to use way

      # dict of name:dim number pairs
      self.assocdim = dict(zip(names,range(len(names))))

      # dict of dim:name pairs
      self.assocname = dict(enumerate(names))

    #==================================
    #Administration stuff
    def __getattr__(self, name):
        """ delegate to self.cpt """
        return getattr(self.cpt,name)

    #===================================
    # Put values into the cpt
    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = na.mlab.rand(*self.shape)

    def AllOnes(self):
        self.cpt = na.ones(self.shape, type='Float32')
    
    def setValues(self, values):
        self.cpt = na.array(values, shape=self.sizes, type='Float32')
    #==================================
    # Indexing
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
      return self.cpt[numIndex]

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
      self.cpt[numIndex] = value

    def _numIndexFromDict(self, d):
      index = []
      for dim in range(len(self.shape)):
         if d.has_key(self.assocname[dim]):
            index.append(d[self.assocname[dim]])
         else:
            index.append(slice(None,None,None))
      return tuple(index) # must convert to tuple in order to work, bug fix

    #=====================================
    # Printing
    def __repr__(self):
      " Return printable representation of instance."
      className= self.__class__.__name__
      className= className.zfill(5).replace('0', ' ')
      rep= className + repr(self.cpt)[5:]
      return rep

    #=====================================
    # Operations
    def __eq__(a,b):
        """ True if a and b have same elements, size and names """
        if b.__class__ == na.NumArray:
            return (na.alltrue(a.cpt.flat == b.flat) \
                    and a.shape == b.shape)
        elif b == None: return False
        else:
            # the b class should better be a Table or something like that
            return (a.shape == b.shape \
                    and a.names_list == b.names_list \
                    and na.alltrue(a.cpt.flat == b.cpt.flat)  \
                    )



    def __mul__(self, other):
        """
        a keeps the order of its dimensions
        
        always use a = a*b or b=b*a, not b=a*b
        """
        
        aa,bb = self.cpt, other.cpt
        
        correspondab,names, shape = self.FindCorrespond(other)
        
        while aa.rank < len(correspondab): aa = aa[..., na.NewAxis]
        while bb.rank < len(correspondab): bb = bb[..., na.NewAxis]

        bb.transpose(correspondab)

        return self.__class__(names, shape, aa*bb)

    def FindCorrespond(self, other):
        correspond = []
        newnames = self.names_list
        newshape = list(self.shape)
        
        aa,bb = self.cpt, other.cpt
        k = bb.rank
        for i in range(aa.rank):
            p = self.assocname[i]   #p=var name in self
            if other.assocdim.has_key(p): 
                correspond.append(other.assocdim[p])
            else:
                correspond.append(k)
                k += 1

        for i in range(bb.rank):
            p = other.assocname[i]  #p=var name in other
            if not self.assocdim.has_key(p):
                correspond.append(other.assocdim[p])
                newnames.append(p)
                newshape.append(other.shape[correspond[-1]])
                
        return correspond, newnames, tuple(newshape)

#==========================================================
# Table inherits numarray
# a numarray is ot really meant to be subclassed since it is written in C++
# by subclassing it, I believe that we lose a lot of time on accessing
# it's elements.
# we could obtain the same behaviour by simply containing an array and delegating
# where we need
class Table_numarray(na.NumArray):
   def __init__(self, names, shape, elements, type):
      ''' n provides the length of each dimension,
      a is the constant value to be plugged.
      '''
      arr=na.array(sequence=elements, shape=shape, type=type)
      self.__setstate__(arr.__getstate__())
      
      self.names = set(names)
      self.names_list = list(names) # just to keep the order in an easy to use way

      # dict of name:dim number pairs
      self.assocdim = dict(zip(names,range(len(names))))

      # dict of dim:name pairs
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

   #=====================================
   # Operations
   def __eq__(a,b):
        """ True if a and b have same elements, size and names """
        if b.__class__ == Table:
            return (a.tolist() == b.tolist() \
                    and a._shape == b._shape \
                    and a.names == b.names)
        elif b.__class__ == na.NumArray:
            return (a.tolist() == b.tolist() \
                    and a._shape == b._shape)
        else:
            print 'Can not compare',a.__class__,'with',b.__class__
            return False

        

    
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
   
   def testEq(self):
       a = Table(['a','b'],[2,3],range(6),'Float32')
       b = Table(['a','b'],[2,3],range(6),'Float32')
       c = Table(['a'],[6],range(6),'Float32')
       d = na.arange(6,shape=(2,3))
       assert(a == b and \
              not a == c and \
              a == d), \
                "__eq__ does not work"

   def testMul(self):
       a = Table(['a','b'],[2,3],range(6),'Float32')
       b = Table(['b','c'],[3,4],range(12),'Float32')

       c = na.arange(6,shape=(2,3))[...,na.NewAxis]
       d = na.arange(12,shape=(3,4))[na.NewAxis,...]

       assert (a*b == Table(['a','b','c'],[2,3,4],c*d)), \
              " Multiplication does not work"

   def testDelegate(self):
       assert (na.alltrue(self.a.flat == self.a.cpt.flat)), \
              " Delegation does not work check __getattr__"
       
       
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

   a = Table(['a','b'],[2,3],range(6),'Float32')
   b = Table(['b','c'],[3,4],range(12),'Float32')

   c = a*b
   print c
