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
from copy import copy
from numarray.ieeespecial import getnan

# same behaviour as your class, but this one is not a children of numarray
# it simply contains an array. Operations are delegated to this array
class Table:
    def __init__(self, names, shape = None, elements = None, type = 'Float32'):
      ''' names = ['a','b',...]
          shape = (2,3,...) (default: binary)
          elements = [0,1,2,....] (a list or a numarray, default: all ones)
          type = 'Float32' or 'Float64' or 'UInt8', etc... (default: Float32)
      '''
      # set default parameters
      assert(isinstance(names,list))
      if shape == None:
          shape = [2]*len(names)
      if elements == None:
          elements = na.ones(shape = shape,type=type)
          
      self.cpt = na.array(sequence=elements, shape=shape, type=type)
      
      self.names = set(names)
      self.names_list = names # just to keep the order in an easy to use way

      # dict of name:dim number pairs
      self.assocdim = dict(zip(names,range(len(names))))

      # dict of dim:name pairs
      self.assocname = dict(enumerate(names))

    #==================================
    #Administration stuff
    def __getattr__(self, name):
        """ delegate to self.cpt """
        return getattr(self.cpt,name)
    
    def __coerce__(self, other):
        assert(isinstance(other, Table))
        return (self,other)

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
      rep += '\nVariables :' + str(self.names_list)
      return rep

    #=====================================
    # Operations
    def __eq__(a,b):
        """ True if a and b have same elements, size and names """
        if b.__class__ == na.NumArray:
            return (na.alltrue(a.cpt.flat == b.flat) \
                    and a.shape == b.shape)
        elif b == None: 
            return False
        elif isinstance(b, (int, float, long)):
            return a.cpt == b
        else:
            # the b class should better be a Table or something like that
            return (a.shape == b.shape \
                    and a.names_list == b.names_list \
                    and na.alltrue(a.cpt.flat == b.cpt.flat)  \
                    )

    def __imul__(self,b): return self*b
    def __idiv__(self,b): return self/b

    def __mul__(self,b):
        """
        in place multiplication
        a = a*b <==> a*b <==> a*=b

        a keeps the order of it's dimensions
        b MUST be a subset of a
        b can have any variable order
        """
        #FIXME: fixed else case get rid of assert
        assert(self.names.issuperset(b.names))
        if self.names.issuperset(b.names):
            cptb = self.prepareDimensions(b)
            # multiply in place, a's values are changed
            self.cpt *= cptb
        #elif b.names.issuperset(self.names):
            #cpta = b.prepareDimensions(self)
            #self.cpt = b.cpt * cpta
        else:
            #FIXME: Make this truly inplace
            #OPTIMIZE: this can be greatly optimized, but it is better than breaking
            namesToAdd = b.names.difference(self.names)
            nShape = self.shape
            nAssocDim = self.assocdim.copy()
            nAssocName = self.assocname.copy()
            nNamesList = copy.copy(self.names_list)
            for var,dim in zip(namesToAdd,range(len(self.names),len(self.names)+len(namesToAdd))):
                nNamesList.append(var)
                nAssocDim[var] = dim
                nAssocName[dim] = var
                nShape += tuple(b.shape[b.assocdim[var]])
            newTable = Table(nNamesList, shape=nShape)
            for dIndex in GenerateDictIndex(nNamesList,na.array(nShape)):
                newTable[dIndex] = self[dIndex] * b[dIndex]
            #CHECK: not sure following line works properly
            self.cpt = newTable.cpt
            self.assocdim = nAssocDim
            self.assocname = nAssocName
            self.names = set(nNamesList)
            self.names_list = nNamesList

        return self

    def __div__(a, b):
        """
        in place multiplication
        a = a/b <==> a/b <==> a/=b

        a keeps the order of it's dimensions
        b MUST be a subset of a
        b can have any variable order

        0/0 are replaced by 0s
        """
        cptb = a.prepareDimensions(b)

        # divide in place, a's values are changed
        a.cpt /= cptb

        ## WARNING, division by zero, avoided using na.Error.setMode(invalid='ignore')
        # replace INFs by 0s
        a.cpt[getnan(a.cpt)] = 0

        return a

    def prepareDimensions(self,b):
        """
        prepares b.cpt for multiplication with a
            - b MUST be a subset of a
            - new dimensions are added for each variable in a that does not
              exist in b
            - variables are put in the correct order
            - does not touch b.cpt

        return b.cpt ready for multiplication with a.cpt
        """
        #Assumes that b is a subset of self
        #if not self.names.issuperset(b.names):
        #    self.addMissingDims(b)
        cptb = b.cpt
        # create new empty dimensions in cptb for the extra a variables
        
        while cptb.rank < self.cpt.rank: cptb = cptb[...,na.NewAxis]
        
        # find correspondance between b vars and a vars
        correspondab = self.findCorrespond(b)

        # put the b dimensions into the right place
        cptb.transpose(correspondab)

        return cptb
    
    def findCorrespond(self, other):
        """ Returns the correspondance vector between the variables of
        two Tables
        other.variables MUST be a subset of self.variables !!!
        eg. a= Pr(A,B,C)
            b= Pr(C,B)
            a.FindCorrespond(b) --> [2,1,0]
            b.FindCorrespond(a) --> error (a not a subset of b)

            a=Pr(A,B,C,D,E)
            b=Pr(B,E)
            a.FindCorrespond(b) --> [1,4,0,2,3]

        any variables in a but do not exist in b are added at the end of list
        """
        #---TODO: ASSERT other must be a subset of self !!!
        #raise str(other.names_list) + " not a subset of " + str(self.names_list)
        
        correspond = []
        for varb in other.names_list:
            # varb is the name of a variable in other
            # self MUST contain all the variables of b
            correspond.append(self.assocdim[varb])

        k = 0
        for vara in self.names_list:
            # vara is the name of a variable in self
            # add all variables contained in a and not in b
            if not other.assocdim.has_key(vara):
                correspond.append(k)
            k += 1

        return correspond
    
    def addMissingDims(self, other):
        """ add blank (size 1) dimensions to self.  Use variable names as specified in newDims.
        """
        newDims = other.names.difference(self.names)
        lnewDims = list(newDims)
        newShape = [other.shape[other.assocdim[var]] for var in lnewDims]
        for var,dim in zip(lnewDims, range(len(self.names),len(self.names)+len(newDims))):
            self.assocdim[var] = dim
            self.assocname[dim] = var
            #self.cpt = self.cpt[...,na.NewAxis]
        self.cpt.resize(self.shape+tuple(newShape))
        self.names.update(newDims)
        self.names_list += lnewDims
        
        
def GenerateDictIndex(names, shape):
    assert(isinstance(shape, na.ArrayType))
    stop = shape - 1
    value = zeros(len(shape))
    value[0] -= 1
    while True:
        while not na.alltrue(value == stop):
            for i in range(len(stop)):
                if value[i] == stop[i]:
                    value[i] = 0
                else:
                    value[i] += 1
                    break
            yield dict(zip(names,value))
        raise StopIteration

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
       self.b = ones(names[1:],shape[1:],type='Float32')
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
        a = Table(['c','d'],[2,3],range(2*3))
        #a = Table(['a','b','c'],[2,3,4],range(2*3*4))
        b = Table(['c','b'],[2,4],range(2*4))
        c = Table(['a','b','c','d'],[2,3,4,5],range(2*3*4*5))
        a*b
        acpt = na.arange(2*3*4*5,shape=(2,3,4,5), type='Float32')
        bcpt = na.arange(12,shape=(4,3),type='Float32')[...,na.NewAxis,na.NewAxis]
        bcpt.transpose([2,1,0,3])
 
        # test the three types of possible notations
        a*b
        c*=c
        b = b*b
 
        assert (a == Table(['a','b','c','d'],[2,3,4,5],acpt*bcpt) and \
                c == Table(['a','b','c','d'],[2,3,4,5],na.arange(2*3*4*5)**2) and \
                b == Table(['c','b'],[4,3],na.arange(12)**2)), \
               " Multiplication does not work"
 
    def testDiv(self):
        a  = Table(['a','b','c','d'],[2,3,4,5],range(2*3*4*5))
        b = Table(['c','b'],[4,3],range(12))
        c = Table(['a','b','c','d'],[2,3,4,5],range(2*3*4*5))
    
        acpt = na.arange(2*3*4*5,shape=(2,3,4,5), type='Float32')
        bcpt = na.arange(12,shape=(4,3),type='Float32')[...,na.NewAxis,na.NewAxis]
        bcpt.transpose([2,1,0,3])
 
        # test the three types of possible notations
        a/b
        c/=c
        b = b/b
 
        cres = na.ones(2*3*4*5)
        cres[0] = 0
        bres = na.ones(12)
        bres[0] = 0
        ares = acpt/bcpt
        ares[getnan(ares)] = 0
        
        assert (a == Table(['a','b','c','d'],[2,3,4,5],ares) and \
                c == Table(['a','b','c','d'],[2,3,4,5],cres) and \
                b == Table(['c','b'],[4,3],bres) ), \
               " Division does not work"
        
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
 
    def testFindCorrespond(self):
         a = Table(['a','b','c','d','e'])
         b = Table(['d','b'])
 
         assert(a.findCorrespond(b) == [3,1,0,2,4]), \
                 "findCorrespond does not work correctly..."
        
      
if __name__ == '__main__':
    suite = unittest.makeSuite(TableTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    a = Table(['a','b'],[2,3],range(6))
    b = Table(['b'],[3],range(3))
    c = Table(['b','e'],[3,2],range(6))
    d = Table(['a','b','c','d','e'],[2,2,2,2,2],range(2**5))
    
    print a
    a/a
    print a
##    a*c
##    print a
##    print c

    #a*b
    #print 'mul'
    #print a

    

  
