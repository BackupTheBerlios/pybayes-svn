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

# avoid divide by zero warnings...
na.Error.setMode(invalid='ignore', dividebyzero='ignore')

class Table:
    def __init__(self, names, shape = None, elements = None, type = 'Float32'):
      ''' names = ['a','b',...]
          shape = (2,3,...) (default: binary)
          elements = [0,1,2,....] (a list or a numarray, default: all ones)
          type = 'Float32' or 'Float64' or 'UInt8', etc... (default: Float32)
      '''
      # set default parameters
      if shape == None:
          shape = [2]*len(names)
      if elements == None:
          elements = na.ones(shape = shape)
          
      self.cpt = na.array(sequence=elements, shape=shape, type=type)

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

    def __copy__(self):
        """ copy method """
        return Table(self.names_list,self.shape,self.cpt,self.cpt.type())

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
    def addDim(self, newDimName):
        """adds a new unary dimension to the table """
        # add a new dimension to the cpt
        self.cpt = self.cpt[...,na.NewAxis]

        self.names.add(newDimName)
        self.names_list.append(newDimName) # just to keep the order in an easy to use way

        # dict of name:dim number pairs
        self.assocdim[newDimName] = len(self.names)-1
        # dict of dim:name pairs
        self.assocname[len(self.names)-1] = newDimName     
        
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

    def __imul__(a,b): return a*b
    def __idiv__(a,b): return a/b

    def __mul__(a,b):
        """
        in place multiplication
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        a = a*b <==> a*b <==> a*=b

        POST:
            a=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   a keeps the order of its existing variables
        -   any new variables in b (d and e) are added at the end of a in the
            order they appear in b
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        """
        # prepare dimensions in a and b for multiplication
        cptb = a.prepareDimensions(b)

        # multiply in place, a's values are changed
        #a.cpt *= cptb  # this does not work correctly for some reason...
        #na.multiply(a.cpt,cptb,a.cpt) # does not work either
        a.cpt = a.cpt * cptb    #this one works fine
                                #is this a numarray BUG????

        return a

    def __div__(a, b):
        """
        in place division
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        a = a/b <==> a/b <==> a/=b

        POST:
            a=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   a keeps the order of its existing variables
        -   any new variables in b (d and e) are added at the end of a in the
            order they appear in b
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        -   0/0 are replaced by 0s
        """
        cptb = a.prepareDimensions(b)

        # divide in place, a's values are changed
        #a.cpt /= cptb  # this does not work correctly for some reason...
        #na.divide(a.cpt,cptb,a.cpt) #does not work either
        a.cpt = a.cpt / cptb    #this one works fine
                                #is this a numarray BUG????
        ## WARNING, division by zero, avoided using na.Error.setMode(invalid='ignore')
        # replace INFs by 0s
        a.cpt[getnan(a.cpt)] = 0
        #---TODO: replace this very SLOW function with a ufunc

        return a

    def prepareDimensions(a,b):
        """ Returns the correspondance vector between the variables of
        two Tables and inserts any extra dimensions needed in both a
        and b. the b dimensions are transposed to correspond with the ones
        in a.
        
        eg. a= Pr(A,B,C,D,E)
            b= Pr(C,G,A,F)
            a.prepareDims(b) --> a = Pr(A,B,C,D,E,G,F)  (G,F added at the end)
                                 b = Pr(A,B,C,D,E,G,F)  (B,D,E added and dimensions
                                                         rearranged)
        Notes:
        -    the operation is destructive for a only, b remains unchanged
        -    a and b must be Table instances
        -    a always keeps the same order of its existing variables
        -    any new variables found in b are added at the end of a in the order
             they appear in b.
        -    new dimensions are added with numarray.NewAxis
        -    a and b have exactly the same dimensions at the end and are ready
             for any kind of operation, *,/,...
        """
        bcpt = b.cpt.copy()     # don't touch b
        
        for varb in b.names_list:
            # varb is the name of a variable in b
            if not a.assocdim.has_key(varb):
                a.addDim(varb) # add new variable to a

        # a now contains all the variables contained in b
        # A = A U B

        correspond = []        
        bnames = copy(b.names_list)
        for vara in a.names_list:
            # vara is the name of a variable in a
            if not b.assocdim.has_key(vara):
                bcpt = bcpt[...,na.NewAxis]
                bnames.append(vara)
            correspond.append(bnames.index(vara))

        # transpose dimensions in b to match those in a
        btr = na.transpose(bcpt, axes = correspond)

        # btr is now ready for any operation for a
        return btr
    
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
       a  = Table(['a','b','c','d'],[2,3,4,5],range(2*3*4*5))
       b = Table(['c','b','e'],[4,3,6],range(12*6))
       c = Table(['a','b','c','d','e'],[2,3,4,5,6],range(2*3*4*5*6))
   
       acpt = copy(a.cpt)[...,na.NewAxis]
       bcpt = copy(b.cpt)[...,na.NewAxis,na.NewAxis]
       bcpt.transpose([3,1,0,4,2])

       # test the three types of possible notations
       a*b
       c*=c
       b = b*b

       assert (a == Table(['a','b','c','d','e'],[2,3,4,5,6],acpt*bcpt) and \
               c == Table(['a','b','c','d','e'],[2,3,4,5,6],na.arange(2*3*4*5*6)**2) and \
               b == Table(['c','b','e'],[4,3,6],na.arange(12*6)**2)), \
              " Multiplication does not work"

   def testDiv(self):
       a  = Table(['a','b','c','d'],[2,3,4,5],range(2*3*4*5))
       b = Table(['c','b','e'],[4,3,6],range(12*6))
       c = Table(['a','b','c','d','e'],[2,3,4,5,6],range(2*3*4*5*6))
   
       acpt = copy(a.cpt)[...,na.NewAxis]
       bcpt = copy(b.cpt)[...,na.NewAxis,na.NewAxis]
       bcpt.transpose([3,1,0,4,2])
       
       # test the three types of possible notations
       a/b
       c/=c
       b = b/b

       cres = na.ones(2*3*4*5*6)
       cres[0] = 0
       bres = na.ones(12*6)
       bres[0] = 0
       ares = acpt/bcpt
       ares[getnan(ares)] = 0
       
       assert (a == Table(['a','b','c','d','e'],[2,3,4,5,6],ares) and \
               c == Table(['a','b','c','d','e'],[2,3,4,5,6],cres) and \
               b == Table(['c','b','e'],[4,3,6],bres) ), \
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

   def testAddDim(self):
        a = Table('a b c'.split())
        a.addDim('d')

        assert(a.names == set('a b c d'.split()) and \
               a.names_list ==  'a b c d'.split() and \
               a.assocdim.has_key('d') and \
               a.assocname.has_key(3)), \
               "add Dim does not work correctly..."
        

   def testPrepareDimensions(self):
        #print 'a:2,b:3,c:4,d:5,e:6,g:7,f:8'
        a = Table('a b c d e'.split(), [2,3,4,5,6])
        b = Table('c g a f'.split(), [4,7,2,8])

        bcpt = a.prepareDimensions(b)

        assert(bcpt.shape == tuple([2,1,4,1,1,7,8]) and \
               a.cpt.shape == tuple([2,3,4,5,6,1,1]) and \
               a.names_list == 'a b c d e g f'.split()), \
               " prepareDimensions does not work..."

      
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

    

  
