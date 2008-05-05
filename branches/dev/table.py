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
#import random
import types
from copy import copy

import numpy

__all__ = ['Table']
# avoid divide by zero warnings...
numpy.seterr(invalid='ignore', divide='ignore')

class Table:
    """
    A table implementation based on numarray
    """
    def __init__(self, names, shape=None, elements=None, dtype='Float32'):
        ''' names = ['a','b',...]
        shape = (2, 3, ...) (default: binary)
        elements = [0, 1, 2,....] (a list or a numarray, default: all ones)
        type = 'Float32' or 'Float64' or 'UInt8', etc... (default: Float32)
        '''
        # set default parameters
        if shape is None:
            shape = [2] * len(names)
        if elements is None:
            elements = numpy.ones(shape=shape)
        self.cpt = numpy.array(elements, dtype=dtype).reshape(shape)
        self.names = set(names)
        # just to keep the order in an easy to use way
        self.names_list = list(names) 
        # dict of name:dim number pairs
        self.assocdim = dict(zip(self.names_list, range(len(self.names_list))))
        # dict of dim:name pairs
        self.assocname = dict(enumerate(self.names_list))

#==============================================================================
#    def normalize(self, dim=-1):
#        """ If dim=-1 all elements sum to 1.  Otherwise sum to specific 
#        dimension, such that sum(Pr(x=i|Pa(x))) = 1 for all values of i 
#        and a specific set of values for Pa(x)
#        """
#        if dim == -1 or len(self.cpt.shape) == 1:
#            self.cpt /= self.cpt.sum()            
#        else:
#            ndim = self.assocdim[dim]
#            order = range(len(self.names_list))
#            order[0] = ndim
#            order[ndim] = 0
#            tcpt = na.transpose(self.cpt, order)
#            t1cpt = na.sum(tcpt, axis=0)
#            t1cpt = na.resize(t1cpt,tcpt.shape)
#            tcpt = tcpt/t1cpt
#            self.cpt = na.transpose(tcpt, order)
#    #======================================================
#    #=== Sampling
#    def sample(self, index={}):
#        """ returns the index of the sampled value
#        eg. a=Pr(A)=[0.5 0.3 0.0 0.2]
#            a.sample() -->  5/10 times will return 0
#                            3/10 times will return 1
#                            2/10 times will return 3
#                            2 will never be returned
#
#            - returns an integer
#            - only works for one variable tables
#              eg. a=Pr(A,B); a.sample() --> ERROR
#        """
#        assert(len(self.names) == 1 or len(self.names - set(index.keys())) == 1),\
#              "Sample only works for one variable tables"
#        if not index == {}:
#            tcpt = self.__getitem__(index)
#        else:
#            tcpt = self.cpt
#        # csum is the cumulative sum of the distribution
#        # csum[i] = na.sum(self.cpt[0:i])
#        # csum[-1] = na.sum(self.cpt)
#        csum = [na.sum(tcpt.flat[0:end+1]) for end in range(tcpt.shape[0])]
#        
#        # sample in this distribution
#        r = random.random()
#        for i,cs in enumerate(csum):
#            if r < cs: return i
#        return i
#==============================================================================
    #==================================
    #Administration stuff
    def __getattr__(self, name):
        """ delegate to self.cpt """
        return getattr(self.cpt, name)
    
    def __coerce__(self, other):
        assert(isinstance(other, Table))
        return (self, other)

    def __copy__(self):
        """ copy method """
        return Table(self.names_list, self.shape, self.cpt, self.cpt.dtype)

    def update(self, other):
        """ updates this Table with the values contained in the other"""
        # check that all variables in self are contained in other
        if self.names != other.names:
            return "error in update, all variables in other should be"\
                   " contained in self"

        # find the correspondance vector
        correspond = []
        for vara in self.names_list:
            correspond.append(other.assocdim[vara])

        self.cpt = copy(numpy.transpose(other.cpt, axes=correspond))
    #===================================
    # Put values into the cpt
    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = numpy.mlab.rand(*self.shape)

    def all_ones(self):
        """
        set all the element to ones
        """
        self.cpt = numpy.ones(self.shape, dtype='Float32')
    
    def set_values(self, values):
        """
        set self.cpt to values
        """
        ###X ???self.sizes is not a atribute, change to self.shape
        self.cpt = numpy.array(values, dtype='Float32').reshape(self.sizes)
    #==================================
    # Indexing
    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs, 
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]
      
        returns the indexed cpt
        """
        if isinstance(index, types.DictType):
            num_index = self._num_index_from_dict(index)
        else:
            num_index = index
        return self.cpt[num_index]

    def __setitem__(self, index, value):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs, 
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]
        """
        if isinstance(index, types.DictType):
            num_index = self._num_index_from_dict(index)
        else:
            num_index = index
        self.cpt[num_index] = value

    def _num_index_from_dict(self, dim_name):
        """
        TODO figure out
        """
        index = []
        for dim in range(len(self.shape)):
            if dim_name.has_key(self.assocname[dim]):###X might be bug
                index.append(dim_name[self.assocname[dim]])
            else:
                index.append(slice(None, None, None))
        return tuple(index) # must convert to tuple in order to work, bug fix

    #=====================================
    # printing
    def __repr__(self):
        " Return printable representation of instance."
        class_name = self.__class__.__name__
        class_name = class_name.zfill(5).replace('0', ' ')
        rep = class_name + repr(self.cpt)[5:]
        rep += '\nVariables :' + str(self.names_list)
        return rep

    #=====================================
    # Operations
    def add_dim(self, new_dim_name):
        ###X bug??? e.g. abc->abcd the no of elements of self.cpt still 8
        """adds a new unary dimension to the table """
        # add a new dimension to the cpt
        self.cpt = self.cpt[..., numpy.newaxis]

        self.names.add(new_dim_name)
        # just to keep the order in an easy to use way
        self.names_list.append(new_dim_name) 
        # dict of name:dim number pairs
        self.assocdim[new_dim_name] = len(self.names) - 1
        # dict of dim:name pairs
        self.assocname[len(self.names) - 1] = new_dim_name     
        
    def __eq__(self, other):
        """ True if a and b have same elements, size and names """
        if other.__class__ == numpy.ndarray:
        # in case b is a just a numarray and not a Table instance
        # in this case, variable should absoltely be at the same order
        # otherwise the Table and numArray are considered as different
            return (numpy.alltrue(self.cpt.flat == other.flat) \
                    and self.shape == other.shape)

        elif other == None:
        # in case b is None type
            return False
        
        elif isinstance(other, (int, float, long)):
        # b is just a number, int, float, long
            return self.cpt == other
        
        else:
        # the b class should better be a Table or something like that
        # order of variables is not important
            # put the variables in the same order
            # first must get the correspondance vector :
            bcpt = self.prepare_other(other)
            return (self.names == other.names and \
                    bcpt.shape == self.shape and \
                    numpy.allclose(bcpt, self.cpt))
        
## This code checks that order is the same
##            return (a.shape == b.shape \
##                    and a.names_list == b.names_list \
##                    and na.alltrue(a.cpt.flat == b.cpt.flat)  \
##                    )

    def __imul__(self, other):
        """
        in place multiplication
        PRE:
            - B must be a subset of A!!!
            eg.
                a=Pr(A); A = {'a','b','c'}
                b=Pr(B); B = {'c','a'}

        usage:
        a*=b 

        POST:
            a=Pr(A)*Pr(B) = Pr(a,b,c)


        Notes :
        -   a keeps the order of its existing variables
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        """
        # prepare dimensions in b for multiplication
        cptb = self.prepare_other(other)

        # multiply in place, a's values are changed
        self.cpt *= cptb  # this does not work correctly for some reason...
        #na.multiply(a.cpt,cptb,a.cpt) # does not work either
        #a.cpt = a.cpt * cptb    #this one works fine
                                #is this a numarray BUG????

        return self

    def __idiv__(self, other):
        """
        in place division
        PRE:
            - B must be a subset of A!!!
            eg.
                a=Pr(A); A = {'a','b','c'}
                b=Pr(B); B = {'c','a'}

        usage:
        a/=b 

        POST:
            a=Pr(A)/Pr(B) = Pr(a,b,c)


        Notes :
        -   a keeps the order of its existing variables
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        """
        # prepare dimensions in b for multiplication
        cptb = self.prepare_other(other)

        # multiply in place, a's values are changed
        #a.cpt /= cptb  # this does not work correctly for some reason...
        #na.divide(a.cpt,cptb,a.cpt) # does not work either
        self.cpt = self.cpt / cptb    #this one works fine
                                #is this a numarray BUG????

        ## WARNING, division by zero, avoided using 
        # na.Error.setMode(invalid='ignore') replace INFs by 0s
        self.cpt[numpy.isnan(self.cpt)] = 0
        #---TODO: replace this very SLOW function with a ufunc
        return self 

    def __mul__(self, other):
        """
        multiplication
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        c = a*b
        c is a NEW Table instance

        POST:
            c=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   c keeps the order of the variables in a
        -   any new variables in b (d and e) are added at the end of c in the
            order they appear in b
        -   a and b are not touched during this operation
        -   return a NEW Table instance
        """
        # prepare dimensions in a and b for multiplication
        new, cptb = self.union(other)

        # multiply
        #new.cpt *= cptb  # this does not work correctly for some reason...
        #na.multiply(new.cpt,cptb,new.cpt) # does not work either
        new.cpt = new.cpt * cptb    #this one works fine
                                #is this a numarray BUG????        

        return new

    def __div__(self, other):
        """
        multiplication
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'c','a','d','e'}

        usage:
        c = a/b
        c is a NEW Table instance

        POST:
            c=Pr(A U B) = Pr(a,b,c,d,e)

        Notes :
        -   c keeps the order of the variables in a
        -   any new variables in b (d and e) are added at the end of c in the
            order they appear in b
        -   a and b are not touched during this operation
        -   return a NEW Table instance
        """
        #########################################
        #---TODO: add division with a number
        #########################################
        
        # prepare dimensions in a and b for multiplication
        new, cptb = self.union(other)

        # multiply
        #new.cpt /= cptb  # this does not work correctly for some reason...
        #na.divide(new.cpt,cptb,new.cpt) # does not work either
        new.cpt = new.cpt / cptb    #this one works fine
                                #is this a numarray BUG????        

        ## WARNING, division by zero, avoided using 
        # na.Error.setMode(invalid='ignore') replace INFs by 0s
        new.cpt[numpy.isnan(new.cpt)] = 0
        #---TODO: replace this very SLOW function with a ufunc

        return new
    
    def prepare_other(self, other):
        """
        Prepares other for inplace multiplication/division with self. Returns
        a *view* of other.cpt ready for an operation. other must contain a
        subset of the variables of self. NON-DESTRUCTIVE!

        eg. a= Pr(A,B,C,D)
            b= Pr(D,B)
            a.prepare_other(b) --> returns a numarray Pr(1,B,1,D)

            a= Pr(A,B,C,D)
            b= Pr(C,B,E)
            a.prepare_other(b) --> ERROR (E not in {A,B,C,D})

        Notes:
        -   a and b are not altered in any way. NON-DESTRUCTIVE
        -   b must contain a subset of a's variables
            a=Pr(X),b=Pr(Y); Y entirely included in X
        """
        #self contains all variables found in other
        if len(other.names - self.names) > 0:
            raise ValueError(str((other.names-self.names)) +
                             "not in" + str(self.names))

        # add new dimensions to b
        bcpt = other.cpt.view()
        b_assocdim = copy(other.assocdim)
        for var in (self.names - other.names):
            #for all variables found in self and not in other
            #add a new dimension to other
            bcpt = bcpt[..., numpy.newaxis]
            b_assocdim[var] = numpy.rank(bcpt) - 1

        #create the transposition vector
        trans = list()
        for var in self.names_list:
            trans.append(b_assocdim[var])

        bcpt_trans = bcpt.transpose(trans)

        # transpose and return bcpt
        return bcpt_trans
        
    def union(self, other):
        """ Returns a new instance of same class as a that contains all
        data contained in a but also has any new variables found in b with unary
        dimensions. Also returns a view of b.cpt ready for an operation with
        the returned instance.
        
        eg. a= Pr(A,B,C,D,E)
            b= Pr(C,G,A,F)
            a.union(b) --> returns (Pr(A,B,C,D,E,1,1),numarray([A,1,C,1,1,G,F]))

            
            
        Notes:
        -    a and b remain unchanged
        -    a and b must be Table instances (or something equivalent)
        -    a always keeps the same order of its existing variables
        -    any new variables found in b are added at the end of a in the order
             they appear in b.
        -    new dimensions are added with numpy.newaxis
        -    the two numarrays objects returns have exactly the same dimensions
             and are ready for any kind of operation, *,/,...
        """
        # make a copy of a
        new = copy(self)

        for varb in other.names_list:
            # varb is the name of a variable in b
            if not new.assocdim.has_key(varb):
                new.add_dim(varb) # add new variable to new

        # new now contains all the variables contained in a and b
        # new = A U B

        correspond = []        
        b_assocdim = copy(other.assocdim)
        bcpt = other.cpt.view()
        for var in new.names_list:
            # var is the name of a variable in new
            if not other.assocdim.has_key(var):
                bcpt = bcpt[..., numpy.newaxis]
                b_assocdim[var] = numpy.rank(bcpt) - 1
            correspond.append(b_assocdim[var])

        # transpose dimensions in b to match those in a
        btr = bcpt.transpose(correspond)

        # btr is now ready for any operation with new
        return new, btr

    def ones(self):
        """ All CPT elements are set to 1 """
        self.cpt = numpy.ones(self.cpt.shape, dtype=self.cpt.dtype)

    def zeros(self):
        """ All CPT elements are set to 0 """
        self.cpt = numpy.zeros(self.cpt.shape, dtype=self.cpt.dtype)

  
