""" This is a set of code for subclassing numpy.ndarray.
It creates a new table class which is similar to numpy's basic array
except that each dimension of the array is associated with a name.
This allows indexing via a dictionary and transposing dimensions
according to an ordered list of dimension names.
"""
# Copyright (C) 2005-2008 by
# Elliot Cohen <elliot.cohen@gmail.com>
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt
import numpy
import numpy.random

from openbayes import __version__, authors
__all__ = ['Table']
__author__ = authors['Cohen'] + '\n' +\
             authors['Gaitanis'] + '\n' +\
             authors['Salamin']

# avoid divide by zero warnings...
numpy.seterr(invalid='ignore', divide='ignore')

class Table(numpy.ndarray):
    """
    A table implementation based on numpy

    Table is a subclass of numpy.ndarray and can therefor be used
    anywhere a numpy array can be
    """

    # __init__ is replaced by __new__ in our case. We are subclassing
    # a numpy.ndarray instance

    def __new__(cls, names, shape=None, elements=None, dtype='Float32',
                copy=False):
        if shape is None:
            shape = [2] * len(names)
        if elements is None:
            elements = numpy.ones(shape=shape)
        if copy:
            subarray = numpy.array(elements, dtype=dtype).reshape(shape)
        else:
            subarray = numpy.asarray(elements, dtype=dtype).reshape(shape)
        subarray = subarray.view(cls)
        subarray.assocname = dict(enumerate(names))
        subarray._compute_internal_dict()
        return subarray

    def get_names_list(self):
        """
        This return a list of the name in the current order
        """
        return [self.assocname[i] for i in range(len(self.shape))]

    names_list = property(get_names_list, None, None,
                          "Return an ordered list of names")

    def get_names_set(self):
        """
        This return the set of names associated with dims
        """
        return set(self.assocdim.keys())

    names = property(get_names_set, None, None,
                     "Return the set of the names present in the array")

    def __array_finalize__(self, obj):
        """
        We must set the default value in this function.
        See numpy documentation
        """
        if hasattr(obj, "assocname"):
            self.assocname = dict(obj.assocname)
        else:
            self.assocname = dict([(x,str(x))
                                  for x in xrange(len(self.shape))])
        self._compute_internal_dict()

    def _compute_internal_dict(self):
        """
        This method is responsible to update self.assocdim. Some
        Checking on the length of assocname and the actual number
        of dimensions is done
        """
        self.assocdim = dict([(x[1], x[0]) for x in self.assocname.items()])
        if self.shape == tuple() and len(self.assocdim) == 1:
            return
        if len(self.assocdim) != len(self.shape):
            raise ValueError("Missing dimension name:"
                             "\n shape: %s\n assocdim %s" %
                              (self.shape, self.assocdim))

    def update(self, other):
        """ updates this Table with the values contained in the other"""
        # check that all variables in self are contained in other
        if self.assocname != other.assocname:
            raise ValueError("Update not possible if names differs")
        correspond = []
        for vara in self.names_list:
            correspond.append(other.assocdim[vara])
        self[:] = numpy.transpose(other, axes=correspond)

    def rand(self):
        ''' put random values into self '''
        self[:] = numpy.random.uniform(size = self.shape)

    def ones(self):
        """
        set all the element to ones
        """
        self[:] = numpy.ones(self.shape, dtype=self.dtype)

    def zeros(self):
        """
        set all element to zero
        """
        self[:] = numpy.zeros(self.shape, dtype = self.dtype)

    def set_values(self, values):
        """
        set self to values
        """
        self[:] = numpy.array(values, dtype=self.dtype).reshape(self.shape)

    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs,
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]

        We also support slices. If any of the return index
        is a slice, then we return a Table object, else we return
        a simple float
        """
        if isinstance(index, dict):
            names, num_index = self._num_index_from_dict(index)
        else:
            num_index = index
            # we never return table if we index by dimension
            names = []
       # we now check if we need to return slices
        # We first
        # we either return a simple numpy object
        if len(names) == 0:
            return numpy.ndarray.__getitem__(self.view(numpy.ndarray),
                                             num_index)
        # or a table if we selcted complete dimension
        else:
            temp = numpy.ndarray.__getitem__(self.view(numpy.ndarray), 
                                             num_index)
            temp = Table(names, temp.shape, temp)
            return temp

    def __setitem__(self, index, value):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs,
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]
        """
        if isinstance(index, dict):
            _, num_index  = self._num_index_from_dict(index)
        else:
            num_index = index
        numpy.ndarray.__setitem__(self.view(numpy.ndarray), num_index, value)

    def _num_index_from_dict(self, dim_name):
        """
        This function convert a dictionnary of index into
        a list of index
        """
        index = []
        names = []
        for dim in range(len(self.shape)):
            if dim_name.has_key(self.assocname[dim]):
                index.append(dim_name[self.assocname[dim]])
            else:
                index.append(slice(None, None, None))
                names.append(self.assocname[dim])
        return names, tuple(index)


    def __str__(self):
        string = "(Dims: " + \
                 " ".join([str(x) for x in self.names_list]) + "\n"
        return string + numpy.ndarray.__str__(self.view(numpy.ndarray))+")"

    def transpose(self, *axis):
        """
        This function get called when we want to transpose the table. We
        need to ensure that the names stay on the right dimension
        """
        new = numpy.ndarray.transpose(self, *axis)
        if len(axis) == 1:
            axis = axis[0]
        if axis is None or len(axis) == 0:
            axis = range(len(self.shape)-1, -1, -1)
        new.assocname = dict([(x, self.assocname[y])
                              for x,y in enumerate(axis)])
        return new.view(Table)

    def add_dim(self, new_dim_name):
        """adds a new unary dimension to the table """
        # add a new dimension to the cpt
        self.assocname[len(self.shape)] = new_dim_name
        self.shape = self.shape + (1,)
        self._compute_internal_dict()

    def __eq__(self, other):
        """ True if a and b have same elements, size and names """
        if not hasattr(other, 'names'):
            return numpy.ndarray.__eq__(self.view(numpy.ndarray), other)
        else:
            # the b class should better be a Table or something like that
            # order of variables is not important
            # put the variables in the same order
            # first must get the correspondance vector :
            if self.names != other.names:
                return numpy.array([False])
            correspond = []
            for vara in self.names_list:
                correspond.append(other.assocdim[vara])
            other_view =  numpy.transpose(other, axes=correspond)
            return numpy.ndarray.__eq__(self, other_view).view(numpy.ndarray)


    def union(self, other):
        """
        Returns a new instance of same class as a that contains all
        data contained in self but also has any new variables found in other with unary
        dimensions. Also returns a view of other ready for an operation with
        the returned instance.

        eg. self = Pr(A,B,C,D,E)
            other = Pr(C,G,A,F)
            self.union(other) --> returns (Pr(A,B,C,D,E,1,1),numarray([A,1,C,1,1,G,F])

        Notes:
        -    self and other remain unchanged
        -    self and other must be Table instances (or something equivalent)
        -    self always keeps the same order of its existing variables
        -    any new variables found in other are added at the end of the result in the order
             they appear in other.
        -    new dimensions are added with numpy.newaxis
        -    the two numpy.ndarray objects returns have exactly the same dimensions
             and are ready for any kind of operation, *,/,...
        """
        new = self.copy()
        for variable in other.names_list:
            if variable not in new.names:
                new.add_dim(variable) # add new variable to new
        # new now contains all the variables contained in a and b
        # new = A U B
        correspond = []
        b_assocdim = other.assocdim.copy()
        bcpt = other.view()
        for var in new.names_list:
            # var is the name of a variable in new
            if not other.assocdim.has_key(var):
                bcpt.add_dim(var)
                b_assocdim[var] = numpy.rank(bcpt) - 1
            correspond.append(b_assocdim[var])
        # transpose dimensions in b to match those in a
        btr = bcpt.transpose(correspond)
        # btr is now ready for any operation with new
        return new, btr


    ##########################################################################
    # What is below must still be implemented
    # TODO understand and reimplement
    ##########################################################################
    def __imul__(self, other):
        """
        Simply call multiplication. No optimisation for inplace
        multiplication
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
        -> code start

        # prepare dimensions in b for multiplication
        cptb = self.prepare_other(other)

        # multiply in place, a's values are changed
        self.cpt *= cptb  # this does not work correctly for some reason...
        #na.multiply(a.cpt,cptb,a.cpt) # does not work either
        #a.cpt = a.cpt * cptb    #this one works fine
                                #is this a numarray BUG????

        return self
        """
        self = self*other
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

        The division is done element by element
        Notes :
        -   a keeps the order of its existing variables
        -   b is not touched during this operation
        -   operation is done in-place for a, a is not the same after the operation
        -> code start
        # prepare dimensions in b for multiplication
        """
        # if we dont have names, then we assume everything was taken care of
        if not hasattr(other, 'names'):
            numpy.ndarray.__idiv__(self, other)
        else:
            cptb = self.prepare_other(other)
            numpy.ndarray.__idiv__(self, cptb)
        self[numpy.isnan(self)] = 0
        return self

    def __div__(self, other):
        """
        Division
        PRE:
            a=Pr(A); A = {'a','b','c'}
            b=Pr(B); B = {'a','b'}

        usage:
        c = a/b
        c is a NEW Table instance

        POST:
            c = Pr(A | B) = Pr(a | b ,c)

        Notes :
        -   c keeps the order of the variables in a
        -   a and b are not touched during this operation
        -   return a NEW Table instance
        -> code
        """
        if not hasattr(other, 'names'):
            ans = numpy.ndarray.__div__(self, other)
        else:
            cptb = self.prepare_other(other)
            ans = numpy.ndarray.__div__(self, cptb)
        ans[numpy.isnan(ans)] = 0
        return ans

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
        -> code start
        # prepare dimensions in a and b for multiplicatio
        """
        a,  b = self.union(other)
        return numpy.ndarray.__mul__(a,  b)

    def prepare_other(self, other):
        """
        Prepares other for inplace multiplication/division with self. Returns
        a *view* of other.cpt ready for an operation. other must contain a
        subset of the variables of self. NON-DESTRUCTIVE!

        eg. a= Pr(A,B,C,D)
            b= Pr(D,B)
            a.prepareOther(b) --> returns a numarray Pr(1,B,1,D)

            a= Pr(A,B,C,D)
            b= Pr(C,B,E)
            a.prepareOther(b) --> ERROR (E not in {A,B,C,D})

        Notes:
        -   a and b are not altered in any way. NON-DESTRUCTIVE
        -   b must contain a subset of a's variables
            a=Pr(X),b=Pr(Y); Y entirely included in X
        """
        #self contains all variables found in other
        if len(other.names - self.names) > 0:
            raise ValueError(str(other.names-self.names) + 
                             " not in " + str(self.names))
        # add new dimensions to b
        new_b = other.copy()
        for var in (self.names - other.names):
            #for all variables found in self and not in other
            #add a new dimension to other
            new_b.add_dim(var)
        #create the transposition vector
        trans = []
        for var in self.names_list:
            trans.append(new_b.assocdim[var])
        return new_b.transpose(trans)

    def sum(self, axis=None, dtype=None, out=None):
        """
        We overload the sum to support of named axis
        """
        names = []
        if not axis is None:
            for i in self.names_list:
                if i != axis:
                    names.append(i)
            axis = self.assocdim[axis]
        temp = numpy.ndarray.sum(self.view(numpy.ndarray), axis, dtype, out)
        if len(names) == 0:
            return temp
        return Table(names, temp.shape, temp)

    # TO-DO: implement some marginalize method (is it useful???) 
    # salamin 20.06.2008
    def normalize(self, axis = None):
        """
        This function normalize the value in the table such that the sum acros
        of dimension dim is 1
        """
        sum_ = self.sum(axis)
        self /= sum_
