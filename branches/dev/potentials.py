"""
This is the potentials module from OpenBayes
"""
# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt


from copy import copy

import numpy

import openbayes.table as table
from openbayes import __version__, authors

__all__ = ['DiscretePotential', 'GaussianPotential']
__author__ = authors['Gaitanis']

class Potential:
    """ General Potential class that will be inherited by all potentials
    Maybe we should delegate to a type of potential, the same we did for the
    Distributions
    """
    def __init__(self, names):
        self.names = set(names)
        self.names_list = list(names)

        #we give an order to variables to avoid manipulation errors with arrays
##        order = [(names,k) for k,names in enumerate(names)]
##        order.sort()
##        self.names_list = [o[0] for o in order]
##        
##        # return the order of sorting
##        return [o[1] for o in order]

    #=====================================================================
    # All potentials should implement all of these functions!!!
    #=====================================================================
    def marginalise(self, varnames):
        """ marginalises out some variables and keeps the rest """
        raise NotImplementedError 

    def retrieve(self, varnames):
        """ retrieves and returns some variables """
        raise NotImplementedError 
    
    def normalise(self):
        """ normalizes the distribution """
        raise NotImplementedError    
    def __mul__(self, other):
        """ multiplication, returns a new potential """
        raise NotImplementedError 

    def __imul__(self, other):
        """ in-place multiplication, destructive for a """
        raise NotImplementedError  

    def __div__(self, other):
        """ division, returns a new potential """
        raise NotImplementedError
    
    def __idiv__(self, other):
        """ in-place division, destructive for a """
        raise NotImplementedError 
  
    #===================================================================== 
    
class DiscretePotential(table.Table, Potential):
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and marginalise().
    """
    def __init__(self, names, shape, elements=None):
        Potential.__init__(self, names)
        # sort shape in the same way names are sorted
        #print names, self.names_list,order
        #shape = na.take(shape,order)
        
        if elements == None:
            elements = numpy.ones(shape)
        #elements = na.transpose(elements, axes=order)
        
        table.Table.__init__(self, self.names_list, shape=shape, \
                             elements=elements, dtype='Float32')

    def __copy__(self):
        return DiscretePotential(self.names_list, self.cpt.shape,
                                 copy(self.cpt))

    #=========================
    # Operations
    def marginalise(self, varnames):
        """ marginalises the variables specified in varnames.
        eg. a = Pr(A,B,C,D)
            a.marginalise(['A','C']) --> Pr(B,D) = Sum(A,C)(Pr(A,B,C,D))

        returns a new DiscretePotential instance
        the variables keep their relative order
        """
        temp = self.cpt.view()
        ax = [self.assocdim[v] for v in varnames]
        # sort and reverse list to avoid inexistent dimensions
        ax.sort(reverse=True)  
        newnames = copy(self.names_list)
        for a in ax:
            temp = numpy.sum(temp, axis=a)
            newnames.pop(a)

        #=================================================
        #---ERROR : In which order ?????
        # remainingNames = self.names - set(varnames)
        # remainingNames_list = [name for name in self.names_list 
        #                             if name in remainingNames]

        return self.__class__(newnames, temp.shape, temp)

    def retrieve(self, varnames):
        """ retrieves the dimensions specified in varnames.
        To do this, we marginalise all the variables EXCEPT those specified
        in varnames.
        E.g.    a = Pr(A,B,C,D)
                a.retrieve(['A','C']) --> Pr(A,C) = Sum(B,D)(Pr(A,B,C,D))
        """
        marginals = self.names - set(varnames)
        return self.marginalise(marginals)
    
    def __add__(self, other):
        """
        sum(X\S)phiX

        marginalise the variables contained in BOTH SepSet AND in Cluster
        returns a new DiscretePotential instance

        eg: a = Pr(A,B,C)
            b = Pr(B,C)

            a + b <=> a.marginalise(set(a.names) - set(b.names))
            = Sum(A)a = Pr(B,C)

        only the names of the variables contained in b are relevant!
        no operation with b is done in practice
        """
        var = set(v for v in self.names) - set(v for v in other.names)
        return self.marginalise(var)

    def normalise(self):
        """
        simply assure the the cpt sum to one
        """
        self.cpt /= numpy.sum(self.cpt.flat)
  
    #================================
    # Initialise
    def uniform(self):
        ' uniform distribution '
        n_dim = numpy.product(self.shape)
        self[:] = 1.0 / n_dim

    #===================================
    # printing
    #def __str__(self): return str(self.cpt)

    def printcpt(self):
        """
        Print the cpt
        """
        string =  str(self.cpt) + '\nshape:' + str(self.cpt.shape) + \
                  '\nnames:' + str(self.names) + '\nsum : ' + \
                  str(numpy.sum(self.cpt.flat))
        print string

class GaussianPotential(Potential):
    """ A Canonical Gaussian Potential 
    Only gaussian variables can be contained in this potential
    
    Reference: "A technique for painless derivation of Kalman Filtering Recursions"
                Ali Taylan Cemgril
                SNN, University of Nijmegen, the netherlands
                June 7, 2001
    
    parameters : - g : scalar
                 - h : (n)    row vector where n = sum(sizes of all variables)
                 - K : (n,n)  square matrix
     
     How to derive these parameters :
         phi(x) = a*N(m,S)    # a general multivariate gaussian potential
                              # a is the normalisation factor, 
                              # m = mean, S = covariance matrix
         
     we can prove that :
         phi(x) = exp(g +h'*x - 1/2*x'*K*x)        #' means transposed
     
     where :
         K = S^-1            # the inverse of the covariance matrix
         h = S^-1*m
         g = log(a) + 1/2log(det(K/2pi)) - 1/2*h'*K^-1*h        #det is the determinant
         
     and the inverse formulae :
         S = K^-1
         m = K^-1*h
         a = exp(g - 1/2log(det(K/2pi)) + 1/2*h'*K^-1*h
    """
    def __init__(self, names, shape, g=None, h=None, K=None):
        Potential.__init__(self, names)
        self.shape = shape
        
        # set parameters to 0s
        self.n = numpy.sum(shape)
        if g is None: 
            self.g = 0.0
        else: 
            self.g = float(g)
        if h is None: 
            self.h = numpy.zeros(self.n, dtype='Float32')     
        else: 
            self.h = numpy.array(h, dtype='Float32').reshape(self.n)
        if K is None: 
            self.K = numpy.zeros((self.n, self.n), dtype='Float32')
        else: 
            self.K = numpy.array(K, dtype='Float32').reshape((self.n,self.n))

    def __str__(self):
        string = 'Gaussian Potential over variables ' + str(self.names)
        string += '\ng = ' + str(self.g)
        string += '\nh = ' + str(self.h)
        string += '\nK = ' + str(self.K) 
        return string

