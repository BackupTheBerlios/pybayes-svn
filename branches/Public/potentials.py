###############################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
###############################################################################
__all__ = ['DiscretePotential','GaussianPotential']

import numarray as na
from copy import copy

import delegate
import table
import unittest

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
    def Marginalise(self, varnames):
        """ Marginalises out some variables and keeps the rest """
        raise "Method is not yet implemented at child level"
    
    def Retrieve(self, varnames):
        """ Retrieves and returns some variables """
        raise "Method is not yet implemented at child level"
    
    
    def __mul__(a,b):
        """ multiplication, returns a new potential """
        raise "Method is not yet implemented at child level"

    def __imul__(a,b):
        """ in-place multiplication, destructive for a """
        raise "Method is not yet implemented at child level"
    
    def __div__(a,b):
        """ division, returns a new potential """
        raise "Method is not yet implemented at child level"
    
    def __idiv__(a,b):
        """ in-place division, destructive for a """
        raise "Method is not yet implemented at child level"   
    #===================================================================== 
    
class DiscretePotential(table.Table, Potential):
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and Marginalise().
    """
    def __init__(self, names, shape, elements=None):
        Potential.__init__(self, names)

        # sort shape in the same way names are sorted
        #print names, self.names_list,order
        #shape = na.take(shape,order)
        
        if elements == None:
            elements = na.ones(shape=shape)
        #elements = na.transpose(elements, axes=order)
        
        table.Table.__init__(self, self.names_list, shape=shape, \
                             elements=elements, type='Float32')

    def __copy__(self):
        return DiscretePotential(self.names_list, self.cpt.shape, copy(self.cpt))

    #=========================
    # Operations
    def Marginalise(self, varnames):
        """ Marginalises the variables specified in varnames.
        eg. a = Pr(A,B,C,D)
            a.Marginalise(['A','C']) --> Pr(B,D) = Sum(A,C)(Pr(A,B,C,D))

        returns a new DiscretePotential instance
        the variables keep their relative order
        """
        temp = copy(self.cpt)
        ax = [self.assocdim[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        newnames = copy(self.names_list)
        for a in ax:
            temp = na.sum(temp, axis = a)
            newnames.pop(a)

        return DiscretePotential(newnames, temp.shape, temp)

    def Normalize(self):
        """ All values are normalized Sum(Pr(A)) = 1     """
        self.cpt /= na.sum(self.cpt.flat)   
        
    def Retrieve(self, varnames):
        """ Retrieves the dimensions specified in varnames.
        To do this, we marginalise all the variables EXCEPT those specified
        in varnames.
        E.g.    a = Pr(A,B,C,D)
                a.Retrieve(['A','C']) --> Pr(A,C) = Sum(B,D)(Pr(A,B,C,D))
        """
        marginals = self.names - set(varnames)
        return self.Marginalise(marginals)
    
    def __add__(self, other):
        """
        sum(X\S)phiX

        sum out the variables contained in self but NOT in other
        returns a new DiscretePotential instance

        eg: a = Pr(A,B,C)
            b = Pr(B,C)

            a + b <=> a.Marginalise(set(a.names) - set(b.names))
            = Sum(A)a = Pr(B,C)

        only the names of the variables contained in b are relevant!
        no operation with the values of b is performed
        
        The result is a new potential containing the variables of other with
        the values of these variables found in self.
        """
        
        var = self.names - other.names
        return self.Marginalise(var)

            
    #================================
    # Initialise
    def Uniform(self):
        ' Uniform distribution '
        N = na.product(self.shape)
        self[:] = 1.0/N

    #===================================
    # Printing
    #def __str__(self): return str(self.cpt)

    def Printcpt(self):
        string =  str(self.cpt) + '\nshape:'+str(self.cpt.shape)+'\nnames:'+str(self.names)+'\nsum : ' +str(na.sum(self.cpt.flat))
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
        self.n = na.sum(shape)
        if not g: self.g = 0.0
        else: self.g = float(g)
        if not h: self.h = na.zeros(shape=(self.n), type='Float32')     
        else: self.h = na.array(h,shape=(self.n), type='Float32')
        if not K: self.K = na.zeros(shape=(self.n,self.n), type='Float32')
        else: self.K = na.array(K,shape=(self.n,self.n), type='Float32')

    def __str__(self):
        string = 'Gaussian Potential over variables ' + str(self.names)
        string += '\ng = ' + str(self.g)
        string += '\nh = ' + str(self.h)
        string += '\nK = ' + str(self.K)
        
        return string

class GaussianPotentialTestCase(unittest.TestCase):
    def setUp(self):
        names = ('a','b')
        shape = (1,2)
        self.a = GaussianPotential(names,shape) 
        
        g=2
        h=[1,2,3]
        K=range(9)
        self.b = GaussianPotential(names,shape,g,h,K)
    
    def testInit(self):
        a = self.a
        b = self.b
        assert(a.g == 0.0 and \
               na.allclose(a.h, na.zeros(3)) and \
               na.allclose(a.K, na.zeros(shape=(3,3)))), \
               " Error with standard initialization "

        assert(b.g == 2.0 and \
               na.allclose(b.h, na.array([1,2,3],type='Float32')) and \
               na.allclose(b.K, na.arange(9,shape=(3,3),type='Float32'))), \
               " Error with standard initialization with parameter setting"    
        
    

class DiscretePotentialTestCase(unittest.TestCase):
    def setUp(self):
      names = ('a','b','c')
      shape = (2,3,4)
      self.a = DiscretePotential(names,shape,na.arange(24))
      self.names = names
      self.shape = shape
   
    def testMarginalise(self):
        def factorial(n):
            if n==1:return 1
            return factorial(n-1)*n
        
        var = set('c')
        b = self.a.Marginalise(var)
        var2 = set(['c','a'])
        c = self.a.Marginalise(var2)
        d = DiscretePotential(['b','c'],[3,4],na.arange(12))

        # extended test
        a = DiscretePotential('a b c d e f'.split(), [2,3,4,5,6,7],na.arange(factorial(7)))
        aa=a.Marginalise('c f a'.split())
      

        assert(b.names == self.a.names - var and \
               b[0,1] == na.sum(self.a[0,1]) and \
               c.names == self.a.names - var2 and \
               na.alltrue(c.cpt.flat == na.sum(na.sum(self.a.cpt,axis=2),axis=0)) and
               aa.shape == (3,5,6) and \
               aa.names_list == 'b d e'.split() and \
               aa[2,4,3] == na.sum(a[:,2,:,4,3,:].flat)), \
               " Marginalisation doesn't work"

    def testAdd(self):
        d = DiscretePotential(['b','c'],[3,4],na.arange(12))
        
        assert(self.a + d == self.a.Marginalise(['a'])), \
               "Addition does not work..."
    
    def testIntEQIndex(self):
        self.a[1,1,1] = -2
        self.a[self.a==-2] = -3
        assert(self.a[1,1,1] == -3), \
              "Set by EQ does not work"

    def testAll(self):
        """ this is actually the Water-sprinkler example """
        c = DiscretePotential(['c'],[2],[0.5,0.5])                  # Pr(C)
        s = DiscretePotential(['s','c'],[2,2],[0.5, 0.9, 0.5, 0.1]) # Pr(S|C)
        r = DiscretePotential(['r','c'],[2,2],[0.8,0.2,0.2,0.8])    # Pr(R|C)
        w = DiscretePotential(['w','s','r'],[2,2,2])                # Pr(W|S,R)
        w[:,0,0]=[0.99, 0.01]
        w[:,0,1]=[0.1, 0.9]
        w[:,1,0]=[0.1, 0.9]
        w[:,1,1]=[0.0, 1.0]

        cr = c*r        # Pr(C,R)     = Pr(R|C) * Pr(C)
        crs = cr*s      # Pr(C,S,R)   = Pr(S|C) * Pr(C,R)
        crsw = crs*w    # Pr(C,S,R,W) = Pr(W|S,R) * Pr(C,R,S)

        # this can be verified using any bayesian network software

        # check the result for the multiplication and marginalisation
        assert(na.allclose(crsw.Marginalise('s r w'.split()).cpt,[0.5,0.5]) and \
               na.allclose(crsw.Marginalise('c r w'.split()).cpt,[0.7,0.3]) and \
               na.allclose(crsw.Marginalise('c s w'.split()).cpt,[0.5,0.5]) and \
               na.allclose(crsw.Marginalise('c s r'.split()).cpt,[0.349099,0.6509])),\
                "Something's wrong on the big Test..."


if __name__ == '__main__':
    suite = unittest.makeSuite(DiscretePotentialTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    suite = unittest.makeSuite(GaussianPotentialTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
    


