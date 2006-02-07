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
        
class DiscretePotential(Potential, table.Table):
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and Marginalise().
    """
    def __init__(self, names, shape, elements=None):
        order = Potential.__init__(self, names)

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
        """ sum(varnames) self.arr
        eg. a = Pr(A,B,C,D)
        a.Marginalise(['A','C']) --> Pr(B,D)

        returns a new DiscretePotential instance
        the variables keep their relative order
        """
        temp = self.cpt.view()
        ax = [self.assocdim[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        newnames = copy(self.names_list)
        for a in ax:
            temp = na.sum(temp, axis = a)
            newnames.pop(a)

        #=================================================
        #---ERROR : In which order ?????
        #remainingNames = self.names - set(varnames)
        #remainingNames_list = [name for name in self.names_list if name in remainingNames]

        return self.__class__(newnames, temp.shape, temp)

    def __add__(self, other):
        """
        sum(X\S)phiX

        marginalise the variables contained in BOTH SepSet AND in Cluster
        returns a new DiscretePotential instance

        eg: a = Pr(A,B,C)
            b = Pr(B,C)

            a + b <=> a.Marginalise(set(a.names) - set(b.names))
            = Sum(A)a = Pr(B,C)

        only the names of the variables contained in b are relevant!
        no operation with b is done in practice
        """
        var = set(v for v in self.names) - set(v for v in other.names)
        return self.Marginalise(var)

    def Normalise(self):
        self.cpt /= na.sum(self.cpt.flat)

            
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
        print crs,crs.names_list
        print crs[:,0,0]
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
    names = ('a','b','c')
    shape = (2,3,4)
    a = DiscretePotential(names,shape,na.arange(24))

    names = ('a','d','b')
    shape = (2,5,3)
    b = DiscretePotential(names,shape,na.arange(2*5*3))

    c = DiscretePotential(['c'],[2],[0.5,0.5])
    s = DiscretePotential(['s','c'],[2,2],[0.5, 0.9, 0.5, 0.1])
    r = DiscretePotential(['r','c'],[2,2],[0.8,0.2,0.2,0.8])
    w = DiscretePotential(['w','s','r'],[2,2,2])
    w[:,0,0]=[0.99, 0.01]
    w[:,0,1]=[0.1, 0.9]
    w[:,1,0]=[0.1, 0.9]
    w[:,1,1]=[0.0, 1.0]

    cr = c*r
    crs = cr*s
    crsw = crs*w

    print 'c:', crsw.Marginalise('s r w'.split())
    print 's:', crsw.Marginalise('c r w'.split())
    print 'r:', crsw.Marginalise('c s w'.split())
    print 'w:', crsw.Marginalise('c s r'.split())

