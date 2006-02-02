import numarray as na

import delegate
import table
import unittest
import copy

class Potential:
    """ General Potential class that will be inherited by all potentials
    Maybe we should delegate to a type of potential, the same we did for the
    Distributions
    """
    def __init__(self, names, shape):
        self.names = set(names)
        self.names_list = list(names)
        self.shape = shape
        
class DiscretePotential(Potential, table.Table):
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and Marginalise().
    """
    def __init__(self, names, shape, elements=None):
        Potential.__init__(self, names, shape)
        
        if elements == None:
            elements = na.ones(shape=shape)
        table.Table.__init__(self, names, shape, elements, 'Float32')

    #=========================
    # Operations
    def Marginalise(self, vars):
        """ sum(varnames) self.arr
        eg. a = Pr(A,B,C,D)
        a.Marginalise(['A','C']) --> Pr(A,C)
        THIS HAS BEEN REVERSED FROM INITIAL WRITE, TO 
        MAKE CLOSER TO SPECIFICATION IN HUANG

        returns a new DiscretePotential instance
        the variables keep their relative order
        """
        assert(self.names.issuperset(vars))
        varnames = self.names.difference(vars)
        temp = self.cpt.view()
        ax = [self.assocdim[v] for v in varnames]
        ox = list(set(self.assocdim.values()) - set(ax))
        ox.sort()
        #extract proper names in proper order
        names_list = [self.assocname[dim] for dim in ox]

        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        
        return self.__class__(names_list, temp.shape, temp.flat)

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
    
    def __copy__(self):
        names = copy.copy(self.names_list)
        shape = copy.copy(self.shape)
        newcpt = self.cpt.copy()
        return self.__class__(names,shape,newcpt)

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
        var = set('c')
        b = self.a.Marginalise(var)
        var2 = set(['c','a'])
        c = self.a.Marginalise(var2)
        d = DiscretePotential(['b','c'],[3,4],na.arange(12))
      

        assert(b.names == self.a.names - var and \
               b[0,1] == na.sum(self.a[0,1]) and \
               c.names == self.a.names - var2 and \
               na.alltrue(c.cpt.flat == na.sum(na.sum(self.a.cpt,axis=2),axis=0))), \
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


if __name__ == '__main__':
    suite = unittest.makeSuite(DiscretePotentialTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
    names = ('a','b','c')
    shape = (2,3,4)
    a = DiscretePotential(names,shape,na.arange(24))
