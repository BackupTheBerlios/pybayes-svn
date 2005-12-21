import numarray as na

import delegate
import table
import unittest

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
    def Marginalise(self, varnames):
        """ sum(varnames) self.arr """
        temp = self.view()
        ax = [self.assocdim[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        remainingNames = self.names - set(varnames)
        return DiscretePotential(remainingNames, temp.shape, elements=temp.flat)

    def Normalise(self):
        self.cpt = na.divide(self.cpt, na.sum(self.cpt.flat), self.cpt)

    #================================
    # Initialise
    def Uniform(self):
        ' Uniform distribution '
        N = na.product(self.shape)
        self[:] = 1.0/N

    #===================================
    # Printing
    def __str__(self): return str(self.arr)

    def Printcpt(self):
        string =  str(self.arr) + '\nshape:'+str(self.arr.shape)+'\nnames:'+str(self.names)+'\nsum : ' +str(na.sum(self.arr.flat))
        print string

# this class should be named discreteJointTree Potential
# we will later create anotther two classes of JoinTreePotential :
# - Continuous and SG Potential (continuous & discrete)
class JoinTreePotential(DiscretePotential):
    """
    The potential of each node/Cluster and edge/SepSet in a
    Join Tree Structure
    
    self.cpt = Pr(X)
    
    where X is the set of variables contained in Cluster or SepSet
    self.vertices contains the graph.vertices instances where the variables
    come from
    """
    def __init__(self, names, shape, cpt=None):
        """ self. vertices must be set """
        DiscretePotential.__init__(self, names, shape, cpt)


    #=========================================
    # Operations
    def __add__(c,s):
        """
        sum(X\S)phiX

        marginalise the variables contained in BOTH SepSet AND in Cluster
        """
        var = set(v.name for v in c.vertices) - set(v.name for v in s.vertices)
        return c.Marginalise(var)


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

        assert(b.names == self.a.names - var and \
               b[0,1] == na.sum(self.a[0,1]) and \
               c.names == self.a.names - var2 and \
               na.alltrue(c.cpt.flat == na.sum(na.sum(self.a.cpt,axis=2),axis=0))), \
               " Marginalisation doesn't work"


if __name__ == '__main__':
    suite = unittest.makeSuite(DiscretePotentialTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
    names = ('a','b','c')
    shape = (2,3,4)
    a = DiscretePotential(names,shape,na.arange(24))
