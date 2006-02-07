import delegate # no longer needed. Only RawCPT uses it, erase it when RawCPT doesn't exist anymore
import types
import numarray as na
from table import Table


# Testing
import unittest

# object gives access to __new__
class Distribution(object):
    """
    Base Distribution Class for all types of distributions
    defines the Pr(x|Pa(x))

    variables :
    -------------
        - self.family = [x, Pa(x)1, Pa(x)2,...,Pa(x)N)]
                        references to the nodes
        - self.names  = [name of x, name of Pa(x)1,..., name of Pa(x)N]
                        strings : the names of the nodes

        - self.order  = dict{var name : index}
                        the order in which the variables are stored

    """
    def __init__(self, v, isAdjustable = False):
        self.vertex = v     # the node to which this distribution is attached
        self.family = [v] + [parent for parent in v.in_v]
        self.ndimensions = len(self.family)
        self.parents = self.family[1:]
        self.names_list = [v.name for v in self.family]

        # the type of distribution
        self.distribution_type = None
       
        #used for learning
        self.isAjustable = isAdjustable
        
    def __str__(self):
        string = 'Distribution for node : '+ self.names_list[0]
        if len(self.names_list)>1: string += '\nParents : ' + str(self.names_list[1:])

        return string


class MultinomialDistribution(Distribution, Table):
    """     Multinomial/Discrete Distribution
    All nodes involved all discrete --> the distribution is represented by
    a Conditional Probability Table (CPT)
    This class now inherits from Distribution and Table.
    """
    def __init__(self, v, cpt = None, isAdjustable=False):
        Distribution.__init__(self, v, isAdjustable=isAdjustable)
        self.distribution_type = "Multinomial"
        
        assert(na.alltrue([v.discrete for v in self.family])), \
              'All nodes in family '+ str(self.names_list)+ ' must be discrete !!!'
        
        self.sizes = [v.nvalues for v in self.family]

        # initialize the cpt
        Table.__init__(self, self.names_list, self.sizes, cpt)

        #Used for Learning
        self.counts = None
    
    
    def setParameters(self, *args, **kwargs):
        ''' put values into self.cpt, delegated to Table class'''
        Table.setValues(self, *args, **kwargs)

    def Normalize(self):
        """ puts the cpt into canonical form : Sum(Pr(x=i|Pa(x)))=1 for each
        i in values(x)
        """
        # add all dimensions except for the first one
        # add a new empty dimension to keep the same rank as self.cpt
        s = self.cpt
        for i in range(1,self.ndimensions):
            s = na.sum(s,axis = 1)[...,na.NewAxis]
        self.cpt /= s
        return self.cpt

    def Convert_to_CPT(self):
        return self.cpt

    #==================================================
    #=== Learning Functions
    def initializeCounts(self):
        ''' initialize counts array to ones '''
        self.counts = na.ones(shape=self.sizes, type='Float32')        

    def addToCounts(self, index, counts):
        self.counts[index] += counts

    def Sample(self, pvals):
        """ Return a sample of the distribution P(V | pvals)
        """
        #FIXME: Currently assumes that all parents are set in pvals, but doesn't enforce that fact, this is important because if they are not all set, then the index of self.cpt[pvals] is not a table that sums to 1
        dist = self[pvals]
        rnum = ra.random()
        probRange = 0
        i = -1
        for prob in dist:
            probRange += prob
            i += 1
            if rnum <= probRange:
                break
        return i
    
    #===================================================
    #=== printing functions
    def __str__(self):
        string = 'Multinomial ' + Distribution.__str__(self)
        string += '\nConditional Probability Table (CPT) :\n'
        #---TODO: should find a nice neat way to represent numarrays
        #         only 3 decimals are sufficient... any ideas?
        string += repr(self.cpt)

        return string

#=================================================================
#=================================================================
class Gaussian_Distribution(Distribution):
    """ Gaussian Continuous Distribution
    
    parents can be either discrete or continuous.
    continuous parents (if any) : X
    discrete parents (if any) : Q
    this node : Y

    Pr(Y|X,Q) =
         - no parents: Y ~ N(mu, Sigma)
         - cts parents : Y|X=x ~ N(mu + W x, Sigma)
         - discrete parents: Y|Q=i ~ N(mu(i), Sigma(i))
         - cts and discrete parents: Y|X=x,Q=i ~ N(mu(i) + W(i) x, Sigma(i))

    
     The list below gives optional arguments [default value in brackets].
    
     mean       - numarray(shape=(len(Q1),len(Q2),...len(Qn))
                  the mean for each combination of parents
                  mean[i1,i2,...,in]
                  
     cov        - Sigma[:,:,i] is the covariance given Q=i [ repmat(100*eye(Y,Y), [1 1 Q]) ]
     weights    - W[:,:,i] is the regression matrix given Q=i [ randn(Y,X,Q) ]
     cov_type   - if 'diag', Sigma[:,:,i] is diagonal [ 'full' ]
     tied_cov   - if True, we constrain Sigma[:,:,i] to be the same for all i [False]
 """
    def __init__(self, v, mu = None, sigma = None, wi = None, \
                 cov_type = 'full', tied_cov = False, isAdjustable = True):
        
        Distribution.__init__(self, v, isAdjustable = isAdjustable)
        self.distribution_type = 'Gaussian'

        # check that current node is continuous
        if v.discrete:
            raise 'Node must be continuous'

        self.discrete_parents = [parent for parent in self.parents if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents if not parent.discrete]

        self.discrete_parents_shape = [dp.nvalues for dp in self.discrete_parents]
                
        # set the parameters : mean, cov, weights
        self.setParameters(mu=mu, sigma=sigma, wi=wi, cov_type=cov_type, \
                           tied_cov=tied_cov, isAdjustable=isAdjustable)
        
    def setParameters(self, mu = None, sigma = None, wi = None, cov_type = 'full', \
                      tied_cov = False, isAdjustable = False):
        
        # set the mean : 
        if mu == None:
            # set all mu to zeros
            mu = na.zeros(shape=self.discrete_parents_shape, type='Float32')
        try:
            mu = na.array(shape=self.discrete_parents_shape,type='Float32')
        except:
            raise 'Could not convert mu to numarray of shape : %s, discrete parents = %s' %(str(self.discrete_parents_shape),
                                                                                            str([dp.name for dp in self.discrete_parents]))
        self.mean = mu

        # set the covariance
        #---TODO:

        # set the wi
        #---TODO:

    def __str__(self):
        string = 'Gaussian ' + Distribution.__str__(self)
        string += '\nDiscrete Parents :' + str([p.name for p in self.discrete_parents])
        string += '\nContinuous Parents :' + str([p.name for p in self.continuous_parents])

        return string
#=================================================================
#   Test case for Gaussian_Distribution class
#=================================================================
class GaussianTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet, Water-Sprinkler
        G = BNet('Test')
        
        a,b,c,d = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('a b c d'.split(),[2,3,4,2])]
        ad,bd,cd,dd = a.distribution, b.distribution, c.distribution, d.distribution
        
        # sizes = (2,3,4,2)
        # a has 3 parents, b,c and d
        for ep in [(b,a), (c,a), (d,a)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()

        a.setCPT(na.arange(48))
        
        self.G = G
        self.a = a
        #print G

    def testSizes(self):
        assert (self.a.distribution.sizes == [2,3,4,2]), "Error with self.sizes"



#=================================================================
#   Test case for Distribution class
#=================================================================
class DistributionTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet
        G = BNet('Water Sprinkler Bayesian Network')
        
        c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('c s r w'.split(),[2,3,4,2])]
        
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()
        
        self.G = G
        #print G

    def testFamily(self):
        """ test parents, family, etc... """
        
        G = self.G
        c,s,r,w = G.v['c'],G.v['s'], \
                  G.v['r'],G.v['w']
        
        assert(c.distribution.parents == [] and \
               set(w.distribution.parents) == set([r,s]) and \
               r.distribution.parents == [c] and \
               s.distribution.parents == [c]), \
               "Error with parents"

        assert(c.distribution.family == [c] and \
               set(s.distribution.family) == set([c,s]) and \
               set(r.distribution.family) == set([r,c]) and \
               set(w.distribution.family) == set([w,r,s])), \
               "Error with family"

        assert(c.distribution.order['c'] == 0 and \
               set([w.distribution.order['w'],w.distribution.order['s'], w.distribution.order['r']]) == set([0,1,2])), \
               "Error with order"
 
#=================================================================
#   Test case for Multinomial_Distribution class
#=================================================================
class MultinomialTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet, Water-Sprinkler
        G = BNet('Test')
        
        a,b,c,d = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('a b c d'.split(),[2,3,4,2])]
        ad,bd,cd,dd = a.distribution, b.distribution, c.distribution, d.distribution
        
        # sizes = (2,3,4,2)
        # a has 3 parents, b,c and d
        for ep in [(b,a), (c,a), (d,a)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()

        a.setCPT(na.arange(48))
        
        self.G = G
        self.a = a
        #print G

    def testSizes(self):
        assert (self.a.distribution.sizes == [2,3,4,2]), "Error with self.sizes"


    # test the indexing of the cpt
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked correctly, by getting things out of the matrix
        """
        assert(na.all(self.a.distribution[0,0,0,:] == na.array([0,1])) and \
               na.all(self.a.distribution[1,0,0,:] == na.array([24,25]))), \
              "Error getting raw cpt"
    
    def testSetCPT(self):
        """ Violate abstraction and check that we can actually set elements.
        """
        self.a.distribution.cpt[0,1,0,:] = na.array([4,5])
        assert(na.all(self.a.distribution[0,1,0,:] == na.array([4,5]))), \
              "Error setting the array when violating abstraction"

    
    def testDictIndex(self):
        """ test that an index using a dictionary works correctly
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        assert(na.all(self.a.distribution[0,0,0,:] == self.a.distribution[index]) and \
               na.all(self.a.distribution[1,0,0,:] == self.a.distribution[index2])), \
              "Error getting with dict index"
    
    def testDictSet(self):
        """ test that an index using a dictionary can set a value within the cpt 
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        index3 = {'a':1,'b':1,'c':0}
        self.a.distribution[index] = -1
        self.a.distribution[index2] = 100
        self.a.distribution[index3] = na.array([-2, -3])
        assert(na.all(self.a.distribution[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.a.distribution[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.a.distribution[1,1,0,:] == na.array([-2, -3]))), \
              "Error setting cpt with dict"
    
    def testNumIndex(self):
        """ test that a raw index of numbers works correctly
        """
        assert(na.all(self.a.distribution[0,:,0,:] == self.a.distribution[0,:,0,:]) and \
               na.all(self.a.distribution[1,0,0,:] == self.a.distribution[1,0,0,:])), \
              "Error getting item with num indices"
    
    def testNumSet(self):
        """ test that a raw index of numbers can access and set a position in the 
        """
        self.a.distribution[0,0,0,:] = -1
        self.a.distribution[1,0,0,:] = 100
        self.a.distribution[1,1,0,:] = na.array([-2, -3])
        assert(na.all(self.a.distribution[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.a.distribution[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.a.distribution[1,1,0,:] == na.array([-2, -3]))), \
              "Error Setting cpt with num indices"


if __name__ == '__main__':
    from bayesnet import *

##    suite = unittest.makeSuite(DistributionTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
##
##    suite = unittest.makeSuite(MultinomialTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
    
    # create a small BayesNet
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('c s r w'.split(),[2,2,2,0])]
    w.discrete = False
    w.nvalues = 0
    
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))

    print G

    G.InitDistributions()
    c.setDistributionParameters([0.5, 0.5])
    s.distribution.setParameters([0.5, 0.9, 0.5, 0.1])
    r.distribution.cpt=na.array([0.8, 0.2, 0.2, 0.8])
##    w.distribution[:,0,0]=[0.99, 0.01]
##    w.distribution[:,0,1]=[0.1, 0.9]
##    w.distribution[:,1,0]=[0.1, 0.9]
##    w.distribution[:,1,1]=[0.0, 1.0]
    wd = w.distribution
    print wd.mean
    
