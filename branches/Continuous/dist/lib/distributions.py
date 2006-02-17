import types
import numarray as na
import numarray.random_array as ra
import random
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
        - vertex =      a reference to the BVertex instance containing the
                        variable quantified in this distribution
                        
        - family =      [x, Pa(x)1, Pa(x)2,...,Pa(x)N)]
                        references to the nodes
                   
        - names =       set(name of x, name of Pa(x)1,..., name of Pa(x)N)
                        set of strings : the names of the nodes
                        this is a set, no order is specified!
                   
        - names_list =  [name of x, name of Pa(x)1,..., name of Pa(x)N]
                        list of strings : the names of the nodes
                        this is a list, order is specified!

        - parents =     [name of Pa(x)1,...,name of Pa(x)N]
                        list of strings : the names of the node's parents
                        this is a list, order is the same as family[1:]

        - ndimensions = number of variables contained in this distribution
                        = len(self.family)

        - distribution_type = a string the type of the distribution
                              e.g. 'Multinomial', 'Gaussian', ...

        - isAdjustable = if True: the parameters of this distribution can
                         be learned.

        - nvalues =     the dimension of the distribution
                        discrete : corresponds to number of states
                        continuous : corresponds to number of dimensions
                        (e.g. 2D gaussian,...)

    """
    vertex = None 
    family = list()
    ndimensions = 0
    parents = list()
    names_list = list()
    #names = set()
    distribution_type = 'None'
    isAdjustable = False
    nvalues = 0
    
    def __init__(self, v, isAdjustable = False):
        """ Creates a new distribution for the given variable.
        v is a BVertex instance
        """
        ###################################################
        #---TODO: Should give an order to the variables, sort them by name for example...
        ###################################################
        self.vertex = v     # the node to which this distribution is attached
        self.family = [v] + [parent for parent in v.in_v]
        self.ndimensions = len(self.family)
        self.parents = self.family[1:]
        self.names_list = [v.name for v in self.family]
        #self.names = set(self.names_list)
        self.nvalues = self.vertex.nvalues

        # the type of distribution
        self.distribution_type = 'None'
       
        #used for learning
        self.isAdjustable = isAdjustable
        
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
    
    def Convert_to_CPT(self):
        return self.cpt

    #======================================================
    #=== Operations on CPT
    def normalize(self, dim=-1):
        """ If dim=-1 all elements sum to 1.  Otherwise sum to specific dimension, such that 
        sum(Pr(x=i|Pa(x))) = 1 for all values of i and a specific set of values for Pa(x)
        """
        if dim == -1 or len(self.cpt.shape) == 1:
            self.cpt /= self.cpt.sum()            
        else:
            ndim = self.assocdim[dim]
            order = range(len(self.names_list))
            order[0] = ndim
            order[ndim] = 0
            tcpt = na.transpose(self.cpt, order)
            t1cpt = na.sum(tcpt, axis=0)
            t1cpt = na.resize(t1cpt,tcpt.shape)
            tcpt = tcpt/t1cpt
            self.cpt = na.transpose(tcpt, order)

    def ones(self):
        """ All CPT elements are set to 1 """
        self.cpt = na.ones(self.cpt.shape, type=self.cpt.type())

    def zeros(self):
        """ All CPT elements are set to 0 """
        self.cpt = na.zeros(self.cpt.shape, type=self.cpt.type())

    def uniform(self):
        """ All CPT elements have equal probability :
            a = Pr(A|B,C,D)
            a.uniform()
            Pr(A=0)=Pr(A=1)=...=Pr(A=N)

            the result is a normalized CPT
            calls self.ones() and then self.normalize()
        """
        self.ones()
        self.normalize()
        
    ######################################################
    #---TODO: Should add some initialisation functions:
    #           all ones, uniform, zeros
    #           gaussian, ...
    ######################################################
    
    #======================================================
    #=== Sampling
    def sample(self, index={}):
        """ returns the index of the sampled value
        eg. a=Pr(A)=[0.5 0.3 0.0 0.2]
            a.sample() -->  5/10 times will return 0
                            3/10 times will return 1
                            2/10 times will return 3
                            2 will never be returned

            - returns an integer
            - only works for one variable tables
              eg. a=Pr(A,B); a.sample() --> ERROR
        """
        assert(len(self.names) == 1 or len(self.names - set(index.keys())) == 1),\
              "Sample only works for one variable tables"
        if not index == {}:
            tcpt = self.__getitem__(index)
        else:
            tcpt = self.cpt
        # csum is the cumulative sum of the distribution
        # csum[i] = na.sum(self.cpt[0:i])
        # csum[-1] = na.sum(self.cpt)
        csum = [na.sum(tcpt.flat[0:end+1]) for end in range(tcpt.shape[0])]
        
        # sample in this distribution
        r = random.random()
        for i,cs in enumerate(csum):
            if r < cs: return i
        return i

    #==================================================
    #=== Learning Functions
    def initializeCounts(self):
        ''' initialize counts array to ones '''
        self.counts = Table(self.names_list, shape=self.shape)
        
    def incrCounts(self, index):
        """ add one to given count """
        self.counts[index] += 1

    def addToCounts(self, index, counts):
        self.counts[index] += counts
    
    def setCounts(self):
        """ set the distributions underlying cpt equal to the counts """
        assert(self.names_list == self.counts.names_list)
        #set to copy in case we later destroy the counts or reinitialize them
        self.cpt = self.counts.cpt.copy()
    
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

    Notes: - this can be a scalar gaussian or multidimensional gaussian
             depending on the value of nvalues of the parent vertex
           - The domain is always defined as ]-inf,+inf[
             TODO: Maybe we should add a somain variable...
          
    parents can be either discrete or continuous.
    continuous parents (if any) : X
    discrete parents (if any) : Q
    this node : Y

    Pr(Y|X,Q) =
         - no parents: Y ~ N(mu(i), Sigma(i,j))     0 <= i,j < self.nvalues
         - cts parents : Y|X=x ~ N(mu + W x, Sigma)
         - discrete parents: Y|Q=i ~ N(mu(i), Sigma(i))
         - cts and discrete parents: Y|X=x,Q=i ~ N(mu(i) + W(i) x, Sigma(i))

    
     The list below gives optional arguments [default value in brackets].
    
     mean       - numarray(shape=(len(Y),len(Q1),len(Q2),...len(Qn))
                  the mean for each combination of DISCRETE parents
                  mean[i1,i2,...,in]
                  
     sigma        - Sigma[:,:,i] is the sigmaariance given Q=i [ repmat(100*eye(Y,Y), [1 1 Q]) ]
     weights      - W[:,:,i] is the regression matrix given Q=i [ randn(Y,X,Q) ]
     sigma_type   - if 'diag', Sigma[:,:,i] is diagonal [ 'full' ]
     tied_sigma   - if True, we constrain Sigma[:,:,i] to be the same for all i [False]
     """
     
    #---TODO: Maybe we should add a domain variable...
    def __init__(self, v, mu = None, sigma = None, wi = None, \
                 sigma_type = 'full', tied_sigma = False, isAdjustable = True):
        
        Distribution.__init__(self, v, isAdjustable = isAdjustable)
        self.distribution_type = 'Gaussian'

        # check that current node is continuous
        if v.discrete:
            raise 'Node must be continuous'

        self.discrete_parents = [parent for parent in self.parents if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents if not parent.discrete]

        self.discrete_parents_shape = [dp.nvalues for dp in self.discrete_parents]
        self.parents_shape = [p.nvalues for p in self.parents]
        if not self.parents_shape:self.parents_shape = [0]
                
        # set the parameters : mean, sigma, weights
        self.setParameters(mu=mu, sigma=sigma, wi=wi, sigma_type=sigma_type, \
                           tied_sigma=tied_sigma, isAdjustable=isAdjustable)
                           
       #---TODO: add support for sigma_type, tied_sigma
       #---TODO: add learning functions
        
    def setParameters(self, mu = None, sigma = None, wi = None, sigma_type = 'full', \
                      tied_sigma = False, isAdjustable = False):
        #============================================================
        # set the mean :
        # self.mean[i] = the mean for dimension i
        # self.mean.shape = (self.nvalues, q1,q2,...,qn)
        #        where qi is the size of discrete parent i
        if mu == None:
            # set all mu to zeros
            mu = na.zeros(shape=([self.nvalues]+self.discrete_parents_shape), \
                          type='Float32')
        try:
            mu = na.array(shape=[self.nvalues]+self.discrete_parents_shape, \
                          type='Float32')
        except:
            raise 'Could not convert mu to numarray of shape : %s, discrete parents = %s' %(str(self.discrete_parents_shape),
                                                                                            str([dp.name for dp in self.discrete_parents]))
        self.mean = mu

        #============================================================
        # set the covariance :
        # self.sigma[i,j] = the covariance between dimension i and j
        # self.sigma.shape = (nvalues,nvalues,q1,q2,...,qn)
        #        where qi is the size of discrete parent i
        if sigma == None:
            eye = na.identity(self.nvalues, type = 'Float32')[...,na.NewAxis]
            if len(self.discrete_parents) > 0:
                q = reduce(lambda a,b:a*b,self.discrete_parents_shape) # number of different configurations for the parents
                sigma = na.concatenate([eye]*q, axis=2)
                sigma = na.array(sigma,shape=[self.nvalues,self.nvalues]+self.discrete_parents_shape) 
        try:
            sigma = na.array(sigma, shape=[self.nvalues,self.nvalues]+self.discrete_parents_shape, type='Float32')
        except:
            raise 'Not a valid covariance matrix'
        self.sigma = sigma

        #============================================================
        # set the weights :
        # self.weights[i,j] = the regression for dimension i and continuous parent j
        # self.weights.shape = (nvalues,x1,x2,...,xn,q1,q2,...,qn)
        #        where xi is the size of continuous parent i)
        #        and qi is the size of discrete parent i
        
        if wi == None:
            wi = na.ones(shape=[self.nvalues]+self.parents_shape, type='Float32') 
        try:
            wi = na.array(wi, shape=[self.nvalues]+self.parents_shape, type='Float32')
        except:
            raise 'Not a valid weight'
        self.weights = wi
        
    #======================================================
    #=== Sampling
    def sample(self, index={}):
        raise "Not Implemented yet!!!"

    #==================================================
    #=== Learning Functions
    def initializeCounts(self):
        ''' initialize counts array to ones '''
        raise "Not Implemented yet!!!"
        
    def incrCounts(self, index):
        """ add one to given count """
        raise "Not Implemented yet!!!"

    def addToCounts(self, index, counts):
        raise "Not Implemented yet!!!"
    
    def setCounts(self):
        """ set the distributions underlying cpt equal to the counts """
        raise "Not Implemented yet!!!"

    #==================================================
    #=== Printing Functions    
    def __str__(self):
        string = 'Gaussian ' + Distribution.__str__(self)
        string += '\nDimensionality : ' + str(self.nvalues)
        string += '\nDiscrete Parents :' + str([p.name for p in self.discrete_parents])
        string += '\nContinuous Parents :' + str([p.name for p in self.continuous_parents])
        string += '\nMu : ' + str(self.mean)
        string += '\nSigma : ' + str(self.sigma)
        string += '\nWeights: ' + str(self.weights)

        return string
    
    
#=================================================================
#   Test case for Gaussian_Distribution class
#=================================================================
class GaussianTestCase(unittest.TestCase):
    """ Unit tests for general distribution class
    """
    def setUp(self):
        from bayesnet import BNet, BVertex, graph
        # create a small BayesNet
        self.G = G = BNet('Test')
        
        self.a = a = G.add_v(BVertex('a',discrete=False,nvalues=1))
        self.b = b = G.add_v(BVertex('b',discrete=False,nvalues=2))
        self.c = c = G.add_v(BVertex('c',discrete=False,nvalues=3))
        self.d = d = G.add_v(BVertex('d',discrete=True,nvalues=2))
        self.e = e = G.add_v(BVertex('e',discrete=True,nvalues=3))
        self.f = f = G.add_v(BVertex('f',discrete=False,nvalues=1))
        self.g = g = G.add_v(BVertex('g',discrete=False,nvalues=1))

        for ep in [(a,c),(b,c),(d,f),(e,f),(a,g),(b,g),(d,g),(e,g)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        #a,b : continuous(2,3), no parents
        #c   : continuous(3), 2 continuous parents (a,b)
        #d,e : discrete(2,3), no parents
        #f   : continuous(1), 2 discrete parents (d,e)
        #g   : continuous(1), 2 discrete parents (d,e) & 2 continuous parents (a,b)
   
        G.InitDistributions()

        self.ad = ad = a.distribution
        self.bd = bd = b.distribution
        self.cd = cd = c.distribution
        self.fd = fd = f.distribution
        self.gd = gd = g.distribution
        
    def testNoParents(self):
        ad = self.ad
        bd = self.bd
        # a and b have no parents, test basic parameters
        assert(ad.mean.shape == (1,) and \
               bd.mean.shape == (2,) ), \
               " Mean does not have the correct shape for no parents "

        assert(ad.sigma.shape == (1,1) and \
               bd.sigma.shape == (2,2) ), \
               " Sigma does not have the correct shape for no parents "

        assert(ad.weights.shape == (1,0) and \
               bd.weights.shape == (2,0)), \
               " Wi does not have the correct shape for no parents "
 
    def testContinuousParents(self):
		 """ test a gaussian with continuous parents """
		 cd = self.cd
		 #c has two continuous parents
		 assert(cd.mean.shape == (3,)), \
				"mean does not have correct shape for continous parents"
		 assert(cd.sigma.shape == (3,3)), \
				"sigma does not have correct shape for continous parents"
		 assert(cd.weights.shape == tuple([cd.nvalues]+cd.parents_shape)), \
				"weights does not have correct shape for continous parents"

    def testDiscreteParents(self):
        """ test a gaussian with discrete parents """
        fd = self.fd
        assert(fd.mean.shape == tuple([fd.nvalues]+fd.discrete_parents_shape)), \
                "mean does not have correct shape for discrete parents"
        assert(fd.sigma.shape == tuple([fd.nvalues,fd.nvalues]+fd.discrete_parents_shape)), \
                "sigma does not have correct shape for discrete parents"
        assert(fd.weights.shape == tuple([fd.nvalues]+fd.discrete_parents_shape)), \
                "weights does not have correct shape for discrete parents"        

    def testDiscrete_and_Continuous_Parents(self):
        """ test a gaussian with discrete and continuous parents """
        gd = self.gd
        assert(gd.mean.shape == tuple([gd.nvalues]+gd.discrete_parents_shape)), \
                "mean does not have correct shape for discrete & continuous parents"
        assert(gd.sigma.shape == tuple([gd.nvalues,gd.nvalues]+gd.discrete_parents_shape)), \
                "sigma does not have correct shape for discrete & continuous parents"
        assert(gd.weights.shape == tuple([gd.nvalues]+gd.parents_shape)), \
                "weights does not have correct shape for discrete & continuous parents"          
         




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

        assert(c.distribution.nvalues == c.nvalues), \
                "nvalues not set properly"

##        assert(c.distribution.order['c'] == 0 and \
##               set([w.distribution.order['w'],w.distribution.order['s'], w.distribution.order['r']]) == set([0,1,2])), \
##               "Error with order"
 
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
        
        self.G = G
        self.a, self.b, self.c, self.d = a,b,c,d
        #print G

    def testNormalize(self):
        a = MultinomialDistribution(self.G.v['a'])
        a.setParameters(range(48))
        a.normalize()
        
    def testSizes(self):
        assert (self.a.distribution.sizes == [2,3,4,2]), "Error with self.sizes"


    # test the indexing of the cpt
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked correctly, by getting things out of the matrix
        """
        assert(na.all(self.a.distribution[0,0,0,:] == self.a.distribution.cpt[0,0,0,:]) and \
               na.all(self.a.distribution[1,0,0,:] == self.a.distribution.cpt[1,0,0,:])), \
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
    suite = unittest.makeSuite(GaussianTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)    
##    from bayesnet import *
##
##    suite = unittest.makeSuite(DistributionTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
####
##
##    suite = unittest.makeSuite(MultinomialTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
##    
##    # create a small BayesNet
##    G = BNet('Water Sprinkler Bayesian Network')
##    
##    c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('c s r w'.split(),[2,2,2,0])]
##    w.discrete = False
##    w.nvalues = 0
##    
##    
##    for ep in [(c,r), (c,s), (r,w), (s,w)]:
##        G.add_e(graph.DirEdge(len(G.e), *ep))
##
##    print G
##
##    G.InitDistributions()
##    c.setDistributionParameters([0.5, 0.5])
##    s.distribution.setParameters([0.5, 0.9, 0.5, 0.1])
##    r.distribution.cpt=na.array([0.8, 0.2, 0.2, 0.8])
####    w.distribution[:,0,0]=[0.99, 0.01]
####    w.distribution[:,0,1]=[0.1, 0.9]
####    w.distribution[:,1,0]=[0.1, 0.9]
####    w.distribution[:,1,1]=[0.0, 1.0]
##    wd = w.distribution
##    print wd.mean
    
