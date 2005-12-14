import delegate # no longer needed. Only RawCPT uses it, erase it when RawCPT doesn't exist anymore
import types
import numarray as na


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
    def __init__(self, isAdjustable = False):
        self.family = [self] + [parent for parent in self.in_v]
        self.ndimensions = len(self.family)
        self.parents = self.family[1:]
        self.names = [v.name for v in self.family]
        # order = dict{ var name : index }
        self.order = dict((k,v) for k,v in zip(self.names, range(len(self.names))))

        self.distribution_type = 'None' # just the name of the distribution
        self.distribution = None

        #used for learning
        self.isAjustable = isAdjustable
        
        # self.names contains the names of the Family of V [V,Pa1(V),Pa2(V),...]
        # self.family contains the corresponding vertices (pointers)
        # only use self.family/names for iterating over dimensions...

        # order is given by the underlying BNet

    def SetDistribution(self, distribution, *args, **kwargs):
        ''' sets and returns the distribution for this node
            distribution = a CLASS, not an instance

            POST : self.distribution = a distribution instance

            # with no arguments, uses defaults            
            >>> bvertex.SetDistribution(Multinomial_Distribution)

            # with some arguments, (uses the order defined in the distribution.__init__)
            >>> bvertex.SetDistribution(Multinomial_Distribution, cpt_array)

            # with some keyword arguments
            >>> bvertex.SetDistribution(Multinomial_Distribution, cpt = cpt_array)
            
        '''
        # creates self.distribution as an instance of the class specified
        # all the work is done by this new distribution. This class simply
        # delegates everything to self.distribution (__getitem__,__setitem__ and all
        # other unknown attributes). This way, this class becomes virtually a
        # ditribution of the type specified
        self.distribution = distribution.__new__(distribution)
        self.distribution.__init__(self, *args, **kwargs)
        return self.distribution

    def __getitem__(self,index):
        # delegate to self.distribution
        return self.distribution[index]

    def __setitem__(self,index, value):
        # delegate to self.distribution
        self.distribution[index] = value

    def __getattr__(self, name):
        """ any attributes not found on this instance are delegated to
        self.distribution, if they exist """
        # delegate only if self.distribution contains that attribute
        if self.distribution.__dict__.has_key(name) or type(self.distribution).__dict__.has_key(name):
            return getattr(self.distribution, name)
        else:
            raise 'Could not find attribute',name
        
    def __str__(self):
        string = 'Distribution for node : '+ self.names[0]
        string += '\nParents : ' + str(self.names[1:])

        return string


class Multinomial_Distribution(object):
    """     Multinomial/Discrete Distribution
    All nodes involved all discrete --> the distribution is represented by
    a Conditional Probability Table (CPT)
    """
    def __init__(self, parent_dist, cpt = None):
        # this class is not a child of Distributions !!!
        # it virtually passes all calls to attributes that do not exist
        # in this class to the parent_distribution class which is defined at
        # __init__.
        # this way, this class is virtually inheriting parent_distribution
        # (attributes like self.family, self.names, etc.. can be accessed
        # by using self.names
        self.parent_dist = parent_dist
        parent_dist.distribution_type = 'Multinomial'

        if not na.alltrue(na.array([v.discrete for v in self.family])):
            error = 'All nodes in family '+ str(self.names)+ ' must be discrete !!!'
            raise error
            
                          
        self.sizes = [v.nvalues for v in self.family]

        # initialize the cpt
        if cpt == None:
            self.cpt = na.ones(shape=self.sizes, type='Float32')
        else:
            self.setCPT(cpt)

        #Used for Learning
        self.counts = None
        
    def __getattr__(self, name):
        # always delegate to parent_dist
        # if it is not found there, an error will occur
        return getattr(self.parent_dist, name)
    
    def setCPT(self, cpt):
        ''' put values into self.cpt'''
        self.cpt = na.array(cpt, shape=self.sizes, type='Float32')


    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = na.mlab.rand(*self.sizes)

    def AllOnes(self):
        self.cpt = na.ones(self.sizes, type='Float32')

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
        strIndex = self._strIndexFromDict(index)
        if isinstance(counts, na.ArrayType):
            exec "self.counts["+strIndex+"]+=na." + repr(counts)
        else:
            exec "self.counts["+strIndex+"]+=" + repr(counts)

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

    #=========================================================
    #=== indexing functions
    #---TODO: this could be part of an indexing class, we will also use it
    #         for potentials
    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.
        Index can be a string as in PBNT ('1,:,1'), a dictionary
        of var name:value pairs, or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]
        """
        if isinstance(index, types.DictType):
            numIndex = self._numIndexFromDict(index)
            return self._getNumItem(numIndex)
        if isinstance(index, types.StringType):
            return self._getStrIndex(index)
        return self._getNumItem(index)
    
    def _getStrIndex(self, index):
        """ Helper function for __getitem__, takes a string as index.
        """
        return eval("self.cpt["+index+"]")
    
    def _getNumItem(self, index):
        """ Helper function for __getitem__, index are numbers as in array[1,:,1]
        """
        return self.cpt[index]
    
    def __setitem__(self, index, value):
        """ Overload array-style indexing and setting behaviour, as in __getitem__ this will take a dictionary, string, or normal set of numbers as index
        """
        if isinstance(index, types.DictType):
            numIndex = self._numIndexFromDict(index)    #use numindexfromdict instaead of strindexfromdict, bug fix
            return self._setNumItem(numIndex, value)
        if isinstance(index, types.StringType):
            return self._setStrIndex(index, value)
        return self._setNumItem(index, value)
    
    def _setStrIndex(self, index, value):
        v = na.array(value) #convert to array
        exec "self.cpt["+index+"]=na." + repr(v)
    
    def _setNumItem(self, index, value):
        self.cpt[index] = value
        return

    # deleted _getStrIndexFromDict(), no longer used
    
    def _numIndexFromDict(self, d):
        index = []
        for vname in self.names:
            if d.has_key(vname):
                index.append(d[vname])
            else:
                index.append(slice(None,None,None))
        return tuple(index) # must convert to tuple in order to work, bug fix

    #===================================================
    #=== printing functions
    def __str__(self):
        string = 'Multinomial ' + Distribution.__str__(self)
        string += 'Conditional Probability Table (CPT) :\n'
        #---TODO: should find a nice neat way to represent numarrays
        #         only 3 decimals are sufficient... any ideas?
        string += repr(self.cpt)

        return string

#=================================================================
#=================================================================
class Gaussian_Distribution(object):
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
    def __init__(self, parent_dist, mu = None, sigma = None, wi = None, cov_type = 'full', tied_cov = False):
        self.parent_dist = parent_dist
        parent_dist.distribution_type = 'Gaussian'

        # check that current node is continuous
        if BVertex.discrete:
            raise 'Node must be continuous'

        self.discrete_parents = [parent for parent in self.parents if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents if not parent.discrete]

        self.discrete_parents_shape = [dp.nvalues for dp in self.discrete_parents]
                
        # set the parameters : mean, cov, weights
        # set the mean : self.mean[:
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
        
    def __getattr__(self, name):
        # always delegate to parent_dist
        # if it is not found there, an error will occur
        return getattr(self.parent_dist, name)        
        
        
    def __str__(self):
        string = 'Gaussian ' + Distribution.__str__(self)
        string += 'Discrete Parents :' + str([p.name for p in self.discrete_parents])
        string += 'Continuous Parents :' + str([p.name for p in self.continuous_parents])

        return string
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
        c,s,r,w = G.v['c'],G.v['s'],G.v['r'],G.v['w']

        assert(c.parents == [] and \
               set(w.parents) == set([r,s]) and \
               r.parents == [c] and \
               s.parents == [c]), \
               "Error with parents"

        assert(c.family == [c] and \
               set(s.family) == set([c,s]) and \
               set(r.family) == set([r,c]) and \
               set(w.family) == set([w,r,s])), \
               "Error with family"

        assert(c.order['c'] == 0 and \
               set([w.order['w'],w.order['s'], w.order['r']]) == set([0,1,2])), \
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
        # sizes = (2,3,4,2)
        # a has 3 parents, b,c and d
        for ep in [(b,a), (c,a), (d,a)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))

        G.InitDistributions()

        for v in G.v.values():
            v.SetDistribution(Multinomial_Distribution)

        a.setCPT(na.arange(48))
        
        self.G = G
        self.a = a
        #print G

    def testSizes(self):
        assert (self.a.sizes == [2,3,4,2]), "Error with self.sizes"


    # test the indexing of the cpt
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked correctly, by getting things out of the matrix
        """
        assert(na.all(self.a[0,0,0,:] == na.array([0,1])) and \
               na.all(self.a[1,0,0,:] == na.array([24,25]))), \
              "Error setting raw cpt"
    
    def testSetCPT(self):
        """ Violate abstraction and check that we can actually set elements.
        """
        self.a.cpt[0,1,0,:] = na.array([4,5])
        assert(na.all(self.a[0,1,0,:] == na.array([4,5]))), \
              "Error setting the array when violating abstraction"
        
    def testStrIndex(self):
        """ test that an index using strings works correctly
        """
        index = '0,0,0,:'
        index2 = '1,0,0,:'
        assert(na.all(self.a[0,0,0,:] == self.a[index]) and \
               na.all(self.a[1,0,0,:] == self.a[index2])), \
              "Error getting with str index"
    
    def testStrSet(self):
        """ test that an index using strings can access and set a value in the cpt
        """
        index = '0,0,0,:'
        index2 = '1,0,0,:'
        index3 = '1,1,0,:'
        self.a[index] = -1
        self.a[index2] = 100
        self.a[index3] = na.array([-2, -3])
        assert(na.all(self.a[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.a[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.a[1,1,0,:] == na.array([-2, -3]))), \
              "Error setting with str indices"
    
    def testDictIndex(self):
        """ test that an index using a dictionary works correctly
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        assert(na.all(self.a[0,0,0,:] == self.a[index]) and \
               na.all(self.a[1,0,0,:] == self.a[index2])), \
              "Error getting with dict index"
    
    def testDictSet(self):
        """ test that an index using a dictionary can set a value within the cpt 
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        index3 = {'a':1,'b':1,'c':0}
        self.a[index] = -1
        self.a[index2] = 100
        self.a[index3] = na.array([-2, -3])
        assert(na.all(self.a[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.a[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.a[1,1,0,:] == na.array([-2, -3]))), \
              "Error setting cpt with dict"
    
    def testNumIndex(self):
        """ test that a raw index of numbers works correctly
        """
        assert(na.all(self.a[0,:,0,:] == self.a[0,:,0,:]) and \
               na.all(self.a[1,0,0,:] == self.a[1,0,0,:])), \
              "Error getting item with num indices"
    
    def testNumSet(self):
        """ test that a raw index of numbers can access and set a position in the 
        """
        self.a[0,0,0,:] = -1
        self.a[1,0,0,:] = 100
        self.a[1,1,0,:] = na.array([-2, -3])
        assert(na.all(self.a[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.a[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.a[1,1,0,:] == na.array([-2, -3]))), \
              "Error Setting cpt with num indices"

#=================================================================
#=================================================================
# old code, didn't touch this part but it's becoming obsolete
# keep it because we can fetch code in that
# I deleted any parts no longer needed
class RawCPT(delegate.Delegate):
    def Marginalise(self, varnames):
        """ sum(varnames) self.cpt """
        temp = self.cpt
        ax = [self.p[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        return temp

    def Uniform(self):
        ' Uniform distribution '
        N = len(self.cpt.flat)
        self.cpt = na.array([1.0/N for n in range(N)], shape = self.cpt.shape, type='Float32')

    def __mul__(a,b):
        """
        a keeps the order of its dimensions
        
        always use a = a*b or b=b*a, not b=a*b
        """
        
        aa,bb = a.cpt, b.cpt
        
        correspondab = a.FindCorrespond(b)
        
        while aa.rank < len(correspondab): aa = aa[..., na.NewAxis]
        while bb.rank < len(correspondab): bb = bb[..., na.NewAxis]

        bb = na.transpose(bb, correspondab)

        return aa*bb

    def __str__(self): return str(self.cpt)

    def FindCorrespond(a,b):
        correspond = []
        k = len(b.p)
        for p in a.Fv:   #p=var name
            if b.p.has_key(p): correspond.append(b.p[p])
            else:
                correspond.append(k)
                k += 1

        for p in b.Fv:
            if not a.p.has_key(p):
                correspond.append(b.p[p])
                
        return correspond

    def Printcpt(self):
        string =  str(self.cpt) + '\nshape:'+str(self.cpt.shape)+'\nFv:'+str(self.Fv)+'\nsum : ' +str(na.sum(self.cpt.flat))
        print string



class CPT(RawCPT):
    def __init__(self, nvalues, parents):
        names = [self.name]
        names.extend([p.name for p in parents])
        shape = [p.nvalues for p in parents]
        shape.insert(0, nvalues)
        
        RawCPT.__init__(self, names, shape)

    def makecpt(self):
        """
        makes a consistent conditional probability distribution
        sum(parents)=1
        """
        shape = self.cpt.shape
        self.cpt.shape = (shape[0], MultiplyElements(shape[1:]))
        self.cpt /= na.sum(self.cpt, axis=0)
        self.cpt.shape = shape

def MultiplyElements(d):
    "multiplies the elements of d between them"
    #this one is the fastest
    nel = 1
    for x in d: nel = nel * x

    return nel


if __name__ == '__main__':
    from bayesnet import *

    suite = unittest.makeSuite(DistributionTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    suite = unittest.makeSuite(MultinomialTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
    
    # create a small BayesNet
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('c s r w'.split(),[2,3,4,5])]
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))

    print G

    G.InitDistributions()

    w.SetDistribution(Multinomial_Distribution)
    index = {'w':0,'c':1}
   
    #ds = s.SetDistribution(Multinomial_Distribution)
    #ds.Normalize()
    #print ds.cpt

    #dw = w.SetDistribution(Gaussian_Distribution,mu=[0,1])
    #print dw

    class A:
        cpt = na.arange(10,shape=(2,5))
        def __getitem__(self,i):
            print i,i.__class__
            return self.cpt[i]

    a=A()
    a[0]
    
    
