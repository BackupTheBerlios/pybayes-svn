import delegate
import numarray as na
import numarray.random_array as ra

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
    def __init__(self, BVertex, isAdjustable=True):
        self.family = [BVertex] + [parent for parent in BVertex.in_v]
        self.parents = self.family[1:]
        self.names = [v.name for v in self.family]
        # order = dict{ var name : index }
        self.order = dict((k,v) for k,v in zip(self.names, range(len(self.names))))

        self.distribution_type = 'None' # just the name of the distribution
        self.isAjustable = isAdjustable
        
        # self.names contains the names of the Family of V [V,Pa1(V),Pa2(V),...]
        # self.family contains the corresponding vertices (pointers)
        # only use self.family for iterating over dimensions...

        # order is given by the underlying BNet

    def __str__(self):
        string = 'Distribution for node : '+ self.names[0]
        string += '\nParents : ' + str(self.names[1:])

        return string


class Multinomial_Distribution(Distribution):
    """     Multinomial/Discrete Distribution
    All nodes involved all discrete --> the distribution is represented by
    a Conditional Probability Table (CPT)
    """
    def __init__(self, BVertex, cpt = None, isAdjustable=True):
        Distribution.__init__(self, BVertex, isAdjustable)
        self.distribution_type = 'Multinomial'

        if not na.alltrue(na.array([v.discrete for v in self.family])):
            error = 'All nodes in family '+ str(self.names)+ ' must be discrete !!!'
            raise error
            
                          
        self.sizes = [v.nvalues for v in self.family]

        if cpt == None:
            self.cpt = na.ones(shape=self.sizes, type='Float32')
        else:
            self.setCPT(cpt)
        
        #Used for Learning
        self.counts = None

    def setCPT(self, cpt):
        ''' put values into self.cpt'''
        self.cpt = na.array(cpt, shape=self.sizes, type='Float32')
    
    def initializeCounts(self):
        ''' initialize counts array to ones '''
        self.counts = na.ones(shape=self.sizes, type='Float32')        

    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = na.mlab.rand(*self.sizes)

    def AllOnes(self):
        self.cpt = na.ones(self.sizes, type='Float32')

    def Normalize(self):
        """ puts the cpt into canonical form : Sum(Pr(x|Pa(x)))=1 for each
        combination of Pa(x)
        """
        self.cpt /= na.sum(self.cpt, axis = 0)
        return self.cpt

    def Convert_to_CPT(self):
        return self.cpt
    
    def addToCounts(self, index, counts):
        strIndex = self._strIndexFromDict(index)
        if isinstance(counts, na.ArrayType):
            exec "self.counts["+strIndex+"]+=na." + repr(counts)
        else:
            exec "self.counts["+strIndex+"]+=" + repr(counts)

    def __str__(self):
        string = 'Multinomial ' + Distribution.__str__(self)
        string += 'Conditional Probability Table (CPT) :\n'
        string += repr(self.cpt)

        return string
    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.  Index can be a string as in PBNT ('1,:,1'), a dictionary of var name:value pairs, or pure numbers as in the standard way of accessing a numarray array array[1,:,1]
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
            strIndex = self._strIndexFromDict(index)
            return self._setStrIndex(strIndex, value)
        if isinstance(index, types.StringType):
            return self._setStrIndex(index, value)
        return self._setNumItem(index, value)
    
    def _setStrIndex(self, index, value):
        if isinstance(value, na.ArrayType):
            exec "self.cpt["+index+"]=na." + repr(value)
        else:
            exec "self.cpt["+index+"]=" + repr(value)
        
    def _setNumItem(self, index, value):
        self.cpt[index] = value
        return
    
    def _strIndexFromDict(self, d):
        index = ''
        for vname in self.Fv:
            if d.has_key(vname):
                index += repr(d[vname]) + ','
            else:
                index += ':,'
        return index[:-1]
            
    def _numIndexFromDict(self, d):
        index = []
        for vname in self.Fv:
            if d.has_key(vname):
                index.append(d[vname])
            else:
                index.append(slice(None,None,None))
        return index

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
    
        
        


#=================================================================
class Gaussian_Distribution(Distribution):
    """ Gaussian Continuous Distribution
    
    parents can be either discrete or continuous
    """
    def __init__(self, BVertex, mu = None, sigma = None, wi = None, isAdjustable=True):
        Distribution.__init__(self, BVertex, isAdjustable)
        self.distribution_type = 'Gaussian'

        # check that current node is continuous
        if BVertex.discrete:
            raise 'Node must be continuous'

        self.discrete_parents = [parent for parent in self.parents if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents if not parent.discrete]
                
        
    def __str__(self):
        string = 'Gaussian ' + Distribution.__str__(self)
        string += 'Discrete Parents :' + str([p.name for p in self.discrete_parents])
        string += 'Continuous Parents :' + str([p.name for p in self.continuous_parents])

        return string

#=================================================================
# old code, didn't touch this part but it's becoming obsolete
# keep it because we can fetch code in that
class RawCPT(delegate.Delegate):
    def __init__(self, names, shape):
        # keys = variable names, values = index
        self.p = dict((k,v) for k,v in zip(names, range(len(names))))
        self.Fv = self.p.items()
        self.Fv.sort(cmp=lambda x,y: cmp(x[1],y[1]))    # sort family by index
        self.Fv = [f[0] for f in self.Fv]   # list of names of vars ordered by index
        # self.Fv contains the names of the Family of V
        # only use self.Fv for iterating over dimensions... not self.p.items()
        self.cpt = na.ones(shape, type='Float32')
        
    def setCPT(self, cpt):
        ''' put values into self.cpt'''
        self.cpt = na.array(cpt, shape=self.cpt.shape, type='Float32')
        
    def rand(self):
        ''' put random values to self.cpt '''
        self.cpt = na.mlab.rand(*self.cpt.shape)

    def Marginalise(self, varnames):
        """ sum(varnames) self.cpt """
        temp = self.cpt
        ax = [self.p[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        return temp

    def AllOnes(self):
        self.cpt = na.ones(self.cpt.shape, type='Float32')

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

    def __getitem__(self, index):
        """ Overload array-style indexing behaviour.  Index can be a string as in PBNT ('1,:,1'), a dictionary of var name:value pairs, or pure numbers as in the standard way of accessing a numarray array array[1,:,1]
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
            strIndex = self._strIndexFromDict(index)
            return self._setStrIndex(strIndex, value)
        if isinstance(index, types.StringType):
            return self._setStrIndex(index, value)
        return self._setNumItem(index, value)
    
    def _setStrIndex(self, index, value):
        exec "self.cpt["+index+"]=na." + repr(value)
    
    def _setNumItem(self, index, value):
        self.cpt[index] = value
        return
    
    def _strIndexFromDict(self, d):
        index = ''
        for vname in self.Fv:
            if d.has_key(vname):
                index += repr(d[vname]) + ','
            else:
                index += ':,'
        return index[:-1]
    
    def _numIndexFromDict(self, d):
        index = []
        for vname in self.Fv:
            if d.has_key(vname):
                index.append(d[vname])
            else:
                index.append(slice(None,None,None))
        return index
        
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

#========================================================================
class CPTIndexTestCase(unittest.TestCase):
    """ Unit tests for indexing and setting the RawCPT class with [] notation.
    """
    def setUp(self):
        names = ['a','b','c','d']
        shape = (2,3,4,2)
        cpt = Multinomial_Distribution(names,shape)
        cpt.setCPT(range(48))
        self.cpt = cpt
    
    def testGetCPT(self):
        """ Violate abstraction and check that setCPT actually worked correctly, by getting things out of the matrix
        """
        assert(na.all(self.cpt.cpt[0,0,0,:] == na.array([0,1])) and \
               na.all(self.cpt.cpt[1,0,0,:] == na.array([24,25]))), \
              "Error setting raw cpt"
    
    def testSetCPT(self):
        """ Violate abstraction and check that we can actually set elements.
        """
        self.cpt.cpt[0,1,0,:] = na.array([4,5])
        assert(na.all(self.cpt.cpt[0,1,0,:] == na.array([4,5]))), \
              "Error setting the array when violating abstraction"
        
    def testStrIndex(self):
        """ test that an index using strings works correctly
        """
        index = '0,0,0,:'
        index2 = '1,0,0,:'
        assert(na.all(self.cpt.cpt[0,0,0,:] == self.cpt[index]) and \
               na.all(self.cpt.cpt[1,0,0,:] == self.cpt[index2])), \
              "Error getting with str index"
    
    def testStrSet(self):
        """ test that an index using strings can access and set a value in the cpt
        """
        index = '0,0,0,:'
        index2 = '1,0,0,:'
        index3 = '1,1,0,:'
        self.cpt[index] = -1
        self.cpt[index2] = 100
        self.cpt[index3] = na.array([-2, -3])
        assert(na.all(self.cpt.cpt[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.cpt.cpt[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.cpt.cpt[1,1,0,:] == na.array([-2, -3]))), \
              "Error setting with str indices"
    
    def testDictIndex(self):
        """ test that an index using a dictionary works correctly
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        assert(na.all(self.cpt.cpt[0,0,0,:] == self.cpt[index]) and \
               na.all(self.cpt.cpt[1,0,0,:] == self.cpt[index2])), \
              "Error getting with dict index"
    
    def testDictSet(self):
        """ test that an index using a dictionary can set a value within the cpt 
        """
        index = {'a':0,'b':0,'c':0}
        index2 = {'a':1,'b':0,'c':0}
        index3 = {'a':1,'b':1,'c':0}
        self.cpt[index] = -1
        self.cpt[index2] = 100
        self.cpt[index3] = na.array([-2, -3])
        assert(na.all(self.cpt.cpt[0,0,0,:] == array([-1, -1])) and \
               na.all(self.cpt.cpt[1,0,0,:] == array([100, 100])) and \
               na.all(self.cpt.cpt[1,1,0,:] == array([-2, -3]))), \
              "Error setting cpt with dict"
    
    def testNumIndex(self):
        """ test that a raw index of numbers works correctly
        """
        assert(na.all(self.cpt.cpt[0,:,0,:] == self.cpt[0,:,0,:]) and \
               na.all(self.cpt.cpt[1,0,0,:] == self.cpt[1,0,0,:])), \
              "Error getting item with num indices"
    
    def testNumSet(self):
        """ test that a raw index of numbers can access and set a position in the 
        """
        self.cpt[0,0,0,:] = -1
        self.cpt[1,0,0,:] = 100
        self.cpt[1,1,0,:] = na.array([-2, -3])
        assert(na.all(self.cpt.cpt[0,0,0,:] == na.array([-1, -1])) and \
               na.all(self.cpt.cpt[1,0,0,:] == na.array([100, 100])) and \
               na.all(self.cpt.cpt[1,1,0,:] == na.array([-2, -3]))), \
              "Error Setting cpt with num indices"
    
    def testAddToCounts(self):
        """ Test that we can add a basic number to the counts.        
        """
        index = {'a':1,'b':2,'c':0,'d':0}
        self.cpt.addToCounts(index, 1)
        assert(self.cpt.counts[1,2,0,0] == 2), \
              "Error adding integer to counts"
        index = {'a':1,'b':2,'c':0}
        self.cpt.addToCounts(index,na.array([2,3]))
        assert(na.all(self.cpt.counts[1,2,0,:] == na.array([4,4]))), \
              "Error adding array to counts"
        
if __name__ == '__main__':
    #suite = unittest.makeSuite(CPTIndexTestCase, 'test')
    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    from bayesnet import *
    
    G = BNet('Water Sprinkler Bayesian Network')
    
    c,s,r,w = [G.add_v(BVertex(nm,discrete=True,nvalues=nv)) for nm,nv in zip('c s r w'.split(),[2,3,4,5])]
    
    for ep in [(c,r), (c,s), (r,w), (s,w)]:
        G.add_e(graph.DirEdge(len(G.e), *ep))

    w.discrete = False
    r.discrete = False
    print G
    
    ds = s.SetDistribution(Multinomial_Distribution)
    ds.Normalize()
    print ds.cpt

    dw = w.SetDistribution(Gaussian_Distribution)
    print dw

    
