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
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
###############################################################################
import types
import random

import numpy

from table import Table

__all__ = ['MultinomialDistribution', 'GaussianDistribution']

# object gives access to __new__
class Distribution(object):
    """
    Base Distribution Class for all types of distributions
    defines the Pr(x|Pa(x))

    variables :
    -------------
        - vertex =     a reference to the BVertex instance containing the
                        variable quantified in this distribution
                        
        - family =		[x, Pa(x)1, Pa(x)2,...,Pa(x)N)]
                        references to the nodes

        - names =      set(name of x, name of Pa(x)1,..., name of Pa(x)N)
                        set of strings : the names of the nodes
                        this is a set, no order is specified!

        - names_list =	[name of x, name of Pa(x)1,..., name of Pa(x)N]
                        list of strings : the names of the nodes
                        this is a list, order is specified!

        - parents =    [name of Pa(x)1,...,name of Pa(x)N]
                        list of strings : the names of the node's parents
                        this is a list, order is the same as family[1:]

        - ndimensions = number of variables contained in this distribution
                        = len(self.family)

        - distribution_type = a string the type of the distribution
                              e.g. 'Multinomial', 'Gaussian', ...

        - is_adjustable = if True: the parameters of this distribution can
                         be learned.

        - nvalues =		the dimension of the distribution
                        discrete : corresponds to number of states
                        continuous : corresponds to number of dimensions
                        (e.g. 2D gaussian,...)

    """

    def __init__(self, vertex, is_adjustable=False, ignore_family=False):
        """ Creates a new distribution for the given variable.
        v is a BVertex instance
        """
        ###################################################
        #---TODO: Should give an order to the variables, sort them by name 
        # for example...
        ###################################################
        self.vertex = vertex		# the node to which this distribution is attached
        if not ignore_family:
            self.family = [vertex] + [parent for parent in vertex.in_v]
        else:
            # ignore the family of the node, simply return a distribution for 
            # this node only used in the MCMC inference engine to create empty
            # distributions
            self.family = [vertex]

        self.ndimensions = len(self.family)
        self.parents = self.family[1:]
        self.names_list = [vertex.name for vertex in self.family]
        #self.names = set(self.names_list)
        self.nvalues = self.vertex.nvalues
        # the type of distribution
        self.distribution_type = 'None'
        #used for learning
        self.is_adjustable = is_adjustable

    def __str__(self):
        list_ = ['Distribution for node : '+ self.vertex.name]
        list_.append('Type : ' + self.distribution_type)
        if len(self.names_list)>1 : 
            list_.append('Parents : ' + str(self.names_list[1:]))
        return '\n'.join(list_)
        # TODO Implements a nice representation, with
        # compatibility down the tree

    #==================================================
    #=== Learning & Sampling Functions
    def initialize_counts(self):
        """ Initialize the counts """
        raise NotImplementedError
    
    def incr_counts(self, index):
        """ add one to given count """
        raise NotImplementedError

    def add_to_counts(self, index, counts):
        """ add counts to count """
        raise NotImplementedError

    def set_counts(self):
        """ set the distributions underlying cpt equal to the counts """
        raise NotImplementedError


class MultinomialDistribution(Distribution, Table):
    """		Multinomial/Discrete Distribution
    All nodes involved all discrete --> the distribution is represented by
    a Conditional Probability Table (CPT)
    This class now inherits from Distribution and Table.
    """
    # TODO: Do a lot of cleaning. This is An extended CPT table
    # Need to reflect on the class hierarchy
    def __init__(self, vertex, cpt=None, is_adjustable=True, 
                 ignore_family=False):
        Distribution.__init__(self, vertex, is_adjustable=is_adjustable, \
                              ignore_family=ignore_family)
        self.distribution_type = "Multinomial"

        assert(numpy.alltrue([vertex.discrete for vertex in self.family])), \
              'All nodes in family ' + str(self.names_list) + \
              ' must be discrete !!!'

        self.sizes = [vertex.nvalues for vertex in self.family]

        # initialize the cpt
        Table.__init__(self, self.names_list, self.sizes, cpt)

        #Used for Learning
        self.counts = None
        self.augmented = None

    def set_parameters(self, *args, **kwargs):
        ''' put values into self.cpt, delegated to Table class'''
        Table.set_values(self, *args, **kwargs)

    def convert_to_cpt(self):
        """ Return the cpt"""
        return self.cpt

    #======================================================
    #=== Operations on CPT
    def normalize(self, dim=-1):
        """ If dim=-1 all elements sum to 1.  Otherwise sum to specific 
        dimension, such that sum(Pr(x=i|Pa(x))) = 1 for all values of i 
        and a specific set of values for Pa(x)
        """
        if dim == -1 or len(self.cpt.shape) == 1:
            self.cpt /= self.cpt.sum()			  
        else:
            ndim = self.assocdim[dim]
            order = range(len(self.names_list))
            order[0] = ndim
            order[ndim] = 0
            tcpt = numpy.transpose(self.cpt, order)
            t1cpt = numpy.sum(tcpt, axis=0)
            t1cpt = numpy.resize(t1cpt, tcpt.shape)
            tcpt = tcpt/t1cpt
            self.cpt = numpy.transpose(tcpt, order)

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
    #			all ones, uniform, zeros
    #			gaussian, ...
    ######################################################

    #======================================================
    #=== Sampling
    def sample(self, index=None):
        """ returns the index of the sampled value
        eg. a=Pr(A)=[0.5 0.3 0.0 0.2]
            a.sample() -->	5/10 times will return 0
                            3/10 times will return 1
                            2/10 times will return 3
                            2 will never be returned

            - returns an integer
            - only works for one variable tables
              eg. a=Pr(A,B); a.sample() --> ERROR
        """
        if index is None:
            index = {}
        assert(len(self.names) == 1 or \
               len(self.names - set(index.keys())) == 1), \
               "Sample only works for one variable tables"
        if not index == {}:
            tcpt = self.__getitem__(index)
        else:
            tcpt = self.cpt
        # csum is the cumulative sum of the distribution
        # csum[i] = numpy.sum(self.cpt[0:i])
        # csum[-1] = numpy.sum(self.cpt)
        csum = [numpy.sum(tcpt.flat[0:end+1]) for end in range(tcpt.shape[0])]

        # sample in this distribution
        r = random.random()
        for i, cs in enumerate(csum):
            if r < cs: 
                return i
        return i

    def random(self):
        """ Returns a random state of this distribution, 
        chosen completely at random, it does not take account of the 
        underlying distribution 
        """

        # CHECK: legal values are 0 - nvalues-1: checked OK
        return random.randint(0, self.nvalues-1)

    #==================================================
    #=== Learning & Sampling Functions
    def initialize_counts(self):
        ''' initialize counts array to zeros '''
        self.counts = Table(self.names_list, shape=self.shape)
        self.counts.zeros()

    def initialize_counts_to_ones(self):
        ''' initialize counts array to ones '''
        self.counts = Table(self.names_list, shape=self.shape)
        self.counts.ones()

    def incr_counts(self, index):
        """ add one to given count """
        self.counts[index] += 1

    def add_to_counts(self, index, counts):
        """ add counts to the given count """
        self.counts[index] += counts

    def set_counts_to(self, index, counts):
        """ set counts to a given value """
        self.counts[index] = counts

    def set_counts(self):
        """ set the distributions underlying cpt equal to the counts """
        assert(self.names_list == self.counts.names_list)
        #set to copy in case we later destroy the counts or reinitialize them
        self.cpt = self.counts.cpt.copy()

    #=== Augmented Bayesian Network
    # The augmented bayesain parameters are used to enforce the 
    # direction in which the CPT's evolve when learning parameters
    #
    # They can also be set to equivalent chances if we do not want to 
    # enforce a CPT, in this case this is useful because it prohibits 
    # eg. the EM-algorithm to output 'nan' when a particular case 
    # doesn't exist in the given data (which cases the counts for that 
    # case to be zero everywhere which causes normalize() to divide by 
    # zero which gives 'nan')
    #
    # More information can be found in "Learning Bayesian Networks" 
    # by Richard E. Neapolitan
    
    def initialize_augmented_eq(self, eqsamplesize=1):
        '''initialize augmented parameters based on the equivalent array size.
        This can be used to give some prior information before learning.
        '''
        ri = self.nvalues
        qi = 1
        for parent in self.family[1:]:
            qi = qi * parent.distribution.nvalues
        self.augmented = Table(self.names_list, shape=self.shape)
        self.augmented[:] = float(eqsamplesize) / (ri * qi)

    def set_augmented(self, index, value):
        '''set the augmented value on position index to value'''
        self.augmented[index] = value

    def set_augmented_and_counts(self):
        ''' set the distributions underlying cpt equal to the 
        counts + the parameters of the augmented network
        '''
        assert(self.names_list == self.counts.names_list)
        # set to copy in case we later destroy the counts or 
        # reinitialize them
        if self.augmented == None:
            # if no augmented parameters are used, only use the counts
            cpt = self.counts.cpt
        else:
            cpt = self.counts.cpt + self.augmented.cpt
        self.cpt = cpt.copy()

    #===================================================
    #=== printing functions
    def __str__(self):
        list_ = [Distribution.__str__(self)]
        list_.append('Conditional Probability Table (CPT) :')
        #---TODO: should find a nice neat way to represent numarrays
        #		  only 3 decimals are sufficient... any ideas?
        list_.append(str(self.cpt))
        return "\n".join(list_)


#=================================================================
#=================================================================
class GaussianDistribution(Distribution):
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
         - no parents: Y ~ N(mu(i), Sigma(i,j))		0 <= i,j < self.nvalues
         - cts parents : Y|X=x ~ N(mu + W x, Sigma)
         - discrete parents: Y|Q=i ~ N(mu(i), Sigma(i))
         - cts and discrete parents: Y|X=x,Q=i ~ N(mu(i) + W(i) x, Sigma(i))


     The list below gives optional arguments [default value in brackets].

     mean		- numarray(shape=(len(Y),len(Q1),len(Q2),...len(Qn))
                  the mean for each combination of DISCRETE parents
                  mean[i1,i2,...,in]

     sigma		  - Sigma[:,:,i] is the sigmaariance given 
                    Q=i [ repmat(100*eye(Y,Y), [1 1 Q]) ]
     weights	  - W[:,:,i] is the regression matrix given 
                    Q=i [ randn(Y,X,Q) ]
     sigma_type	  - if 'diag', Sigma[:,:,i] is diagonal [ 'full' ]
     tied_sigma	  - if True, we constrain Sigma[:,:,i] to be the same 
                    for all i [False]
     """

    #---TODO: Maybe we should add a domain variable...
    #---TODO: ADD 'set attribute' for private variables mu, sigma, 
    #         weights: they muist always be a numarray!!!!
    def __init__(self, vertex, mu = None, sigma = None, wi = None, \
                   sigma_type = 'full', tied_sigma = False, \
                   is_adjustable = True, ignore_family = False):

        Distribution.__init__(self, vertex, is_adjustable=is_adjustable, \
                              ignore_family=ignore_family)
        self.distribution_type = 'Gaussian'
        self.samples = None

        # check that current node is continuous
        if vertex.discrete:
            raise TypeError('Node must be continuous')

        self.discrete_parents = [parent for parent in self.parents \
                                 if parent.discrete]
        self.continuous_parents = [parent for parent in self.parents \
                                   if not parent.discrete]

        self.discrete_parents_shape = [dp.nvalues for dp \
                                       in self.discrete_parents]
        self.parents_shape = [p.nvalues for p in self.parents]
        if not self.parents_shape:
            self.parents_shape = [0]

        # set defaults
        # set all mu to zeros
        self.mean = numpy.zeros(shape=([self.nvalues] + \
                             self.discrete_parents_shape), dtype='Float32')

        # set sigma to ones along the diagonal	
        eye = numpy.identity(self.nvalues, dtype = 'Float32')[..., numpy.newaxis]
        if len(self.discrete_parents) > 0:            
            # number of different configurations for the parents
            q = reduce(lambda a, b:a * b, self.discrete_parents_shape) 
            sigma = numpy.concatenate([eye] * q, axis=2)
            self.sigma = numpy.array(sigma).reshape([self.nvalues, self.nvalues] + \
                                  self.discrete_parents_shape) 

        # set weights to 
        self.weights = numpy.ones(shape=[self.nvalues] + self.parents_shape, 
                               dtype='Float32')

        # set the parameters : mean, sigma, weights
        self.set_parameters(mu=mu, sigma=sigma, wi=wi, sigma_type=sigma_type, \
                           tied_sigma=tied_sigma, is_adjustable=is_adjustable)

        #---TODO: add support for sigma_type, tied_sigma
        #---TODO: add learning functions
    
    def set_parameters(self, mu=None, sigma=None, wi=None, sigma_type='full', \
                         tied_sigma=False, is_adjustable=False):
        """ Set the distribution parameters

        This method set the difeerent parameters of the distribution
        """
        #============================================================
        # set the mean :
        # self.mean[i] = the mean for dimension i
        # self.mean.shape = (self.nvalues, q1,q2,...,qn)
        #		 where qi is the size of discrete parent i
        new_shape = [self.nvalues] + self.discrete_parents_shape
        if mu is None:
            mu = numpy.zeros(new_shape)
        self.mean = numpy.array(mu, dtype='Float32').reshape(new_shape)

        #============================================================
        # set the covariance :
        # self.sigma[i,j] = the covariance between dimension i and j
        # self.sigma.shape = (nvalues,nvalues,q1,q2,...,qn)
        #		 where qi is the size of discrete parent i
        new_shape = [self.nvalues, self.nvalues] + self.discrete_parents_shape
        if sigma is None:
            sigma = numpy.zeros(new_shape)
        self.sigma = numpy.array(sigma,  dtype='Float32').reshape(new_shape)
        #============================================================
        # set the weights :
        # self.weights[i,j] = the regression for dimension i 
        #                     and continuous parent j
        # self.weights.shape = (nvalues,x1,x2,...,xn,q1,q2,...,qn)
        #		 where xi is the size of continuous parent i)
        #		 and qi is the size of discrete parent i
        new_shape = [self.nvalues] + self.parents_shape
        if wi is None:
            wi = numpy.zeros(new_shape)
        self.weights = numpy.array(wi,  dtype='Float32').reshape(new_shape)
        
    def normalize(self):
        """ do nothing """
        pass

    #=================================================================
    # Indexing Functions
    def __getitem__(self, index):
        """
        similar indexing to the Table class
        index can be a number, a slice instance, or a dict ,

        returns a tuple (mean, variance, weights)
        """

        if isinstance(index, types.DictType):
            d_index, c_index = self._num_index_from_dict(index)
        else:
            raise ValueError("Type of index not supported")
#        elif isinstance(index, types.TupleType):
#	    	 numIndex = list(index)		
#        else:
#			  numIndex = [index]
        
        return tuple([self.mean[tuple([slice(None, None, None)] + d_index)], \
                      self.sigma[tuple([slice(None, None, None)] * 2 + d_index)],
                      self.weights[tuple([slice(None, None, None)] + c_index)]])

    def __setitem__(self, index, value):
        """ Overload array-style indexing behaviour.
        Index can be a dictionary of var name:value pairs, 
        or pure numbers as in the standard way
        of accessing a numarray array array[1,:,1]

        value must be a dict with keys ('mean', 'variance' or 'weights')
        and values the corresponding values to be introduced
        """

        if isinstance(index, types.DictType):
            d_index, c_index = self._num_index_from_dict(index)
        else: 
            raise TypeError("not supported...")

#		 elif isinstance(index, types.TupleType):
#			 numIndex = list(index)		
#		 else:
#			 numIndex = [index]

        if value.has_key('mean'):
            self.mean[tuple([slice(None, None, None)] + d_index)] = \
                      value['mean']
        if value.has_key('sigma'):
            self.sigma[tuple([slice(None, None, None)] * 2 + d_index)] = \
                      value['sigma']	 
        if value.has_key('weights'):
            self.weights[tuple([slice(None, None, None)] + c_index)] = \
                      value['weights']

    def _num_index_from_dict(self, distr):
        """
        This is supposed to return indexes into the underlaying
        table. It returns a tuple for discrete and continuous parents
        """
        # first treat the discrete parents
        d_index = []
        for dp in self.discrete_parents:
            if distr.has_key(dp.name):
                d_index.append(distr[dp.name])	  
            else:
                d_index.append(slice(None, None, None))

        # now treat the continuous parents
        c_index = []
        for dp in self.continuous_parents:
            if distr.has_key(dp.name):
                c_index.append(distr[dp.name])	  
            else:
                c_index.append(slice(None, None, None))

        return (d_index, c_index)

    #======================================================
    #=== Sampling
    def sample(self, index=None):
        """ 
        in order to sample from this distributions, all parents must be 
        known 
        """
#		 mean = self.mean.copy()
#		 sigma = self.sigma.copy()
##		  if index:
##			  # discrete parents
##			  for v,i in enumerate(reversed(self.discrete_parents)):
##			  # reverse: avoid missing axes when taking in random
##			  # we start from the end, that way all other dimensions keep 
##            # the same index
##				  if index.has_key(v.name): 
##					  # take the corresponding mean; +1 because first axis is the mean
##					  mean = numpy.take(mean, index[v], axis=(i+1) )
##					  # take the corresponding covariance; +2 because first 2 
##                    # axes are the cov
##					  sigma = numpy.take(sigma, index[v], axis=(i+2) )
##			  
##			  # continuous parents
##			  for v in reversed(self.continuous_parents):
##				  if index.has_key(v):
        if index is None:
            index = {}
        d_index, c_index = self._num_index_from_dict(index)
        mean  = numpy.array(self.mean[tuple([slice(None, None, None)] + d_index)])
        sigma = self.sigma[tuple([slice(None, None, None)] * 2 +d_index)]
        wi = numpy.sum(self.weights * numpy.array(c_index)[numpy.newaxis, ...], axis=1)

#		 if self.continuous_parents:
#			 wi = numpy.array(self.weights[tuple([slice(None,None,None)]+c_index)])
#		 else: wi = 0.0

        # return a random number from a normal multivariate distribution
        return float(numpy.random.multivariate_normal(mean + wi, sigma))

    def random(self):
        """
        Returns a random state of this distribution using a uniform 
        distribution
        """
        # legal values are from -inf to + inf
        # we restrain to mu-5*s --> mu+5*s
        return [(5 * sigma * (random.random() - 0.5) + mu) for mu, sigma in \
                 zip(self.mean, self.sigma.diagonal())]

    #==================================================
    #=== Learning & Sampling Functions
    def initialize_counts(self):
        ''' initialize counts array to empty '''
        self.samples = list()
        # this list will contain all the sampled values

    def incr_counts(self, index):
        """ add the value to list of counts """
        if index.__class__.__name__ == 'list':
            self.samples.extend(index)
        elif index.__class__.__name__ == 'dict':
            # for the moment only take index of main variable
            # ignore the value of the parents, 
            # the value of the parents is not necessary for MCMC but it is very
            # important for learning!
            #TODO: Add support for values of the parents
            self.samples.append(index[self.names_list[0]])
        else:
            self.samples.append(index)

    def add_to_counts(self, index, counts):
        raise TypeError("What's the meaning of add_to_counts for a"
                        " gaussian distribution ???")

    def set_counts(self):
        """ set the distributions underlying parameters (mu, sigma) 
        to match the samples 
        """
        assert(self.samples), "No samples given..."

        samples = numpy.array(self.samples, dtype='Float32')

        self.mean = numpy.sum(samples) / len(samples)

        deviation = samples - self.mean
        squared_deviation = deviation * deviation
        sum_squared_deviation = numpy.sum(squared_deviation)

        self.sigma = (sum_squared_deviation / (len(samples)-1.0)) ** 0.5

    def __str__(self):
        """
        This function is used to print an object. Every new distribution
        should first print the info given by Distribution.__str__. Then
        new information can be appended. 

        It is better to construct a list a then join the list (for
        people using other implementation than CPython
        """
        list_ =[Distribution.__str__(self)]
        list_.append('Dimensionality : ' + str(self.nvalues))
        list_.append('Discrete Parents :' + \
                   str([p.name for p in self.discrete_parents]))
        list_.append('Continuous Parents :' + \
                  str([p.name for p in self.continuous_parents]))
        list_.append('Mu : ' + str(self.mean))
        list_.append('Sigma : ' + str(self.sigma))
        list_.append('Weights: ' + str(self.weights))
        return "\n".join(list_)
	

