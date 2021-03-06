"""
Bayesian network implementation.  Influenced by Cecil Huang's and 
Adnan Darwiche's "Inference in Belief Networks: A Procedural Guide," 
International Journal of Approximate Reasoning, 1994.
"""
# Copyright (C) 2005-2008 by
# Elliot Cohen <elliot.cohen@gmail.com>
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

import copy

import numpy

from openbayes import __version__, authors
import openbayes.graph as graph
import openbayes.distributions as distributions
import openbayes.inference as inference

__all__ = ['BVertex', 'BNet']
__author__ = authors['Cohen'] + '\n' + authors['Gaitanis']

class BVertex(object):
    """
    This class implement a vertex of a Bayesian Network. Please note that
    self.name == self so that it is possible to retrieve value in dict and 
    set with either the name or the original object (or any object with the
    same name)

    This class should never be used directly. Instead, using the subclass
    in vertex is the recommended way.
    """

    def __init__(self, name, observed=True):
        '''
        name needn't be a string but must be hashable and should be immutable.
        observed = True means that this node CAN be observed
        '''
        self.name = name
        self.observed = observed
        
    def set_parents(self, parents):
        """
        This function get called when the graph construction is finished. The list
        of parents should be enough for every distribution to be initialiased
        """
        raise NotImplementedError

    def sample(self, parents):
        """
        This function should return a random value of the node based on the value of
        the parents. Parents is a dictionnary of value (that may contains value for
        node that are not parent) that is guarantee to contains one value for every
        parents
        """
        raise NotImplementedError


    def set_distribution_parameters(self, *args, **kwargs):
        """
        sets any parameters for the distribution of this node
        TODO: replace this proxy by actual ineritance
        """
        raise NotImplementedError
        
        
    def get_sampling_distribution(self):
        """
        This is used for the MCMC engine
        returns a new distributions of the correct type, containing only
        the current without its family
        """
        #TODO: refactor in distribution
        if self.discrete:
            d = distributions.MultinomialDistribution(self, 
                                                      ignore_family = True)
        else:
            d = distributions.GaussianDistribution(self, ignore_family = True)
        
        return d
     
    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function. Furthermore,
    # by implementing __eq__ and __hash__, we can use BVertex as
    # key. The name attribute is equal to the vertex in question
    # This is black magic and should maybe be removed

    def __cmp__(self, other):
        ''' sort by name, any other criterion can be used '''
        if hasattr(other, 'name'):
            return cmp(self.name, other.name)
        return cmp(self.name, other)

    def __eq__(self, other):
        '''
        Test wether to object are equal. This is used for retrieving
        graph element by name
        '''
        if hasattr(other, 'name'):
            return self.name == other.name
        return self.name == other

    def __neq__(self, other):
        '''
        A simple fall back for comparing if two nodes are not equal
        '''
        return not self.__eq__(other)

    def __hash__(self):
        """
        This function is used to hash the value (for putting in a graph).
        It must be defined to count only on the name to be coeherent with
        the previously defined __cmp__
        """
        return hash(self.name)


    def __str__(self):
        if self.discrete:
            return str(self.name) + \
                   '    (discrete, %d)' %self.nvalues
        else:
            return str(self) + '    (continuous)'




class BNet(graph.DirectedGraph):
    """
    This class implements a bayesian Network
    """
    def __init__(self, name=None):
        graph.DirectedGraph.__init__(self)
        self.name = name

    def copy(self):
        ''' 
        returns a deep copy of this BNet 
        '''
        g_new = copy.deepcopy(self)
        g_new.finalize()
        for v in self.vertices():
            g_new[v.name].set_parameters(v.get_parameters()) 
        return g_new
        
    def observed(self):
        """
        This method return the list of observed vertices. Every vertex in a
        BNet must have an observed attribute
        """
        return [v for v in self.vertices if v.observed]

    def split_into_components(self):
        """ 
        returns a list of BNets with the connected components of this BNet
        """
        components = self.connex_components()

        b_nets = []
        i = 0
        # create a BNet for each element in components
        for comp in components:
            new = BNet(self.name + ' (' + str(i+1) + '/' + \
                       str(len(components)) + ')')
            b_nets.append(new)
            
            #add vertices to this new BNet
            for v in comp:
                new.add_v(v)
            #add all edges into this BNet
            for v in comp:
                for e in v.out_e:
                    new.add_e(e)
            i += 1
        return b_nets
    
    def finalize(self):
        """ Finalizes the network, all edges must be added. A 
        distribution (unknown) is added to each node of the network
        """
        if not self.is_dag():
            raise graph.GraphError("Not acyclic graph not supported")
        for v in self.vertices: 
            v.set_parents(self.predecessors(v)) 
    
    def sample(self, nbr_samples=1):
        """ 
        Generate a sample of the network, n is the number of 
        samples to generate
        """
        samples = []
        topological = self.topological_order()
        for _ in xrange(nbr_samples):
            sample = {}
            for v in topological:
                sample[v.name] = v.sample(sample)
            samples.append(sample)
        return samples

    def dimensions(self, node):
        ''' Computes the dimension of node
        = (nbr of state - 1)*nbr of state of the parents
        #TODO what for
        '''
        q = 1
        for pa in self.v[node.name].distribution.parents:
            q = q * self.v[pa.name].nvalues
        dim = (self.v[node.name].nvalues-1) * q
        return dim
