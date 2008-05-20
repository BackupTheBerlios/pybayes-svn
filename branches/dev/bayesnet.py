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

class BVertex(graph.Vertex):
    """
    This class implement a vertex of a Bayesian Network
    """
    def __init__(self, name, discrete=True, nvalues=2, observed=True):
        '''
        Name needn't be a string but must be hashable and immutable.
        if discrete = True:
            nvalues = number of possible values for variable contained \
                      in Vertex
        if discrete = False:
            nvalues is not relevant = 0
        observed = True means that this node CAN be observed
        '''
        graph.Vertex.__init__(self, name)
        self.distribution = None
        self.nvalues = int(nvalues)
        
        self.discrete = discrete
            # a continuous node can be scalar (self.nvalues=1)
            # or vectorial (self.nvalues=n)
            # n=2 is equivalent to 2D gaussian for example

        # True if variable can be observed
        self.observed = observed
        self.family = [self] + list(self.in_v)

    def init_distribution(self, *args, **kwargs):
        """ Initialise the distribution, all edges must be added"""
        # first decide which type of Distribution
        # if all nodes are discrete, then Multinomial)
        if numpy.alltrue([v.discrete for v in self.family]):
            # print self.name,'Multinomial'
            # FIX: should be able to pass through 'is_adjustable=True'
            # and it work
            self.distribution = distributions.MultinomialDistribution(self, 
                                                            *args, **kwargs) 
            return

        # gaussian distribution
        if not self.discrete:
            # print self.name,'Gaussian'
            self.distribution = distributions.GaussianDistribution(self, 
                                                            *args, **kwargs)
            return
        raise ValueError("Some distribution were not init") 
        # other cases go here
    
    def set_distribution_parameters(self, *args, **kwargs):
        """
        sets any parameters for the distribution of this node
        """
        self.distribution.set_parameters(*args, **kwargs)
        
    def __str__(self):
        if self.discrete:
            return graph.Vertex.__str__(self) + \
                   '    (discrete, %d)' %self.nvalues
        else:
            return graph.Vertex.__str__(self) + '    (continuous)'

    def get_sampling_distribution(self):
        """
        This is used for the MCMC engine
        returns a new distributions of the correct type, containing only
        the current without its family
        """
        if self.discrete:
            d = distributions.MultinomialDistribution(self, 
                                                      ignore_family = True)
        else:
            d = distributions.GaussianDistribution(self, ignore_family = True)
        
        return d
     
    # This function is necessary for correct Message Pass
    # we fix the order of variables, by using a cmp function
    def __cmp__(self, other):
        ''' sort by name, any other criterion can be used '''
        return cmp(self.name, other.name)

    def __hash__(self):
        """
        This function is used to hash the value (for putting in set).
        It must be defined to count only on the name to be coeherent with
        the previously defined __cmp__
        """
        return hash(self.name)


class BNet(graph.Graph):
    """
    This class implements a bayesian Network
    """
    def __init__(self, name=None):
        graph.Graph.__init__(self, name)

    def copy(self):
        ''' returns a deep copy of this BNet '''
        g_new = copy.deepcopy(self)
        g_new.init_distributions()
        for v in self.all_v:
            g_new.v[v.name].distribution.set_parameters(
                                           v.distribution.convert_to_cpt()) 
        return g_new

    def add_e(self, edge):
        """
        This add an edge to the Bayesian Net
        """
        if edge.__class__.__name__ == 'DirEdge':
            graph.Graph.add_e(self, edge)
            #e._v[1] = [e._v[1]] + [parent for parent in e._v[1].in_v]
            for v in edge.all_v:
                v.family = [v] + list(v.in_v)
        else:
            raise TypeError("All edges should be directed")

    def del_e(self, edge):
        """
        This method is used to remove an edge from the network
        """
        # remove the parent from the child node
        edge.all_v[1].family.pop(edge.all_v[1].family.index(edge.all_v[0]))
        graph.Graph.del_e(self, edge.name)
  
    def inv_e(self, edge):
        """
        This methode change the direction of the edge
        """
        self.e[edge].invert()
        # change the families of the corresponding nodes
        edge.all_v[0].family.append(edge.all_v[1])
        edge.all_v[1].family.pop(edge.all_v[1].family.index(edge.all_v[0]))
    
    def moralize(self):
        """
        moralize ??? the Network
        """
        tree = inference.MoralGraph(name='Moralized '+str(self.name))
        
        # for each vertice, create a corresponding vertice
        for v in self.v.values():
            tree.add_v(BVertex(v.name, v.discrete, v.nvalues))

        # create an UndirEdge for each DirEdge in current graph
        for e in self.e.values():
            # get corresponding vertices in G (and not in self!)
            v1 = tree.v[e.all_v[0].name]
            v2 = tree.v[e.all_v[1].name]
            tree.add_e(graph.UndirEdge(len(tree.e), v1, v2))

        # add moral edges
        # connect all pairs of parents for each node
        for v in self.v.values():
            # get parents for each vertex
            self.log.debug('Node : ' + str(v))
            parents = [tree.v[p.name] for p in list(v.in_v)]
            self.log.debug('parents: ' + str([p.name for p in list(v.in_v)]))
            
            for p1, i in zip(parents, range(len(parents))):
                for p2 in parents[i+1:]:
                    if not p1.connecting_e(p2):
                        self.log.debug('adding edge '+ str(p1) + 
                                       ' -- ' + str(p2))
                        tree.add_e(graph.UndirEdge(len(tree.e), p1, p2))

        return tree
    
    @graph._roprop('List of observed vertices.')
    def observed(self):
        """
        This method return the list of observed vertices
        """
        return [v for v in self.v.values() if v.observed]

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
    
    def init_distributions(self):
        """ Finalizes the network, all edges must be added. A 
        distribution (unknown) is added to each node of the network
        """
        #---TODO: test if DAG (fdebrouc)
        # this replaces the InitCPTs() function
        for v in self.v.values(): 
            v.init_distribution()
    
##    def InitCPTs(self):
##        for v in self.v.values(): v.InitCPT()

    def randomize_cpts(self):
        """
        This randomize the cpts
        """
        for v in self.v.values():
            v.rand()
            v.makecpt()
    
    def sample(self, nbr_samples=1):
        """ 
        Generate a sample of the network, n is the number of 
        samples to generate
        """
        assert(len(self.v) > 0)
        samples = []
        topological = self.topological_sort()
        for _ in xrange(nbr_samples):
            sample = {}
            for v in topological:
                assert(not v.distribution == None), \
                "vertex's distribution is not initialized"
                sample[v.name] = v.distribution.sample(sample)
            samples.append(sample)
        return samples

    def dimensions(self, node):
        ''' Computes the dimension of node
        = (nbr of state - 1)*nbr of state of the parents
        '''
        q = 1
        for pa in self.v[node.name].distribution.parents:
            q = q * self.v[pa.name].nvalues
        dim = (self.v[node.name].nvalues-1) * q
        return dim
