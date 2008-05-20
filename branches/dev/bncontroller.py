########################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta, Sebastien Arnaud
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
########################################################################
# encoding: utf-8

"""
BNController.py

Created by Sebastien Arnaud on 2006-11-20.
Copyright (c) 2006 All rights reserved.

See Examples/bncontroller.py for usage
"""

__all__ = ['BNController']
__version__ = "0.1"
__author__ = "Sebastien Arnaud"


from openbayes import BNet, BVertex, DirEdge 
from openbayes import learning, MCMCEngine, LoadXBN, SaveXBN


class BNController(object):
    """
    This is used to wrap around a Bayesian network
    """
    def __init__(self, name=None, nodes_def=None):
        self._network = None
        self._nodes_def = None
        if nodes_def is not None:
            self._init_network(name, nodes_def)

    def _init_network(self, name, nodes_def):
        """
        This responsible for the initilisation of the network
        """
        nodes = {}
        connections = []
        network = BNet(name)  

        # Save the nodesdef for future use (saving)
        self._nodes_def = nodes_def

        for (nodename, isdiscrete, numstates, leafnode)  in nodes_def:
            nodes[nodename] = network.add_v(BVertex(nodename, isdiscrete, 
                                                 numstates))

        # TODO: Find a way to improve this and avoid to have to loop a
        # second time 
        # reason is because BNnodes[leafnode] is not sure to be there when
        # going through the first loop

        for (nodename, isdiscrete, numstates, leafnode) in nodes_def:
            if type(leafnode) == type([]):
                for r in leafnode:
                    if r != None:
                        connections.append((nodes[nodename], nodes[r]))
            elif leafnode != None:
                connections.append((nodes[nodename], nodes[leafnode]))
            else:
                #do nothing 
                pass

        for ep in connections:
            network.add_e(DirEdge(len(network.e), *ep))

        # Ok our Bnet has been created, let's save it in the controller
        self._network = network

        # Let's not forget to initialize the distribution
        self._network.init_distributions()

    def show_graph(self):
        """
        This method print the network to stdout
        """
        print self._network

    def show_distribution(self):
        """
        This method print all the distributions in the 
        network to stdout
        """
        for v in self._network.all_v: 
            print v.distribution, '\n'

    def load(self, filename):
        """
        this load the network from a XBN file
        """
        xbnparser = LoadXBN(filename)
        self._network = xbnparser.Load()

    def save(self, filename):
        """
        this save the network to a XBN file
        """
        SaveXBN(filename, self._network)

    def train(self, cases):
        """
        This train the network using the MLLearningEngine
        """
        engine = learning.MLLearningEngine(self._network)
        engine.learn_ml_params(cases)

    def eval(self, evidences, node):
        """
        This use MCMC engine to compute the marginal on a given node
        based on the given evidence
        """
        #TODO: node could be a list if more than 1 node needs to be guessed
        ie = MCMCEngine(self._network)

        for e in evidences:
            ie.set_obs(e)
        return ie.marginalise(node).convert_to_cpt()
