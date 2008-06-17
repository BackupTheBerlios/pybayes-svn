"""
BNController.py

Created by Sebastien Arnaud on 2006-11-20.

See Examples/bncontroller.py for usage

[Salamin] This is a very good idea to have around. May need more
work to get working. Waiting until the structure is finalized
"""
# Copyright (C) 2005-2008 by
# Sebastien Arnaud
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from openbayes import BNet, BVertex, __version__, authors
from openbayes import learning, MCMCEngine

__all__ = ['BNController']
__author__ = authors['Arnaud']



class BNController(object):
    """
    This is used to wrap around a Bayesian network
    """
    def __init__(self, name=None, nodes_def=None):
        self._network = None
        if nodes_def is not None:
            self._init_network(name, nodes_def)

    def _init_network(self, name, nodes_def):
        """
        This responsible for the initilisation of the network
        """
        nodes = {}
        connections = []
        network = BNet(name)   
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
            network.add_e(ep)
        self._network = network
        # Let's not forget to initialize the distribution
        self._network.finalize()

    def __str__(self):
        ans = [str(self._network)]
        for c in self._network.vertices:
            ans.append(str(c))
        return "\n".join(ans)

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
