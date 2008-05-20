#!/usr/bin/env python
"""
This module is the graph module. It implementation is influenced by
networkx. However, only a subset of the features of networkx are implemented.
Bayesian network are usually quite simple graph. 

We also restrict us to the directed case.
"""
# Copyright (C) 2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from openbayes import __version__, authors

__author__ = authors['Salamin']

class GraphError(StandardError):
    """
    This exception is raised in case of error in graph manipulating
    method
    """
    pass

class Graph(object):
    """
    This class implements a directed graph. The only constraints is that
    vertices are hashable
    """
    def __init__(self):
        # This dictionary contains all the predecesor of a vertices.
        # They are stored as ordered list
        self._pred = {}
        # This dictionary contains all the successor of a vertices.
        # They are also stored as an ordered list
        self._succ = {}

    def add_vertice(self, vertice):
        """
        This method add a vertice to the graph. It hte vertice already exist,
        an GraphError is raised
        """
        if vertice in self._pred or vertice in self._succ:
            raise GraphError("Vertices already exists")
        self.pred[vertice] = []
        self._succ[vertice] = []

    def add_edge(self, edge):
        # We start by updating the self._prec
        if edge[1] not in self._pred:
            self._pred[edge[1]] = [edge[0]]
        else:
            self._pred[edge[1]].append(edge[0])
        if edge[0] not in self._pred:
            self._pred[edge[0]] = []
        # We then update self._succ
        if edge[0] not in self._succ:
            self._succ[edge[0]] = [edge[1]]
        else:
            self._succ[edge[0]].append(edge[1])
        if edge[1] not in self._succ:
            self._succ[edge[1]] = []
        # The invariant that
        #   self._prec.keys() == self._succ.keys() 
        # remains valid

    def __getitem__(self, vertice):
        return self._pred[vertice].key()+self._succ[vertice].key()

    def nodes(self):
        return self._succ.keys()

    def edges(self):
        """
        This function return a list of edges
        """
        ans = []
        for start, succ in self._succ.iteritems():
            for end in succ:
                ans.append((start, end))
        return ans

    def succesors(self, vertice):
        return self._succ[vertice]

    def predecessors(self, vertice):
        return self._pred[vertice]

    def __str__(self):
        ans = ["Nodes:"+ str(self.nodes())]
        ans.append("Edges:"+ str(self.edges()))
        return '\n'.join(ans)

if __name__ == "__main__":
    g = Graph()
    g.add_edge( (1,2))
    g.add_edge((2,3))
    g.add_edge((3,1))
    print g
    print __author__
