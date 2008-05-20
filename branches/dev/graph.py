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

    def add_vertex(self, vertex):
        """
        This method add a vertice to the graph. It hte vertice already exist,
        an GraphError is raised
        """
        if vertex in self._pred or vertex in self._succ:
            raise GraphError("Vertices already exists")
        self._pred[vertex] = []
        self._succ[vertex] = []
        return vertex

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

    def del_edge(self, edge):
        """
        This simply remove an edge from the graph. The set of vertice is 
        untouched. return false if no edge was removed, else return true
        """
        if edge[0] in self._succ:
            if edge[1] in self._succ[edge[0]]:
                self._succ[edge[0]].remove(edge[1])
                self._pred[edge[1]].remove(edge[0])
                return True
        return False

    def inv_edge(self, edge):
        """
        This method inverse an edge in the graph. If the edge is not
        present, then we raise an exception
        """
        if self.del_edge(edge):
            self.add_edge( (edge[1], edge[0]))
        else:
            raise GraphError("Impossible to invert inexistant edge %s"%str(edge))

    def __getitem__(self, vertex):
        return self._pred[vertex].key()+self._succ[vertex].key()

    def vertices(self):
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

    def successors(self, vertex):
        return self._succ[vertex]

    def predecessors(self, vertex):
        return self._pred[vertex]

    def family(self, vertex):
        return [vertex] + self._pred[vertex]

    def topological_order(self):
        """
        This function return a list of vertex in topological order. If the graph is
        not DAG, then the empty list is returned
        """
        ans = []
        choices = [x for x in self._pred if len(self._pred[x]) == 0]
        while choices:
            v = choices.pop()
            ans.append(v)
            for candidate in self._succ[v]:
                if len([i for i in self._pred[candidate] if i not in ans]) == 0:
                    choices.append(candidate)
        if len(ans) == len(self._succ):
            return ans
        return False

    def is_dag(self):
        """
        This function try to do a topolgical order sort and
        then return True if the graph is dag, False else.
        """
        if self.topological_order():
            return True
        return False

    def __str__(self):
        ans = ["Nodes:"+ str(self.vertices())]
        ans.append("Edges:"+ str(self.edges()))
        return '\n'.join(ans)
