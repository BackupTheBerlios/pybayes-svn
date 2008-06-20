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
from copy import deepcopy, copy

from openbayes import __version__, authors

__author__ = authors['Salamin']

class GraphError(StandardError):
    """
    This exception is raised in case of error in graph manipulating
    method
    """
    pass

class DirectedGraph(object):
    """
    This class implements a directed graph. The only constraints is that
    vertices are hashable

    The parents and children of a nodes are ordered
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
        This method add a vertice to the graph. It the vertice already exist,
        nothing is done
        """
        self._pred.setdefault(vertex, [])
        self._succ.setdefault(vertex, [])
        return vertex

    def add_edge(self, edge):
        """
        This method can be used to add an edge in the
        graph. The needed vertices are inserted
        """
        self._succ.setdefault(edge[0], []).append(edge[1])
        self._succ.setdefault(edge[1], [])
        self._pred.setdefault(edge[1], []).append(edge[0])
        self._pred.setdefault(edge[0], [])
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
            raise GraphError("Impossible to invert inexistant edge %s" 
                             % str(edge))

    def copy(self):
        ans = DirectedGraph()
        ans._succ = copy(self._succ)
        ans._pred = copy(self._pred)

    def __getitem__(self, vertex):
        """
        This function return all the neighbour of a given
        vertex
        """
        return self._pred[vertex].key()+self._succ[vertex].key()

    def get_vertices(self):
        """
        This return the list of vertices present in the graph
        """
        return set(self._succ.keys())
    vertices = property(get_vertices)

    def get_edges(self):
        """
        This function return a list of edges
        """
        ans = []
        for start, succ in self._succ.iteritems():
            for end in succ:
                ans.append((start, end))
        return ans
    edges = property(get_edges)

    def successors(self, vertex):
        """
        This return all the direct successor of vertex
        """
        return self._succ[vertex]

    def predecessors(self, vertex):
        """
        This return the parent of vertex
        """
        return self._pred[vertex]

    def family(self, vertex):
        """
        This return the vertex and its parents
        """
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
                if len([i for i in self._pred[candidate] 
                          if i not in ans]) == 0:
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

    def get_moral(self):
        """
        This method return a new undirected graph that is the moral graph of 
        self. The moral graph is by definition obtained by connecting all
        the parent of the node.

        Every node is copied, so that modifying the moral graph does
        not modify the original graph
        """
        ans = UndirectedGraph()
        # we make a copy of the vertices. We store only one copy
        # for every vertices (we dont want multiple copy around)
        vertices = dict([(x, deepcopy(x)) for x in self.vertices])
        for node, parents in self._pred.iteritems():
            for i, parent in enumerate(parents):
                # we add the edge between parent and child
                ans.add_edge((vertices[node], vertices[parent]))
                # we fully connect the parents pair-wise
                for p_2 in parents[i+1:]:
                    ans.add_edge((vertices[parent], vertices[p_2]))
        return ans


    def __str__(self):
        ans = ["Nodes:"+ str(self.vertices)]
        ans.append("Edges:"+ str(self.edges))
        return '\n'.join(ans)

    def __cmp__(self, other):
        return self._succ.__cmp__(other._succ)


class UndirectedGraph(object):
    """
    This class implements a directed graph. The only constraints is that
    vertices are hashable.

    The neighbors are unordered
    """

    def __init__(self):
        self._neighbors = {}

    def add_vertex(self, vertex):
        """
        This method add a vertice to the graph. It the vertice already exist,
        nothing is done
        """
        self._neighbors.setdefault(vertex, set())
        return vertex

    def del_vertex(self, vertex):
        """
        This method remove a vertex and all the adjacent edge from a graph
        """
        # we remove adjacent edges
        for n in self[vertex]:
            self[n].remove(vertex)
        del self._neighbors[vertex]

    def add_edge(self, edge):
        """
        This method add an edge in the graph and the corresponding vertex
        if needed
        """
        self._neighbors.setdefault(edge[0], set()).add(edge[1])
        self._neighbors.setdefault(edge[1], set()).add(edge[0])

    def del_edge(self, edge):
        """
        This simply remove an edge from the graph. The set of vertice is 
        untouched. return false if no edge was removed, else return true
        """
        if edge[0] in self._neighbors.get(edge[1], set()):
            self._neighbors[edge[0]].remove(edge[1])
            self._neighbors[edge[1]].remove(edge[0])
            return True
        return False

       
    def copy(self):
        """
        This return a shallow copy of the graph. Only the dictionnary
        are copied but the vertices are not deep copied
        """
        ans = UndirectedGraph()
        ans._neighbors = copy(self._neighbors)
        return ans

    def get_vertices(self):
        """
        This return a list of the vertices present in the graph
        """
        return set(self._neighbors.keys())

    vertices = property(get_vertices)

    def get_edges(self):
        """
        This return a list of edge present in the graph
        """
        ans = set()
        for x, nbs in self._neighbors.iteritems():
            for y in nbs:
                if x < y:
                    ans.add(frozenset([x, y]))
        return ans
    edges = property(get_edges)


    def is_connected(self, v1, v2):
        return v2 in self._neighbors[v1]

    def empty(self):
        return len(self._neighbors) == 0

    def triangulate(self):
        """
        This method add the edge to triangulate a graph. The algorithm
        used is simple and not optimal for Bnet (but we dont
        have at this level enough info to implement a better one)

        We simply select a vertice, join all the neighbour, remove the vertex
        and its edges, do-it again. The vertex is selected to minimize the
        number of added edges
        """
        # this is a copy of the graph containing all the live
        # edges
        graph_copy = self.copy()
        while not graph_copy.empty():
            # we select a vertices that lead to the smallest number of added
            # edges
            value = []
            for v in graph_copy.vertices:
                value.append([0,v])
                neighbors = list(graph_copy[v])
                for i, n_1 in enumerate(neighbors):
                    for n_2 in neighbors[i+1:]:
                        if not graph_copy.is_connected(n_1, n_2):
                            value[-1][0] += 1
            value.sort()
            best_vertex = value[0][1]
            # we now have the best vertices
            neighbors = list(graph_copy[best_vertex])
            for i, n_1 in enumerate(neighbors):
                for n_2 in neighbors[i+1:]:
                    if not graph_copy.is_connected(n_1, n_2):
                        # we triangulate self
                        self.add_edge((n_1, n_2))
            graph_copy.del_vertex(best_vertex)

    def __getitem__(self, vertex):
        """
        This function return all the neighbour of a given
        vertex
        """
        return self._neighbors[vertex]

    def __eq__(self, other):
        return self._neighbors.__eq__(other._neighbors)

    def __le__(self, other):
        return self._neighbors.__le__(other._neighbors)

    def __str__(self):
        ans = ["Nodes:"+ str(self.vertices)]
        ans.append("Edges:"+ str(self.edges))
        return '\n'.join(ans)


