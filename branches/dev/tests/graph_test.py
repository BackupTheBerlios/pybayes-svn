#!/usr/bin/env python
"""
This module contains all the test fro graph.py
"""
import unittest

from openbayes.graph import DirectedGraph, UndirectedGraph


class TestDirectedGarph(unittest.TestCase):
    """
    This is the test case for the directed graph
    """
    def setUp(self):
        self.graph = DirectedGraph()
        self.graph.add_edge((1, 2))
        self.graph.add_edge((2, 3))
        self.graph.add_edge((3, 1))

    def test_vertices(self):   
        self.assertEqual(self.graph.vertices, set([1, 2, 3]))
        
    def test_edges(self):
        self.assertEqual(set(self.graph.edges), 
                         set([(1, 2), (2, 3), (3, 1)]))

    def test_predecessor(self):
        self.assertEqual(self.graph.predecessors(1), [3])
        self.assertEqual(self.graph.predecessors(2), [1])
        self.assertEqual(self.graph.predecessors(3), [2])

    def test_successor(self):
        self.assertEqual(self.graph.successors(1), [2])
        self.assertEqual(self.graph.successors(2), [3])
        self.assertEqual(self.graph.successors(3), [1])

    def test_delete(self):
        self.assert_(self.graph.del_edge((1, 2)))
        self.assertEquals(set(self.graph.edges), 
                          set([(2, 3), (3, 1)]))

    def test_inverse(self):
        self.graph.inv_edge((1, 2))
        self.assertEquals(set(self.graph.edges), 
                          set([(2, 1), (2, 3), (3, 1)]))

    def test_topo(self):
        self.assertEqual(self.graph.is_dag(), False)
        self.graph.del_edge((3, 1))
        self.assertEqual(self.graph.topological_order(), [1, 2, 3])
        graph = DirectedGraph()
        graph.add_edge((1, 2))
        graph.add_edge((1, 3))
        graph.add_edge((2, 4))
        graph.add_edge((3, 4))
        self.assertEqual(self.graph.is_dag(), True)

class TestUndirectedGarph(unittest.TestCase):
    """
    This is the testcase for the UndirectedGraph class
    """
    def setUp(self):
        self.graph = UndirectedGraph()
        self.graph.add_edge((1, 2))
        self.graph.add_edge((2, 3))
        self.graph.add_edge((3, 1))
    
    def test_vertices(self):   
        self.assertEqual(self.graph.vertices, set([1, 2, 3]))
        
    def test_edges(self):
        self.assertEqual(self.graph.edges, set([frozenset([1, 2]),
                                                frozenset([2,3]),
                                                frozenset([3,1])]))

    def test_getitem(self):
        self.assertEqual(self.graph[1], set([2,3]))
        self.assertEqual(self.graph[2], set([1,3]))
        self.assertEqual(self.graph[3], set([1,2]))






       
if __name__ == "__main__":
    unittest.main()
