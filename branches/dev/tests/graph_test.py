#!/usr/bin/env python
import unittest

from openbayes.graph import Graph


class TestGarph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_edge((1,2))
        self.graph.add_edge((2,3))
        self.graph.add_edge((3,1))

    def test_vertices(self):   
        self.assertEqual(set(self.graph.vertices()), set([1, 2, 3]))
        
    def test_edges(self):
        self.assertEqual(set(self.graph.edges()), set([(1,2),(2,3),(3,1)]))

    def test_predecessor(self):
        self.assertEqual(self.graph.predecessors(1), [3])
        self.assertEqual(self.graph.predecessors(2), [1])
        self.assertEqual(self.graph.predecessors(3), [2])

    def test_successor(self):
        self.assertEqual(self.graph.successors(1), [2])
        self.assertEqual(self.graph.successors(2), [3])
        self.assertEqual(self.graph.successors(3), [1])

    def test_delete(self):
        self.assert_(self.graph.del_edge((1,2)))
        self.assertEquals(set(self.graph.edges()), set([(2,3), (3,1)]))

    def test_inverse(self):
        self.graph.inv_edge((1,2))
        self.assertEquals(set(self.graph.edges()), set([(2,1),(2,3), (3,1)]))

    def test_topo(self):
        self.assertEqual(self.graph.is_dag(), False)
        self.graph.del_edge((3,1))
        self.assertEqual(self.graph.topological_order(), [1,2,3])
        graph = Graph()
        graph.add_edge((1,2))
        graph.add_edge((1,3))
        graph.add_edge((2,4))
        graph.add_edge((3,4))
        self.assertEqual(self.graph.is_dag(), True)





       
if __name__ == "__main__":
    unittest.main()
