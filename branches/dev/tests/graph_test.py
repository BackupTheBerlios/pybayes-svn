#!/usr/bin/env python
import unittest

from openbayes.graph import Graph


class TestGarph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_edge((1,2))
        self.graph.add_edge((2,3))
        self.graph.add_edge((3,1))

    def test_nodes(self):   
        self.assertEqual(set(self.graph.nodes()), set([1, 2, 3]))
        
    def test_edges(self):
        self.assertEqual(set(self.graph.edges()), set([(1,2),(2,3),(3,1)]))

    def test_predecessor(self):
        self.assertEqual(self.graph.predecessors(1), [3])
        self.assertEqual(self.graph.predecessors(2), [1])
        self.assertEqual(self.graph.predecessors(3), [2])

       
if __name__ == "__main__":
    unittest.main()
