#!/usr/bin/env python
import unittest

from OpenBayes.graph import Graph, UndirEdge
from OpenBayes import BVertex


class TestGarph(unittest.TestCase):
    def test_nothing(self):   
        G = Graph()
        a, b, c, d, e, f, g = [G.add_v(BVertex(nm)) 
                               for nm in 'a b c d e f g'.split()]
        for ep in [(a,b), (a,c), (b,d), (b,f), (b,e), (c,e), (d,f), (e,f),
        (f, g)]:
            G.add_e(UndirEdge(len(G.e), *ep))

        print G
        #print 'DFS:', map(str, G.depth_first_search(a))
        #print 'BFS:', map(str, G.breadth_first_search(a))
        #print 'top sort:', map(str, G.topological_sort(a))

        #T = G.minimal_span_tree()
        #print T
        #print [(map(str, k), map(str, v)) for k, v in T.path_dict().items()]

        #S = G.shortest_tree(a)
        #print S

        print a

if __name__ == "__main__":
    unittest.main()
