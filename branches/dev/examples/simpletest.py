#!/usr/bin/env python
"""
This is a simple test
"""
# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from openbayes import BNet, BVertex, DirEdge
from openbayes import learning
from copy import deepcopy
from time import time

def main():
    """
    The main function
    """
# create the network
    graph = BNet('A Simple Bayesian Network')
    node_a, node_b = [graph.add_v(BVertex(nm, True, 2)) 
                      for nm in 'a b'.split()]
    graph.add_e(DirEdge(len(graph.e), node_a, node_b))


    graph_copy = deepcopy(graph)
    print graph


# finalize the bayesian network once all edges have been added   
    graph.init_distributions()

    print node_a.distribution.cpt, node_b.distribution.cpt
    print '+' * 20, '\n'
    graph_copy.init_distributions()
    print graph_copy.all_v[0].distribution.cpt

    engine = learning.MLLearningEngine(graph)
    try:
        cases = engine.read_file('test.xls')
    except ImportError:
        print "No support for excel. Install the xlrd module"
        print "This test does not continue"
        return
    print 'cases:', cases
    start_time = time()
    engine.learn_ml_params(cases, 0)
    print 'Learned from 4 cases in %1.3f secs' % ((time() - start_time))
    # print the parameters
    for vertex in graph.all_v: 
        print vertex.distribution,'\n'

##engine.save_in_file('simpletest.txt', G, engine.BNet, engine)

if __name__ == "__main__":
    main()
