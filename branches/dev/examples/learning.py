#!/usr/bin/env python
""" 
this example shows how to Learn parameters from a set of observations
using Maximum Likelihood Estimator 
"""
 
from openbayes import MCMCEngine
#fron OpenBayes import JoinTree
from copy import deepcopy
from time import time
import watersprinkler

def main():
    """
    This is the main function
    """
    # first create a beyesian network
    graph = watersprinkler.main()
    nbr_samples = 1000
    # sample the network N times
    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
    cases = graph.sample(nbr_samples)   
    # create a new bayesian network with all parameters set to 1
    graph_copy  = deepcopy(graph)
    # set all parameters to 1s
    graph_copy.init_distributions()

    # create an inference Engine
    # choose the one you like by commenting/uncommenting the appropriate line
    # engine = JoinTree(graph_copy)
    engine = MCMCEngine(graph_copy)

    # Learn the parameters from the set of cases
    start_time = time()
    engine.learn_ml_params(cases)
    print 'Learned from %d cases in %1.3f secs' % \
           (nbr_samples,(time()-start_time))

    # print the learned parameters
    for vertex in graph_copy.all_v: 
        print vertex.distribution,'\n'

    # print the learned parameters
    for vertex in graph.all_v: 
        print vertex.distribution,'\n'
        
if __name__ == "__main__":
    main()
