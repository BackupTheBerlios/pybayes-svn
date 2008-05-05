#!/usr/bin/env python
"""
This is a simple script to demonstrate EM learning
"""

from copy import deepcopy
from time import time
import random

from OpenBayes import learning

# first create a bayesian network
import WaterSprinkler

def main():
    """
    This is the main function
    """
    graph = WaterSprinkler.main()
    nbr_samples = 2000
    # sample the network N times
    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
    cases = graph.sample(nbr_samples)    
    # delete some observations
    for i in range(500):
        case = cases[3*i]
        rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
        case[rand] = '?' 
    for i in range(50):
        case = cases[3*i]
        rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
        case[rand] = '?'
    
    # copy the BN
    graph_copy = deepcopy(graph)
    # set all parameters to 1s
    graph_copy.init_distributions()
    # Learn the parameters from the set of cases
    engine = learning.EMLearningEngine(graph_copy)
    # cases = engine.read_file('file.xls') #To use the data of file.xls
    start_time = time()
    engine.em_learning(cases, 10)
    print 'Learned from %d cases in %1.3f secs' % \
          (nbr_samples, (time() - start_time))
    
    # print the learned parameters
    print "Learned paramters"
    for vertex in graph_copy.all_v: 
        print vertex.name, vertex.distribution.cpt, '\n'
    ### print the parameters 
    print "Orignal Parameters"   
    for vertex in graph.all_v: 
        print vertex.name, vertex.distribution.cpt, '\n'
        
if __name__ == "__main__":
    main()
