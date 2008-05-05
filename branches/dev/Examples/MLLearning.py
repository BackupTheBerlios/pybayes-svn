#!/usr/bin/env python
""" this example shows how to Learn parameters from a set of observations
 using Maximum Likelihood Estimator """ 
####from OpenBayes import JoinTree, MCMCEngine
from copy import deepcopy
from time import time

from OpenBayes import learning####, bayesnet

# first create a beyesian network
import WaterSprinkler

def main():
    """
    The main function
    """
    graph = WaterSprinkler.main()
    nbr_samples = 1000
    # sample the network N times
    cases = graph.sample(nbr_samples)  
    # create a new bayesian network with all parameters set to 1
    graph_copy = deepcopy(graph)
    graph_copy.init_distributions()
    # Learn the parameters from the set of cases
    engine = learning.MLLearningEngine(graph_copy)
    # cases = engine.read_file('file.xls') #To use the data of file.xls
    start_time = time()
    engine.learn_ml_params(cases)
    print 'Learned from %d cases in %1.3f secs' % \
          (nbr_samples, (time() - start_time))

    # print the learned parameters
    for vertex in graph_copy.all_v: 
        print vertex.name, vertex.distribution.cpt,'\n'

    # print the parameters
    for vertex in graph.all_v: 
        print vertex.distribution,'\n'

if __name__ == "__main__":
    print main()
