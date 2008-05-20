#!/usr/bin/env python
"""
This example demonstrate the capacity of sem_learningEngine
"""
import random

from openbayes import learning, bayesnet
import watersprinkler

def main():
    """
    This is the main function
    """
    graph = watersprinkler.main()
    nbr_samples = 2000
    # sample the network several times
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
        
# Create a new BNet with no edgraphes
    graph_copy = bayesnet.BNet('Water Sprinkler Bayesian Network2')
    for name in "c s r w".split():
        graph_copy.add_v(bayesnet.BVertex(name, True, 2)) 
    graph_copy.init_distributions()

# Learn the structure
    struct_engine = learning.SEMLearningEngine(graph_copy)
    struct_engine.sem_learning(cases)
    print 'learned structure: ', struct_engine.network
    print 'total bic score: ', \
          struct_engine.global_bic_score(nbr_samples, cases, 0)

# Learn the structure
    struct_engine = learning.SEMLearningEngine(graph_copy)
    struct_engine.sem_learning_approx(cases)
    print 'learned structure: ', struct_engine.network
    print 'total bic score: ', \
          struct_engine.global_bic_score(nbr_samples, cases, 1)

if __name__ == "__main__":
    main()
