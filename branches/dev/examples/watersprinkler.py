#!/usr/bin/env python
""" this example will create the Water-Sprinkler example 
        Cloudy
         /  \
        /    \
       /      \
 Sprinkler  Rainy
     \        /
      \      /
       \    /
    Wet grass
    
all edges are pointing downwards

"""
from openbayes import BNet, BVertex, DirEdge

def main():
    """
    This is the main function. It simply create a baysian network and
    return it.
    """
    # We start by creating the network
    graph = BNet( 'Water Sprinkler Bayesian Network' )
    cloudy, sprinkler, rainy, wet = [graph.add_v(BVertex(nm, True, 2)) 
                                     for nm in 'c s r w'.split()]
    for edge_pair in [(cloudy, rainy), (cloudy, sprinkler), 
                      (rainy, wet), (sprinkler, wet)]:
        graph.add_e(DirEdge(len(graph.e), edge_pair[0], edge_pair[1]))
    print graph
    # finalize the bayesian network once all edges have been added   
    graph.init_distributions()
    # c | Pr(c)
    #---+------
    # 0 |  0.5
    # 1 |  0.5
    cloudy.set_distribution_parameters([0.5, 0.5])
    # c s | Pr(s|c)
    #-----+--------
    # 0 0 |   0.5
    # 1 0 |   0.9
    # 0 1 |   0.5
    # 1 1 |   0.1
    sprinkler.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
    # c r | Pr(r|c)
    #-----+--------
    # 0 0 |   0.8
    # 1 0 |   0.2
    # 0 1 |   0.2
    # 1 1 |   0.8
    rainy.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
    # s r w | Pr(w|c,s)
    #-------+------------
    # 0 0 0 |   1.0
    # 1 0 0 |   0.1
    # 0 1 0 |   0.1
    # 1 1 0 |   0.01
    # 0 0 1 |   0.0
    # 1 0 1 |   0.9
    # 0 1 1 |   0.9
    # 1 1 1 |   0.99
    # to verify the order of variables use :
    #>>> w.distribution.names_list
    #['w','s','r']
    
    # we can also set up the variables of a cpt with the following way
    # again the order is w.distribution.names_list
    wet.distribution[:, 0, 0] = [0.99, 0.01]
    wet.distribution[:, 0, 1] = [0.1, 0.9]
    wet.distribution[:, 1, 0] = [0.1, 0.9]
    wet.distribution[:, 1, 1] = [0.0, 1.0]
    
    # or even this way , using a dict:
    #w.distribution[{'s':0,'r':0}]=[0.99, 0.01]
    #w.distribution[{'s':0,'r':1}]=[0.1, 0.9]
    #w.distribution[{'s':1,'r':0}]=[0.1, 0.9]
    #w.distribution[{'s':1,'r':1}]=[0.0, 1.0]
    return graph

if __name__ == "__main__":
    main()

