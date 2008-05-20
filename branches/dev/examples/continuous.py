#!/usr/bin/env python
""" this examples creates a continuous bayesian network
        A
       / 
      B      
       \ 
        C
all edges point downwards
A,B and C are univariate gaussian distributions
"""

from openbayes import BNet, BVertex, DirEdge, MCMCEngine

def main():
    """
    Simply the main function. We watn to be clean
    """
    # create the network
    graph = BNet( 'Water Sprinkler Bayesian Network' )
    node_a, node_b, node_c = [graph.add_v(BVertex(nm, False, 1)) 
                              for nm in 'a b c'.split()]
    for start, end in [(node_a, node_b), (node_b, node_c)]:
        graph.add_e(DirEdge(len(graph.e), start, end ))
    print graph
    # finalize the bayesian network once all edges have been added   
    graph.init_distributions()
    # fill in the parameters
    node_a.distribution.set_parameters(mu=1.0, sigma=0.5)
    node_b.distribution.set_parameters(mu=2.0, sigma=1.0, wi=2.0)
    node_c.distribution.set_parameters(mu=2.0, sigma=1.0, wi=1.0)
    # NOTE : for the moment only MCMCEngine can work for continuous networks
    engine = MCMCEngine(graph)
    res = engine.marginalise_all()
    for res in res.values(): 
        print res,'\n'

if __name__ == "__main__":
    main()
