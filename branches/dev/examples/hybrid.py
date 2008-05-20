#!/usr/bin/env python
""" this examples creates a hybrid bayesian network
        A
       / 
      B      
       \ 
        C
all edges point downwards
B and C are univariate gaussian distributions
A is a boolean discrete distribution
"""

from openbayes import BNet, BVertex, DirEdge, MCMCEngine

def main():
    """
    The main function
    """
    # create the network
    graph = BNet( 'Water Sprinkler Bayesian Network' )
    b,c = [graph.add_v( BVertex( nm, False, 1 ) ) for nm in 'b c'.split()]
    a = graph.add_v(BVertex('a',True,2))
    
    for start, end in [( a, b ), ( b, c )]:
        graph.add_e( DirEdge( len( graph.e ), start, end ) )
    
    print graph
    
    # finalize the bayesian network once all edges have been added   
    graph.init_distributions()
    
    # fill in the parameters
    a.distribution.set_parameters([0.7,0.3])
    b.distribution.set_parameters(mu=[2.0,0.0], sigma=[1.0,1.0])
    c.distribution.set_parameters(mu=2.0, sigma=1.0, wi=1.0)
    
    case = graph.sample(1)        #---TODO: correct this bug...
    print case
    
    # NOTE : for the moment only MCMCEngine can work for continuous networks
    engine = MCMCEngine(graph)
    
    res = engine.marginalise_all()
    
    for res in res.values(): 
        print res

if __name__ == "__main__":
    main()
