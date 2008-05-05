""" this examples creates a continuous bayesian network
        A
       / 
      B      
       \ 
        C
all edges point downwards
A,B and C are univariate gaussian distributions
"""

from OpenBayes import BNet, BVertex, DirEdge, MCMCEngine

# create the network
G = BNet( 'Water Sprinkler Bayesian Network' )
a,b,c = [G.add_v( BVertex( nm, False, 1 ) ) for nm in 'a b c'.split()]
for ep in [( a, b ), ( b, c )]:
    G.add_e( DirEdge( len( G.e ), *ep ) )

print G

# finalize the bayesian network once all edges have been added   
G.InitDistributions()

# fill in the parameters
a.distribution.setParameters(mu=1.0, sigma=0.5)
b.distribution.setParameters(mu=2.0, sigma=1.0, wi=2.0)
c.distribution.setParameters(mu=2.0, sigma=1.0, wi=1.0)

# NOTE : for the moment only MCMCEngine can work for continuous networks
ie = MCMCEngine(G)

res = ie.MarginaliseAll()

for r in res.values(): print r,'\n'
