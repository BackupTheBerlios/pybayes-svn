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

from OpenBayes import BNet, BVertex, DirEdge, MCMCEngine

# create the network
G = BNet( 'Water Sprinkler Bayesian Network' )
b,c = [G.add_v( BVertex( nm, False, 1 ) ) for nm in 'b c'.split()]
a = G.add_v(BVertex('a',True,2))

for ep in [( a, b ), ( b, c )]:
    G.add_e( DirEdge( len( G.e ), *ep ) )

print G

# finalize the bayesian network once all edges have been added   
G.InitDistributions()

# fill in the parameters
a.distribution.setParameters([0.7,0.3])
b.distribution.setParameters(mu=[2.0,0.0], sigma=[1.0,1.0])
c.distribution.setParameters(mu=2.0, sigma=1.0, wi=1.0)

case = G.Sample(1)        #---TODO: correct this bug...
print case

# NOTE : for the moment only MCMCEngine can work for continuous networks
ie = MCMCEngine(G)

res = ie.MarginaliseAll()

for r in res.values(): print r,'\n'
