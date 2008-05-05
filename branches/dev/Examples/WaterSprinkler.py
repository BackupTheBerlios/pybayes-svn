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
from OpenBayes import BNet, BVertex, DirEdge

# create the network
G = BNet( 'Water Sprinkler Bayesian Network' )
c, s, r, w = [G.add_v( BVertex( nm, True, 2 ) ) for nm in 'c s r w'.split()]
for ep in [( c, r ), ( c, s ), ( r, w ), ( s, w )]:
    G.add_e( DirEdge( len( G.e ), *ep ) )

print G

# finalize the bayesian network once all edges have been added   
G.InitDistributions()

# c | Pr(c)
#---+------
# 0 |  0.5
# 1 |  0.5
c.setDistributionParameters([0.5, 0.5])
# c s | Pr(s|c)
#-----+--------
# 0 0 |   0.5
# 1 0 |   0.9
# 0 1 |   0.5
# 1 1 |   0.1
s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
# c r | Pr(r|c)
#-----+--------
# 0 0 |   0.8
# 1 0 |   0.2
# 0 1 |   0.2
# 1 1 |   0.8
r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
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
w.distribution[:,0,0]=[0.99, 0.01]
w.distribution[:,0,1]=[0.1, 0.9]
w.distribution[:,1,0]=[0.1, 0.9]
w.distribution[:,1,1]=[0.0, 1.0]

# or even this way , using a dict:
#w.distribution[{'s':0,'r':0}]=[0.99, 0.01]
#w.distribution[{'s':0,'r':1}]=[0.1, 0.9]
#w.distribution[{'s':1,'r':0}]=[0.1, 0.9]
#w.distribution[{'s':1,'r':1}]=[0.0, 1.0]



