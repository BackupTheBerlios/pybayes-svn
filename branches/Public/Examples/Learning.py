""" this example shows how to Learn parameters from a set of observations
 using Maximum Likelihood Estimator """
 
from OpenBayes import JoinTree, MCMCEngine
from copy import deepcopy
from time import time


# first create a beyesian network
from WaterSprinkler import *

N = 1000
# sample the network N times
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]

# create a new bayesian network with all parameters set to 1
G2 = deepcopy(G)
# set all parameters to 1s
G2.InitDistributions()

# create an inference Engine
# choose the one you like by commenting/uncommenting the appropriate line
ie = JoinTree(G2)
#ie = MCMCEngine(G)

# Learn the parameters from the set of cases
t =time()
ie.LearnMLParams(cases)
print 'Learned from %d cases in %1.3f secs' %(N,(time()-t))

# print the learned parameters
for v in G2.all_v: 
    print v.distribution,'\n'

<<<<<<< .mine
# print the learned parameters
for v in G.all_v: 
    print v.distribution,'\n'
    

=======
    

>>>>>>> .r145
