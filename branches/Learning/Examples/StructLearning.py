""" this example shows how to Learn the structure from a set of observations
 and known parameters using a greedy search method """
 
from OpenBayes import learning, bayesnet
from copy import deepcopy
from time import time


# first create a bayesian network
from WaterSprinkler import *

N = 2000
# sample the network N times
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]

# Create a new BNet with no edges
G2 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
c,s,r,w = [G2.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
G2.InitDistributions()

#The distributions are known (given by G.marginaliseAll())
c.setDistributionParameters([0.5, 0.5])
s.setDistributionParameters([0.7, 0.3])
r.setDistributionParameters([0.5, 0.5])
w.setDistributionParameters([0.35, 0.65])

# Learn the structure
struct_engine = learning.GreedyStructLearningEngine(G2)
struct_engine.StructLearning(cases)
print 'learned structure: ', struct_engine.BNet
print 'total bic score: ', struct_engine.GlobalBICScore(N, cases)
