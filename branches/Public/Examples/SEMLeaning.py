from copy import deepcopy
from time import time
import random

from OpenBayes import learning, bayesnet

# first create a bayesian network
from WaterSprinkler import *

N = 2000
# sample the network N times
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
# delete some observations
for i in range(500):
    case = cases[3*i]
    rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
    case[rand] = '?' 
for i in range(50):
    case = cases[3*i]
    rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
    case[rand] = '?'
    
# Create a new BNet with no edges
G2 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
c, s, r, w = [G2.add_v(bayesnet.BVertex(nm, True, 2)) for nm in 'c s r w'.split()]
G2.InitDistributions()

# Learn the structure
struct_engine = learning.SEMLearningEngine(G2)
struct_engine.SEMLearning(cases)
print 'learned structure: ', struct_engine.BNet
print 'total bic score: ', struct_engine.GlobalBICScore(N, cases, 0)

# Learn the structure
struct_engine = learning.SEMLearningEngine(G2)
struct_engine.SEMLearningApprox(cases)
print 'learned structure: ', struct_engine.BNet
print 'total bic score: ', struct_engine.GlobalBICScore(N, cases, 1)