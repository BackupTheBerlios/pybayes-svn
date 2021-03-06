""" this example shows how to Learn parameters from a set of incomplete 
 observations using Maximum Likelihood Estimator """
 
from OpenBayes import learning
from copy import deepcopy
from time import time
import random

# first create a bayesian network
from WaterSprinkler import *

N = 2000
# sample the network N times
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
# delete some observations
for i in range(500):
    case = cases[3*i]
    rand = random.sample(['c','s','r','w'],1)[0]
    case[rand] = '?' 
for i in range(50):
    case = cases[3*i]
    rand = random.sample(['c','s','r','w'],1)[0]
    case[rand] = '?'

# create a new bayesian network with all parameters set to 1
G2 = deepcopy(G)
# set all parameters to 1s
G2.InitDistributions()

# Learn the parameters from the set of cases
engine = learning.EMLearningEngine(G2)
t = time()
engine.EMLearning(cases, 10)
print 'Learned from %d cases in %1.3f secs' %(N,(time()-t))

# print the learned parameters
for v in G2.all_v: 
    print v.name, v.distribution.cpt,'\n'

