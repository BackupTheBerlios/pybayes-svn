""" this example shows how to Learn parameters from a set of observations
 using Maximum Likelihood Estimator """ 
####from OpenBayes import JoinTree, MCMCEngine
from copy import deepcopy
from time import time
import random

from OpenBayes import learning####, bayesnet

# first create a beyesian network
from WaterSprinkler import *

N = 1000
# sample the network N times
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]

# create a new bayesian network with all parameters set to 1
G2 = deepcopy(G)
# set all parameters to 1s
G2.InitDistributions()

# Learn the parameters from the set of cases
engine = learning.MLLearningEngine(G2)
# cases = engine.ReadFile('file.xls') #To use the data of file.xls
t = time()
engine.LearnMLParams(cases)
print 'Learned from %d cases in %1.3f secs' %(N,(time()-t))

# print the learned parameters
for v in G2.all_v: 
    print v.name, v.distribution.cpt,'\n'

### print the parameters
##for v in G.all_v: 
##    print v.distribution,'\n'