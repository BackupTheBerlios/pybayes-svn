""" this example shows how to Learn the structure and the parameters from a set
 of incomplete observations using the Structural EM algorithm """
 
from OpenBayes import learning, bayesnet
from copy import deepcopy
from time import time
import random


# first create a beyesian network
from WaterSprinkler import *

N = 2000
# sample the network N times, delete some data
cases = G.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]       
for i in range(500):
    case = cases[3*i]
    rand = random.sample(['c','s','r','w'],1)[0]
    case[rand] = '?' 
for i in range(50):
    case = cases[3*i]
    rand = random.sample(['c','s','r','w'],1)[0]
    case[rand] = '?'

# Create a new BNet with no edges, and initialise all the distribution to 1
G2 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
c,s,r,w = [G2.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
G2.InitDistributions()

# Learn the structure and the parameters
struct_engine = learning.SEMLearningEngine(G2)
struct_engine.SEMLearning(cases)
print 'learned structure: ', struct_engine.BNet
print 'total bic score: ', struct_engine.GlobalBICScore(N, cases)
    

