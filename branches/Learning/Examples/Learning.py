""" this example shows how to Learn parameters from a set of observations
 using Maximum Likelihood Estimator """
 
from OpenBayes import JoinTree, MCMCEngine
from copy import deepcopy
from time import time
import math

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

G3 = deepcopy(G)
for e in G3.v['c'].out_e:
    G3.del_e(e)
    G3.InitDistributions()
    engine = JoinTree(G3)
    engine.LearnMLParams(cases) #ne faire cela que pour le noeud fils
    break
G4 = deepcopy(G)
for e in G4.v['c'].out_e:
    G4.del_e(e)
    break
for case in cases :
    if G4.v['c'].distribution.isAdjustable:
        G4.v['c'].distribution.incrCounts(case)
if G4.v['c'].distribution.isAdjustable:
    G4.v['c'].distribution.setCounts()
    G4.v['c'].distribution.normalize(dim=v.name)
   



# print the learned parameters
for v in G2.all_v: 
    print v.name, ' G2: ', v.distribution.cpt,'\n'
    print G3.v[v.name].name, ' G3: ', G3.v[v.name].distribution.cpt,'\n'
    print G4.v[v.name].name, ' G4: ', G4.v[v.name].distribution.cpt,'\n'

