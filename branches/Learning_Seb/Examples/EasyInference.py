""" this example shows how to perform Inference using any Inference Engine """
from OpenBayes import JoinTree, MCMCEngine

# first create a beyesian network
from WaterSprinkler import *

# create an inference Engine
# choose the one you like by commenting/uncommenting the appropriate line
ie = JoinTree(G)
#ie = MCMCEngine(G)

# perform inference with no evidence
results = ie.MarginaliseAll()

for n,r in results.items(): print n,r

# add some evidence
ie.SetObs({'s':1})    # s=1

# perform inference with evidence
results = ie.MarginaliseAll()

# notice that the JoinTree engine does not perform a full message pass but only distributes 
# the new evidence...

for r in results.values(): 
    print r, '\n'

print 'ok'
print w.distribution.cpt