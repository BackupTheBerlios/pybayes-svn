from OpenBayes import BNet, BVertex, DirEdge
from OpenBayes import learning
from copy import deepcopy
from time import time

# create the network
G = BNet( 'A Simple Bayesian Network' )
a, b = [G.add_v( BVertex( nm, True, 2 ) ) for nm in 'a b'.split()]
for ep in [( a, b )]:
    G.add_e( DirEdge( len( G.e ), *ep ) )


G2 = deepcopy(G)
print G


# finalize the bayesian network once all edges have been added   
G.InitDistributions()

print a.distribution.cpt, b.distribution.cpt

print '+++++++++++++++++++++++++++++++++\n'
G2.InitDistributions()
print G2.all_v[0].distribution.cpt

engine = learning.MLLearningEngine(G)

cases = engine.ReadFile('test.xls')
print 'cases:', cases
##cases = []
t = time()


engine.LearnMLParams(cases,0)
print 'Learned from 4 cases in %1.3f secs' %((time()-t))


# print the parameters
for v in G.all_v: 
    print v.distribution,'\n'

##engine.SaveInFile('simpletest.txt', G, engine.BNet, engine)