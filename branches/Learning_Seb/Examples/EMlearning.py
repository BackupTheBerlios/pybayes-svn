from OpenBayes import BNet, BVertex, DirEdge, JoinTree, MCMCEngine, MultinomialDistribution
G = BNet( 'debordement de la Seine si il pleut' )
pluie = G.add_v( BVertex('pluie', True, 2))
seine = G.add_v( BVertex('seine', True, 2))
G.add_e( DirEdge( len(G.e), pluie, seine) )
print G

G.InitDistributions()
pluie.setDistributionParameters([0.5, 0.5])
seine.distribution[{'pluie':1}]=[0.7, 0.3]
seine.distribution[{'pluie':0}]=[0.6, 0.4]
print seine.distribution.cpt

old = seine.distribution.cpt
#Algo commence ici
ie = JoinTree(G)
old_bnet = G 
cases=[{'pluie':1, 'seine':'?'},{'pluie':0, 'seine':'?'},{'pluie':1, 'seine':0},{'pluie':0, 'seine':0},{'pluie':1, 'seine':1}]    
for v in G.v.values():
    v.distribution.initializeCounts()

temp = 0
for case in cases:
    if case['seine'] != '?' :
        for v in G.v.values():
            if v.distribution.isAdjustable:
                v.distribution.incrCounts(case)
    else:
        ie.SetObs({'pluie':case['pluie']})
        temp = ie.Marginalise('seine').cpt
        for v in G.v.values():
            for state in range(v.nvalues):
                v.distribution.addToCounts({'pluie':case['pluie'],'seine':state}, temp[state])

for v in G.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)

##for v in G.v.values():
##    v.distribution.setCounts()
##
# reinitialize the JunctionTree to take effect of new parameters learned
ie.Initialization()
ie.GlobalPropagation()
##dis = seine.distribution.cpt
##print dis 
##t = dis[1][1]/(dis[1][1]+dis[0][1])
##u = dis[1][0]/(dis[1][0]+dis[0][0])
##print t
##print u
###G.InitDistributions()
##seine.distribution[{'pluie':1}]=[1-t, t]
##seine.distribution[{'pluie':0}]=[1-u, u]
print seine.distribution.cpt
print pluie.distribution.cpt
##print old-seine.distribution.cpt
##test = old-seine.distribution.cpt
##print max(abs(test[0]))
##print len(seine.distribution.parents)
##test2 = test[0]
##print max(abs(test2))
test1 = {}
test2 = {}
i = 0
j = 0
m = 0
final = 0
for v in G.v.values():
    test1[i]=v.distribution.cpt
    i += 1
for v in old_bnet.v.values():
    test2[j]=v.distribution.cpt
    j += 1
for v in G.v.values():
    temp1 = test1[m]
    if len(v.distribution.family) != 1:
        for k in range(len(v.distribution.parents)):
            temp1 = temp1[0]
    temp2 = test2[m]
    if len(v.distribution.family) != 1:
        for k in range(len(v.distribution.parents)):
            temp2 = temp2[0]
    temp3 = temp1-temp2
    final = max(max(abs(temp3)),final)
    m += 1
print final
#ON RECOMMENCE
