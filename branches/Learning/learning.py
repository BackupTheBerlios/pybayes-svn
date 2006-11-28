import graph
import bayesnet
import distributions
from inference import JoinTree
import random
import unittest
import copy
import numarray as na

import logging
# show INFO messages
#logging.basicConfig(level= logging.INFO)
#uncomment the following to remove all messages
logging.basicConfig(level = logging.NOTSET)

class EMLearningEngine:
    """ EM learning algorithm
    Learns the parameters of a known bayesian structure from incomplete data.
    """   
    BNet = None # The underlying bayesian network
    
    def __init__(self, BNet):
        self.BNet = BNet
        self.engine = JoinTree(BNet)
        #self.engine = MCMCEngine(BNet)
    
    def EMLearning(self, cases, max_iter):
        """ cases = [{'c':0,'s':1,'r':'?','w':1},{...},...]
        Put '?' when the data is unknown.
        Will estimate  the '?' by inference.
        """
        iter = 0
        old = None
        new = self.BNet
        precision = 0.005
##        engine = JoinTree(self.BNet)
##        #engine = MCMCEngine(self.BNet)
        while self.hasntConverged(old, new, precision) and iter < max_iter:
            iter += 1
##            old = {}
##            new = {}
##            for j,v in enumerate(self.BNet.v.values()):
##                old[j]=v.distribution.cpt
            old = copy.copy(new)
            self.LearnEMParams(cases)
            # reinitialize the JunctionTree to take effect of new parameters learned
            self.engine.Initialization()
            self.engine.GlobalPropagation()
##            for j,v in enumerate(self.BNet.v.values()):
##                new[j]=v.distribution.cpt
            new = copy.copy(self.BNet)
            if old == new:
                print 'y a un stress ici'
            print iter
    
    def LearnEMParams(self, cases):
        """ First part of the algorithm : Estimation of the unknown 
        data. (E-part)
        """ 
        for v in self.BNet.v.values():
                v.distribution.initializeCounts()
        for case in cases:
            known={}
            unknown={}
            for key in case.iterkeys():
                if case[key] != '?':
                    known[key] = case[key] #Contient tous les elements de case sauf l'element inconnu
                else:
                    unknown[key] = case[key]
            if len(case) == len(known): #Then all the data is known
                for v in self.BNet.v.values():
                    if v.distribution.isAdjustable: 
                        v.distribution.incrCounts(case)
            else:
                self.engine.SetObs(known)
                likelihood = {}
                for key in unknown.iterkeys():
                    likelihood[key] = self.engine.Marginalise(key).cpt
                for v in self.BNet.v.values():
                    if v.distribution.isAdjustable:
                        self.IterUnknown(known, unknown, likelihood, v)
                self.engine.Initialization() 

        """ Second part of the algorithm : Estimation of the parameters. 
        (M-part)
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)
    
    def IterUnknown(self, known, unknown, likelihood, v):
                #la recursion devrait commencer ici
                for state in range(self.BNet.v[unknown.keys()[0]].nvalues):
                    known[unknown.keys()[0]] = state
                    v.distribution.addToCounts(known, likelihood[unknown.keys()[0]][state])
                    del known[unknown.keys()[0]]
               
    
    def hasntConverged(self, old, new, precision):
        if not old :
            return True   
        else:
##            print not  na.alltrue([na.allclose(v.distribution.cpt, new.v[v.name].distribution.cpt, atol=precision) for v in old.v.values()])
##            return not  na.alltrue([na.allclose(v.distribution.cpt, new.v[v.name].distribution.cpt, atol=precision) for v in old.v.values()])
            return True
##            final = 0
##            m = 0
##            for v in self.BNet.v.values():
##                new_dist = new[m]
##                if len(v.distribution.family) != 1:
##                    for k in range(len(v.distribution.parents)):
##                        new_dist = new_dist[0]
##                old_dist = old[m]
##                if len(v.distribution.family) != 1:
##                    for k in range(len(v.distribution.parents)):
##                        old_dist = old_dist[0]
##                difference = new_dist-old_dist #EST-IL POSSIBLE QUE LES NOEUDS AIENT CHANGE DE PLACE??
##                result = max(max(abs(difference)),final)
##                m += 1
##            if result > precision:
##                return True
##            else:
##                return False

    
    

class EMLearningTestCase(unittest.TestCase):
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([0.5, 0.5])
        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:,0,0]=[0.99, 0.01]
        w.distribution[:,0,1]=[0.1, 0.9]
        w.distribution[:,1,0]=[0.1, 0.9]
        w.distribution[:,1,1]=[0.0, 1.0]
        
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        self.BNet = G
    
    def testEM(self):
        # sample the network 2000 times
        cases = self.BNet.Sample(2000)
        
        # delete some observations
        for i in range(500):
            case = cases[3*i]
            rand = random.sample(['c','s','r','w'],1)[0]
            case[rand] = '?' 
        
        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        G = copy.deepcopy(self.BNet)
        
        G.InitDistributions()
        
        engine = EMLearningEngine(G)
        engine.EMLearning(cases, 3)
        
        tol = 0.05
        assert(na.alltrue([na.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
               for v in G.all_v])), \
                " Learning does not converge to true values "
        print 'ok!!!!!!!!!!!!'
                
if __name__ == '__main__':
    suite = unittest.makeSuite(EMLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)