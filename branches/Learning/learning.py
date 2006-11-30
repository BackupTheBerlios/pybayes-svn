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
        precision = 0.05
##        engine = JoinTree(self.BNet)
##        #engine = MCMCEngine(self.BNet)
        while self.hasntConverged(old, new, precision) and iter < max_iter:
            iter += 1
##            old = {}
##            new = {}
##            for j,v in enumerate(self.BNet.v.values()):
##                old[j]=v.distribution.cpt
            old = copy.deepcopy(new)
            self.LearnEMParams(cases)
            # reinitialize the JunctionTree to take effect of new parameters learned
            self.engine.Initialization()
            self.engine.GlobalPropagation()
##            for j,v in enumerate(self.BNet.v.values()):
##                new[j]=v.distribution.cpt
            new = copy.deepcopy(self.BNet)
            print 'EM iteration: ', iter
    
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
                k = 0
                possible_list = []
                temp_dic = {}
                temp_unknown = copy.copy(unknown)
                for key in temp_unknown.iterkeys():
                    for state in range(self.BNet.v[key].nvalues):
                        k += 1
                old_k = k
                first_k = k
                print 'k=4: ', k
                k = k/self.BNet.v[temp_unknown.keys()[0]].nvalues
                print 'k=2: ', k
                print 'temp_unknown: ',temp_unknown
                for state in range(self.BNet.v[temp_unknown.keys()[0]].nvalues):
                    temp_dic[temp_unknown.keys()[0]]=state
                    temp = copy.copy(temp_dic)
                    for j in range(k):
                        possible_list.append(temp)
                    del temp_dic[temp_unknown.keys()[0]]
                del temp_unknown[temp_unknown.keys()[0]]
                print 'possible_list 1: ', possible_list
                states_list = self.DetermineList(possible_list, temp_unknown, k, old_k, first_k)
                print 'states_list: ', states_list
                likelihood_list = self.DetermineLikelihood(known, states_list)
                for j in range(first_k):
                    index = copy.copy(known)
                    index.update(states_list[j])
                    for v in self.BNet.v.values():
                        if v.distribution.isAdjustable:
                            v.distribution.addToCounts(index, likelihood_list[j])
                
##                self.engine.SetObs(known)
##                likelihood = {}
##                for key in unknown.iterkeys():
##                    likelihood[key] = self.engine.ExtractCPT(key)
##                for v in self.BNet.v.values():
##                    if v.distribution.isAdjustable:
##                        self.IterUnknown(known, unknown, likelihood, v)
##                self.engine.Initialization() 

        """ Second part of the algorithm : Estimation of the parameters. 
        (M-part)
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)

    def DetermineLikelihood(self, known, states_list):
        likelihood = []
        for states in states_list:
            like = 1
            temp_dic = {}
            copy_states = copy.copy(states)
            for key in states.iterkeys():
                self.engine.SetObs(known)
                temp_dic[key] = (copy_states[key])
                del copy_states[key]
                if len(copy_states) != 0:
                    self.engine.SetObs(copy_states)
                like = like*self.engine.ExtractCPT(key)[temp_dic[key]]
                copy_states.update(temp_dic)               
                del temp_dic[key]
                self.engine.Initialization()
            likelihood.append(like)
        return likelihood

    def DetermineList (self, possible_list, temp_unknown, k, old_k, first_k):
        print 'temp_unknown detlist: ',temp_unknown
        if len(temp_unknown) != 0:
            m = k/self.BNet.v[temp_unknown.keys()[0]].nvalues
            print 'm=1: ', m
            temp_list = []
            temp_dic = {}
            for state in range(self.BNet.v[temp_unknown.keys()[0]].nvalues):
                temp_dic[temp_unknown.keys()[0]]=state
                print 'temp_dic: ', temp_dic
                temp = copy.copy(temp_dic)
                print 'temp:', temp
                for j in range(m):
                    temp_list.append(temp)
                del temp_dic[temp_unknown.keys()[0]]
            del temp_unknown[temp_unknown.keys()[0]] 
            print 'temp_list: ', temp_list
            for j in range(old_k/k - 1):
                temp_list.extend(temp_list)
            print 'temp_list:', temp_list
            print 'temp_list[0]:', temp_list[0]
            print 'possible_list[0]', possible_list[0]
            for j in range(first_k):
                print 'temp_list[j]): ', temp_list[j]
                possible_list[j].update(temp_list[j]) ##Y A UN STRESS
                print 'possible_list', possible_list
            old_k = k
            k = m
            print 'possible_list len(unknown) different de 0: ', possible_list
            self.DetermineList(possible_list, temp_unknown, k, old_k, first_k)
        else:
            print 'possible_list len(unknown) = 0: ', possible_list
            return possible_list
    
##    def IterUnknown(self, known, unknown, likelihood, v):
##                #la recursion devrait commencer ici
##                for state in range(self.BNet.v[unknown.keys()[0]].nvalues):
##                    known[unknown.keys()[0]] = state
##                    v.distribution.addToCounts(known, likelihood[unknown.keys()[0]][state])
##                    del known[unknown.keys()[0]]
    
    def hasntConverged(self, old, new, precision):
        if not old :
            return True   
        else:
            return not  na.alltrue([na.allclose(v.distribution, new.v[v.name].distribution, atol=precision) for v in old.v.values()])
    

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
##        cases = self.BNet.Sample(2000)
##        
##        # delete some observations
##        for i in range(500):
##            case = cases[3*i]
##            rand = random.sample(['c','s','r','w'],1)[0]
##            case[rand] = '?' 
##        for i in range(50):
##            case = cases[3*i]
##            rand = random.sample(['c','s','r','w'],1)[0]
##            case[rand] = '?'
        cases = [{'c':'?','s':1,'r':'?','w':0}]
        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        G = copy.deepcopy(self.BNet)
        
        G.InitDistributions()
        
        engine = EMLearningEngine(G)
        engine.EMLearning(cases, 10)
        
        tol = 0.05
        assert(na.alltrue([na.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
               for v in G.all_v])), \
                " Learning does not converge to true values "
        print 'ok!!!!!!!!!!!!'
                
if __name__ == '__main__':
    suite = unittest.makeSuite(EMLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)