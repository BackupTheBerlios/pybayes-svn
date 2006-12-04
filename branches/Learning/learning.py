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
    engine = None
    
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
        while self.hasntConverged(old, new, precision) and iter < max_iter:
            iter += 1
            old = copy.deepcopy(new)
            self.LearnEMParams(cases)
            # reinitialize the JunctionTree to take effect of new parameters learned
            self.engine.Initialization()
            self.engine.GlobalPropagation()
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
                k = 1
                possible_list = []
                temp_dic = {}
                temp_unknown = copy.copy(unknown)
                for key in temp_unknown.iterkeys():
                    k = k*self.BNet.v[key].nvalues
                first_k = k
                k = first_k/self.BNet.v[temp_unknown.keys()[0]].nvalues
                old_k = k
                iteration = 2
                for state in range(self.BNet.v[temp_unknown.keys()[0]].nvalues):
                    temp_dic[temp_unknown.keys()[0]]=state
                    temp = copy.copy(temp_dic)
                    for j in range(k):
                        foo = copy.copy(temp)
                        possible_list.append(foo)
                    del temp_dic[temp_unknown.keys()[0]]
                del temp_unknown[temp_unknown.keys()[0]]
                states_list = self.DetermineList(possible_list, temp_unknown, old_k, iteration, first_k)
                likelihood_list = self.DetermineLikelihood(known, states_list)
                for j in range(first_k):
                    index = copy.copy(known)
                    index.update(states_list[j])
                    for v in self.BNet.v.values():
                        if v.distribution.isAdjustable:
                            v.distribution.addToCounts(index, likelihood_list[j])
                

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

    def DetermineList (self, possible_list, temp_unknown, old_k, iteration, first_k):
        if len(temp_unknown) != 0:
            m = old_k/self.BNet.v[temp_unknown.keys()[0]].nvalues
            temp_list = []
            temp_dic = {}
            for state in range(self.BNet.v[temp_unknown.keys()[0]].nvalues):
                temp_dic[temp_unknown.keys()[0]]=state
                temp = copy.copy(temp_dic)
                for j in range(m):
                    foo = copy.copy(temp)
                    temp_list.append(foo)
                del temp_dic[temp_unknown.keys()[0]]
            del temp_unknown[temp_unknown.keys()[0]] 
            for j in range(iteration):
                foo = copy.copy(temp_list)
                temp_list.extend(foo)
            for i in range(first_k):
                possible_list[i].update(temp_list[i])
            old_k = m
            iteration = iteration*2
            new_possible_list = copy.copy(possible_list)
            return self.DetermineList(new_possible_list, temp_unknown, old_k, iteration, first_k)
            
        else:
            return possible_list
    
    def hasntConverged(self, old, new, precision):
        if not old :
            return True   
        else:
            for v in old.v.values():
                print v.distribution.names_list
                print len(new.v[v.name].distribution)
            
            return not  na.alltrue([na.allclose(v.distribution, new.v[v.name].distribution, atol=precision) for v in old.v.values()])
    

class StructLearningEngine:
    """ Structural learning algorithm
    Learns the structure of a bayesian network from the known parameters.
    """   
    BNet = None # The underlying bayesian network
    engine = None
    
    def __init__(self, BNet):
        self.BNet = BNet
        self.engine = JoinTree(BNet)
        #self.engine = MCMCEngine(BNet) 

    def DetOptStruct (self):
        """Determine the optimal structure"""

    def ChangeStruct(self, change, edge):
        """Changes the edge (add, remove or reverse)"""
        if change == 'del':
            self.BNet.del_e(edge)
            # EST-CE QU'IL FAUT AUSSI DELETER LE NOEUD SI LE SEUL ARC QUI LE LIE EST EDGE??
        elif change == 'add':
            self.BNet.add_e(edge)
            # FAUT-IL VERIFIER QUE LE NOUVEAU BNET EST ACYCLIQUE??
        elif change == 'inv':
            self.BNet.inv_e(edge)
            # FAUT-IL VERIFIER QUE LE NOUVEAU BNET EST ACYCLIQUE??
        else:
            assert(False), "The asked change of structure is not possible. Only 'del' for delete, 'add' for add, and 'inv' for invert"

    


class StructLearningTestCase(unittest.TestCase):
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
    
    def testStruct(self):
        # sample the network 2000 times
        cases = self.BNet.Sample(2000)
        
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
                


class EMLearningTestCase(unittest.TestCase):
    def setUp(self):
        # create a discrete network
        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
        s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 's r w'.split()]
        c = G.add_v(bayesnet.BVertex('c',True,3))
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitDistributions()
        c.setDistributionParameters([1/3, 1/3, 1/3])
        s.setDistributionParameters([0.5, 0.9, 0.2, 0.5, 0.1, 0.8])
        r.setDistributionParameters([0.8, 0.2, 0.4, 0.2, 0.8, 0.6])
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
        for i in range(50):
            case = cases[3*i]
            rand = random.sample(['c','s','r','w'],1)[0]
            case[rand] = '?'

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

def Combinations(vertices):
    ''' vertices is a list of BVertex instances
        
        in the case of the water-sprinkler BN :
        >>> Combinations([c,r,w])
        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
    '''
    if len(vertices) > 1:
        list_comb = Combinations(vertices[:-1])
        new_list_comb = []
        for el in list_comb:
            for i in range(vertices[-1].nvalues):
                temp = copy.copy(el)
                temp[vertices[-1].name]=i
                new_list_comb.append(temp)
        return new_list_comb
    else:
        return [{vertices[0].name:el} for el in range(vertices[0].nvalues)]
                
if __name__ == '__main__':
##    suite = unittest.makeSuite(StructLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)    
    
    suite = unittest.makeSuite(EMLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
