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
        """ 
        First part of the algorithm : Estimation of the unknown 
        data. (E-part)
        """ 
        # Initialise the counts of each vertex
        for v in self.BNet.v.values():
                v.distribution.initializeCounts()
        for case in cases:
            known={} # will contain all the known data of case
            unknown=[] # will contain all the unknown keys of case
            for key in case.iterkeys():
                if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                    known[key] = case[key]
                else:
                    unknown.append(key)
            if len(case) == len(known): # Then all the data is known -> proceed as LearnMLParams (inference.py)
                for v in self.BNet.v.values():
                    if v.distribution.isAdjustable: 
                        v.distribution.incrCounts(case)
            else:
                states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                likelihood_list = self.DetermineLikelihood(known, states_list) # Give a list with the likelihood to have the states in states_list                
                for j, index_unknown in enumerate(states_list):
                    index = copy.copy(known)
                    index.update(index_unknown)
                    for v in self.BNet.v.values():
                        if v.distribution.isAdjustable:
                            v.distribution.addToCounts(index, likelihood_list[j]) 
        """ 
        Second part of the algorithm : Estimation of the parameters. 
        (M-part)
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)
    
    def DetermineLikelihood(self, known, states_list):
        """ 
        Give a list with the likelihood to have the states in states_list
        I think this function could be optimized
        """        
        likelihood = []
        for states in states_list:
            # states = {'c':0,''r':1} for example (c and r were unknown in the beginning)
            like = 1
            temp_dic = {}
            copy_states = copy.copy(states)
            for key in states.iterkeys():
                """ 
                It has to be done for all keys because we have to set every 
                observation but key to compute the likelihood to have key in
                his state. The multiplication of all the likelihoods gives the
                likelihood to have states.                 
                """
                self.engine.SetObs(known) # Has to be done at each iteration because of the self.engine.Initialization() below 
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

    def Combinations(self, vertices):
        ''' vertices is a list of BVertex instances
        
        in the case of the water-sprinkler BN :
        >>> Combinations([c,r,w])
        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
        '''
        if len(vertices) > 1:
            list_comb = self.Combinations(vertices[:-1])
            new_list_comb = []
            for el in list_comb:
                for i in range(self.BNet.v[vertices[-1]].nvalues):
                    temp = copy.copy(el)
                    temp[self.BNet.v[vertices[-1]].name]=i
                    new_list_comb.append(temp)
            return new_list_comb
        else:
            return [{self.BNet.v[vertices[0]].name:el} for el in range(self.BNet.v[vertices[0]].nvalues)]
    
    def hasntConverged(self, old, new, precision):
        '''
        Return true if the difference of distribution of at least one vertex 
        of the old and new BNet is bigger than precision
        '''
        if not old :
            return True   
        else:
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

if __name__ == '__main__':
##    suite = unittest.makeSuite(StructLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)    
    
    suite = unittest.makeSuite(EMLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
