import graph
import bayesnet
import distributions
from inference import ConnexeInferenceJTree
import random
import unittest
import copy
import numarray as na
import math

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
            # states = {'c':0,'r':1} for example (c and r were unknown in the beginning)
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
                like = like*self.engine.CPT(key)[temp_dic[key]]
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
    

class GreedyStructLearningEngine:
    """ Greedy Structural learning algorithm
    Learns the structure of a bayesian network from the known parameters.
    """   
    BNet = None # The underlying bayesian network
    engine = None
    
    def __init__(self, BNet):
        self.BNet = copy.deepcopy(BNet)
        self.engine = ConnexeInferenceJTree(BNet)
        self.BNet.InitDistributions()
        for v in BNet.all_v:
                self.BNet.v[v.name].distribution.setParameters(self.engine.ExtractCPT(v.name))       
         

    def StructLearning(self, cases):
        N = len(cases)
        new_score = self.GlobalBICScore(self.BNet, N)
        old_score = None
        while self.Converged(old_score, new_score):
            old_score = copy.copy(new_score)
            print 'old_score', iter, old_score
            self.LearnStruct(cases, N)
            self.engine = ConnexeInferenceJTree(self.BNet)
            new_score = copy.copy(self.GlobalBICScore(self.BNet, N))
            print 'new_score', iter, new_score

    def Converged(self, old_score, new_score):
        result = False
        if not old_score:
            result = True
        if new_score > old_score:
            result = True
        return result

    def GlobalBICScore(self, G, N, engine):
        score = 0
        for v in G.all_v:
            cpt_matrix = engine.ExtractCPT(v.name)
            dim = G.Dimension(v)
            score = score + self.ScoreBIC(N, dim, cpt_matrix) 
        return score

    def LearnStruct(self, cases, N):
        """Greedy search for optimal structure (all the data in cases are known).
        It will go through every node of the BNet. At each node, it will delete 
        every outgoing edge, or add every possible edge, or reverse every 
        possible edge. It will compute the BIC score each time and keep the BNet
        with the highest score.
        """
        G_initial = copy.deepcopy(self.BNet)
        G_initial.InitDistributions()
        for v in G_initial.all_v:
                G_initial.v[v.name].distribution.setParameters(self.engine.ExtractCPT(v.name))
        infengine_init = ConnexeInferenceJTree(G_initial)
        G_best = copy.deepcopy(G_initial)
        G_best.InitDistributions()
        for v in G_initial.all_v:
                G_best.v[v.name].distribution.setParameters(self.engine.ExtractCPT(v.name))        
        infengine_best = ConnexeInferenceJTree(G_best)
        engine_init =  GreedyStructLearningEngine(G_initial)
        prec_var_score = 0
        
        for v in self.BNet.all_v:
            G = copy.deepcopy(G_initial)
            edges = copy.deepcopy(G.v[v.name].out_e)
            
            # delete the outgoing edges
            while edges:
                edge = edges.pop(0)
                node = edge._v[1] #node is the child node, the only node for which the cpt table changes
                cpt_matrix_init = copy.deepcopy(infengine_init.ExtractCPT(node.name))
                dim_init = G_initial.Dimension(node)
                score_init = engine_init.ScoreBIC(N, dim_init, cpt_matrix_init, infengine_init)
                self.ChangeStruct('del', edge) #delete the current edge
                self.SetNewDistribution(G_initial, node, cases, infengine_init)
                cpt_matrix = self.engine.ExtractCPT(node.name)
                dim = self.BNet.Dimension(node)
                score = self.ScoreBIC(N, dim, cpt_matrix, self.engine)
                var_score = score - score_init
                if var_score > prec_var_score:
                    prec_var_score = var_score
                    G_best = copy.deepcopy(self.BNet)
                    G_best.InitDistributions()
                    for vert in G_initial.all_v:
                        G_best.v[vert.name].distribution.setParameters(self.engine.ExtractCPT(vert.name)) 
                    infengine_best = ConnexeInferenceJTree(G_best)
                self.BNet = copy.deepcopy(G_initial) #re-initialise the BNet such that it deletes only one edge at a time
                self.BNet.InitDistributions()
                for verti in G_initial.all_v:
                    self.BNet.v[verti.name].distribution.setParameters(infengine_init.ExtractCPT(verti.name))
                self.engine = ConnexeInferenceJTree(self.BNet)
            
            # Add all possible edges
            G = copy.deepcopy(G_initial)
            nodes = []
            for node in G.all_v:
                if (not (node.name in [vv.name for vv in G.v[v.name].out_v])) and (not (node.name == v.name)):
                    nodes.append(node)
            while nodes:
                node = nodes.pop(0)
                if G.e.keys():
                    edge = graph.DirEdge(max(G.e.keys())+1, copy.deepcopy(self.BNet.v[v.name]), copy.deepcopy(self.BNet.v[node.name]))
                else:
                    edge = graph.DirEdge(0, copy.deepcopy(self.BNet.v[v.name]), copy.deepcopy(self.BNet.v[node.name]))
                self.ChangeStruct('add', edge)
                if self.BNet.HasNoCycles(node):
                    cpt_matrix_init = copy.deepcopy(infengine_init.ExtractCPT(node.name))
                    dim_init = G_initial.Dimension(node)
                    score_init = engine_init.ScoreBIC(N, dim_init, cpt_matrix_init, infengine_init)
                    self.SetNewDistribution(G_initial, node, cases, infengine_init)
                    cpt_matrix = self.engine.ExtractCPT(node.name)
                    dim = self.BNet.Dimension(node)
                    score = self.ScoreBIC(N, dim, cpt_matrix, self.engine)
                    var_score = score - score_init
                    if var_score > prec_var_score:
                        prec_var_score = var_score
                        G_best = copy.deepcopy(self.BNet)
                        G_best.InitDistributions()
                        for vert in G_initial.all_v:
                            G_best.v[vert.name].distribution.setParameters(self.engine.ExtractCPT(vert.name))
                        infengine_best = ConnexeInferenceJTree(G_best)
                self.BNet = copy.deepcopy(G_initial) #re-initialise the BNet such that it deletes only one edge at a time
                self.BNet.InitDistributions()
                for verti in G_initial.all_v:
                    self.BNet.v[verti.name].distribution.setParameters(infengine_init.ExtractCPT(verti.name))
                self.engine = ConnexeInferenceJTree(self.BNet)
        
##            # Invert all possible edges
##            G = copy.deepcopy(G_initial)
##            edges = copy.deepcopy(G.v[v.name].out_e)
##            while edges:
##                edge = edges.pop(0)
##                node = self.BNet.v[edge._v[1].name] #node is the child node
##                self.ChangeStruct('del', edge)
##                self.SetNewDistribution(G_initial, node, cases, infengine_init)
##                G_invert = copy.deepcopy(self.BNet)
##                G_invert.InitDistributions()
##                for vert in G_initial.all_v:
##                    G_invert.v[vert.name].distribution.setParameters(self.engine.ExtractCPT(vert.name))     
##                infengine_invert = ConnexeInferenceJTree(G_invert)
##                inverted_edge = graph.DirEdge(max(G.e.keys())+1, copy.deepcopy(self.BNet.v[node.name]),copy.deepcopy(self.BNet.v[v.name]))
##                self.ChangeStruct('add', inverted_edge)
##                if self.BNet.HasNoCycles(node):
##                    cpt_matrix_init = copy.deepcopy(infengine_init.ExtractCPT(v.name))
##                    dim_init = G_initial.Dimension(v)
##                    score_init = engine_init.ScoreBIC(N, dim_init, cpt_matrix_init, infengine_init)
##                    self.SetNewDistribution(G_invert, v, cases, infengine_init)
##                    cpt_matrix = self.engine.ExtractCPT(v.name)
##                    dim = self.BNet.Dimension(v)
##                    score = self.ScoreBIC(N, dim, cpt_matrix, self.engine)
##                    var_score = score - score_init
##                    if var_score > prec_var_score:
##                        prec_var_score = var_score
##                        G_best = copy.deepcopy(self.BNet)
##                        G_best.InitDistributions()
##                        for vert in G_initial.all_v:
##                            G_best.v[vert.name].distribution.setParameters(self.engine.ExtractCPT(vert.name))     
##                        infengine_best = ConnexeInferenceJTree(G_best)
##                self.BNet = copy.deepcopy(G_initial) #re-initialise the BNet such that it deletes only one edge at a time
##                self.BNet.InitDistributions()
##                for verti in G_initial.all_v:
##                    self.BNet.v[verti.name].distribution.setParameters(infengine_init.ExtractCPT(verti.name))
##                self.engine = ConnexeInferenceJTree(self.BNet)
        
        #self.BNet is the optimal graph structure
        self.BNet = copy.deepcopy(G_best)
        self.BNet.InitDistributions()
        for v in G_initial.all_v:
            self.BNet.v[v.name].distribution.setParameters(infengine_best.ExtractCPT(v.name))
    
    def SetNewDistribution(self, G_initial, node, cases, engine):#Refaire tout le EMLearning mais pour un seul noeud?
        self.BNet.InitDistributions()
        for v in G_initial.all_v:
            if v != node:
                self.BNet.v[v.name].distribution.setParameters(engine.ExtractCPT(v.name))
        if self.BNet.v[node.name].distribution.isAdjustable:
            self.BNet.v[node.name].distribution.initializeCounts()
            for case in cases :
                self.BNet.v[node.name].distribution.incrCounts(case)#To change if case has unknown data
            self.BNet.v[node.name].distribution.setCounts()
            self.BNet.v[node.name].distribution.normalize(dim=node.name)
    
    def ScoreBIC (self, N, dim, cpt_matrix):
        ''' This function computes the BIC score of one node.
        N is the size of the data from which we learn the structure
        dim is the dimension of the node, = (nbr of state - 1)*nbr of state of the parents
        cpt_matrix is the cpt table of the node
        return the BIC score
        '''
        score = self.ForBIC(N, cpt_matrix)
        score = score - 0.5*dim*math.log10(N)
        return score

    
    def ForBIC (self, N, cpt_matrix):
        score = 0
        if not isinstance(cpt_matrix[0],float):
            for i in range(len(cpt_matrix)):
                score = score + self.ForBIC(N, cpt_matrix[i])
            return score
        else :
            for i in range(len(cpt_matrix)):
                if cpt_matrix[i] != 0:
                    score = score + N*cpt_matrix[i]*math.log10(cpt_matrix[i])
                else:
                    score = score - 100000000
            return score

    def ChangeStruct(self, change, edge):
        """Changes the edge (add, remove or reverse)"""
        if change == 'del':
            self.BNet.del_e(edge)
        elif change == 'add':
            self.BNet.add_e(edge)
        elif change == 'inv':
            self.BNet.inv_e(edge)
        else:
            assert(False), "The asked change of structure is not possible. Only 'del' for delete, 'add' for add, and 'inv' for invert"

    


class GreedyStructLearningTestCase(unittest.TestCase):
##    #TEST SCORE
##    def setUp(self):
##        # create a discrete network
##        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
##        c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
##        for ep in [(c,r), (c,s), (r,w), (s,w)]:
##            G.add_e(graph.DirEdge(len(G.e), *ep))
##        G.InitDistributions()
##        c.setDistributionParameters([0.5, 0.5])
##        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
##        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
##        w.distribution[:,0,0]=[0.99, 0.01]
##        w.distribution[:,0,1]=[0.1, 0.9]
##        w.distribution[:,1,0]=[0.1, 0.9]
##        w.distribution[:,1,1]=[0.0, 1.0]
##        
##        self.c = c
##        self.s = s
##        self.r = r
##        self.w = w
##        self.BNet = G  
##    
##    def testStruct(self):
##        N = 1000
##        # sample the network N times
##        cases = self.BNet.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
##        G = copy.deepcopy(self.BNet)
##        G.InitDistributions()
##        for v in self.BNet.all_v:
##            G.v[v.name].distribution.setParameters(self.BNet.v[v.name].distribution.cpt)
##        struct_engine = GreedyStructLearningEngine(G)
##        dim = G.Dimension(G.v['w'])
##        cpt_matrix = G.v['w'].distribution.cpt
##        score_calc = struct_engine.ScoreBIC(N,dim,cpt_matrix)
##        s1 = N*G.v['w'].distribution.cpt[0][0][0]*math.log10(G.v['w'].distribution.cpt[0][0][0])
##        s2 = N*G.v['w'].distribution.cpt[0][0][1]*math.log10(G.v['w'].distribution.cpt[0][0][1])
##        s3 = N*G.v['w'].distribution.cpt[0][1][0]*math.log10(G.v['w'].distribution.cpt[0][1][0])
##        #s4 = N*G.v['w'].distribution.cpt[0][1][1]*math.log10(G.v['w'].distribution.cpt[0][1][1])
##        s5 = N*G.v['w'].distribution.cpt[1][0][0]*math.log10(G.v['w'].distribution.cpt[1][0][0])
##        s6 = N*G.v['w'].distribution.cpt[1][0][1]*math.log10(G.v['w'].distribution.cpt[1][0][1])
##        s7 = N*G.v['w'].distribution.cpt[1][1][0]*math.log10(G.v['w'].distribution.cpt[1][1][0])
##        s8 = N*G.v['w'].distribution.cpt[1][1][1]*math.log10(G.v['w'].distribution.cpt[1][1][1])
##        s9 = math.log10(N)*0.5*4
##        score = s1+s2+s3+s5+s6+s7+s8-s9
##        print 'score_calc',score_calc
##        print 'score',score
    
    # TEST ASIA
    def setUp(self):
        # create the network
        G = bayesnet.BNet( 'Asia Bayesian Network' )
        visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = [G.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in 'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
        for ep in [(visit,tuberculosis), (smoking,lung), (lung, ou), (tuberculosis, ou), (ou, Xray), (smoking, bronchitis), (bronchitis, dyspnoea), (ou, dyspnoea)]:
            G.add_e( graph.DirEdge( len( G.e ), *ep ) )

        G.InitDistributions()
        visit.setDistributionParameters([0.99, 0.01])
        tuberculosis.distribution[:,0]=[0.99, 0.01]
        tuberculosis.distribution[:,1]=[0.95, 0.05]
        smoking.setDistributionParameters([0.5, 0.5])
        lung.distribution[:,0]=[0.99, 0.01]
        lung.distribution[:,1]=[0.9, 0.1]
        ou.distribution[:,0,0]=[1, 0]
        ou.distribution[:,0,1]=[0, 1]
        ou.distribution[:,1,0]=[0, 1]
        ou.distribution[:,1,1]=[0, 1]
        Xray.distribution[:,0]=[0.95, 0.05]
        Xray.distribution[:,1]=[0.02, 0.98]
        bronchitis.distribution[:,0]=[0.7, 0.3]
        bronchitis.distribution[:,1]=[0.4, 0.6]
        dyspnoea.distribution[{'bronchitis':0,'ou':0}]=[0.9, 0.1]
        dyspnoea.distribution[{'bronchitis':1,'ou':0}]=[0.2, 0.8]
        dyspnoea.distribution[{'bronchitis':0,'ou':1}]=[0.3, 0.7]
        dyspnoea.distribution[{'bronchitis':1,'ou':1}]=[0.1, 0.9]
                
        self.v = visit
        self.t = tuberculosis
        self.s = smoking
        self.l = lung
        self.o = ou
        self.x = Xray
        self.b = bronchitis
        self.d = dyspnoea
        self.BNet = G
    
    def testStruct(self):
        N = 1000
        # sample the network N times
        cases = self.BNet.Sample(N)    # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
        
        # create two new bayesian network with the same parameters as self.BNet
        G1 = bayesnet.BNet( 'Asia Bayesian Network2' )
        visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = [G1.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in 'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
        G1.InitDistributions()
        visit.setDistributionParameters([0.99, 0.01])
        tuberculosis.setDistributionParameters([0.99, 0.01])
        smoking.setDistributionParameters([0.5, 0.5])
        lung.setDistributionParameters([0.95, 0.05])
        ou.setDistributionParameters([0.94, 0.06])
        Xray.setDistributionParameters([0.89, 0.11])
        bronchitis.setDistributionParameters([0.55, 0.45])
        dyspnoea.setDistributionParameters([0.56, 0.44])
        
        # Test StructLearning
        struct_engine1 = GreedyStructLearningEngine(G1)
        struct_engine1.StructLearning(cases)
        print struct_engine1.BNet
        
##    #TEST DE DISTRIBUTION.CPT    
##    def setUp(self):
##        # create a discrete network
##        G = bayesnet.BNet('Water Sprinkler Bayesian Network')
##        c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
##        for ep in [(c,r), (c,s), (r,w), (s,w)]:
##            G.add_e(graph.DirEdge(len(G.e), *ep))
##        G.InitDistributions()
##        c.setDistributionParameters([0.5, 0.5])
##        s.setDistributionParameters([0.5, 0.9, 0.5, 0.1])
##        r.setDistributionParameters([0.8, 0.2, 0.2, 0.8])
##        w.distribution[:,0,0]=[0.99, 0.01]
##        w.distribution[:,0,1]=[0.1, 0.9]
##        w.distribution[:,1,0]=[0.1, 0.9]
##        w.distribution[:,1,1]=[0.0, 1.0]
##        
##        self.c = c
##        self.s = s
##        self.r = r
##        self.w = w
##        self.BNet = G
##        
##    def testStruct(self):
##        cases = self.BNet.Sample(2000)
##        G = copy.deepcopy(self.BNet)
##        G.InitDistributions()
##        for v in self.BNet.all_v:
##            G.v[v.name].distribution.setParameters(self.BNet.v[v.name].distribution.cpt)
##        G1 = copy.deepcopy(self.BNet)
##        G1.add_e(graph.DirEdge(len(G1.e), G1.v['c'], G1.v['w']))
##        G1.InitDistributions()
##        G1.v['c'].distribution.setParameters(self.BNet.v['c'].distribution.cpt)
##        G1.v['s'].distribution.setParameters(self.BNet.v['s'].distribution.cpt)
##        G1.v['r'].distribution.setParameters(self.BNet.v['r'].distribution.cpt)
##        if G1.v['w'].distribution.isAdjustable:
##            G1.v['w'].distribution.initializeCounts()
##        for case in cases :
##            if G1.v['w'].distribution.isAdjustable:
##                G1.v['w'].distribution.incrCounts(case)
##        if G1.v['w'].distribution.isAdjustable:
##            G1.v['w'].distribution.setCounts()
##            G1.v['w'].distribution.normalize(dim='w')   
##        for v in self.BNet.all_v: 
##            print v.name, ' self: ', v.distribution.cpt,'\n'
##            print G.v[v.name].name, ' G: ', G.v[v.name].distribution.cpt,'\n'
##            print G1.v[v.name].name, ' G1: ', G1.v[v.name].distribution.cpt,'\n'

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
    suite = unittest.makeSuite(GreedyStructLearningTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)    
    
##    suite = unittest.makeSuite(EMLearningTestCase, 'test')
##    runner = unittest.TextTestRunner()
##    runner.run(suite)
