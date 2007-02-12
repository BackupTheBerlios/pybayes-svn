###############################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta, Francois de Brouchoven
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
###############################################################################
import graph
import bayesnet
import distributions
from inference import ConnexeInferenceJTree, JoinTree
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
	
	def EMLearning(self, cases, max_iter, initializeFirst=False):
		""" cases = [{'c':0,'s':1,'r':'?','w':1},{...},...]
		Put '?' when the data is unknown.
		Will estimate  the '?' by inference.
		"""
		
		# if wanted the network is initialized (previously learned CPT's are
		# resetted, the previous learned data is lost)
		if initializeFirst:
			for v in self.BNet.v.values():
				if v.distribution.isAdjustable:
					v.distribution.initializeAugmentedEq() # sets all a_ij and b_ij to equivalent sample size
					v.distribution.initializeCounts() # sets all counts to zero
					v.distribution.setAugmentedCounts()
					v.distribution.normalize(dim=v.name) # set the initial Pr's to a_ij/(a_ij+b_ij)
			
		iter = 0
		old = None
		new = self.BNet
		precision = 0.05
		while self.hasntConverged(old, new, precision) and iter < max_iter:
			iter += 1
			print 'EM iteration: ', iter
			old = copy.deepcopy(new)
			self.LearnEMParamsWM(cases)
			# reinitialize the JunctionTree to take effect of new parameters learned
			self.engine.Initialization()
			# self.engine.GlobalPropagation()
			new = copy.deepcopy(self.BNet)
	
	def LearnEMParamsWM(self, cases):
		# Initialise the counts of each vertex
		for v in self.BNet.v.values():
			if v.distribution.isAdjustable:
				v.distribution.initializeCounts() # sets all counts to zero
		
		# First part of the algorithm : Estimation of the unknown 
		# data. (E-part)
		for case in cases:
			#assert(set(case.keys()) == set(self.BNet.v.keys())), "Not all values of 'case' are set"
			
			known = dict() # will contain al the known data of the case
			for key in case.iterkeys():
				if case[key] != '?':
					known[key] = case[key]
			
			for v in self.BNet.v.values():
				if v.distribution.isAdjustable:
					names = [parent.name for parent in v.family[1:]]
					nvals = [parent.nvalues for parent in v.family[1:]]
					names.append(v.name)
					nvals.append(v.nvalues)
					self.calcExpectation(v,known,names,nvals)
		
		# Second part of the algorithm : Estimation of the parameters. 
		# (M-part)
		for v in self.BNet.v.values():
			if v.distribution.isAdjustable:
				v.distribution.setAugmentedAndCounts()
				v.distribution.normalize(dim=v.name)
	
	def calcExpectation(self,v,known,names,nvals,cumPr=1):
		"""	calculate and set expectations of node v
			based on chain rule: Pr(names | known) = Pr(names[0]|names[1] .. names[end], known).Pr(names[1]|names[2]..names[end], known)...
			
			in:
			v           : node to calc and set
			known       : the known nodes of the net
			names       : the names of the nodes necessary to calcute P(X_v = val_v, pa_v = val_ij| known, prior_CPT)
			nvals       : the nvalues ...
			cumPr       : the cumulative chance of the chain rule
			
			out:
			Counts of node v are adjusted based on the known nodes
		"""
		if cumPr == 0:
			# chance will be remain zero so no use in calculating any further
			return
		
		newnames = list(names)
		name = newnames.pop()
		newnvals = list(nvals)
		nval = newnvals.pop()
		
		if known.has_key(name):
			# Value of P(name=X | pa_v(i+1) ... , known) is 1 because of the known information
			if len(newnames) == 0:
				v.distribution.addToCounts(known,cumPr)
			else:
				self.calcExpectation(v,known,newnames,newnvals,cumPr)
		else:
			self.engine.Initialization() #Only Initializiatino is enough to reset because SetObs calls GlobalPropagation
			self.engine.SetObs(known)
			marg = self.engine.Marginalise(name)
			
			for value in range(nval):
				newknown = dict(known)
				newknown[name] = value
				thisPr = marg[value]
				if not(str(thisPr) == 'nan' or thisPr == 0):
					if len(newnames) == 0:
						v.distribution.addToCounts(newknown,cumPr*thisPr)
					else:
						self.calcExpectation(v,newknown,newnames,newnvals,cumPr*thisPr)

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
				temp_dic[key] = (copy_states[key])
				del copy_states[key]
				if len(copy_states) != 0:
					# first update copy_states because calling SetObs(known) and SetObs(copy_states)
					# isn't the same as calling copy_states.update(known) and SetObs(copy_states)
					# BUG?
					copy_states.update(known) 
					self.engine.SetObs(copy_states)
				else:
					self.engine.SetObs(known) # Has to be done at each iteration because of the self.engine.Initialization() below 
				#print 'finished	to try setobs'
				like = self.engine.Marginalise(key)[temp_dic[key]]#like = like*self.BNet.v[key].distribution.Convert_to_CPT()[temp_dic[key]]
				#like = like*self.engine.ExtractCPT(key)[temp_dic[key]]
				if str(like) == 'nan':
					like = 0
					
				copy_states.update(temp_dic)			   
				del temp_dic[key]
				self.engine.Initialization()
			likelihood.append(like)
		return likelihood
	
	def DetermineLikelihoodIncorrect(self, known, states_list):
		""" 
		Give a list with the likelihood to have the states in states_list.
		6 to 10 time faster than the DetermineLikelihood function above, but 
		has to be tested more...
		"""		   
		likelihood = []
		for states in states_list:
			# states = {'c':0,'r':1} for example (c and r were unknown in the beginning)
			like = 1
			parents_state = copy.copy(known)
			parents_state.update(copy.copy(states))
			for key in states.iterkeys():
				cpt = self.BNet.v[key].distribution[parents_state]
				like = like * cpt				
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
			return not	na.alltrue([na.allclose(v.distribution, new.v[v.name].distribution, atol=precision) for v in old.v.values()])
	

class GreedyStructLearningEngine(EMLearningEngine):
        """ Greedy Structural learning algorithm
        Learns the structure of a bayesian network from known parameters and an 
        initial structure.
        """	  
        BNet = None # The underlying bayesian network
        engine = None
        
        def __init__(self, BNet):
            self.BNet = BNet.copy()
            #print 'begin connexeInferenceJTree'
            self.engine = ConnexeInferenceJTree(self.BNet)
            #print 'end connexeInferenceJTree'
            self.converged = False

        def StructLearning(self, cases):
            N = len(cases)
            iter = 0
            print 'begin iteration'
            while (not self.converged):
                self.LearnStruct(cases, N)
                iter +=1
                print 'Structure iteration:', iter

        def GlobalBICScore(self, N, cases):
            '''Computes the BIC score of self.BNet'''
            score = 0
            for v in self.BNet.all_v:
                cpt_matrix = v.distribution.Convert_to_CPT()
                dim = self.BNet.Dimension(v)
                score = score + self.ScoreBIC(N, dim, self.BNet, v, cases) 
            return score

        def LearnStruct(self, cases, N):
            """Greedy search for optimal structure (all the data in cases are known).
            It will go through every node of the BNet. At each node, it will delete 
            every outgoing edge, or add every possible edge, or reverse every 
            possible edge. It will compute the BIC score each time and keep the BNet
            with the highest score.
            """
            G_initial = self.BNet.copy()
            engine_init = GreedyStructLearningEngine(G_initial)
            G_best = self.BNet.copy()	  
            prec_var_score = 0
            
            for v in self.BNet.all_v:
                G = copy.deepcopy(engine_init.BNet)
                edges = copy.deepcopy(G.v[v.name].out_e)
                
                # delete the outgoing edges
                while edges:
                    print 'try to delete an edge'
                    edge = edges.pop(0)
                    node = edge._v[1] #node is the child node, the only node for which the cpt table changes
                    dim_init = G_initial.Dimension(node)
                    #print 'try calculating score init'
                    score_init = engine_init.ScoreBIC(N, dim_init, G_initial, G_initial.v[node.name], cases)
                    self.ChangeStruct('del', edge) #delete the current edge
                    self.SetNewDistribution2(engine_init.BNet, node, cases)
                    dim = self.BNet.Dimension(node)
                    #print 'try calculating score'
                    score = self.ScoreBIC(N, dim, self.BNet, self.BNet.v[node.name], cases)
                    var_score = score - score_init
                    if var_score > prec_var_score:
                        print 'deleted:', v.name, node.name, var_score
                        prec_var_score = var_score
                        G_best = self.BNet.copy()
                    self.BNet = G_initial.copy()
                    print 'finish trying deleting an edge'
                
                # Add all possible edges
                G = copy.deepcopy(engine_init.BNet)
                nodes = []
                for node in G.all_v:
                    if (not (node.name in [vv.name for vv in self.BNet.v[v.name].out_v])) and (not (node.name == v.name)):
                        nodes.append(node)
                while nodes:
                    print 'try to add an edge'
                    node = nodes.pop(0)
                    if G.e.keys():
                        edge = graph.DirEdge(max(G.e.keys())+1, self.BNet.v[v.name], self.BNet.v[node.name])
                    else:
                        edge = graph.DirEdge(0, self.BNet.v[v.name], self.BNet.v[node.name])
                    self.ChangeStruct('add', edge)
                    if self.BNet.HasNoCycles(self.BNet.v[node.name]):
                        dim_init = engine_init.BNet.Dimension(node)
                        score_init = engine_init.ScoreBIC(N, dim_init, G_initial, G_initial.v[node.name], cases)
                        self.SetNewDistribution2(engine_init.BNet, node, cases)
                        dim = self.BNet.Dimension(node)
                        score = self.ScoreBIC(N, dim, self.BNet, self.BNet.v[node.name], cases)
                        var_score = score - score_init
                        if var_score > prec_var_score:
                            print 'added: ', v.name, node.name, var_score
                            prec_var_score = var_score
                            G_best = self.BNet.copy()
                    self.BNet = G_initial.copy()
                    print 'finish trying adding an edge'
            
                # Invert all possible edges
                G = copy.deepcopy(G_initial)
                edges = copy.deepcopy(G.v[v.name].out_e)
                while edges:
                    print 'try to invert an edge'
                    edge = edges.pop(0)
                    node = self.BNet.v[edge._v[1].name] #node is the child node
                    dim_init1 = G_initial.Dimension(node)
                    score_init1 = engine_init.ScoreBIC(N, dim_init1, G_initial, G_initial.v[node.name], cases)
                    self.ChangeStruct('del', edge)
                    self.SetNewDistribution2(engine_init.BNet, node, cases)
                    dim1 = self.BNet.Dimension(node)
                    score1 = self.ScoreBIC(N, dim1, self.BNet, self.BNet.v[node.name], cases)
                    G_invert = self.BNet.copy() 
                    engine_invert = GreedyStructLearningEngine(G_invert)  
                    inverted_edge = graph.DirEdge(max(G.e.keys())+1, self.BNet.v[node.name],self.BNet.v[v.name])
                    self.ChangeStruct('add', inverted_edge)
                    if self.BNet.HasNoCycles(self.BNet.v[node.name]):
                        dim_init = G_initial.Dimension(v)
                        score_init = engine_init.ScoreBIC(N, dim_init, G_initial, G_initial.v[v.name], cases)
                        self.SetNewDistribution2(engine_invert.BNet, v, cases)
                        dim = self.BNet.Dimension(v)
                        score = self.ScoreBIC(N, dim, self.BNet, self.BNet.v[v.name], cases)
                        var_score = score1 - score_init1 + score - score_init
                        if var_score > prec_var_score + 5: #+ 5 is to avoid recalculation due to round errors
                            print 'inverted:', v.name, node.name, var_score
                            prec_var_score = var_score
                            G_best = self.BNet.copy()
                    self.BNet = G_initial.copy()
                    print 'finish trying inverting an edge'
            
            #self.BNet is the optimal graph structure
            if prec_var_score == 0:
                self.converged = True
            self.BNet = G_best.copy()
            #self.engine = ConnexeInferenceJTree(self.BNet)
        
        def SetNewDistribution2(self, G_initial, node, cases):
            '''Set the new distribution of the node node. The other distributions
            are the same as G_initial (only node has a new parent, so the other 
            distributions don't change). Works also with incomplete data'''
            self.BNet.InitDistributions()
            for v in G_initial.all_v:
                if v.name != node.name:
                    cpt = G_initial.v[v.name].distribution.Convert_to_CPT()
                    self.BNet.v[v.name].distribution.setParameters(cpt)
                #else:
            if self.BNet.v[node.name].distribution.isAdjustable:
                self.BNet.v[node.name].distribution.initializeCounts()
                for case in cases :
                    known = dict() # will contain al the known data of the case
                    for key in case.iterkeys():
                        if case[key] != '?':
                            known[key] = case[key]
                    for v in self.BNet.v.values():
                        if v.distribution.isAdjustable:
                            names = [parent.name for parent in self.BNet.v[node.name].family[1:]]
                            nvals = [parent.nvalues for parent in self.BNet.v[node.name].family[1:]]
                            names.append(self.BNet.v[node.name].name)
                            nvals.append(self.BNet.v[node.name].nvalues)
                            self.calcExpectation(self.BNet.v[node.name],known,names,nvals)
                    self.BNet.v[node.name].distribution.setAugmentedAndCounts()
                    self.BNet.v[node.name].distribution.normalize(dim=self.BNet.v[node.name].name)

        def SetNewDistribution(self, G_initial, node, cases):
            '''Set the new distribution of the node node. The other distributions
            are the same as G_initial (only node has a new parent, so the other 
            distributions don't change). Works also with incomplete data'''
            self.BNet.InitDistributions()
            for v in G_initial.all_v:
                if v.name != node.name:
                    cpt = G_initial.v[v.name].distribution.Convert_to_CPT()
                    self.BNet.v[v.name].distribution.setParameters(cpt)
                else:
                    if self.BNet.v[node.name].distribution.isAdjustable:
                        self.BNet.v[node.name].distribution.initializeCounts()
                        for case in cases :
                            known={} # will contain all the known data of case
                            unknown=[] # will contain all the unknown keys of case
                            for key in case.iterkeys():
                                if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                                    known[key] = case[key]
                                else:
                                    unknown.append(key)
                            if len(case) == len(known): # Then all the data is known -> proceed as LearnMLParams (inference.py)
                                self.BNet.v[node.name].distribution.incrCounts(case)
                            else:
                                states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                                likelihood_list = self.DetermineLikelihood(known, states_list) # Give a list with the likelihood to have the states in states_list				  
                                for j, index_unknown in enumerate(states_list):
                                    index = copy.copy(known)
                                    index.update(index_unknown)
                                    self.BNet.v[node.name].distribution.addToCounts(index, likelihood_list[j])
                        self.BNet.v[node.name].distribution.setCounts()
                        self.BNet.v[node.name].distribution.normalize(dim=node.name)
	
        def ScoreBIC (self, N, dim, G, node, data):
            ''' This function computes the BIC score of one node.
            N is the size of the data from which we learn the structure
            dim is the dimension of the node, = (nbr of state - 1)*nbr of state of the parents
            data is the list of cases
            return the BIC score
            Works also with incomplete data!
            '''
            #print 'begin forbic'
            score = self.ForBIC(G, data, node)
            #print 'end forbic, score = ', score
            score = score - 0.5*dim*math.log(N)
            return score

        def ForBIC (self, G, cases, node):
            ''' Computes for each case the probability to have node and his parents
            in the case state, take the log of that probability and add them.'''
            score = 0
            for case in cases :
                cpt = 0 
                known={} # will contain all the known data of case
                unknown=[] # will contain all the unknown data of case
                for key in case.iterkeys():
                    if case[key] != '?': # It's the only part of code you have to change if you want to have another 'unknown sign' instead of '?'
                        known[key] = case[key]
                    else:
                        unknown.append(key)
                if len(case) == len(known): # Then all the data is known
                    #print 'all the data is known, node.distribution'
                    cpt = node.distribution[case]
                    #print 'cpt, all data is known: ', cpt
                else:
                    #print 'begin Combinations'
                    states_list = self.Combinations(unknown) # Give a dictionary list of all the possible states of the unknown parameters
                    #print 'begin determinelikelihood'
                    likelihood_list = self.DetermineLikelihood(known, states_list) # Give a list with the likelihood to have the states in states_list	
                    #print 'end determinelikelihood'			  
                    for j, index_unknown in enumerate(states_list):
                        index = copy.copy(known)
                        index.update(index_unknown)
                        cpt = cpt + likelihood_list[j]*node.distribution[index]
                        #print 'cpt: ', cpt
                if cpt == 0: # To avoid log(0)
                    cpt = math.exp(-700)
                score = score + math.log(cpt)
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

class SEMLearningEngine(GreedyStructLearningEngine, EMLearningEngine):	
	""" Structural EM learning algorithm
	Learns the structure and the parameters of a bayesian network from 
	unknown parameters and an initial structure.
	""" 
	def __init__(self, BNet):
		self.BNet = BNet
		self.engine = ConnexeInferenceJTree(self.BNet)
		self.converged = False
	
	def SEMLearning(self, cases, max_iter = 30):
		"""Structural EM for optimal structure and parameters if some of the 
		data is unknown (put '?' for unknown data).
		"""
		N = len(cases)
		iter = 0
		while (not self.converged) and iter < max_iter:
			#First we estimate the distributions of the initial structure
			self.BNet.InitDistributions()
			self.EMLearning(cases, 10)
			#Then we find a better structure in the neighborhood of self.BNet
			self.LearnStruct(cases, N)
			iter +=1
			print 'Structure Expectation-Maximisation iteration:', iter
  

class SEMLearningTestCase(unittest.TestCase):
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

	def testSEM(self):
		N = 2000
		# sample the network N times, delete some data
		cases = self.BNet.Sample(N)	   # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]		 
		for i in range(500):
			case = cases[3*i]
			rand = random.sample(['c','s','r','w'],1)[0]
			case[rand] = '?' 
		for i in range(50):
			case = cases[3*i]
			rand = random.sample(['c','s','r','w'],1)[0]
			case[rand] = '?' 
		G = bayesnet.BNet('Water Sprinkler Bayesian Network2')
		c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
		G.InitDistributions()
		# Test SEMLearning
		struct_engine = SEMLearningEngine(G)
		struct_engine.SEMLearning(cases)
		print 'learned structure: ', struct_engine.BNet
		print 'total bic score: ', struct_engine.GlobalBICScore(N, cases)

class GreedyStructLearningTestCase(unittest.TestCase):
	#TEST Waterspringler
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
		N = 50
		# sample the network N times, delete some data
		cases = self.BNet.Sample(N)	   # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]		 
		for i in range(10):
			case = cases[3*i]
			rand = random.sample(['c','s','r','w'],1)[0]
			case[rand] = '?' 
		for i in range(5):
			case = cases[3*i]
			rand = random.sample(['c','s','r','w'],1)[0]
			case[rand] = '?' 
##		G = bayesnet.BNet('Water Sprinkler Bayesian Network2')
##		c,s,r,w = [G.add_v(bayesnet.BVertex(nm,True,2)) for nm in 'c s r w'.split()]
##		G.InitDistributions()
##		c.setDistributionParameters([0.5, 0.5])
##		s.setDistributionParameters([0.7, 0.3])
##		r.setDistributionParameters([0.5, 0.5])
##		w.setDistributionParameters([0.35, 0.65])
		# Test StructLearning
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
		struct_engine = GreedyStructLearningEngine(G)
		struct_engine.StructLearning(cases)
		print 'learned structure: ', struct_engine.BNet
		print 'total bic score: ', struct_engine.GlobalBICScore(N, cases)


##	  # TEST ASIA
##	  def setUp(self):
##		  # create the network
##		  G = bayesnet.BNet( 'Asia Bayesian Network' )
##		  visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = [G.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in 'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
##		  for ep in [(visit,tuberculosis), (tuberculosis, ou), (smoking,lung), (lung, ou), (ou, Xray), (smoking, bronchitis), (bronchitis, dyspnoea), (ou, dyspnoea)]:
##			  G.add_e( graph.DirEdge( len( G.e ), *ep ) )
##		  G.InitDistributions()
##		  visit.setDistributionParameters([0.99, 0.01])
##		  tuberculosis.distribution[:,0]=[0.99, 0.01]
##		  tuberculosis.distribution[:,1]=[0.95, 0.05]
##		  smoking.setDistributionParameters([0.5, 0.5])
##		  lung.distribution[:,0]=[0.99, 0.01]
##		  lung.distribution[:,1]=[0.9, 0.1]
##		  ou.distribution[:,0,0]=[1, 0]
##		  ou.distribution[:,0,1]=[0, 1]
##		  ou.distribution[:,1,0]=[0, 1]
##		  ou.distribution[:,1,1]=[0, 1]
##		  Xray.distribution[:,0]=[0.95, 0.05]
##		  Xray.distribution[:,1]=[0.02, 0.98]
##		  bronchitis.distribution[:,0]=[0.7, 0.3]
##		  bronchitis.distribution[:,1]=[0.4, 0.6]
##		  dyspnoea.distribution[{'bronchitis':0,'ou':0}]=[0.9, 0.1]
##		  dyspnoea.distribution[{'bronchitis':1,'ou':0}]=[0.2, 0.8]
##		  dyspnoea.distribution[{'bronchitis':0,'ou':1}]=[0.3, 0.7]
##		  dyspnoea.distribution[{'bronchitis':1,'ou':1}]=[0.1, 0.9]
##		  self.visit = visit
##		  self.tuberculosis = tuberculosis
##		  self.smoking = smoking
##		  self.lung = lung
##		  self.ou = ou
##		  self.Xray = Xray
##		  self.bronchitis = bronchitis
##		  self.dyspnoea = dyspnoea
##		  self.BNet = G
##	  
##	  def testStruct(self):
##		  N = 10000
##		  # sample the network N times
##		  cases = self.BNet.Sample(N)	 # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
##		  # create two new bayesian network with the same parameters as self.BNet
##		  G1 = bayesnet.BNet( 'Asia Bayesian Network2' )
##		  visit, smoking, tuberculosis, bronchitis, lung, ou, Xray, dyspnoea = [G1.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in 'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
##		  G1.InitDistributions()
##		  visit.setDistributionParameters([0.99, 0.01])
##		  tuberculosis.setDistributionParameters([0.99, 0.01])
##		  smoking.setDistributionParameters([0.5, 0.5])
##		  lung.setDistributionParameters([0.95, 0.05])
##		  ou.setDistributionParameters([0.94, 0.06])
##		  Xray.setDistributionParameters([0.89, 0.11])
##		  bronchitis.setDistributionParameters([0.55, 0.45])
##		  dyspnoea.setDistributionParameters([0.56, 0.44])
##		  # Test StructLearning
##		  struct_engine1 = GreedyStructLearningEngine(G1)
##		  struct_engine1.StructLearning(cases)
##		  print struct_engine1.BNet
	

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
		
		tol = 0.08
		assert(na.alltrue([na.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
			   for v in engine.BNet.all_v])), \
				" Learning does not converge to true values "
		print 'ok!!!!!!!!!!!!'

if __name__ == '__main__':
##	suite = unittest.makeSuite(SEMLearningTestCase, 'test')
##	runner = unittest.TextTestRunner()
##	runner.run(suite) 

	  suite = unittest.makeSuite(GreedyStructLearningTestCase, 'test')
	  runner = unittest.TextTestRunner()
	  runner.run(suite)	   
	
##	  suite = unittest.makeSuite(EMLearningTestCase, 'test')
##	  runner = unittest.TextTestRunner()
##	  runner.run(suite)
