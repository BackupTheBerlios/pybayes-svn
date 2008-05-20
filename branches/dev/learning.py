########################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network 
## library
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
########################################################################

import copy
import time
import logging
import math

import numpy

import openbayes.graph as graph
import openbayes.readexcel as readexcel
from openbayes.inference import ConnexeInferenceJTree 

__all__ = ["LearningEngine","MLLearningEngine", "EMLearningEngine", "SEMLearningEngine" ]
# show INFO messages
#logging.basicConfig(level= logging.INFO)
#uncomment the following to remove all messages
logging.basicConfig(level = logging.NOTSET)


class LearningEngine:
    """
    This class implements the learn engine
    """
    def __init__(self, network):
        self.network = network #.copy()#Xue
        self.engine = ConnexeInferenceJTree(self.network)    
    
    def read_file(self, file_):
        """ file is an excel file. This method reads the file and return
        a list of dictionaries (ex: [{'c':0,'s':1,'r':'?','w':1},{...},...]).
        Each dictionary represents a row of the excell table (= a case for the
        learning methods)
        """
        xl = readexcel.readexcel(file_)
        sheetnames = xl.worksheets()
        cases = []
        for sheet in sheetnames:
            for row in xl.getiter(sheet):
                cases.append(row)
        return cases
    
    def save_in_file(self, file_, g_initial=None, g_learned=None, engine=None):
        """
        This save the learned graph to a file
        """
        f = open(file_, 'w')
        if g_initial:
            f.write('Initial structure:' + '\n' + '\n')
            s = str((g_initial))
            f.write(s + '\n' + '\n')
        if g_learned:
            f.write('Learned structure:' + '\n' + '\n')
            s = str((g_learned))
            f.write(s + '\n' + '\n')
        if engine:
            for node in engine.network.all_v:
                pa = []
                for i in node.distribution.parents:
                    pa.append(i.name)
                if len(pa) > 0:
                    # TODO Figure out what the hell is this combination
                    combi = self.combinations(pa)
                    for cas in combi:
                        s = str(('node: ', node.name, ', parents: ', cas, ','
                                'distribution: ', node.distribution[cas]))
                        f.write(s + '\n')
                else:
                    s = str(('node: ', node.name, ', distribution: ', \
                            node.distribution))
                    f.write(s + '\n')
        f.close()  

class MLLearningEngine(LearningEngine):
    """
    This is a concrete implementation of the Machine Learning Engine
    """
    def __init__(self, network):
        LearningEngine.__init__(self, network)
    
##    def combinations(self, vertices):
##        ''' vertices is a list of BVertex instances
##
##        in the case of the water-sprinkler BN :
##        >>> combinations(['c','r','w']) ##Xue
##        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
##        '''
##        if len(vertices) > 1:
##            list_comb = self.combinations(vertices[:-1])
##            new_list_comb = []
##            for el in list_comb:
##                for i in range(self.network.v[vertices[-1]].nvalues):
##                    temp = copy.copy(el)
##                    temp[self.network.v[vertices[-1]].name]=i
##                    new_list_comb.append(temp)
##            return new_list_comb
##        else:
##            return [{self.network.v[vertices[0]].name:el} for 
##                            el in range(self.network.v[vertices[0]].nvalues)]
        
    def learn_ml_params(self, cases, augmented=0):
        """ Learn and set the parameters of the network to the ML estimate
        contained in cases.
       
        Warning: this is destructive, it does not take any prior parameters
                 into account. Assumes that all evidence is specified.
        """
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                if augmented:
                    # sets all a_ij and b_ij to equivalent sample size #momo
                    v.distribution.initialize_augmented_eq()
                v.distribution.initialize_counts()
####                if augmented:
####                    # set the initial Pr's to a_ij/(a_ij+b_ij)
####                    v.distribution.normalize(dim=v.name) 
         
        for case in cases:
            assert(set(case.keys()) == set(self.network.v.keys())), \
                   "Not all values of 'case' are set"
            for v in self.network.v.values():
                if v.distribution.is_adjustable:
                    v.distribution.incr_counts(case)
####                    if augmented:
####                        v.distribution.set_augmented_and_counts() #added
           
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                if augmented:
                    v.distribution.set_augmented_and_counts() #added
                else:
                    v.distribution.set_counts()
                v.distribution.normalize(dim=v.name)

class EMLearningEngine(LearningEngine):
    """ EM learning algorithm
    Learns the parameters of a known bayesian structure from incomplete data.
    """

    def __init__(self, network):
        LearningEngine.__init__(self, network)
        #self.engine = ConnexeInferenceJTree(network)

    def em_learning(self, cases, max_iter, initialize=False, precision = 0.05):
        """ cases = [{'c':0,'s':1,'r':'?','w':1},{...},...]
        Put '?' when the data is unknown.
        Will estimate  the '?' by inference.

        This function return the number of iteration done.
        """
        # if wanted the network is initialized (previously learned CPT's are
        # resetted, the previous learned data is lost)
        if initialize:
            for v in self.network.v.values():
                if v.distribution.is_adjustable:
                    # sets all a_ij and b_ij to equivalent sample size
                    v.distribution.initialize_augmented_eq() 
                    # sets all counts to zero
                    v.distribution.initialize_counts() 
                    v.distribution.set_augmentedCounts()
                    # set the initial Pr's to a_ij/(a_ij+b_ij)
                    v.distribution.normalize(dim=v.name) 

        iter_ = 0
        old = None
        new = self.network
        while self.hasnt_converged(old, new, precision) and iter_ < max_iter:
            iter_ += 1
            old = copy.deepcopy(new)
            self.learn_em_params(cases) ##Xue
            # reinitialize the JunctionTree to take effect of new 
            # parameters learned
            self.engine.initialization()
            # self.engine.global_propagation()
            new = copy.deepcopy(self.network)
        return iter_

    def learn_em_params_wm(self, cases):
        # Initialise the counts of each vertex
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                v.distribution.initialize_counts() # sets all counts to zero

        # First part of the algorithm : Estimation of the unknown 
        # data. (E-part)
        for case in cases:
            # assert(set(case.keys()) == set(self.network.v.keys())), 
            # "Not all values of 'case' are set"

            known = dict() # will contain al the known data of the case
            for key in case.iterkeys():
                if case[key] != '?':
                    known[key] = case[key]

            for v in self.network.v.values():
                if v.distribution.is_adjustable:
                    names = [parent.name for parent in v.family[1:]]
                    nvals = [parent.nvalues for parent in v.family[1:]]
                    names.append(v.name)
                    nvals.append(v.nvalues)
                    self.calc_expectation(v, known, names, nvals)

        # Second part of the algorithm : Estimation of the parameters. 
        # (M-part)
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                v.distribution.set_augmented_and_counts()
                v.distribution.normalize(dim=v.name)

    def calc_expectation(self, vertex, known, names, nvals, cum_prob=1):
        """	calculate and set expectations of node v
            based on chain rule: Pr(names | known) = Pr(names[0]|names[1] .. 
            names[end], known).Pr(names[1]|names[2]..names[end], known)...

            in:
            vertex      : node to calc and set
            known       : the known nodes of the net
            names       : the names of the nodes necessary to calcute P(X_v = 
                          val_v, pa_v = val_ij| known, prior_CPT)
            nvals       : the nvalues ...
            cumPr       : the cumulative chance of the chain rule

            out:
            Counts of node v are adjusted based on the known nodes
        """
        if cum_prob == 0:
            # chance will be remain zero so no use in calculating any further
            return

        newnames = list(names)
        name = newnames.pop()
        newnvals = list(nvals)
        nval = newnvals.pop()

        if known.has_key(name):
            # Value of P(name=X | pa_v(i+1) ... , known) is 1 because of the 
            # known information
            if len(newnames) == 0:
                vertex.distribution.add_to_counts(known, cum_prob)
            else:
                self.calc_expectation(vertex, known, newnames, newnvals, cum_prob)
        else:
            # Only Initializiatino is enough to reset because set_obs 
            # calls global_propagation
            self.engine.initialization() 
            self.engine.set_obs(known)
            marg = self.engine.marginalise(name)

            for value in range(nval):
                newknown = dict(known)
                newknown[name] = value
                this_pr = marg[value]
                if not(str(this_pr) == 'nan' or this_pr == 0):
                    if len(newnames) == 0:
                        vertex.distribution.add_to_counts(newknown, cum_prob*this_pr)
                    else:
                        self.calc_expectation(vertex, newknown, newnames, \
                                             newnvals, cum_prob*this_pr)


    def learn_em_params(self, cases):
        """ 
        First part of the algorithm : Estimation of the unknown 
        data. (E-part)
        """ 
        # Initialise the counts of each vertex
        for v in self.network.v.values():
            v.distribution.initialize_counts()##Xue
        for case in cases:
            known = {} # will contain all the known data of case
            unknown = [] # will contain all the unknown keys of case
            for key in case.iterkeys():
                # It's the only part of code you have to change if you want 
                # to have another 'unknown sign' instead of '?'
                if case[key] != '?': 
                    known[key] = case[key]
                else:
                    unknown.append(key)
            # Then all the data is known -> proceed as learn_ml_params 
            # (inference.py)
            if len(case) == len(known):
                for v in self.network.v.values():
                    if v.distribution.is_adjustable: 
                        v.distribution.incr_counts(case)
            else:
                # Give a dictionary list of all the possible states of the 
                # unknown parameters
                states_list = self.combinations(unknown) 
                # Give a list with the likelihood to have the states in 
                # states_list
                likelihood_list = self.determine_likelihood(known, 
                                                           states_list) 				  
                for j, index_unknown in enumerate(states_list):
                    index = copy.copy(known)
                    index.update(index_unknown)
                    for v in self.network.v.values():
                        if v.distribution.is_adjustable:
                            v.distribution.add_to_counts(index, 
                                                         likelihood_list[j]) 
        # Second part of the algorithm : Estimation of the parameters. 
        # (M-part)
        for v in self.network.v.values():
            if v.distribution.is_adjustable:
                v.distribution.set_counts()
                v.distribution.normalize(dim=v.name)

    def determine_likelihood(self, known, states_list):
        """ 
        Give a list with the likelihood to have the states in states_list
        I think this function could be optimized
        """		   
        likelihood = []
        for states in states_list:
            # states = {'c':0,'r':1} for example (c and r were unknown in 
            # the beginning)
            like = 1
            temp_dic = {}
            copy_states = copy.copy(states)
            for key in states.iterkeys():
                # It has to be done for all keys because we have to set every 
                # observation but key to compute the likelihood to have key in
                # his state. The multiplication of all the likelihoods gives the
                # likelihood to have states.				   
                temp_dic[key] = (copy_states[key])
                del copy_states[key]
                if len(copy_states) != 0:
                    copy_states.update(known) 
                    self.engine.set_obs(copy_states)
                else:
                    # Has to be done at each iteration because of the 
                    # self.engine.initialization() below
                    self.engine.set_obs(known)  
                # print 'finished	to try setobs'
                # like = like*self.network.v[key].distribution.convert_to_cpt()
                # [temp_dic[key]]
                like = self.engine.marginalise(key)[temp_dic[key]]
                #like = like*self.engine.extract_cpt(key)[temp_dic[key]]
                if str(like) == 'nan':
                    like = 0

                copy_states.update(temp_dic)			   
                del temp_dic[key]
                self.engine.initialization()
            likelihood.append(like)
        return likelihood

    def determine_likelihood_approx(self, known, states_list):
        """ 
        Give a list with the likelihood to have the states in states_list.
        6 to 10 time faster than the determine_likelihood function above, but 
        has to be tested more... comme algo loopy belief propagation: pearl 
        wikipedia
        """		   
        likelihood = []
        for states in states_list:
            # states = {'c':0,'r':1} for example (c and r were unknown in the 
            # beginning)
            like = 1
            parents_state = copy.copy(known)
            parents_state.update(copy.copy(states))
            for key in states.iterkeys():
                cpt = self.network.v[key].distribution[parents_state]
                like = like * cpt				
            likelihood.append(like)
        return likelihood

    def combinations(self, vertices):
        ''' vertices is a list of BVertex instances

        in the case of the water-sprinkler BN :
        >>> combinations(['c','r','w']) ##Xue
        [{'c':0,'r':0,'w':0},{'c':0,'r':0,'w':1},...,{'c'=1,'r':1,'w':1}]
        '''
        if len(vertices) > 1:
            list_comb = self.combinations(vertices[:-1])
            new_list_comb = []
            for el in list_comb:
                for i in range(self.network.v[vertices[-1]].nvalues):
                    temp = copy.copy(el)
                    temp[self.network.v[vertices[-1]].name] = i
                    new_list_comb.append(temp)
            return new_list_comb
        else:
            return [{self.network.v[vertices[0]].name:el} for el \
                    in range(self.network.v[vertices[0]].nvalues)]

    def hasnt_converged(self, old, new, precision):
        '''
        Return true if the difference of distribution of at least one vertex 
        of the old and new network is bigger than precision
        '''
        if not old :
            return True	  
        else:
            return not	numpy.alltrue([numpy.allclose(v.distribution, 
                                   new.v[v.name].distribution, 
                                   atol=precision) for v in old.v.values()])

   

class SEMLearningEngine(LearningEngine, EMLearningEngine):
    """ Greedy Structural learning algorithm
    Learns the structure of a bayesian network from known parameters and an 
    initial structure.
    """   
##    network = None # The underlying bayesian network
##    engine = None
    
    def __init__(self, network):
        LearningEngine.__init__(self, network)
        #self.engine = ConnexeInferenceJTree(self.network)
        self.converged = False
        #[{'NCEP':'BMI'}, {'age':'comp'}, {'TABAC':'sexe'}, {'BMI':'HOMA'}, 
        # {'comp':'albu'}, {'sexe':'HDL'}, {'NCEP':'HDL'}, {'HbA1c':'NHDL'}]
        self.inverted = []
        self.changed = []

    def sem_learning(self, cases, alpha=0.5, max_iter = 30):
        """Structural EM for optimal structure and parameters if some of the 
        data is unknown (put '?' for unknown data).

        This function return the number of iterations done
        """
        nbr_cases = len(cases)
        iter_ = 0
        self.network.init_distributions()
        self.em_learning(cases, 15)
        while (not self.converged) and iter_ < max_iter:
            #First we estimate the distributions of the initial structure
            #self.network.init_distributions()
            #self.em_learning(cases, 15)
            #Then we find a better structure in the neighborhood of self.network
            self.learn_struct(cases, nbr_cases, alpha, False)
            iter_ += 1
        return iter_

    def sem_learning_approx(self, cases, alpha=0.5, max_iter=30):
        """Structural EM for optimal structure and parameters if some of the 
        data is unknown (put '?' for unknown data).

        This methods returns the number of iterations done
        """
        nbr_cases = len(cases)
        iter_ = 0
        #self.network.init_distributions()
        #self.em_learning(cases, 15)
        while (not self.converged) and iter_ < max_iter:
            #First we estimate the distributions of the initial structure
            self.network.init_distributions()
            self.em_learning(cases, 15)
            #Then we find a better structure in the neighborhood of self.network
            self.learn_struct(cases, nbr_cases, alpha, True)
            iter_ += 1
        self.em_learning(cases, 15)
        return iter_
    
    def global_bic_score(self, nbr_cases, cases, alpha = 0.5, approx = 0): #Xue
        '''Computes the BIC score of self.network'''
        score = 0
        for v in self.network.all_v:
            cpt_matrix = v.distribution.convert_to_cpt()
            dim = self.network.dimensions(v)
            score = score + self.score_bic(nbr_cases, dim, self.network, v, cases, 
                                          alpha, approx) 
        return score

    def learn_struct(self, cases, nbr_cases, alpha, approx):
        """Greedy search for optimal structure (all the data in cases are 
        known). It will go through every node of the network. At each node, it 
        will delete every outgoing edge, or add every possible edge, or 
        reverse every possible edge. It will compute the BIC score each time 
        and keep the network with the highest score.
        """
        g_initial = self.network.copy()
        engine_init = SEMLearningEngine(g_initial)
        g_best = self.network.copy()     
        prec_var_score = 0
        invert = {}
        change = {}
        
        for v in self.network.all_v:
            g = copy.deepcopy(engine_init.network)
            edges = copy.deepcopy(g.v[v.name].out_e)
            temp = {}
            # delete the outgoing edges
            while edges:
                edge = edges.pop(0)
                # node is the child node, the only node for which the cpt 
                # table changes               
                node = edge._v[1] 
                dim_init = g_initial.dimensions(node)
                score_init = engine_init.score_bic(nbr_cases, dim_init, g_initial, \
                             g_initial.v[node.name], cases, alpha, approx)
                self.change_struct('del', edge) #delete the current edge
                self.set_new_distribution(engine_init.network, node, cases, approx)
                dim = self.network.dimensions(node)
                score = self.score_bic(nbr_cases, dim, self.network, 
                                      self.network.v[node.name],
                                      cases, alpha, approx)
                var_score = score - score_init
                if var_score > prec_var_score:
                    change = {}
                    invert = {}
                    change[v.name] = node.name
                    prec_var_score = var_score
                    g_best = self.network.copy()
                self.network = g_initial.copy()
            
            # Add all possible edges
            g = copy.deepcopy(engine_init.network)
            nodes = []
            for node in g.all_v:
                if (not(node.name in [vv.name for vv in self.network.v[v.name].out_v])) and \
                    (not (node.name == v.name)):
                    nodes.append(node)
            while nodes:
                node = nodes.pop(0)
                if g.e.keys():
                    edge = graph.DirEdge(max(g.e.keys()) + 1, \
                           self.network.v[v.name], self.network.v[node.name])
                else:
                    edge = graph.DirEdge(0, self.network.v[v.name], \
                           self.network.v[node.name])
                self.change_struct('add', edge)
                if self.network.HasNoCycles(self.network.v[node.name]):
                    dim_init = engine_init.network.dimensions(node)
                    score_init = engine_init.score_bic(nbr_cases, dim_init, g_initial, \
                                 g_initial.v[node.name], cases, alpha, approx)
                    self.set_new_distribution(engine_init.network, node, 
                                            cases, approx)
                    dim = self.network.dimensions(node)
                    score = self.score_bic(nbr_cases, dim, self.network, \
                                          self.network.v[node.name], cases, \
                                          alpha, approx)
                    var_score = score - score_init
                    if var_score > prec_var_score:
                        change = {}
                        invert = {}
                        change[v.name] = node.name
                        prec_var_score = var_score
                        g_best = self.network.copy()
                self.network = g_initial.copy()
        
            # Invert all possible edges
            g = copy.deepcopy(g_initial)
            edges = copy.deepcopy(g.v[v.name].out_e)
            while edges:
                edge = edges.pop(0)
                node = self.network.v[edge._v[1].name] #node is the child node
                temp[v.name] = node.name
                if temp not in self.inverted:
                    dim_init1 = g_initial.dimensions(node)
                    score_init1 = engine_init.score_bic(nbr_cases, dim_init1, g_initial,
                                  g_initial.v[node.name], cases, alpha, approx)
                    self.change_struct('del', edge)
                    self.set_new_distribution(engine_init.network, node, \
                                            cases, approx)
                    dim1 = self.network.dimensions(node)
                    score1 = self.score_bic(nbr_cases, dim1, self.network, \
                             self.network.v[node.name], cases, alpha, approx)
                    g_invert = self.network.copy() 
                    engine_invert = SEMLearningEngine(g_invert)  
                    inverted_edge = graph.DirEdge(max(g.e.keys()) + 1,
                                                  self.network.v[node.name], 
                                                  self.network.v[v.name])
                    self.change_struct('add', inverted_edge)
                    if self.network.HasNoCycles(self.network.v[node.name]):
                        dim_init = g_initial.dimensions(v)
                        score_init = engine_init.score_bic(nbr_cases, dim_init, \
                                     g_initial, g_initial.v[v.name], cases, \
                                     alpha, approx)
                        self.set_new_distribution(engine_invert.network, v, \
                                                cases, approx)
                        dim = self.network.dimensions(v)
                        score = self.score_bic(nbr_cases, dim, self.network, \
                                self.network.v[v.name], cases, alpha, approx)
                        var_score = score1 - score_init1 + score - score_init
                        #+ 5 is to avoid recalculation due to round errors
                        if var_score > prec_var_score + 5: 
                            invert = {}
                            change = {}
                            invert[node.name] = v.name
                            prec_var_score = var_score
                            g_best = self.network.copy()
                    self.network = g_initial.copy()
        
        #self.network is the optimal graph structure
        if prec_var_score == 0:
            self.converged = True
        self.network = g_best.copy()
        self.inverted.append(invert)
        self.changed = []
        self.changed.append(change)
        #self.engine = ConnexeInferenceJTree(self.network)
    
    def set_new_distribution(self, g_initial, node, cases, approx):
        '''Set the new distribution of the node node. The other distributions
        are the same as G_initial (only node has a new parent, so the other 
        distributions don't change). Works also with incomplete data'''
        self.network.init_distributions()
        for v in g_initial.all_v:
            if v.name != node.name:
                cpt = g_initial.v[v.name].distribution.convert_to_cpt()
                self.network.v[v.name].distribution.set_parameters(cpt)
            else:
                if self.network.v[node.name].distribution.is_adjustable:
                    self.network.v[node.name].distribution.initialize_counts()
                    for case in cases :
                        known = {} # will contain all the known data of case
                        # will contain all the unknown keys of case
                        unknown = [] 
                        for key in case.iterkeys():
                            # It's the only part of code you have to change 
                            # if you want to have another 'unknown sign' 
                            # instead of '?'
                            if case[key] != '?': 
                                known[key] = case[key]
                            else:
                                unknown.append(key)
                        # Then all the data is known -> proceed as 
                        # learn_ml_params (inference.py)
                        if len(case) == len(known): 
                            self.network.v[node.name].distribution.incr_counts(case)
                        else:
                            states_list = self.combinations(unknown) 
                            # Give a dictionary list of all the possible 
                            # states of the unknown parameters
                            if approx:
                                # Give a list with the likelihood to have the
                                # states in states_list
                                likelihood_list = self.determine_likelihood_approx(known, states_list)                
                            else:
                                likelihood_list = self.determine_likelihood(known, states_list)
                            for j, index_unknown in enumerate(states_list):
                                index = copy.copy(known)
                                index.update(index_unknown)
                                self.network.v[node.name].distribution.add_to_counts(index, likelihood_list[j])
                    self.network.v[node.name].distribution.set_counts()
                    self.network.v[node.name].distribution.normalize(dim=node.name)
    
    def score_bic (self, nbr_cases, dim, g, node, data, alpha, approx):
        ''' This function computes the BIC score of one node.
        N is the size of the data from which we learn the structure
        dim is the dimension of the node, = (nbr of state - 1)*nbr of state of the parents
        data is the list of cases
        return the BIC score
        Works also with incomplete data!
        '''
        score = self.for_bic(g, data, node, approx)
        score = score - alpha * dim * math.log(nbr_cases)
        return score

    def for_bic(self, g, cases, node, approx):
        ''' Computes for each case the probability to have node and his parents
        in the case state, take the log of that probability and add them.'''
        score = 0
        for case in cases :
            cpt = 0 
            known = {} # will contain all the known data of case
            unknown = [] # will contain all the unknown data of case
            for key in case.iterkeys():
                # It's the only part of code you have to change if you want
                # to have another 'unknown sign' instead of '?'
                if case[key] != '?': 
                    known[key] = case[key]
                else:
                    unknown.append(key)
            if len(case) == len(known): # Then all the data is known
                cpt = node.distribution[case]
            else:
                # Give a dictionary list of all the possible states of the 
                # unknown parameters
                states_list = self.combinations(unknown) 
                if approx:
                    # Give a list with the likelihood to have the states in 
                    # states_list 
                    likelihood_list = self.determine_likelihood_approx(known, states_list)       
                else:
                    likelihood_list = self.determine_likelihood(known, states_list)
                for j, index_unknown in enumerate(states_list):
                    index = copy.copy(known)
                    index.update(index_unknown)
                    cpt = cpt + likelihood_list[j] * node.distribution[index]
            if cpt == 0: # To avoid log(0)
                cpt = math.exp(-700)
            score = score + math.log(cpt)
        return score

    def change_struct(self, change, edge):
        """Changes the edge (add, remove or reverse)"""
        if change == 'del':
            self.network.del_e(edge)
        elif change == 'add':
            self.network.add_e(edge)
        elif change == 'inv':
            self.network.inv_e(edge)
        else:
            raise ValueError("The asked change of structure is not possible."
                             " Only 'del' for delete, 'add' for add, and "
                             "'inv' for invert")


