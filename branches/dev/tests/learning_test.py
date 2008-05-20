#!/usr/bin/env python
"""
This module is the test for learning.py
"""

import unittest
import random
from copy import deepcopy


import numpy

from openbayes.inference import ConnexeInferenceJTree
from openbayes.learning import *
from openbayes import bayesnet, graph

class MLLearningTestCase(unittest.TestCase):
    '''ML Learning Test Case'''
    def setUp(self):
        # create a discrete network
        g = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [g.add_v(bayesnet.BVertex(nm, True, 2)) 
                      for nm in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            g.add_e(graph.DirEdge(len(g.e), *ep))
        g.init_distributions()
        c.set_distribution_parameters([0.5, 0.5])
        s.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
        r.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:, 0, 0] = [0.99, 0.01]
        w.distribution[:, 0, 1] = [0.1, 0.9]
        w.distribution[:, 1, 0] = [0.1, 0.9]
        w.distribution[:, 1, 1] = [0.0, 1.0]
        self.network = g
        
        # create a simple continuous network
        g2 = bayesnet.BNet('Gaussian Bayesian Network')
        a, b = [g2.add_v(bayesnet.BVertex(nm, False, 1)) 
                for nm in 'a b'.split()]
        for ep in [(a, b)]:
            g2.add_e(graph.DirEdge(len(g2.e), *ep))
        
        g2.init_distributions()
        a.set_distribution_parameters(mu=1.0, sigma=1.0)
        b.set_distribution_parameters(mu=1.0, sigma=1.0, wi=2.0)
        self.continuous_net = g2

    def test_ml(self):
        # sample the network 2000 times
        cases = self.network.sample(2000)
        
        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        g = deepcopy(self.network)
        
        g.init_distributions()
        
        # create an infeence engine
        engine = MLLearningEngine(g)
        
        # learn according to the test cases
        engine.learn_ml_params(cases)
        
        tol = 0.05
        assert(numpy.alltrue([numpy.allclose(v.distribution.cpt, \
               self.network.v[v.name].distribution.cpt, atol=tol) \
               for v in g.all_v])), \
                " Learning does not converge to true values "

class SEMLearningTestCase(unittest.TestCase):
    """
    Test sem learning
    """
    def setUp(self):
        # create a discrete network
        g = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [g.add_v(bayesnet.BVertex(nm, True, 2)) for \
                      nm in 'c s r w'.split()]
        for ep in [(c, r), (c, s), (r, w), (s, w)]:
            g.add_e(graph.DirEdge(len(g.e), *ep))
        g.init_distributions()
        c.set_distribution_parameters([0.5, 0.5])
        s.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
        r.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
        w.distribution[:, 0, 0] = [0.99, 0.01]
        w.distribution[:, 0, 1] = [0.1, 0.9]
        w.distribution[:, 1, 0] = [0.1, 0.9]
        w.distribution[:, 1, 1] = [0.0, 1.0]
        self.network = g 

    def test_sem(self):
        nbr_samples = 700
        # sample the network N times, delete some data
        # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]     
        cases = self.network.sample(nbr_samples)      
        for i in range(25):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?' 
        for i in range(3):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?' 
        g = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        for name in "c s r w".split():
            g.add_v(bayesnet.BVertex(name, True, 2))
        g.init_distributions()

        # Test SEMLearning
        struct_engine = SEMLearningEngine(g)
        struct_engine.sem_learning_approx(cases)
        struct_engine.save_in_file('./output/testSEM05.txt', g, \
                                 struct_engine.network, struct_engine)
        struct_engine.em_learning(cases, 10)
        struct_engine.save_in_file('./output/testSEM205.txt', g, \
                                 struct_engine.network, struct_engine)
        g1 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        for name in 'c s r w'.split():
            g1.add_v(bayesnet.BVertex(name, True, 2))
        g1.init_distributions()
        struct_engine1 = SEMLearningEngine(g1)
        struct_engine1.sem_learning(cases, 0)
        struct_engine1.save_in_file('./output/testSEM0.txt', g1, \
                                  struct_engine1.network, struct_engine1)
        struct_engine1.em_learning(cases, 10)
        struct_engine1.save_in_file('./output/testSEM20.txt', g1, \
                                  struct_engine1.network, struct_engine1)
        g2 = bayesnet.BNet('Water Sprinkler Bayesian Network2')
        for name in 'c s r w'.split():
            g2.add_v(bayesnet.BVertex(name, True, 2))
        g2.init_distributions()
        struct_engine2 = SEMLearningEngine(g2)
        struct_engine2.sem_learning(cases, 10)
        struct_engine2.save_in_file('./output/testSEM10.txt', g2, \
                                  struct_engine2.network, struct_engine2)
        struct_engine2.em_learning(cases, 10)
        struct_engine2.save_in_file('./output/testSEM210.txt', g2, \
                                  struct_engine2.network, struct_engine2)

class GreedyStructLearningTestCase(unittest.TestCase):
    """
    TEST ASIA
    """
    def setUp(self):
        # create the network
        g = bayesnet.BNet( 'Asia Bayesian Network' )
        
        visit, smoking, tuberculosis, bronchitis, lung, ou, xray, dyspnoea = \
        [g.add_v(bayesnet.BVertex( nm, True, 2)) for nm in \
        'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
        
        for ep in [(visit, tuberculosis), (tuberculosis, ou), (smoking, lung),
                   (lung, ou), (ou, xray), (smoking, bronchitis),
                   (bronchitis, dyspnoea), (ou, dyspnoea)]:
            g.add_e(graph.DirEdge(len(g.e), *ep))
        g.init_distributions()
        visit.set_distribution_parameters([0.99, 0.01])
        tuberculosis.distribution[:, 0] = [0.99, 0.01]
        tuberculosis.distribution[:, 1] = [0.95, 0.05]
        smoking.set_distribution_parameters([0.5, 0.5])
        lung.distribution[:, 0] = [0.99, 0.01]
        lung.distribution[:, 1] = [0.9, 0.1]
        ou.distribution[:, 0, 0] = [1, 0]
        ou.distribution[:, 0, 1] = [0, 1]
        ou.distribution[:, 1, 0] = [0, 1]
        ou.distribution[:, 1, 1] = [0, 1]
        xray.distribution[:, 0] = [0.95, 0.05]
        xray.distribution[:, 1] = [0.02, 0.98]
        bronchitis.distribution[:, 0] = [0.7, 0.3]
        bronchitis.distribution[:, 1] = [0.4, 0.6]
        dyspnoea.distribution[{'bronchitis':0, 'ou':0}] = [0.9, 0.1]
        dyspnoea.distribution[{'bronchitis':1, 'ou':0}] = [0.2, 0.8]
        dyspnoea.distribution[{'bronchitis':0, 'ou':1}] = [0.3, 0.7]
        dyspnoea.distribution[{'bronchitis':1, 'ou':1}] = [0.1, 0.9]
        self.network = g
    
    def test_struct(self):
        nbr_samples = 5000
        # sample the network N times
        # cases = [{'c':0,'s':1,'r':0,'w':1},{...},...]
        cases = self.network.sample(nbr_samples)    
        for i in range(25):
            case = cases[3*i]
            rand = random.sample(['visit', 'smoking', 'tuberculosis', \
                   'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'], 1)[0]
            case[rand] = '?' 
        for i in range(3):
            case = cases[3 * i]
            rand = random.sample(['visit', 'smoking', 'tuberculosis', \
                   'bronchitis', 'lung', 'ou', 'Xray', 'dyspnoea'], 1)[0]
            case[rand] = '?' 
        # create two new bayesian network with the same parameters as self.BNet
        g1 = bayesnet.BNet( 'Asia Bayesian Network2' )

        visit, smoking, tuberculosis, bronchitis, lung, ou, xray, dyspnoea = \
        [g1.add_v( bayesnet.BVertex( nm, True, 2 ) ) for nm in \
        'visit smoking tuberculosis bronchitis lung ou Xray dyspnoea'.split()]
        
        for start, end in [(visit, tuberculosis), (tuberculosis, ou), (smoking, lung), \
                   (lung, ou), (ou, xray), (smoking, bronchitis), \
                   (bronchitis, dyspnoea), (ou, dyspnoea)]:
            g1.add_e( graph.DirEdge(len(g1.e), start, end))
        g1.init_distributions()
##        tuberculosis.distribution[:,0]=[0.99, 0.01]
##        tuberculosis.distribution[:,1]=[0.95, 0.05]
##        smoking.set_distribution_parameters([0.5, 0.5])
##        lung.distribution[:,0]=[0.99, 0.01]
##        lung.distribution[:,1]=[0.9, 0.1]
##        ou.distribution[:,0,0]=[1, 0]
##        ou.distribution[:,0,1]=[0, 1]
##        ou.distribution[:,1,0]=[0, 1]
##        ou.distribution[:,1,1]=[0, 1]
##        Xray.distribution[:,0]=[0.946, 0.054]
##        Xray.distribution[:,1]=[0.0235, 0.9765]
##        bronchitis.distribution[:,0]=[0.7, 0.3]
##        bronchitis.distribution[:,1]=[0.4, 0.6]
##        dyspnoea.distribution[{'bronchitis':0,'ou':0}]=[0.907, 0.093]
##        dyspnoea.distribution[{'bronchitis':1,'ou':0}]=[0.201, 0.799]
##        dyspnoea.distribution[{'bronchitis':0,'ou':1}]=[0.322, 0.678]
##        dyspnoea.distribution[{'bronchitis':1,'ou':1}]=[0.132, 0.868]
        # Test StructLearning
        struct_engine = SEMLearningEngine(g1)
##        struct_engine.SEMLearningApprox(cases)
        struct_engine.em_learning(cases, 10)
        struct_engine.save_in_file('./output/asiaEM.txt', g1, \
                                 struct_engine.network, struct_engine)
        casestemp = self.network.sample(1000)
##        for i in range(25):
##            case = casestemp[3*i]
##            rand = random.sample(['visit', 'smoking', 'tuberculosis', 
##                                  'bronchitis', 'lung', 'ou', 'Xray', 
##                                  'dyspnoea'],1)[0]
##            case[rand] = '?' 
##        for i in range(3):
##            case = casestemp[3*i]
##            rand = random.sample(['visit', 'smoking', 'tuberculosis', 
##                                  'bronchitis', 'lung', 'ou', 'Xray', 
##                                  'dyspnoea'],1)[0]
##            case[rand] = '?'
        cases21 = []
        cases20 = []
        i = 1
        j = 1
        for cas in casestemp:
            if cas['ou'] == 1:
                del cas['ou']
                cases21.append(cas)
                i = i + 1
            elif cas ['ou'] == 0:
                del cas['ou']
                cases20.append(cas)
                j = j + 1
        ie = ConnexeInferenceJTree(struct_engine.network)
##        #print struct_engine.engine.marginalise('tuberculosis')
##        print ie.marginalise('tuberculosis')
##        print ie.marginalise('lung')
##        print ie.marginalise('dyspnoea')
##        ie.set_obs(cases2[0])
##        print cases2[0]
##        print ie.marginalise('tuberculosis')
##        print ie.marginalise('lung')
##        print ie.marginalise('dyspnoea')
                
##        g_copy = G1.copy()
        f = open('./output/testvalidationasia1.txt', 'w')
        nbr1 = 0
        for truc in cases21:
            cases3 = {}
            ie.initialization()
##            for v in ie.BNet.all_v:
##                cpt = Gcopy.v[v.name].distribution.convert_to_cpt()
##                ie.BNet.v[v.name].distribution.set_parameters(cpt)
            for iter_ in truc:
                if truc[iter_] != '?':
                    cases3[iter_] = truc[iter_]
            ie.set_obs(cases3)
            if ie.marginalise('ou')[1] < 0.055:
                f.write(str((0, ie.marginalise('ou')[1])))
            else:
                f.write(str((1, ie.marginalise('ou')[1])))
                nbr1 = nbr1 + 1
            f.write(str(('nombre de 1: ', nbr1)))
            f.write('\n')
        pourcentage = nbr1 * 100 / len(cases21)
        f.write(str((pourcentage)))
        f.close()

        f = open('./output/testvalidationasia0.txt', 'w')
        nbr0 = 0
        for truc in cases20:
            cases3 = {}
            ie.initialization()
##            for v in ie.BNet.all_v:
##                cpt = Gcopy.v[v.name].distribution.convert_to_cpt()
##                ie.BNet.v[v.name].distribution.set_parameters(cpt)
            for iter_ in truc:
                if truc[iter_] != '?':
                    cases3[iter_] = truc[iter_]
            ie.set_obs(cases3)
            if ie.marginalise('ou')[1] < 0.055:
                f.write(str((0, ie.marginalise('ou')[1])))
                nbr0 = nbr0 + 1
            else:
                f.write(str((1, ie.marginalise('ou')[1])))
            f.write(str(('nombre de 0: ', nbr0)))
            f.write('\n')
        pourcentage = nbr0 * 100 / len(cases20)
        f.write(str((pourcentage)))
        f.close()

class EMLearningTestCase(unittest.TestCase):
    """
    This test the EMLearning Engine
    """
    def setUp(self):
        # create a discrete network
        network = bayesnet.BNet('Water Sprinkler Bayesian Network')
        c, s, r, w = [network.add_v(bayesnet.BVertex(nm, True, 2)) for nm \
                   in 'c s r w'.split()]
        for ep in [(c, r), (c, s)]:
            network.add_e(graph.DirEdge(len(network.e), *ep))
        network.init_distributions()
        c.set_distribution_parameters([0.5, 0.5])
        s.set_distribution_parameters([0.5, 0.9, 0.5, 0.1])
        r.set_distribution_parameters([0.8, 0.2, 0.2, 0.8])
        w.set_distribution_parameters([0.5, 0.5])
        self.network = network
    
    def test_em(self):
        # sample the network 2000 times
        cases = self.network.sample(200)
        # delete some observations
        for i in range(50):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?' 
        for i in range(5):
            case = cases[3 * i]
            rand = random.sample(['c', 's', 'r', 'w'], 1)[0]
            case[rand] = '?'
        # create a new BNet with same nodes as self.BNet but all parameters
        # set to 1s
        bnet = deepcopy(self.network)
        
        bnet.init_distributions()
        
        engine = EMLearningEngine(bnet)
        engine.em_learning(cases, 10)
        engine.save_in_file('./output/testerdddddddd.txt', bnet, None, engine)
        tol = 0.08
        self.assert_(numpy.alltrue([numpy.allclose(v.distribution.cpt,
               self.network.v[v.name].distribution.cpt, atol=tol) 
               for v in engine.network.all_v]),
                " Learning does not converge to true values")

if __name__ == "__main__":
    unittest.main()
