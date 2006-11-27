import graph
import bayesnet
import distributions
import inference
import random

class EMLearningEngine:
    """ EM learning algorithm
    Learns the parameters of a known bayesian structure from incomplete data.
    """   
    BNet = none # The underlying bayesian network
    
    def __init__(self, BNet):
        self.BNet = BNet
    
    def EMLearning(self, cases, max_iter):
        """ cases = [{'c':0,'s':1,'r':'?','w':1},{...},...]
        Put '?' when the data is unknown.
        Will estimate  the '?' by inference.
        """
        iter = 0
        old = 'not yet'
        new = 0
        precision = 0.001
        while self.hasntConverged(old, new, precision) and iter < max_iter:
            iter += 1
            old = {}
            new = {}
            i = 0
            j = 0
            for v in self.BNet.v.values():
                old[j]=v.distribution.cpt
                j += 1
            old_param = new_param
            engine = JoinTree(self.BNet)
            #engine = MCMCEngine(self.BNet)
            self.LearnEMParams(self, cases, engine)
            # reinitialize the JunctionTree to take effect of new parameters learned
            engine.Initialization()
            engine.GlobalPropagation()
            for v in self.BNet.v.values():
                new[j]=v.distribution.cpt
                j += 1
            
    
    def LearnEMParams(self, cases, engine):
        """ First part of the algorithm : Estimation of the unknown 
        data. (E-part)
        """  
        for v in self.BNet.v.values():
                v.distribution.initializeCounts()
        for case in cases:
            if not self.hasUnknown(case):
                for v in self.BNet.v.values():
                    if v.distribution.isAdjustable: #NECESSAIRE??
                            v.distribution.incrCounts(case)
            else:
                new_dict={}
                for key in case.iterkeys():
                    if case[key] != '?':
                        engine.setObs({key:case[key]})
                        new_dict[key]=case[key] #Contient tous les éléments de case sauf l'élément inconnu
                for key in case.iterkeys():
                    if case[key] == '?': #NE MARCHE QU'AVEC UNE DONNEE MANQUANTE PAR CASE!!(pour l'instant)
                        likelihood = engine.Marginalise(key).cpt
                        for v in self.BNet.v.values():
                            if v.distribution.isAdjustable: #NECESSAIRE??
                                for state in range(v.nvalues):
                                    new_dict[key]=state
                                    v.distribution.addToCounts(new_dict, likelihood[state])
                                    del new_dict[key]
        
        """ Second part of the algorithm : Estimation of the parameters. 
        (M-part)
        """
        for v in self.BNet.v.values():
            if v.distribution.isAdjustable:
                v.distribution.setCounts()
                v.distribution.normalize(dim=v.name)

    def hasUnknown(self, case):
        result = False
        for key in case.iterkeys():
            if case[key] == '?':
                result = True
                break
        return result
    
    def hasntConverged(self, old, new, precision):
        #Voir avec 
        if old == 'not yet':
            return True   
        else:
            final = 0
            m = 0
            for v in self.BNet.v.values():
                new_dist = new[m]
                if len(v.distribution.family) != 1:
                    for k in range(len(v.distribution.parents)):
                        new_dist = new_dist[0]
                old_dist = old[m]
                if len(v.distribution.family) != 1:
                    for k in range(len(v.distribution.parents)):
                        old_dist = old_dist[0]
                difference = new_dist-old_dist #EST-IL POSSIBLE QUE LES NOEUDS AIENT CHANGE DE PLACE??
                result = max(max(abs(difference)),final)
                m += 1
            if result > precision:
                return True
            else:
                return False
    
    

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
        
        self.EMLearning(cases, 10)
        
        tol = 0.05
        assert(na.alltrue([na.allclose(v.distribution.cpt, self.BNet.v[v.name].distribution.cpt, atol=tol) \
               for v in G.all_v])), \
                " Learning does not converge to true values "