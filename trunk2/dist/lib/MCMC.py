import unittest

import graph.graph as graph

class MCMCEngine(graph.Graph):
        """ Implementation of MCMC (aka Gibbs Sampler), as described on p.517 of Russell and Norvig
        """
        def __init__(self, BNet, cut=100):
            graph.Graph.__init__(BNet.name)
            self.BNet = Bnet
            self.cut = cut
            self.evidence = {}
        
        def SetObs(self, v, val):
            """ Incorporate new evidence 
            """
            self.evidence = dict((vi,vali) for vi,vali in zip(v,val))
                
        def Marginalise(self, v, N):
            """ Compute the Pr(v) where v is a variable name, N is the number of iterations of MCMC to perform.
            """
            # the return distribution
            vDist = RawCPT(v, (self.BNet.v[v].nvalues,1))
            nonEvidence = []
            # find out which values are not specified in evidence
            for vv in self.BNet.v.values():
                if not self.evidence.has_key(vv.name): nonEvidence.append(vv.name)
            # CHECK: this copy is deep enough
            state = copy.copy(self.evidence)
            for vname in nonEvidence:
                # CHECK: legal values are 0 - nvalues-1
                state[vname] = random.randint(0, self.BNet.v[vname].nvalues-1)
            for i in range(N):
                if i > self.cut:
                        # RESEARCH: how to index into CPT
                        vDist[state[v]] += 1
                for vname in nonEvidence:
                        state[vname] = self.sampleGivenMB(self.BNet.v[vname], state)
            # CHECK: how do i normalize? for now assume that RawCPT now has a makecpt() method
            vDist.makecpt()
            return vDist
        
        def sampleGivenMB(self, v, state):
            MBval = RawCPT(v.name, (v.nvalues,1))
            children = v.out_v()
            index = {}
            for vert in [v,v.in_v()]:
                    index[vert.name] = state[vert.name]
            childrenAndIndex = []
            for child in children:
                cindex = {}
                for cvert in [child,child.in_v()]:
                    cindex[cvert.name] = state[cvert.name]       
                childrenAndIndex.append((child,cindex))
            #OPTIMIZE: could vectorize this code
            for value in range(v.nvalues):
                index[v.name] = value
                cindex[v.name] = value
                MBval[value] = v[index]
                for child,cindex in childrenAndIndex:
                    MBval[value] *= child[cindex]
            MBval.makecpt()                
            val = MBval.sample()
            return val

        
        
class MCMCTestCase(unittest.TestCase):
    """ MCMC unit tests.
    """
    def setUp(self):
        G = BNet('Water Sprinkler Bayesian Network')
        c,s,r,w = [G.add_v(BVertex(nm,2,True)) for nm in 'c s r w'.split()]
        for ep in [(c,r), (c,s), (r,w), (s,w)]:
            G.add_e(graph.DirEdge(len(G.e), *ep))
        G.InitCPTs()
        c.setCPT([0.5, 0.5])
        s.setCPT([0.5, 0.9, 0.5, 0.1])
        r.setCPT([0.8, 0.2, 0.2, 0.8])
        w.setCPT([1, 0.1, 0.1, 0.01, 0.0, 0.9, 0.9, 0.99])
        self.engine = MCMCEngine(G,cut=100)
        self.c = c
        self.s = s
        self.r = r
        self.w = w
        
    def testUnobserved(self):
        """ Compute and check the probability of c=true and r=true given no evidence
        """
        N=1000
        cprob = self.engine.Marginalise(self.c.name, N)
        rprob = self.engine.Marginalise(self.r.name, N)
        #FIX: fill in actual value
        assert(cprob[True] == 'value' and \
               rprob[True] == 'value'), \
              "Incorrect probability of Cloudy or Rain being true"
    
    def testObserved(self):
        """ Compute and check the probability of w=true|r=false,c=true and s=false|w=true,c=false
        """
        N=1000
        self.engine.SetObs([self.c.name,self.r.name],[True,False])
        wprob = self.engine.Marginalise(self.w.name,N)
        #Violates abstraction
        self.engine.evidence.clear()
        self.engine.SetObs([self.c.name,self.w.name],[False,True])
        sprob = self.engine.Marginalise(self.s.name,N)
        #FIX: fill in actual value
        assert(wprob[True] == 'value' and \
               sprob[False] == 'value'), \
              "Either P(w=true|c=true,r=false) or P(s=false|c=false,w=true) was incorrect"
    
