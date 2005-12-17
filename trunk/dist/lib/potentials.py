import delegate

class DiscretePotential:
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and Marginalise().
    """
    def __init__(self, names, shape):
        
    def Marginalise(self, varnames):
        """ sum(varnames) self.cpt """
        temp = self.cpt
        ax = [self.p[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        return temp

    def Uniform(self):
        ' Uniform distribution '
        N = len(self.cpt.flat)
        self.cpt = na.array([1.0/N for n in range(N)], shape = self.cpt.shape, type='Float32')

    def __mul__(a,b):
        """
        a keeps the order of its dimensions
        
        always use a = a*b or b=b*a, not b=a*b
        """
        
        aa,bb = a.cpt, b.cpt
        
        correspondab = a.FindCorrespond(b)
        
        while aa.rank < len(correspondab): aa = aa[..., na.NewAxis]
        while bb.rank < len(correspondab): bb = bb[..., na.NewAxis]

        bb = na.transpose(bb, correspondab)

        return aa*bb

    def __str__(self): return str(self.cpt)

    def FindCorrespond(a,b):
        correspond = []
        k = len(b.p)
        for p in a.Fv:   #p=var name
            if b.p.has_key(p): correspond.append(b.p[p])
            else:
                correspond.append(k)
                k += 1

        for p in b.Fv:
            if not a.p.has_key(p):
                correspond.append(b.p[p])
                
        return correspond

    def Printcpt(self):
        string =  str(self.cpt) + '\nshape:'+str(self.cpt.shape)+'\nFv:'+str(self.Fv)+'\nsum : ' +str(na.sum(self.cpt.flat))
        print string

class JoinTreePotential(DiscretePotential):
    """
    The potential of each node/Cluster and edge/SepSet in a
    Join Tree Structure
    
    self.cpt = Pr(X)
    
    where X is the set of variables contained in Cluster or SepSet
    self.vertices contains the graph.vertices instances where the variables
    come from
    """
    def __init__(self):
        """ self. vertices must be set """
        names = [v.name for v in self.vertices]
        shape = [v.nvalues for v in self.vertices]
        DiscretePotential.__init__(self, names, shape)


    def __add__(c,s):
        """
        sum(X\S)phiX

        marginalise the variables contained in BOTH SepSet AND in Cluster
        """
        var = set(v.name for v in c.vertices) - set(v.name for v in s.vertices)
        return c.Marginalise(var)

    # result has the same variable order as c (cluster) (without some variables)
    # result has also the same variable order as s (SepSet)
    # this is because variables are sorted at initialisation
    
    def Normalise(self):
        self.cpt = na.divide(self.cpt, na.sum(self.cpt.flat), self.cpt)
