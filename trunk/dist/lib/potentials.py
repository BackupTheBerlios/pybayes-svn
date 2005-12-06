from distributions import RawCPT

class JoinTreePotential(RawCPT):
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
        RawCPT.__init__(self, names, shape)


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
