import numarray as na

import delegate
import table

class DiscretePotential(table.Table):
    """ This is a basic potential to represent discrete potentials.
    It is very similar to a MultinomialDistribution except that 
    it defines several operations such as __mult__, __add__, 
    and Marginalise().
    """
    def __init__(self, names, shape, elements=None):
        if elements == None:
            elements = na.product(shape)*[1]
        table.Table.__init__(self,names,shape,elements,'Float32')
               
    def Marginalise(self, varnames):
        """ sum(varnames) self.cpt """
        temp = self.view()
        ax = [self.assocdim[v] for v in varnames]
        ax.sort(reverse=True)  # sort and reverse list to avoid inexistent dimensions
        for a in ax:
            temp = na.sum(temp, axis = a)
        remainingNames = self.names - set(varnames)
        return DiscretePotential(remainingNames, temp.shape, elements=temp.flat)

    def Uniform(self):
        ' Uniform distribution '
        N = na.product(self.shape)
        self[:] = 1.0/N

    def __mul__(self, other):
        #FIXME: Needs to be reimplemented now that inherits from table
        """
        a keeps the order of its dimensions
        
        always use a = a*b or b=b*a, not b=a*b
        """
        
        aa,bb = a.cpt, b.cpt
        
        correspondab = self.FindCorrespond(other)
        
        while self.rank < len(correspondab): self = self[..., na.NewAxis]
        while other.rank < len(correspondab): other = other[..., na.NewAxis]

        other.transpose(correspondab)

        return self*other

    def __str__(self): return str(self.cpt)

    def FindCorrespond(self, other):
        correspond = []
        k = other.rank
        for i in range(self.rank):
            p = self.assocname[i]   #p=var name
            if other.assocdim.has_key(p): 
                correspond.append(other.assocdim[p])
            else:
                correspond.append(k)
                k += 1

        for i in range(other.rank):
            p = other.assocname[i]  #p=var name
            if not self.assocdim.has_key(p):
                correspond.append(other.assocdim[p])
                
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
