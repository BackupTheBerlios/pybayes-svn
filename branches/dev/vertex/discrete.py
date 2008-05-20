'''
This file contains all the code relating to discrete variable in the bayesian
network. Any discreted variable should define at least the following to
attribute:
  discrete = True
  nvalues = Nombre of possible states
'''
# Copyright (C) 2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt
import numpy

from openbayes import __version__, authors
from openbayes.bayesnet import BVertex
from openbayes.table import Table

__author__ = authors['Salamin']
__all__ = ['DiscreteVertex']

class DiscreteVertex(BVertex):
    """
    This class represents a discrete variable in a bayesian networks. It can
    take value between 0 and nvalue -1. This is the simplest possible form
    """

    def __init__(self, name, nvalue = 2):
        BVertex.__init__(self, name)
        self. discrete = True
        self.nvalue = nvalue
        self.cpt = None

    def set_parents(self, parents):
        """
        We can now finalize the node. Parents is a list of vertices that
        are the parent of self in the graph. We need to check that all
        the parent are discrete.

        We can then create the conditional probability table self.cpt
        """
        for x in parents:
            try:
                if not x.discrete:
                    raise ValueError("Parents contains not discrete vertex")
            except AttributeError:
                raise ValueError("Parents contains not discrete vertex")
        names = [self.name] + [x.name for x in parents]
        dims = [self.nvalue] + [x.nvalue for x in parents]
        self.cpt = Table(names, dims)

    def sample(self, parents):
        """
        This function return a random value between 0 and nvalue-1, according to the
        cpt and the value of the parents
        """
        # we start by getting the correct vector in the table
        proba_vector =  self.cpt[parents]
        cumul = 0.0
        rand_float = numpy.random.rand()
        for candidate, proba in enumerate(proba_vector):
            cumul += proba
            if rand_float < cumul:
                break
        return candidate

