"""
This package implements bayesian network in python. Only static
bayesian network are for now supported
"""
# Copyright (C) 2005-2008 by
# Elliot Cohen <elliot.cohen@gmail.com>
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from openbayes.setup import __author__, __version__, authors


from openbayes.bayesnet import *
from openbayes.inference import *
from openbayes.distributions import *
from openbayes.potentials import *
from openbayes.table import *
from openbayes.graph import *
# from openbayes.xbn import *
from openbayes.bncontroller import *

__all__ = ['bayesnet', 'distributions', 'inference', 'potentials',
           'table', 'graph', 'OpenBayesXBN', 'BNController']
