#!/usr/bin/env python
""" this example shows how to Learn parameters from a set of observations
 using Maximum Likelihood Estimator """ 

# Copyright (C) 2005-2008 by
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from copy import deepcopy
from time import time

from openbayes import learning

# first create a beyesian network
import watersprinkler

def main():
    """
    The main function
    """
    graph = watersprinkler.main()
    nbr_samples = 1000
    # sample the network N times
    cases = graph.sample(nbr_samples)  
    # create a new bayesian network with all parameters set to 1
    graph_copy = deepcopy(graph)
    graph_copy.init_distributions()
    # Learn the parameters from the set of cases
    engine = learning.MLLearningEngine(graph_copy)
    # cases = engine.read_file('file.xls') #To use the data of file.xls
    start_time = time()
    engine.learn_ml_params(cases)
    print 'Learned from %d cases in %1.3f secs' % \
          (nbr_samples, (time() - start_time))

    # print the learned parameters
    for vertex in graph_copy.all_v: 
        print vertex.name, vertex.distribution.cpt,'\n'

    # print the parameters
    for vertex in graph.all_v: 
        print vertex.distribution,'\n'

if __name__ == "__main__":
    print main()
