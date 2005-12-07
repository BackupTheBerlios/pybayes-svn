import numarray as na
import logging
import unittest

#Library Specific Modules
import graph.graph as graph

def learn_params_em(engine, cases, iterations=10):
    """ Expectation-Maximization as described by Michael Jordan in Chapter 11 of his unpublished book "An Introduction to Probabilistic Graphical Models"
    """
    converged = False
    while iterations > 0 and not converged:
        for case in cases:
            engine.EnterEvidence(case)
            #E step
        
    
