#!/usr/bin/env python
""" this example shows how to perform Inference using any Inference 
Engine 
"""
from openbayes import JoinTree
# from OpenBayes import MCMCEngine

# first create a beyesian network
import watersprinkler

def main():
    """
    The main function
    """
    graph = watersprinkler.main()
    # create an inference Engine
    # choose the one you like by commenting/uncommenting the appropriate line
    engine = JoinTree(graph)
    #ie = MCMCEngine(G)
    
    # perform inference with no evidence
    results = engine.marginalise_all()
    
    print '=============================================================='
    print 'Without evidence:\n'
    for name, res in results.items():
        print name, res, '\n'
    
    print '=============================================================='
    print 'With evidence:\n'
    
    # add some evidence
    engine.set_obs({'s':1})    # s=1
    
    # perform inference with evidence
    results = engine.marginalise_all()
    
    # notice that the JoinTree engine does not perform a full message pass 
    # but only distributes the new evidence...
    for res in results.values(): 
        print res, '\n'
        
if __name__ == "__main__":
    main()
