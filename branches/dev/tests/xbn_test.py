#!/usr/bin/env python
"""
This is the test module for xnb
"""
# Copyright (C) 2005-2008 by
# Ronald Moncarey <rmoncarey@gmail.com>  
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt


import unittest

from openbayes.xbn import *
__version__ = '0.1'
__author__ = 'Ronald Moncarey'
__author_email__ = 'rmoncarey@gmail.com'


class XBNTestCase(unittest.TestCase):   
         
    def setUp(self):
        """ reads the WetGrass.xbn """
        file_name_in = './WetGrass.xbn'
        xbn = LoadXBN(file_name_in)
        self.G = xbn.Load()


    def testGeneral(self):
        ' tests general data'
        assert(self.G.name == 'bndefault'), \
                " General BN data is not read correctly"
                
    def testDistributions(self):
        ' test the distributions'
        r = self.G.v['Rain']
        s = self.G.v['Sprinkler']
        w = self.G.v['Watson']
        h = self.G.v['Holmes']
        
        assert(allclose(r.distribution.cpt, [0.2, 0.8]) and \
                allclose(s.distribution.cpt, [0.1, 0.9]) and \
                allclose(w.distribution[{'Rain':1}], [0.2, 0.8])), \
                " Distribution values are not correct"

    def testInference(self):
        """ Loads the RainWatson BN, performs inference and checks the results """
        self.engine = JoinTree(self.G)
        r = self.engine.marginalise('Rain')        
        s = self.engine.marginalise('Sprinkler')
        w = self.engine.marginalise('Watson')
        h = self.engine.marginalise('Holmes')

        assert(allclose(r.cpt,[0.2,0.8]) and \
                allclose(s.cpt,[0.1,0.9]) and \
                allclose(w.cpt,[0.36,0.64]) and \
                allclose(h.cpt,[ 0.272, 0.728]) ), \
               " Somethings wrong with JoinTree inference engine"
            
    def testSave(self):
        ''' reads an xbn, then writes it again, then reads it again and then 
        perform inference and checks the results'''
        file_name_out = './test.xbn'
        SaveXBN(file_name_out, self.G)
        xbn = LoadXBN(file_name_out)
        self.G = xbn.Load()
        
        self.testInference()
        
        import os
        # delete the test.xbn file
        os.remove(file_name_out)


#---MAIN   
if __name__ == '__main__':
    unittest.main() 
