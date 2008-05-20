#!/usr/bin/env python
"""
This is an example of use BNController
"""
# Copyright (C) 2005-2008 by
# Sebastien Arnaud
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt


from openbayes import BNController

def main():
    """
    The main function
    """
    # =================================
    # PHASE 1: BN Definition & Training
    # =================================
    print "Phase 1"

    # 1] Create a test BN 
    gname = 'a simple test'
    gdef = [('a', True, 3, ['b']), ('b', True, 2, None)]
    bayesian_net = BNController(gname, gdef)
    # show the structure of BN
    bayesian_net.show_graph()
    print '\n'

    # 2] Train the test BN with data
    training_data = [{'a':0, 'b':0}, {'a':1, 'b':1}, {'a':2, 'b':1}]
    bayesian_net.train(training_data)
    bayesian_net.show_distribution()

    # 3] Save & Destory the test BN to conclude
    bayesian_net.save('./output/my_test_bn.xbn')
    bayesian_net = None

    # =====================================
    # PHASE 2: BN Evaluation with test data
    # =====================================
    print "Phase 2"

    # 1] Recreate quickly the test BN & CPTs learned during training
    bayesian_net = BNController()
    bayesian_net.load('./output/my_test_bn.xbn')
    bayesian_net.show_distribution()

    # 2] Present evidences and see what the BN returns
    test_data = [{'a': 2}]
    result_cpt = bayesian_net.eval(test_data, 'b')

    # Voila ;)
    print "Presented the following test_data: %s\n"\
           "BN returned the following:\n"\
           "P(b=0)= %.2f \nP(b=1)= %.2f" % \
           (test_data, result_cpt[0], result_cpt[1])

if __name__ == "__main__":
    main()
