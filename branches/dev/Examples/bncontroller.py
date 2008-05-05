from OpenBayes import BNController

# =================================
# PHASE 1: BN Definition & Training
# =================================
print "Phase 1"

# 1] Create a test BN 
gname = 'a simple test'
gdef = [('a', True, 3, ['b']), ('b', True, 2, None)]
myBN = BNController(gname, gdef)
# show the structure of BN
myBN.show_graph()
print '\n'

# 2] Train the test BN with data
training_data = [{'a':0, 'b':0}, {'a':1, 'b':1}, {'a':2, 'b':1}]
myBN.train(training_data)
myBN.show_distribution()

# 3] Save & Destory the test BN to conclude
myBN.save('./output/my_test_bn.xbn')
myBN = None

# =====================================
# PHASE 2: BN Evaluation with test data
# =====================================
print "Phase 2"

# 1] Recreate quickly the test BN & CPTs learned during training
myBN = BNController()
myBN.load('./output/my_test_bn.xbn')
myBN.show_distribution()

# 2] Present evidences and see what the BN returns
test_data = [{'a': 2}]
result_cpt = myBN.eval(test_data, 'b')

# Voila ;)
print "Presented the following test_data: %s \nBN returned the following:\n\
P(b=0)= %.2f \nP(b=1)= %.2f" % (test_data, result_cpt[0], result_cpt[1])