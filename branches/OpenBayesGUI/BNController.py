#!/usr/bin/env python2.4
# encoding: utf-8

"""
bayesiannetworks.py

Created by Sébastien Arnaud on 2006-11-20.
Copyright (c) 2006 All rights reserved.
"""

from OpenBayes import BNet, BVertex, DirEdge, graph
from OpenBaye#!/usr/bin/env python2.4
# encoding: utf-8

"""
bayesiannetworks.py

Created by Sébastien Arnaud on 2006-11-20.
Copyright (c) 2006 All rights reserved.
"""

from OpenBayes import BNet, BVertex, DirEdge, graph
from OpenBayes import MCMCEngine, JoinTree
import cPickle, gzip
import types

class BNController(object):
	_BN = None
	_BNDef = None
	
	def __init__(self, BNnodesdef=None):
		if BNnodesdef!=None:
			self._init_BN(BNnodesdef)

	def _init_BN(self, BNnodesdef):
		BNnodes = {}
		BNconnections = []
		BN = BNet()  

		# Save the nodesdef for future use (saving)
		self._BNDef = BNnodesdef

		for node in BNnodesdef:
			(nodename , isdiscrete, numstates, leafnode) = node
			BNnodes[nodename] = BN.add_v( BVertex(nodename, isdiscrete, numstates))
		
		#TODO: Find a way to improve this and avoid to have to loop a second time
		# reason is because BNnodes[leafnode] is not sure to be there when going through the first loop
		for node in BNnodesdef:
			(nodename , isdiscrete, numstates, leafnode) = node
			if type(leafnode)==type([]):
				for r in leafnode:
					if r!=None:	BNconnections.append( (BNnodes[nodename], BNnodes[r]) )
			elif leafnode!=None:
				BNconnections.append( (BNnodes[nodename], BNnodes[leafnode]) )
			else:
				#do nothing 
				pass
		
		for ep in BNconnections:
			BN.add_e( DirEdge( len( BN.e ), *ep ) )
			
		# Ok our Bnet has been created, let's save it in the controller
		self._BN = BN

		# Let's not forget to initialize the distribution
		self._BN.InitDistributions()
			
	def show_graph(self):
		print self._BN

	def show_distribution(self):
		for v in self._BN.all_v: 
			print v.distribution,'\n'

	def load(self, filename):
		file = gzip.GzipFile('%s.bnz' % filename, 'rb')
		
		# Read the nodesdef (always saved first)
		nodesdef = cPickle.load(file)
		
		# Only create the BN is it does not exists
		if self._BN==None: 
			self._init_BN(nodesdef)

		# Now we just load the CPTs
		vdict = {}
		while 1:
			try:
				node = cPickle.load(file)
				lv, ld = node
				vdict[lv] = ld			
			except EOFError:
				break
		# Assign now back the CPT to each node of our BN
		for v in self._BN.all_v:
			v.distribution.cpt = vdict[graph.Vertex.__str__(v)]

		# Ok, done!
		file.close()
	
	def save(self, filename):
		if self._BN!=None: #only save if BN exists!
			file = gzip.GzipFile('%s.bnz' % filename, 'wb')
			
			# Let's save the nodesdef first
			file.write(cPickle.dumps(self._BNDef, 1))
			
			for v in self._BN.all_v: 
				node = (graph.Vertex.__str__(v), v.distribution.Convert_to_CPT())
				#REMINDER: node type = (str, numarray.numarraycore.NumArray)
				file.write(cPickle.dumps(node, 1))
			
			file.close()

	def train(self, ie_engine, cases):
		if ie_engine=="MCMC":
			ie = MCMCEngine(self._BN)
		elif ie_engine=="JTREE":
			ie = JoinTree(self._BN)

		ie.LearnMLParams(cases)

	def eval(self, evidences, node):
		#TODO: node could be a list if more than 1 node needs to be guessed
		ie = MCMCEngine(self._BN)
		
		for e in evidences:
			ie.SetObs(e)
		return ie.Marginalise(node).Convert_to_CPT()


if __name__ == "__main__":
	"""
	Simple example on how to use the class above
	"""	
	output_trace = False
	
	# =================================
	# PHASE 1: BN Definition & Training
	# =================================
	if output_trace: print "Phase 1"
	
	# 1] Create a test BN 
	gdef = [('a', True, 3, ['b']), ('b', True, 2, None)]
	myBN = BNController(gdef)
	if output_trace: myBN.show_graph()
	
	# 2] Train the test BN with data
	training_data = [{'a':0, 'b':0}, {'a':1, 'b':1}, {'a':2, 'b':1}]
	myBN.train('MCMC', training_data)
	if output_trace: myBN.show_distribution()

	# 3] Save & Destory the test BN to conclude
	myBN.save('my_test_bn')
	myBN = None

	# =====================================
	# PHASE 2: BN Evaluation with test data
	# =====================================
	if output_trace: print "Phase 2"
	
	# 1] Recreate quickly the test BN & CPTs learned during training
	myBN = BNController()
	myBN.load('my_test_bn')
	if output_trace: myBN.show_distribution()
	
	# 2] Present evidences and see what the BN returns
	test_data = [{'a': 2}]
	result_cpt = myBN.eval(test_data, 'b')

	# Voila ;)
	print "Presented the following test_data: %s \n BN returned the following: \n P(b=0)= %.2f \n P(b=1)= %.2f" % (test_data, result_cpt[0], result_cpt[1])s import MCMCEngine, JoinTree
import cPickle, gzip
import types

class BNController(object):
	_BN = None
	_BNDef = None
	
	def __init__(self, BNnodesdef=None):
		if BNnodesdef!=None:
			self._init_BN(BNnodesdef)

	def _init_BN(self, BNnodesdef):
		BNnodes = {}
		BNconnections = []
		BN = BNet()  

		# Save the nodesdef for future use (saving)
		self._BNDef = BNnodesdef

		for node in BNnodesdef:
			(nodename , isdiscrete, numstates, leafnode) = node
			BNnodes[nodename] = BN.add_v( BVertex(nodename, isdiscrete, numstates))
		
		#TODO: Find a way to improve this and avoid to have to loop a second time
		# reason is because BNnodes[leafnode] is not sure to be there when going through the first loop
		for node in BNnodesdef:
			(nodename , isdiscrete, numstates, leafnode) = node
			if type(leafnode)==type([]):
				for r in leafnode:
					if r!=None:	BNconnections.append( (BNnodes[nodename], BNnodes[r]) )
			elif leafnode!=None:
				BNconnections.append( (BNnodes[nodename], BNnodes[leafnode]) )
			else:
				#do nothing 
				pass
		
		for ep in BNconnections:
			BN.add_e( DirEdge( len( BN.e ), *ep ) )
			
		# Ok our Bnet has been created, let's save it in the controller
		self._BN = BN

		# Let's not forget to initialize the distribution
		self._BN.InitDistributions()
			
	def show_graph(self):
		print self._BN

	def show_distribution(self):
		for v in self._BN.all_v: 
			print v.distribution,'\n'

	def load(self, filename):
		file = gzip.GzipFile('%s.bnz' % filename, 'rb')
		
		# Read the nodesdef (always saved first)
		nodesdef = cPickle.load(file)
		
		# Only create the BN is it does not exists
		if self._BN==None: 
			self._init_BN(nodesdef)

		# Now we just load the CPTs
		vdict = {}
		while 1:
			try:
				node = cPickle.load(file)
				lv, ld = node
				vdict[lv] = ld			
			except EOFError:
				break
		# Assign now back the CPT to each node of our BN
		for v in self._BN.all_v:
			v.distribution.cpt = vdict[graph.Vertex.__str__(v)]

		# Ok, done!
		file.close()
	
	def save(self, filename):
		if self._BN!=None: #only save if BN exists!
			file = gzip.GzipFile('%s.bnz' % filename, 'wb')
			
			# Let's save the nodesdef first
			file.write(cPickle.dumps(self._BNDef, 1))
			
			for v in self._BN.all_v: 
				node = (graph.Vertex.__str__(v), v.distribution.Convert_to_CPT())
				#REMINDER: node type = (str, numarray.numarraycore.NumArray)
				file.write(cPickle.dumps(node, 1))
			
			file.close()

	def train(self, ie_engine, cases):
		if ie_engine=="MCMC":
			ie = MCMCEngine(self._BN)
		elif ie_engine=="JTREE":
			ie = JoinTree(self._BN)

		ie.LearnMLParams(cases)

	def eval(self, evidences, node):
		#TODO: node could be a list if more than 1 node needs to be guessed
		ie = MCMCEngine(self._BN)
		
		for e in evidences:
			ie.SetObs(e)
		return ie.Marginalise(node).Convert_to_CPT()


if __name__ == "__main__":
	"""
	Simple example on how to use the class above
	"""	
	output_trace = False
	
	# =================================
	# PHASE 1: BN Definition & Training
	# =================================
	if output_trace: print "Phase 1"
	
	# 1] Create a test BN 
	gdef = [('a', True, 3, ['b']), ('b', True, 2, None)]
	myBN = BNController(gdef)
	if output_trace: myBN.show_graph()
	
	# 2] Train the test BN with data
	training_data = [{'a':0, 'b':0}, {'a':1, 'b':1}, {'a':2, 'b':1}]
	myBN.train('MCMC', training_data)
	if output_trace: myBN.show_distribution()

	# 3] Save & Destory the test BN to conclude
	myBN.save('my_test_bn')
	myBN = None

	# =====================================
	# PHASE 2: BN Evaluation with test data
	# =====================================
	if output_trace: print "Phase 2"
	
	# 1] Recreate quickly the test BN & CPTs learned during training
	myBN = BNController()
	myBN.load('my_test_bn')
	if output_trace: myBN.show_distribution()
	
	# 2] Present evidences and see what the BN returns
	test_data = [{'a': 2}]
	result_cpt = myBN.eval(test_data, 'b')

	# Voila ;)
	print "Presented the following test_data: %s \n BN returned the following: \n P(b=0)= %.2f \n P(b=1)= %.2f" % (test_data, result_cpt[0], result_cpt[1])