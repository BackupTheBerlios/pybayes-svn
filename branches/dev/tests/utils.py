"""
A simple unittest.TestCase extension to allow
for simplier test of array equality
"""
# Copyright (C) 2005-2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt
import unittest

import numpy

class ExtendedTestCase(unittest.TestCase):
    def assertAllEqual(self, a, b, message = None):
        if not (a==b).all():
            if message is None:
                self.fail("%s and %s are not equal"%(str(a), str(b)))
            else:
                self.fail(message)

    def assertNotAllEqual(self, a, b, message = None):
        if (a==b).all():
            if message is None:
                self.fail("%s and %s should not be equal"%(str(a), str(b)))
            else:
                self.fail(message)

    def assertAllClose(self, a, b, tol=7, message = None):
        if not numpy.allclose(a,b,tol):
            if message is None:
                self.fail("%s and %s not close enough"%(str(a), str(b)))
            self.fail(message)


