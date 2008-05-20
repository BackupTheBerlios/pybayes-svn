"""
A simple unittest.TestCase extension to allow
for simplier test of array equality
"""
# Copyright (C) 2005-2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt


import unittest

class ExtendedTestCase(unittest.TestCase):
    def assertAllEqual(self, a, b, message = None):
        self.assert_( ( a==b).all(), message)
    def assertNotAllEqual(self, a, b, message = None):
        self.failIf( ( a==b).all(), message)

