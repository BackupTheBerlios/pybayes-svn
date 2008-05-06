"""
A simple unittest.TestCase extension to allow
for simplier test of array equality
"""

import unittest

class ExtendedTestCase(unittest.TestCase):
    def assertAllEqual(self, a, b, message = None):
        self.assert_( ( a==b).all(), message)
    def assertNotAllEqual(self, a, b, message = None):
        self.failIf( ( a==b).all(), message)

