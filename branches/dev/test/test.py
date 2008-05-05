#!/usr/bin/env python
"""
This module run all the test found in the directory
"""

__version__ = "0.1"
__author__ = "Hugues Salamin"
__author_email__ = "hugues.salamin@gmail.com"


from glob import glob
import unittest

def main():
    """
    Simply run all the test in the current directory
    """
    suite = unittest.TestSuite()
    for file_ in glob("*_test.py"):
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(file_[:-3]))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    return result

if __name__ == "__main__":
    main()