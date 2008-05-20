#!/usr/bin/env python
"""
This module run all the test found in the directory
"""
# Copyright (C) 2008 by
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt
from glob import glob
import unittest

from openbayes import __version__, authors

__version__ = "0.1"
__author__ = authors['Salamin']

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
