"""
This script is the installation script. It also contains
all the info about the differents authors
"""
# Copyright (C) 2005-2008 by
# Elliot Cohen <elliot.cohen@gmail.com>
# Kosta Gaitanis <gaitanis@tele.ucl.ac.be>  
# Hugues Salamin <hugues.salamin@gmail.com>
# Distributed under the terms of the GNU Lesser General Public License
# http://www.gnu.org/copyleft/lesser.html or LICENSE.txt

from distutils.core import setup
import sys
import time
__version__ = '0.1'
__author__ = 'Ronald Moncarey'
__author_email__ = 'rmoncarey@gmail.com'


authors = {'Arnaud' : 'Sebastien Arnaud',
           'Brouchoven' : 'Francois de Brouchoven',
           'Cohen' : 'Elliot Cohen (elliot.cohen@gmail.com)',
           'Gaitanis' : 'Kosta Gaitanis (gaitanis@tele.ucl.ac.be)',
           'Moncarey' : 'Ronald Moncarey (rmoncarey@gmail.com)',
           'Salamin' : 'Hugues Salamin (hugues.salamin@gmail.com)'
          }

__author__ = "\n".join([x for _, x in sorted(authors.items())])

name = 'openbayes'

__version__ = '0.1.0'

description = 'An OpenSource Python implementation of bayesian networks'\
              ' inspired by BNT.'

long_description = \
"""OpenBayes is a library that allows users to easily create a bayesian 
network and perform inference on it. It is mainly inspired from the Bayes 
Net Toolbox (BNT) which is available for MatLAB, but uses python as a base 
language which provides many benefits : fast execution, portability and 
ease to use and maintain. Any inference engine can be implemented by 
inheriting a base class. In the same way, new distributions can be added 
to the package by simply defining the data contained in the distribution 
and some basic probabilistic operations. 

The project is mature enough to be used for static bayesian networks and 
we are currently developping the dynamical aspect.
"""

__license__ = 'LGPL'
url = 'http://www.openbayes.org/'
download_url = 'http://svn.berlios.de/wsvn/pybayes/branches/Public/dist/'\
              '?rev=0&sc=0'
platforms = ['Linux', 'Mac OSX', 'Windows XP/2000/NT']
keywords = ['bayesian network', 'machine learning']
classifiers = [
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: Free for non-commercial use',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',]

packages = ['openbayes'],

package_dir = {'openbayes':'.', 
               'openbayes.examples':'./examples', 
               'openbayes.tests':'./tests'},

# Get date dynamically
date = time.asctime()

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
if __name__ == "__main__": 
    setup(name = name,
        version = __version__,
        description = description,
        author = 'Hugues Salamin', #The maintainer
	    author_email = 'hugues.salamin@gmail.com',
	    url = url,
        download_url = download_url,
        keywords = keywords,
        packages = packages,
        package_dir = package_dir,
	    license = __license__,
	    classifiers=classifiers,
        long_description = long_description)
