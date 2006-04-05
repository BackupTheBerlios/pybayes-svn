from distutils.core import setup
import sys

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
if sys.version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

setup(  name='OpenBayes',
        version='0.1.0',
        description='An OpenSource Python implementation of bayesian networks inspired by BNT.',
        author = 'Kosta Gaitanis, Elliot Cohen',
	    author_email = 'gaitanis@tele.ucl.ac.be, elliot.cohen@gmail.com',
	    url = 'http://www.openbayes.org',
        packages = ['OpenBayes'],
        package_dir = {'OpenBayes':'./'},
	    license = 'modified Python',
	    classifiers=[
          'Development Status :: First Public Release',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Python Software Foundation License',
	      'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Bayesian Networks',
          'Topic :: Probabilistic Inference',
          'Topic :: Learning Graphical models',
          ],
)
