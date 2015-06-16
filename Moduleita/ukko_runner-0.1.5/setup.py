#!/usr/bin/env python

import os
from distutils.core import setup
from glob import glob

README = open(os.path.join(os.path.dirname(__file__), 'README.txt')).read()

setup(name='ukko_runner',
      version='0.1.5',
      license='BSD',
      author='Antti Honkela',
      author_email='antti.honkela@hiit.fi',
      maintainer='Antti Honkela',
      maintainer_email='antti.honkela@hiit.fi',
      url='http://github.com/ahonkela/',
      description='Utilities for running stuff on Ukko cluster',
      long_description=README,
      packages=['ukko_runner'],
      classifiers=[
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
      ],
      )
