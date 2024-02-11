#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='ex3',
      version='1.0',
      url='https://github.com/saarkatz/numerical-astrophysics/tree/ex3',
      packages=find_packages(where="src"),
      package_dir={'': 'src'}
      )
