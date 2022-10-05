from setuptools import setup

setup(
   name='omp_uq',
   version='1.0',
   description='',
   author='Kyle Beyer',
   author_email='beykyle@umich.edu',
   packages=['omp_uq'],  # would be the same as name
   install_requires=['CGMFtk', 'nudel'], #external packages acting as dependencies
)
