from setuptools import setup

setup(
   name='omp_uq',
   version='1.0',
   description='UQ for fission MCHF fragment de-excitation using optical model potentials',
   author='Kyle Beyer',
   author_email='beykyle@umich.edu',
   packages=['omp_uq', 'sample_omparams'],
   install_requires=['CGMFtk', 'nudel'],
)
