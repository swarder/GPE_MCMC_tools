from setuptools import setup

setup(name='gpe_mcmc_tools',
      version='0.1',
      description='Tools for basic GPE training and MCMC',
      url='https://github.com/swarder/GPE_MCMC_tools',
      author='Simon Warder',
      author_email='s.warder15@imperial.ac.uk',
      license='MIT',
      packages=['gpe_mcmc_tools'],
      zip_safe=False,
      install_requires=['GPy'])
