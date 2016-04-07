"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='discreteMarkovChain',
    version='0.1',
    description='Solve Markov chains with a discrete state space.',
    long_description=long_description,
    url='https://github.com/gvanderheide/discreteMarkovChain',
    author='Gerlach van der Heide',
    author_email='g.van.der.heide@rug.nl',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research'
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='Markov chain stochastic stationary steady-state',
    packages=find_packages(),
    install_requires=['numpy','scipy'],
)
