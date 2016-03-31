# discreteMarkovChain
This Python module calculates the steady state distribution of a Markov chain with a discrete state space. The module handles both continous time and discrete time Markov chains (CTMC or DTMC). Markov chains with several million states can be solved. This module is for Python 2.7 and its only dependency is numpy/scipy.

The module has the following features. 
-There are two alternative methods for determining the generator matrix of the Markov chain: a direct and an indirect method. Both methods require the user to specify a transition function. 
-The indirect method requires only an initial state. By repeatedly calling the transition function on unvisited states, reachable states are determined automatically. The generator matrix is built up on the fly.
-The direct method requires the state space to be derived beforehand. States are translated into unique codes that can be used to identify reachable states. This has some computational and memory advantages for vector states.
-Memory consumption is reduced by using sparse matrices. 
-Checks are included to see whether all states in the Markov chain are connected
-Four different methods for calculating steady state distributions are included. The power method, linear algebra solver, the first eigenvector and a method searching in Krylov subspace.

The examples.py file shows some examples of a one- and multi-dimensional random walk.








