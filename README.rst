discreteMarkovChain
=======================
This Python module based on numpy/scipy calculates the steady state distribution of a Markov chain with a discrete state space. Markov chains with several million states can be solved. 
The module introduces the `markovChain` class which has the following features. 

* States in the state space can be integers or integer vectors (1d numpy arrays). 
* The Markov chain can be a continous time Markov chains (CTMC) as well as discrete time Markov chains (DTMC). 
* At the moment, four different methods for calculating steady state distributions are included: 
   * The power method,
   * Linear algebra solver,
   * The first left eigenvector, 
   * Krylov subspace method.
* There are two alternative methods for obtaining the generator/transition matrix of the Markov chain: a direct and an indirect method.
   * Both methods require the user to specify a transition function. 
   * The indirect method requires only an initial state. 
     * By repeatedly calling the transition function on unvisited states, reachable states are determined automatically. 
     * The generator matrix is built up on the fly.
   * The direct method requires a function that gives the state space on beforehand. 
     * States are translated into unique codes that can be used to identify reachable states. 
     * This has some computational and memory advantages for vector states.
* Memory consumption is reduced by using sparse matrices. 
* Checks are included to see whether all states in the Markov chain are connected.