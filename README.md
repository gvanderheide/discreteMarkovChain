# discreteMarkovChain
This Python module calculates the steady state distribution of a Markov chain with a discrete state space. Markov chains with several million states can be solved. This module works in the latest versions of Python 2 and Python 3 and its only dependency is numpy/scipy.

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

###Example
The `markovChain` class can be used to initialize your own Markov chains. This is an example of a one-dimensional random walk between integers m and M, using the indirect method. While numpy is obligatory for the direct method, the indirect method allows for a transition function that returns a dictionary.
```python
from discreteMarkovChain import markovChain

class randomWalk(markovChain):
    #A random walk where we move up and down with rate 0.5 in each state between bounds m and M.
    #Uses the linear algebra solver for determing the steady-state.
    def __init__(self,m,M,direct=False,method='linear'):
        super(randomWalk, self).__init__(direct=direct,method=method)
        self.initialState = m
        self.m = m
        self.M = M
        self.uprate = 0.5
        self.downrate = 0.5
        
    def transition(self,state):
        #Specify the reachable states from 'state' and their rates.
        #A dictionary is easy here!
        rates = {}
        if self.m < state < self.M:
            rates[state+1] = self.uprate 
            rates[state-1] = self.downrate 
        elif state == self.m:
            rates[state+1] = self.uprate 
        elif state == self.M:
            rates[state-1] = self.downrate 
        return rates
```
Now initialize the random walk and calculate the steady-state vector pi.
```python
mc = randomWalk(0,10)
mc.computePi()
mc.printPi()
```
The examples.py file shows also multi-dimensional random walks using the direct method.








