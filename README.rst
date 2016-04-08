discreteMarkovChain
=======================
While for statistical and scientific programming languages such as R various packages are available for analyzing Markov chains, equivalent packages in Python are rather scarce. This discreteMarkovChain package for Python addresses the problem of obtaining the steady state distribution of a Markov chain, also known as the stationary distribution, limiting distribution or an invariant measure. The package is for Markov chains with discrete and finite state spaces, which are most common. 

This package is based on numpy and scipy for efficient computations and limited use of resources. Markov chains with several million states can be solved. The package introduces the `markovChain` class which has the following features. 

* States can be either integers or vectors of integers.
* Steady state distributions can be calculated for continous time Markov chains (CTMC) as well as discrete time Markov chains (DTMC).
* The user does not have to specify the generator/transition matrix of the Markov chain. This is done automatically using either an indirect or direct method.
* The indirect method requires the user to specify an initial state and transition function (giving for each state the reachable states and their probabilities). 
   * By repeatedly calling the transition function on unvisited states, the state space and the generator matrix are built up automatically.
   * This makes it easy to implement your own Markov chains!
* The direct method requires the user to specify a transition function and a function that gives the complete state space. 
   * While the implementation is typically more complex, this may have some computational advantage for large state spaces with vector states over de indirect method. 
* The steady state distribution can be calculated by a method of choice: 
   * The power method,
   * Solving a system of linear equations,
   * Determing the first left eigenvector, 
   * Searching in Krylov subspace.
* Checks are included to see whether all states in the Markov chain are connected.
* Memory consumption is reduced by using sparse matrices. 

When the user calls a certain solution method, the `markovChain` object gets the attribute `pi` which specifies the steady state probability of each state. The attribute `mapping` is a dictionary that links each index of `pi` with a corresponding state. Using the `mapping` and `pi`, it becomes simple to calculate performance measures for your Markov chain, such as the average cost per time unit or the number of blocked customers in a queue with blocking.

--------------
Installation
--------------
The package can be installed by calling

::

    pip install discreteMarkovChain

or by downloading the source and installing manually with

::

    python setup.py install

------------
Example
------------
The `markovChain` class can be used to initialize your own Markov chains. This is an example of a one-dimensional random walk between integers m and M, using the indirect method. While numpy is obligatory for the direct method, the indirect method allows for a transition function that returns a dictionary.

::

    from discreteMarkovChain import markovChain

    class randomWalk(markovChain):
        #A random walk where we move up and down with rate 0.5 in each state between bounds m and M.
        #We use the indirect method for obtaining the state space and the linear algebra solver for determing the steady-state.
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

Now initialize the random walk and calculate the steady-state vector pi.

::

    mc = randomWalk(0,10)
    mc.computePi()
    mc.printPi()


The examples.py file shows also multi-dimensional random walks using the direct method. 
