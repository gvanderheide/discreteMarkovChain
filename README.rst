discreteMarkovChain
======================= 

While for statistical and scientific programming languages such as R
various packages are available for analyzing Markov chains, equivalent
packages in Python are rather scarce. This discreteMarkovChain package
for Python addresses the problem of obtaining the steady state
distribution of a Markov chain, also known as the stationary
distribution, limiting distribution or invariant measure. The package
is for Markov chains with discrete and finite state spaces, which are
most commonly encountered in practical applications.

This package is based on numpy and scipy for efficient computations
and limited use of resources. Markov chains with several million
states can be solved. The package introduces the ``markovChain`` class
which has the following features.

* States can be either integers or vectors of integers.
* Steady state distributions can be calculated for continous time
  Markov chains (CTMC) as well as discrete time Markov chains (DTMC).
* The user can either manually specify the probability/rate matrix of
  the Markov chain, or let the program do this automatically using an
  indirect or direct method.
* The indirect method requires the user to specify an initial state
  and transition function (giving for each state the reachable states
  and their probabilities/rates).

   - By repeatedly calling the transition function on unvisited
     states, the state space and the probability matrix are built up
     automatically.
   - This makes it easy to implement your own Markov chains!
   
* The direct method requires the user to specify a transition function
  and a function that gives the complete state space.

   - While the implementation is typically more complex, this may have
     some computational advantage over the indirect method for large
     state spaces with vector states.
   
* The steady state distribution can be calculated by a method of choice: 

   - The power method,
   - Solving a system of linear equations,
   - Determing the first left eigenvector, 
   - Searching in Krylov subspace.
   
* Checks are included to see whether all states in the Markov chain
  are connected.
* Memory consumption is reduced by using sparse matrices. 

When the user calls a certain solution method, the ``markovChain``
object gets the attribute ``pi`` which specifies the steady state
probability of each state. When the user uses the direct or indirect
method, the object gets the ``mapping`` attribute which is a dictionary
that links each index of ``pi`` with a corresponding state. Using the
``mapping`` and ``pi``, it becomes simple to calculate performance
measures for your Markov chain, such as the average cost per time unit
or the number of blocked customers in a queue with blocking.

--------------
Installation
--------------
The package can be installed with the command

::

    pip install discreteMarkovChain

or by downloading the source distribution and installing manually with

::

    python setup.py install

------------
Examples
------------

Computing the steady state distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``markovChain`` class can be used to initialize your own Markov
chains. We import it by using

.. code:: python

    import numpy as np
    from discreteMarkovChain import markovChain

First, lets consider a simple Markov chain with two states, where we
already know the probability matrix ``P``.

.. code:: python

    P = np.array([[0.5,0.5],[0.6,0.4]])
    mc = markovChain(P)
    mc.computePi('linear') #We can also use 'power', 'krylov' or 'eigen'
    print(mc.pi)

We get the following steady state probabilities:

.. code:: python

    [ 0.54545455  0.45454545]


A One-dimensional Random Walk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

Now we show an example of a one-dimensional random walk in continuous
time between integers ``m`` and ``M``. We move up and down with
rates 1. We will use the indirect method to determine the rate matrix
for us automatically. The indirect method is rather flexible, and
allows the transition function to return a dictionary with reachable
states and rates. We first introduce our ``randomWalk`` class.

.. code:: python

    class randomWalk(markovChain):

        """
        A random walk where we move up and down with rate 1.0 in each
        state between bounds m and M.

        For the transition function to work well, we define some
        class variables in the __init__ function.
        """
    
        def __init__(self,m,M):
            super(randomWalk, self).__init__() 
            self.initialState = m
            self.m = m
            self.M = M
            self.uprate = 1.0
            self.downrate = 1.0
        
        def transition(self,state):
            #Specify the reachable states from state and their rates.
            #A dictionary is extremely easy here!
            rates = {}
            if self.m < state < self.M:
                rates[state+1] = self.uprate 
                rates[state-1] = self.downrate 
            elif state == self.m:
                rates[state+1] = self.uprate 
            elif state == self.M:
                rates[state-1] = self.downrate 
            return rates

Now we initialize the random walk with some values for ``m`` and ``M`` and
calculate the steady-state vector ``pi``.

.. code:: python

    mc = randomWalk(0,5)
    mc.computePi()
    mc.printPi()

The stationary probabilities are given below.

.. code:: python

    0 0.166666666667
    1 0.166666666667
    2 0.166666666667
    3 0.166666666667
    4 0.166666666667
    5 0.166666666667

Not unexpectedly, they are the same for each state.

A Multi-dimensional Random Walk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can repeat this for a multi-dimensional random walk. Now we use the
direct method. Here, we need to use a transition function returning
numpy arrays and we need to define a function that calculates the
state space. For explanation of the transition function, see `this
example <docs/multirandomwalk.ipynb>`_.

.. code:: python 

    from discreteMarkovChain import partition 

    class randomWalkNumpy(markovChain):
        """
        Now we do the same thing with a transition function that returns a 2d numpy array.
        We also specify the statespace function so we can use the direct method.
        This one is defined immediately for general n.
        """
        def __init__(self,m,M,n,direct=True):
            super(randomWalkNumpy, self).__init__(direct=direct)
            self.initialState = m*np.ones(n,dtype=int)
            self.n = n
            self.m = m
            self.M = M
            self.uprate = 1.0
            self.downrate = 1.0        

            #It is useful to define the variable 'events' for the the
            #transition function.  The possible events are 'move up'
            #or 'move down' in one of the random walks.  The rates of
            #these events are given in 'eventRates'.
            self.events = np.vstack((np.eye(n,dtype=int),-np.eye(n,dtype=int)))
            self.eventRates = np.array([self.uprate]*n+[self.downrate]*n)  
        
        def transition(self,state):
            #First check for the current state which of the 'move up'
            #and 'move down' events are possible.
            up = state < self.M
            down = state > self.m
            possibleEvents = np.concatenate((up,down))  #Combine into one boolean array. 
            
            #The possible states after the transition follow by adding
            #the possible 'move up'/'move down' events to the current
            #state.
            newstates = state+self.events[possibleEvents]
            rates = self.eventRates[possibleEvents]
            return newstates,rates   
            
          def statespace(self):
              #Each random walk can be in a state between m and M.
              #The function partition() gives all partitions of integers
              #between min_range and max_range.
              min_range = [self.m]*self.n
              max_range = [self.M]*self.n
              return partition(min_range,max_range) 
        

Now we initialize ``n=2`` random walks between ``m=0`` and ``M=2`` and print
the stationary distribution.

.. code:: python

    mc = randomWalkNumpy(0,2,n=2)
    mc.computePi('linear')
    mc.printPi()
    
    [0 0] 0.111111111111
    [1 0] 0.111111111111
    [2 0] 0.111111111111
    [0 1] 0.111111111111
    [1 1] 0.111111111111
    [2 1] 0.111111111111
    [0 2] 0.111111111111
    [1 2] 0.111111111111
    [2 2] 0.111111111111

We could also solve much larger models. The example below has random
walks in 5 dimensions with 100.000 states. For these larger models, it
is often better to use the power method. The linear algebra solver may
run into memory problems.

.. code:: python

    mc = randomWalkNumpy(0,9,n=5)
    mc.computePi('power')

On a dual core computer from 2006, the rate matrix and ``pi`` can be
calculated within 10 seconds.

Expected Hitting Times
^^^^^^^^^^^^^^^^^^^^^^

The expected hitting time to a certain set ``A`` can be computed with
this package too.

Suppose ``k`` is the vector of hitting times, so ``k[i]`` is the expected
time from state ``i`` to the set ``A``. Then, clearly, ``k[i]=0`` for any
state ``i`` in ``A``. When state ``i`` not in ``A`` we wait one time step,
make one transition from ``i`` to ``j`` with probability ``P[i,j]``, and
then see how long it takes from state ``j`` to reach ``A``. Thus, the
vector ``k`` must be such that ``k[i]=0`` for ``i`` in ``A`` and ``k = 1 +
P.dot(k)`` for ``k`` not in ``A``.

For the example of the one-dimensional random walk, suppose we want to
know the time to hit the origion. Then ``A=set([0])``, i.e., the hitting
set ``A`` only contains the state ``0``. In the computations we use a mask
to set ``k[0]=0`` after each iteration. 

.. code:: python

    mc = randomWalk(0,5)
    P = mc.getTransitionMatrix()

    hittingset=[0]  # the hitting set A

    one = np.ones(mc.size)
    one[hittingset] = 0

    k = np.zeros(mc.size)
    for i in range(100):
        k = one + P.dot(k)
        k[hittingset]  = 0
    print(k)

    [  0.           9.86851962  17.74650115  23.64447789  27.57120126
  29.53293175]



A more proper stopping criterion is when the difference between the
old and new value of ``k`` is below some threshold. In the computations we use a mask
to set `k[0]=0` after each iteration. 

.. code:: python

    from numpy import linalg as LA
    
    mask = np.zeros(mc.size)
    for i in range(mc.size):
        if i in hittingset:
            mask[i]=1

    k1 = np.zeros(mc.size)
    k2 = one + P.dot(k1)
    i = 0
    while(LA.norm(k1-k2)>1e-6):
        k1=k2
        k2 = one + P.dot(k1)
        np.putmask(k2, mask, 0)
        i += 1
    print(k2)
    print(i)

    [  0.          10.00999607  18.01799247  24.02398947  28.02798732
  30.02998621]


See ``hitting_time.py`` in the ``discreteMarkovchain`` directory for a full example.

----------------
Changes in v0.22
----------------
* Added documentation for the ``markovChain`` class and all its methods,
  including examples.
* Added the function ``partition`` that can be used to determine the
  state space when states are consists of all integers between
  ranges. The optional parameter ``max_sum`` can be specified if the
  state vectors should sum up to less than ``max_sum`` (useful in some
  queueing and inventory applications).
* Fixed an error when calling ``krylovMethod()``, ``linearMethod()`` and
  ``eigenMethod()`` on Markov chains with one state.
* Included a workaround for an error when calling ``eigenMethod()`` on
  Markov chains with two states.




