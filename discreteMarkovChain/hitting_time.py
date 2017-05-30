import numpy as np
from numpy import linalg as LA

from discreteMarkovChain import markovChain


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

mc = randomWalk(0,5)
P = mc.getTransitionMatrix()

hittingset=[0]

one = np.ones(mc.size)
one[hittingset] = 0

k = np.zeros(mc.size)
for i in range(100):
    k = P.dot(k)+one
    k[hittingset]  = 0

print(k)

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
        

