from __future__ import print_function
import numpy as np
from itertools import product
from markovChain import markovChain

class randomWalk(markovChain):
    #A random walk where we move up and down with rate 0.5 in each state between bounds m and M.
    def __init__(self,m,M,direct=False,method='linear'):
        super(randomWalk, self).__init__(direct=direct,method=method)
        self.initialState = m
        self.m = m
        self.M = M
        self.uprate = 0.5
        self.downrate = 0.5
        
        self.computePi()        
        
    def transition(self,state):
        #Using a dictionary here is quite simple! 
        rates = {}
        if self.m < state < self.M:
            rates[state+1] = self.uprate 
            rates[state-1] = self.downrate 
        elif state == self.m:
            rates[state+1] = self.uprate 
        elif state == self.M:
            rates[state-1] = self.downrate 
        return rates
            
class randomWalkMulti(markovChain):
    #Now we move up an down on multiple dimensions. 
    def __init__(self,m,M,n,direct=False,method='linear'):
        super(randomWalkMulti, self).__init__(direct=direct,method=method)
        assert n > 1 and isinstance(n,int), "n should be an integer greater than 1"
        self.initialState = tuple([m]*n)
        self.n = n
        self.m = m
        self.M = M
        self.uprate = 0.5
        self.downrate = 0.5 
        
        self.computePi()

    def tupleAdd(self,state,i,b):
        #add amount 'b' to entry 'i' of tuple 'state'.
        newstate = list(state)
        newstate[i] += b
        return tuple(newstate)

    def transition(self,state):
        #now we need to loop over the states
        rates = {}
        for i in range(n):
            if self.m < state[i] < self.M:
                rates[self.tupleAdd(state,i,1)] = self.uprate 
                rates[self.tupleAdd(state,i,-1)] = self.downrate 
            elif state[i] == self.m:
                rates[self.tupleAdd(state,i,1)] = self.uprate 
            elif state[i] == self.M:
                rates[self.tupleAdd(state,i,-1)] = self.downrate 
        return rates               
            
class randomWalkNumpy(markovChain):
    #Now we do the same thing with a transition function that returns a 2d numpy array.
    #We also specify the statespace function so we can use the direct method.
    #This one is defined immediately for general n.
    def __init__(self,m,M,n,direct=True,method='linear'):
        super(randomWalkNumpy, self).__init__(direct=direct,method=method)
        self.initialState = m*np.ones(n,dtype=int)
        self.n = n
        self.m = m
        self.M = M
        self.uprate = 0.5
        self.downrate = 0.5        
        
        #Useful to define for the transition function.
        self.events = np.vstack((np.eye(n,dtype=int),-np.eye(n,dtype=int)))
        self.eventRates = np.array([self.uprate]*n+[self.downrate]*n)  
        
        self.computePi()

    def transition(self,state):
        up = state < self.M
        down = state > self.m
        possibleEvents = np.concatenate((up,down))
        newstates = state+self.events[possibleEvents]
        rates = self.eventRates[possibleEvents]
        return newstates,rates   
        
    def statespace(self):
        minvalues = [self.m]*self.n
        maxvalues = [self.M]*self.n
        return np.array([i for i in product(*(list(range(i,j+1)) for i,j in zip(minvalues,maxvalues)))],dtype=int)  
        
if __name__ == '__main__': 
    import time
    
    #for a one-dimensional state space, calculate the steady state distribution.
    #provided uprate and downrate are the same, each state is equally likely
    m = 0; M = 5    
    print(randomWalk(m,M).pi) 
    print(randomWalkNumpy(m,M,n=1,direct=True).pi)
    
    #When states are scalar integers, the indirect method is faster here. 
    #The linear algebra solver is quite fast for these one-dimensional problems (here, krylov and power method have really poor performance)
    M = 100000
    tm=time.clock(); randomWalk(m,M,method='linear'); print("Indirect:",time.clock()-tm)
    tm=time.clock(); randomWalkNumpy(m,M,n=1,method='linear'); print("Direct:", time.clock()-tm)       
         
    
    #Now a multidimensional case with an equal number of states.
    #Since building the state space is much more complex, the direct approach is faster. 
    #Here the krylov method and power method seem to work best. 
    #The linear algebra solver has memory problems, likely due to fill up of the sparse matrix.
    n = 5; m = 0; M = 9
    tm=time.clock(); randomWalkMulti(m,M,n,method='krylov'); print("Indirect:", time.clock()-tm)
    tm=time.clock(); randomWalkNumpy(m,M,n,method='krylov'); print("Direct:",time.clock()-tm)
