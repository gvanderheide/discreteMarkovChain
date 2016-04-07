""" 
Possible additions:
-Check that the state codes do not suffer from integer overflow.
-Determine whether the transition function leads to an infinite state space.
"""
from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix, csgraph, eye, vstack
from scipy.sparse.linalg import eigs, gmres, spsolve
from numpy.linalg import norm
from collections import OrderedDict,defaultdict
try: #For python 3 functionality.
    from itertools import imap
except ImportError:
    imap = map

def uniqueStates(states,rates):
    """
    Returns unique states and sums up the corresponding rates.
    States should be a 2d numpy array with on each row a state, and rates a 1d numpy array with length equal to the number of rows in states.      
    
    This may be helpful in the transition function for summing up the rates of different transitions that lead to the same state            
    """        
    order     = np.lexsort(states.T)
    states    = states[order]
    diff      = np.ones(len(states), 'bool')
    diff[1:]  = (states[1:] != states[:-1]).any(-1)
    sums      = np.bincount(diff.cumsum() - 1, rates[order])
    return states[diff], sums

class markovChain(object):
    """
    This class calculates the steady state distribution for a Markov chain with a finite and discrete state space.
    The Markov chain can be defined on continuous time or discrete time.
    A state can be represented as an integer number or a vector consisting of integer numbers.
    
    There are two ways to build the generator matrix of the Markov chain: direct and indirect.
    For both ways, the user needs to a transition function that calculates for each state the reachable states and the corresponding rates.
       
    For the direct method the user needs to generate the statespace in advance. 
    States in the statespace are translated into unique codes, which can be used to identify the indices of new states.
    These indices are used to construct the sparse generator matrix.
     
    For the indirect method the user needs to specify an initial state.
    By repeatedly calling the transition function, all reachable states are determined starting from this initial state.
    On the fly, we construct the sparse generator matrix of the Markov chain.

    Steady state distributions can be calculated using various techniques. 
    This includes the power method, using a linear algebra solver, finding the first eigenvector, and searching in Krylov space.
    """    
    def __init__(self,direct=True,method='power'):
        """ 
        By default, use the direct method combined with the power method.
        The indirect method is called with direct=False
        """
        self.direct         = direct  
        self.pi             = None #steady state probability vector
        self.mapping        = {}   #mapping used to identify states
        self.initialState   = None #a dummy initial state
        
        methodSet = ['power','eigen','linear','krylov']
        assert method in methodSet, "Incorrect method specified. Choose from %r" % methodSet
        self.method         = method         
        
    @property
    def size(self):
        """ 
        Return the number of states in the state space
        """
        return len(self.mapping)
       
    def statespace(self):
        """
        To be provided by the subclass. Return the state space
        in an integer 2d numpy array with a state on each row.
        """
        raise NotImplementedError('Implement the function statespace() in the subclass')    

    def transition(self, state):
        """
        To be provided by the subclass. 
        Return a 2d numpy array with reachable states and a 1d numpy array with transition rates.
        For the iterative method, it is also allowed to return a dictionary where the keys are tuples with the state and the values are the transition rates.
        It is preferred to return unique states (for the iterative method, the returned states MUST be unique).
        """
        raise NotImplementedError('Implement the function transition() in the subclass')  

    def checkInitialState(self,initialState):
        """
        Check whether the initial state is of the correct type.
        The state should be either an int, list, tuple or np.array and all its elements must be integer.
        Returns an int if the state is an integer, otherwise a tuple.
        """
        assert initialState is not None, "Initial state has not been specified."
        assert isinstance(initialState,(int,list,tuple,np.ndarray)), "initialState %r is not an int, tuple, list, or numpy array" % initialState

        if isinstance(initialState,(list,tuple)):
            assert all(isinstance(i, int) for i in initialState), "initialState %r is not integer" % initialState 
            initialState = int(initialState) if len(initialState)==1 else tuple(initialState) 
        elif isinstance(initialState,np.ndarray):
            assert issubclass(initialState.dtype.type, np.integer) and initialState.ndim==1, "initialState %r is not a one-dimensional integer numpy array" % initialState 
            initialState = int(initialState) if len(initialState)==1 else tuple(initialState) 

        return initialState


    def checkTransitionType(self,state):
        """
        Check whether the transition function returns output of the correct types.
        This can be either a dictionary with as keys ints/tuples and values floats.
        Or a tuple consisting of a 2d integer numpy array with states and a 1d numpy array with rates.  
        """        
        test = self.transition(state)
        assert isinstance(test,(dict,tuple)), "Transition function does not return a dict or tuple"
        
        if isinstance(test,dict):
            assert all(isinstance(states, (int,tuple)) for states in test.keys()), "Transition function returns a dict, but states are not represented as tuples or integers"
            assert all(isinstance(rates, float) for rates in test.values()), "Transition function returns a dict, but the rates should be floats."
            usesNumpy=False
            
        if isinstance(test,tuple):
            assert len(test)==2, "The transition function should return two variables: states and rates."
            states,rates = test
            assert isinstance(states, np.ndarray) and states.ndim==2 and issubclass(states.dtype.type, np.integer), "The states returned by the transition function need to be an integer 2d numpy array: %r" %states
            assert isinstance(rates, np.ndarray) and rates.ndim==1, "The rates returned by the transition function need to be a 1d numpy array: %r" % rates
            usesNumpy = True
            
        return usesNumpy         
    
    def convertToTransitionDict(self,transitions):
        """
        If numpy is used, then this turns the output from transition() into a dict.  
        """  
        states,rates = transitions
        rateDict = defaultdict(float)
        if states.shape[1] == 1:
            for idx,state in enumerate(states):
                rateDict[int(state)] += rates[idx]
        else:
            for idx,state in enumerate(states):
                rateDict[tuple(state)] += rates[idx]        
        return rateDict         
    
    def indirectInitialMatrix(self, initialState):
        """
        Given some initial state, this code iteratively determines new states.
        We repeatedly call the transition function on unvisited states in the frontier set.
        Each newly visited state is put in a dictionary called 'mapping' and the rates are stored in a dictionary.
        """
        
        #Check whether the initial state is defined and of the correct type 
        initState               = self.checkInitialState(initialState)   
                
        #Now test if the transition function returns a dict or a numpy array.
        #It is more robust to call this after every transition. However, we do it once to save time.
        usesNumpy               = self.checkTransitionType(initialState)

        mapping                 = {}           
        mapping[initState]      = 0
        frontier                = set( [initState] )
        rates                   = OrderedDict()
        
        while len(frontier) > 0:
            fromstate = frontier.pop()
            fromindex = mapping[fromstate]    
            
            if usesNumpy: #If numpy is used, convert to a dictionary with tuples and rates.
                transitions = self.transition(np.array(fromstate))
                transitions = self.convertToTransitionDict(transitions) 
            else:
                transitions = self.transition(fromstate)
                
            for tostate,rate in transitions.items():
                if tostate not in mapping:
                    frontier.add(tostate)
                    mapping[tostate] = len(mapping)
                toindex                     = mapping[tostate]
                rates[(fromindex, toindex)] = rate

        #Inverse the keys and values in mapping to get a dictionary with indices and states.
        self.mapping = {value: key for key, value in list(mapping.items())}
        
        #Now use the keys and values of the rates dictionary to fill up a sparse coo_matrix.
        rateArray = np.array(list(rates.keys()))
        rows      = rateArray[:,0]
        cols      = rateArray[:,1]
        return coo_matrix((np.array(list(rates.values())),(rows,cols)),shape=(self.size,self.size),dtype=float).tocsr()
       
    def getStateCode(self,state):
        """                
        Calculates the state code for a specific state or set of states.
        We transform the states so that they are nonnegative and take an inner product.
        The resulting number is unique because we use numeral system with a large enough base.
        """
        return np.dot(state-self.minvalues,self.statecode)
            
    def setStateCodes(self):    
        """                
        Generates (sorted) codes for the states in the statespace
        This is used to quickly identify which states occur after a transition/action
        """

        #calculate the statespace and determine the minima and maxima each element in the state vector     
        statespace      = self.statespace()     
        self.minvalues  = np.amin(statespace,axis=0)
        self.maxvalues  = np.amax(statespace,axis=0)
        
        #calculate the largest number of values and create a state code        
        statesize       = statespace.shape[1]  
        largestRange    = 1+np.max(self.maxvalues-self.minvalues) 
        self.statecode  = np.power(largestRange, np.arange(statesize),dtype=int) 
   
        #Calculate the codes, sort them, and store them in self.codes
        codes           = self.getStateCode(statespace)         
        sorted_indices  = np.argsort(codes)
        self.codes      = codes[sorted_indices]  
        if np.unique(self.codes).shape != self.codes.shape:
            raise "Non-unique coding of states, results are unreliable"
            
        #For the end results, it is useful to put the indices and corresponding states in a dictionary        
        mapping = OrderedDict()
        for index,state in enumerate(statespace[sorted_indices]):
            mapping[index]  = state    
            
        self.mapping        = mapping  

    def getStateIndex(self,state):
        """
        Returns the index of a state by calculating the state code and searching for this code a sorted list.
        Can be called on multiple states at once.
        """
        statecodes = self.getStateCode(state)
        return np.searchsorted(self.codes,statecodes).astype(int)  
     
    def transitionStates(self,state):
        """
        Return the indices of new states, the rates, and the number of transitions. 
        """
        newstates,rates         = self.transition(state)              
        newindices              = self.getStateIndex(newstates)  
        return newindices,rates

    def directInitialMatrix(self):   
        """
        We generate an initial sparse matrix with all the transition rates (or probabilities).
        We later transform this matrix into a rate or probability matrix depending on the preferred method of obtaining pi.
        """
        
        #First initialize state codes and the mapping with states. 
        self.setStateCodes()  

        #For each state, calculate the indices of reached states and rates using the transition function.
        results  = imap(self.transitionStates, self.mapping.values())

        #preallocate memory for the rows, cols and rates of the sparse matrix      
        rows = np.empty(self.size,dtype=int)
        cols = np.empty(self.size,dtype=int)
        rates = np.empty(self.size,dtype=float)        
        
        #now fill the arrays with the results, increasing their size if current memory is too small.
        right = 0
        for index,(col,rate) in enumerate(results): #more robust alternative: in izip(self.mapping.keys(),results)
            left = right
            right += len(col)
            if right >= len(cols):
                new_capacity = int(round(right * 1.5))  #increase the allocated memory if the vectors turn out to be too small.
                cols.resize(new_capacity)
                rates.resize(new_capacity)
                rows.resize(new_capacity)
            rows[left:right] = index #since states are sorted, the index indeed corresponds to the state.
            cols[left:right] = col
            rates[left:right] = rate   
           
        #Place all data in a coo_matrix and convert to a csr_matrix for quick computations.
        return coo_matrix((rates[:right],(rows[:right],cols[:right])),shape=(self.size,self.size)).tocsr() 

    def convertToRateMatrix(self, Q):
        """
        Converts the initial matrix to a rate matrix.
        We make all rows in Q sum to zero by subtracting the row sums from the diagonal.
        """
        rowSums             = Q.sum(axis=1).getA1()
        Qdiag               = coo_matrix((rowSums,(np.arange(self.size),np.arange(self.size))),shape=(self.size,self.size)).tocsr()
        return Q-Qdiag

    def convertToProbabilityMatrix(self, Q):
        """
        Converts the initial matrix to a probability matrix
        We calculate P = I + Q/l, with l the largest diagonal element.
        Even if Q is already a probability matrix, this step helps for numerical stability. 
        By adding a small probability on the diagional (0.001), periodicity can be prevented.
        """
        rowSums             = Q.sum(axis=1).getA1()
        l                   = np.max(rowSums)*1.001
        diagonalElements    = 1.-rowSums/l
        Qdiag               = coo_matrix((diagonalElements,(np.arange(self.size),np.arange(self.size))),shape=(self.size,self.size)).tocsr()
        return Qdiag+Q.multiply(1./l)

    def assertSingleClass(self,P):
        """ 
        Check whether the rate/probability matrix consists of a single connected class.
        Otherwise, the steady state distribution is not well defined.
        """
        components, _ = csgraph.connected_components(P, directed=True, connection='weak')   
        assert components==1, "The Markov chain has %r communicating classes. Make sure there is a single communicating class." %components
           
    def getTransitionMatrix(self,probabilities=True):
        """
        Depending on whether the iterative method it chosen, calculate the generator matrix Q of the Markov chain
        Since most methods use stochastic matrices, by default we return a stochastic matrix
        """
        if self.direct == True:
            P = self.directInitialMatrix()
        else:
            P = self.indirectInitialMatrix(self.initialState)             
            
        if probabilities:    
            P = self.convertToProbabilityMatrix(P)
        else: 
            P = self.convertToRateMatrix(P)
            
        self.assertSingleClass(P)    
        
        return P
                     
    def power(self, tol = 1e-8, numIter = 1e5):
        """
        Carry out the power method. Repeatedly take the dot product to obtain pi.
        """
        P = self.getTransitionMatrix().T #take transpose now to speed up dot product.
        pi = np.zeros(self.size);  pi1 = np.zeros(self.size)
        pi[0] = 1;
        n = norm(pi - pi1,1); i = 0;
        while n > tol and i < numIter:
            pi1 = P.dot(pi)
            pi = P.dot(pi1)
            n = norm(pi - pi1,1); i += 1
        self.pi = pi

    def eigen(self, tol = 1e-8, numIter = 1e5):  
        """
        Determines the eigenvector corresponding to the first eigenvalue.
        The speed of convergence depends heavily on the choice of the initial guess for pi.
        For now, we let the initial pi be a vector of ones.
        """
        Q = self.getTransitionMatrix(probabilities=False)
        guess = np.ones(self.size,dtype=float)
        w, v = eigs(Q.T, k=1, v0=guess, sigma=1e-6, which='LM',tol=tol, maxiter=numIter)
        pi = v[:, 0].real
        pi /= pi.sum()
        
        self.pi = pi
        
    def linear(self): 
        """
        Here we use the standard linear algebra solver to obtain pi from a system of linear equations. 
        The first equation isreplaced by the normalizing condition.
        Consumes a lot of memory.
        Code due to http://stackoverflow.com/questions/21308848/
        """
        P       = self.getTransitionMatrix()
        dP      = P - eye(self.size)
        A       = vstack([np.ones(self.size), dP.T[1:,:]]).tocsr()
        rhs     = np.zeros((self.size,))
        rhs[0]  = 1
        
        self.pi = spsolve(A, rhs)

    def krylov(self,tol=1e-8): 
        """
        Here we use the 'gmres' solver for the system of linear equations. 
        It searches in Krylov subspace for a vector with minimal residual. 
        Code due to http://stackoverflow.com/questions/21308848/
        """
        P       = self.getTransitionMatrix()
        dP      = P - eye(self.size)
        A       = vstack([np.ones(self.size), dP.T[1:,:]]).tocsr()
        rhs     = np.zeros((self.size,))
        rhs[0]  = 1
                
        pi, info = gmres(A, rhs, tol=tol)
        if info != 0:
            raise RuntimeError("gmres did not converge")
        self.pi = pi
        
    def computePi(self):
        """
        Calculate the steady state distribution using the preferred method.
        """
        return getattr(self,self.method)()

    def printPi(self):
        """
        Prints all states state and their steady state probabilities.
        Not recommended for large state spaces.
        """
        if self.pi is not None:
            for key,state in self.mapping.items():
                print(state,self.pi[key])
