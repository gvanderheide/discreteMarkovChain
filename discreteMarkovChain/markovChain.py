""" 
Possible fixes:
-Check that the state codes do not suffer from integer overflow.
-Improve memory usage.
"""
from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix, csgraph, eye, vstack, isspmatrix, isspmatrix_csr
from scipy.sparse.linalg import eigs, gmres, spsolve, inv
from numpy.linalg import norm
from collections import OrderedDict,defaultdict


from scipy.sparse import dok_matrix

try: #For python 3 functionality.
    from itertools import imap
except ImportError:
    imap = map

class markovChain(object): 
    """
    A class for calculating the steady state distribution of a Markov chain with a finite and discrete state space.
    The Markov chain can be defined on continuous time or discrete time and states can be integers or vectors of integers.
    
    Summary
    -------
    If the transition matrix ``P`` is specified by the user, we use that for calculating the steady-state distribution.
    Otherwise, we derive ``P`` automatically using an indirect or a direct method. 
    
    Both the indirect and direct method require the function :func:`transition` to be defined within the class, calculating for each state the reachable states and corresponding rates/probabilities.
    Implementing this function is similar in difficulty as constructing ``P`` manually, since when you construct ``P`` you also have to determine where you can go from any given state. 
    
    For the indirect method the user needs to specify an initial state in the class attribute ``initialState``.
    By repeatedly calling the transition function on unvisited states, all reachable states are determined starting from this initial state.

    For the direct method the function :func:`statespace` is required, giving the complete state space in a 2d numpy array. 
    We build up ``P`` by calling :func:`transition` on each state in the statespace.  

            
    Steady state distributions can be calculated by calling :func:`computePi` with a method of choice.

    Parameters
    ----------
    P : array(float,ndim=2), optional(default=None)
        Optional argument. The transition matrix of the Markov chain. Needs to have an equal number of columns and rows. Can be sparse or dense. 
    direct : bool, optional(default=False)
        Specifies whether the indirect method is used or the direct method in case ``P`` is not defined. By default, ``direct=False``.
    
    Attributes
    ----------
    pi : array(float)
        The steady state probabilities for each state in the state space.    
    mapping : dict
        The keys are the indices of the states in ``P`` and ``pi``,  the values are the states. Useful only when using the direct/indirect method.    
    size : int
        The size of the state space.    
    P : scipy.sparse.csr_matrix
        The sparse transition/rate matrix.    
    initialState : int or array_like(int)
        State from which to start the indirect method. Should be provided in the subclass by the user.  
    
    Methods
    -------
    transition(state)
        Transition function of the Markov chain, returning the reachable states from `state` and their probabilities/rates. 
        Should be be provided in the subclass by the user when using the indirect/direct method.
    statespace()
        Returns the state space of the Markov chain. Should be be provided in the subclass by the user when using the direct method.
    computePi(method='power') 
        Call with 'power','linear','eigen' or 'krylov' to use a certain method for obtaining ``pi``.
    printPi() 
        Print all states and their corresponding steady state probabilities. Not recommended for large state spaces.
    linearMethod() 
        Use :func:`spsolve`, the standard linear algebra solver for sparse matrices, to obtain ``pi``.
    powerMethod(tol=1e-8,numIter=1e5) 
        Use repeated multiplication of the transition matrix to obtain ``pi``.
    eigenMethod(tol=1e-8,numIter=1e5)  
        Search for the first left eigenvalue to  obtain ``pi``.
    krylovMethod(tol=1e-8)    
        Search for ``pi`` in Krylov subspace using the :func:`gmres` procedure for sparse matrices.

    Example
    -------
    >>> P = np.array([[0.5,0.5],[0.6,0.4]])
    >>> mc = markovChain(P)
    >>> mc.computePi('power') #alternative: 'linear','eigen' or 'krylov'.   
    >>> print(mc.pi)
    [ 0.54545455  0.45454545]
    
    """
    def __init__(self,P=None,direct=False):
        self.P              = P
        self.direct         = direct  
        self.pi             = None #steady state probability vector
        self.mapping        = {}   #mapping used to identify states
        self.initialState   = None #a dummy initial state             
        
    @property
    def size(self):
        """ 
        Return the number of states in the state space, if ``self.mapping`` is defined.
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
        Ensure that unique states are returned.
        """
        raise NotImplementedError('Implement the function transition() in the subclass')  

    def checkInitialState(self,initialState):
        """
        Check whether the initial state is of the correct type.
        The state should be either an int, list, tuple or np.array and all its elements must be integer.
        Returns an int if the state is an integer, otherwise a tuple.
        """
        assert initialState is not None, "Initial state has not been specified."
        assert isinstance(initialState,(int,list,tuple,np.ndarray,set)), "initialState %r is not an int, tuple, list, set or numpy array" % initialState

         
        if isinstance(initialState,list): 
            #Check whether all entries of the list are ints. Return an int if the len ==1, otherwise a tuple.
            assert all(isinstance(i, int) for i in initialState), "initialState %r is not integer" % initialState 
            initialState = int(initialState) if len(initialState)==1 else tuple(initialState)          

        elif isinstance(initialState,tuple): 
            #Check whether all entries are ints. Return an int if the len ==1, otherwise a tuple.
            assert all(isinstance(i, int) for i in initialState), "initialState %r is not integer" % initialState
            if len(initialState)==1:
                initialState = int(initialState)     
        
        elif isinstance(initialState,np.ndarray):
            #Check whether the state is a 1d numpy array. Return an int if it has length 1.
            assert issubclass(initialState.dtype.type, np.integer) and initialState.ndim==1, "initialState %r is not a one-dimensional integer numpy array" % initialState 
            initialState = int(initialState) if len(initialState)==1 else tuple(initialState)
            
        elif isinstance(initialState,set):
            #If we have a set, then check whether all elements are ints or tuples.
            for state in initialState:                
                assert isinstance(state,(tuple,int)), "the set initialState %r should contain tuples or ints" % initialState 
                if isinstance(state,tuple):
                    assert all(isinstance(i,int) for i in state), "the state %r should be integer" % initialState 
                
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
        If numpy is used, then this converts the output from transition() into a dict.  
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
        Given some initial state, this iteratively determines new states.
        We repeatedly call the transition function on unvisited states in the frontier set.
        Each newly visited state is put in a dictionary called 'mapping' and the rates are stored in a dictionary.
        """
        mapping                 = {}   
        rates                   = OrderedDict()
        
        #Check whether the initial state is defined and of the correct type, and convert to a tuple or int. 
        convertedState           = self.checkInitialState(initialState)        

        if isinstance(convertedState,set):
            #If initialstates is a set, include all states in the set in the mapping.
            frontier = set( convertedState )
            for idx,state in enumerate(convertedState):  
                mapping[state] = idx
                if idx == 0: #Test the return type of the transition function (dict or numpy).
                    usesNumpy = self.checkTransitionType(initialState)                
        else:
            #Otherwise include only the single state.
            frontier                = set( [convertedState] )
            usesNumpy               = self.checkTransitionType(initialState)
            mapping[convertedState] = 0       
     

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
        
        #Use the `rates` dictionary to fill a sparse dok matrix.
        D = dok_matrix((self.size,self.size))
        D.update(rates)
        return D.tocsr()
        

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
        Return the indices of new states and their rates. 
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
        
         #Simpler alternative that uses less memory. 
         #Would be competitive if the conversion from dok to csr is faster.  
#        D = dok_matrix((self.size,self.size),dtype=float)
#        for index,(col,rate) in enumerate(results):
#            D.update({(index,c): r for c,r in zip(col,rate)})
#        return D.tocsr()

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
        idxRange            = np.arange(Q.shape[0])
        Qdiag               = coo_matrix((rowSums,(idxRange,idxRange)),shape=Q.shape).tocsr()
        return Q-Qdiag

    def convertToProbabilityMatrix(self, Q):
        """
        Converts the initial matrix to a probability matrix
        We calculate P = I + Q/l, with l the largest diagonal element.
        Even if Q is already a probability matrix, this step helps for numerical stability. 
        By adding a small probability on the diagonal (0.00001), periodicity can be prevented.
        """
        rowSums             = Q.sum(axis=1).getA1()
        l                   = np.max(rowSums)*1.00001
        diagonalElements    = 1.-rowSums/l
        idxRange            = np.arange(Q.shape[0])
        Qdiag               = coo_matrix((diagonalElements,(idxRange,idxRange)),shape=Q.shape).tocsr()
        return Qdiag+Q.multiply(1./l)

    def assertSingleClass(self,P):
        """ 
        Check whether the rate/probability matrix consists of a single connected class.
        If this is not the case, the steady state distribution is not well defined.
        """
        components, _ = csgraph.connected_components(P, directed=True, connection='weak')   
        assert components==1, "The Markov chain has %r communicating classes. Make sure there is a single communicating class." %components
           
    def getTransitionMatrix(self,probabilities=True):
        """
        If self.P has been given already, we will reuse it and convert it to a sparse csr matrix if needed.
        Otherwise, we will generate it using the direct or indirect method.         
        Since most solution methods use a probability matrix, this is the default setting. 
        By setting probabilities=False we can also return a rate matrix.
        """
        if self.P is not None:               
            if isspmatrix(self.P): 
                if not isspmatrix_csr(self.P):
                    self.P = self.P.tocsr() 
            else:
                assert isinstance(self.P, np.ndarray) and self.P.ndim==2 and self.P.shape[0]==self.P.shape[1],'P needs to be a 2d numpy array with an equal number of columns and rows'                     
                self.P = csr_matrix(self.P)   
                
        elif self.direct == True:
            self.P = self.directInitialMatrix()
            
        else:
            self.P = self.indirectInitialMatrix(self.initialState)   

        if probabilities:    
            P = self.convertToProbabilityMatrix(self.P)
        else: 
            P = self.convertToRateMatrix(self.P)   
        
        return P
        
    def getIrreducibleTransitionMatrix(self,probabilities=True):
        #Gets the transitionmatrix and assert that it consists of a single irreducible class.
        P = self.getTransitionMatrix(probabilities=True)
        self.assertSingleClass(P)   
        return P
                     
    def powerMethod(self, tol = 1e-8, maxiter = 1e5):
        """
        Carry out the power method and store the result in the class attribute ``pi``. 
        Repeatedly takes the dot product between ``P`` and ``pi`` until the norm is smaller than the prespecified tolerance ``tol``.
          
        Parameters
        ----------
        tol : float, optional(default=1e-8)
            Tolerance level for the precision of the end result. A lower tolerance leads to more accurate estimate of ``pi``.
        maxiter : int, optional(default=1e5)
            The maximum number of power iterations to be carried out.            
        
        Example
        -------
        >>> P = np.array([[0.5,0.5],[0.6,0.4]])
        >>> mc = markovChain(P)
        >>> mc.powerMethod()
        >>> print(mc.pi) 
        [ 0.54545455  0.45454545]
        
        Remarks
        -------
        The power method is robust even when state space becomes large (more than 500.000 states), whereas the other methods may have some issues with memory or convergence.
        The power method may converge slowly for Markov chains where states are rather disconnected. That is, when the expected time to go from one state to another is large.        
        """
        P = self.getIrreducibleTransitionMatrix().T #take transpose now to speed up dot product.
        size = P.shape[0]        
        pi = np.zeros(size);  pi1 = np.zeros(size)
        pi[0] = 1;
        n = norm(pi - pi1,1); i = 0;
        while n > tol and i < maxiter:
            pi1 = P.dot(pi)
            pi = P.dot(pi1)
            n = norm(pi - pi1,1); i += 1
        self.pi = pi

    def eigenMethod(self, tol = 1e-8, maxiter = 1e5):
        """
        Determines ``pi`` by searching for the eigenvector corresponding to the first eigenvalue, using the :func:`eigs` function. 
        The result is stored in the class attribute ``pi``.         
        
        Parameters
        ----------
        tol : float, optional(default=1e-8)
            Tolerance level for the precision of the end result. A lower tolerance leads to more accurate estimate of ``pi``.
        maxiter : int, optional(default=1e5)
            The maximum number of iterations to be carried out.            
        
        Example
        -------
        >>> P = np.array([[0.5,0.5],[0.6,0.4]])
        >>> mc = markovChain(P)
        >>> mc.eigenMethod()
        >>> print(mc.pi) 
        [ 0.54545455  0.45454545]
        
        Remarks
        -------
        The speed of convergence depends heavily on the choice of the initial guess for ``pi``.
        Here we let the initial ``pi`` be a vector of ones. 
        For large state spaces, this method may not work well.
        At the moment, we call :func:`powerMethod` if the number of states is 2.
        Code is due to a colleague: http://nicky.vanforeest.com/probability/markovChains/markovChain.html
        """
        Q = self.getIrreducibleTransitionMatrix(probabilities=False)
        
        if Q.shape == (1, 1):
            self.pi = np.array([1.0]) 
            return          
        
        if Q.shape == (2, 2):
            self.pi= np.array([Q[1,0],Q[0,1]]/(Q[0,1]+Q[1,0]))
            return                 
        
        size = Q.shape[0]
        guess = np.ones(size,dtype=float)
        w, v = eigs(Q.T, k=1, v0=guess, sigma=1e-6, which='LM',tol=tol, maxiter=maxiter)
        pi = v[:, 0].real
        pi /= pi.sum()
        
        self.pi = pi
        
    def linearMethod(self): 
        """
        Determines ``pi`` by solving a system of linear equations using :func:`spsolve`. 
        The method has no parameters since it is an exact method. The result is stored in the class attribute ``pi``.   
     
        Example
        -------
        >>> P = np.array([[0.5,0.5],[0.6,0.4]])
        >>> mc = markovChain(P)
        >>> mc.linearMethod()
        >>> print(mc.pi) 
        [ 0.54545455  0.45454545]
        
        Remarks
        -------
        For large state spaces, the linear algebra solver may not work well due to memory overflow.
        Code due to http://stackoverflow.com/questions/21308848/
        """    
        P       = self.getIrreducibleTransitionMatrix()        
      
        #if P consists of one element, then set self.pi = 1.0
        if P.shape == (1, 1):
            self.pi = np.array([1.0]) 
            return  
     
        size    = P.shape[0]
        dP      = P - eye(size)
        #Replace the first equation by the normalizing condition.
        A       = vstack([np.ones(size), dP.T[1:,:]]).tocsr()  
        rhs     = np.zeros((size,))
        rhs[0]  = 1   
        
        self.pi = spsolve(A, rhs)

    def krylovMethod(self,tol=1e-8): 
        """
        We obtain ``pi`` by using the :func:``gmres`` solver for the system of linear equations. 
        It searches in Krylov subspace for a vector with minimal residual. The result is stored in the class attribute ``pi``.   

        Example
        -------
        >>> P = np.array([[0.5,0.5],[0.6,0.4]])
        >>> mc = markovChain(P)
        >>> mc.krylovMethod()
        >>> print(mc.pi) 
        [ 0.54545455  0.45454545]
        
        Parameters
        ----------
        tol : float, optional(default=1e-8)
            Tolerance level for the precision of the end result. A lower tolerance leads to more accurate estimate of ``pi``.        
        
        Remarks
        -------
        For large state spaces, this method may not always give a solution. 
        Code due to http://stackoverflow.com/questions/21308848/
        """            
        P       = self.getIrreducibleTransitionMatrix()
        
        #if P consists of one element, then set self.pi = 1.0
        if P.shape == (1, 1):
            self.pi = np.array([1.0]) 
            return
            
        size    = P.shape[0]
        dP      = P - eye(size)
        #Replace the first equation by the normalizing condition.
        A       = vstack([np.ones(size), dP.T[1:,:]]).tocsr()
        rhs     = np.zeros((size,))
        rhs[0]  = 1
                
        pi, info = gmres(A, rhs, tol=tol)
        if info != 0:
            raise RuntimeError("gmres did not converge")
        self.pi = pi
        
    def computePi(self,method='power'):
        """
        Calculate the steady state distribution using your preferred method and store it in the attribute `pi`. 
        By default uses the most robust method, 'power'. Other methods are 'eigen','linear', and 'krylov'
        
        Parameters
        ----------
        method : string, optional(default='power')
            The method for obtaining ``pi``. The possible options are 'power','eigen','linear','krylov'. 
        
        Example
        -------
        >>> P = np.array([[0.5,0.5],[0.6,0.4]])
        >>> mc = markovChain(P)
        >>> mc.computePi('power') 
        >>> print(mc.pi)
        [ 0.54545455  0.45454545]        
        
        See Also
        --------
        For details about the specific methods see
        :func:`powerMethod`,
        :func:`eigenMethod`,
        :func:`linearMethod`, and
        :func:`krylovMethod` .       
        """
        methodSet = ['power','eigen','linear','krylov']
        assert method in methodSet, "Incorrect method specified. Choose from %r" % methodSet
        method = method + 'Method'         
        return getattr(self,method)()

    def printPi(self):
        """
        Prints all states state and their steady state probabilities.
        Not recommended for large state spaces.
        """
        assert self.pi is not None, "Calculate pi before calling printPi()"
        assert len(self.mapping)>0, "printPi() can only be used in combination with the direct or indirect method. Use print(mc.pi) if your subclass is called mc."        
        for key,state in self.mapping.items():
            print(state,self.pi[key])

class finiteMarkovChain(markovChain):
    def __init__(self,P=None):
        super(finiteMarkovChain,self).__init__(P)    
    
    def absorbTime(self):
        P = self.getTransitionMatrix(probabilities=True)
        components,labels = csgraph.connected_components(P, directed=True, connection='strong',return_labels=True)
        if components == 1:
            print("no absorbing states")
            return
            
        transientStates = np.ones(P.shape[0],dtype=bool)

        for component in range(components):
            indices = np.where(labels==component)[0]
            n = len(indices)
            if n==1:
                probSum = P[indices,indices].sum()
            else:            
                probSum = P[np.ix_(indices,indices)].sum()
            if np.isclose(probSum,n):
                transientStates[indices] = False

        indices = np.where(transientStates)[0]
        n = len(indices)
        if n==1:
            Q = P[indices,indices]
        else:
            Q = P[np.ix_(indices,indices)]
        #N will be dense  
        N = inv(eye(n)-Q).A
        N2 = N*(2*N[np.arange(n),np.arange(n)]-np.eye(n))-np.power(N,2)
        t = np.zeros(P.shape[0])
        t[indices] = np.sum(N,axis=1)
        for index in indices:
            print( self.mapping[index],t[index] )
