from __future__ import print_function
import numpy as np

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

def number_of_partitions(max_range, max_sum):
    '''
    Returns an array arr of the same shape as max_range, where
    arr[j] = number of admissible partitions for 
             j summands bounded by max_range[j:] and with sum <= max_sum
    '''
    M = max_sum + 1
    N = len(max_range) 
    arr = np.zeros(shape=(M,N), dtype = int)    
    arr[:,-1] = np.where(np.arange(M) <= min(max_range[-1], max_sum), 1, 0)
    for i in range(N-2,-1,-1):
        for j in range(max_range[i]+1):
            arr[j:,i] += arr[:M-j,i+1] 
    return arr.sum(axis = 0)

def partition(max_range, max_sum, out = None, n_part = None):
    '''
    Function that can be helpful for obtaining the state space of a discrete Markov chain or Markov decision processes. 
    Returns a 2d-array with on the rows all possible partitions of the ranges `0,...,max_range[j]` that add up to at most `max_sum`.

    Code due to ptrj, see http://stackoverflow.com/a/36563744/1479342.  
   
    Parameters
    ----------
    max_range : array or list of ints
        Gives the ranges for each element in the output array. Element `j` has range `np.arange(max_range[j]+1)`. 
    max_sum : int
        The maximum sum for each partition in the output array. 
       
    Returns
    -------
    out : array
        2d array with all possible partitions of the ranges `0,...,max_range[j]` summing up to at most `max_sum`.
        
    Example
    -------
    >>> max_range=np.array([1,3,2])    
    >>> max_sum = 3    
    >>> partition(max_range,max_sum)    
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 2],
           [0, 1, 0],
           [0, 1, 1],
           [0, 1, 2],
           [0, 2, 0],
           [0, 2, 1],
           [0, 3, 0],
           [1, 0, 0],
           [1, 0, 1],
           [1, 0, 2],
           [1, 1, 0],
           [1, 1, 1],
           [1, 2, 0]])
    '''
    if out is None:
        max_range = np.asarray(max_range, dtype = int).ravel()
        n_part = number_of_partitions(max_range, max_sum)
        out = np.zeros(shape = (n_part[0], max_range.size), dtype = int)

    if(max_range.size == 1):
        out[:] = np.arange(min(max_range[0],max_sum) + 1, dtype = int).reshape(-1,1)
        return out

    P = partition(max_range[1:], max_sum, out=out[:n_part[1],1:], n_part = n_part[1:])        

    S = np.minimum(max_sum - P.sum(axis = 1), max_range[0])
    offset, sz  = 0, S.size
    out[:sz,0] = 0
    for i in range(1, max_range[0]+1):
        ind, = np.nonzero(S)
        offset, sz = offset + sz, ind.size
        out[offset:offset+sz, 0] = i
        out[offset:offset+sz, 1:] = P[ind]
        S[ind] -= 1
    return out