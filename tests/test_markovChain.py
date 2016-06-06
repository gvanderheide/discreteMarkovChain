from __future__ import print_function
import unittest
import numpy as np
from discreteMarkovChain.markovChain import markovChain


class TestMarkovChain(unittest.TestCase):

    def test_example1(self):
        P = np.array([[0.5,0.5], [0.6,0.4]])
        mc = markovChain(P)
        mc.computePi('linear')
 
