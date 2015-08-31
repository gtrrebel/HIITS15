from LDA_creator_tester import *

K0c = [0.5, 1, 2]
Vc = [2**(i) for i in xrange(-2, 11)]
K = [2**(i) for i in xrange(2, 9)]
D = [10**(i) for i in xrange(1, 4)]
V = [100*2**(i) for i in xrange(8)]
methods = ['steepest', 'FR']