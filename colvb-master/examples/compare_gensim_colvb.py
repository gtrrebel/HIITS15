import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/Gensim/')
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
from parse_vocab import *
from LDA import LDA

def test1(length = 100, K = 10):
	docs, voc = get_colvb_corpus(length = length)
	m = LDA(docs, voc, K)
	m.optimize()
