import LDA_demo4
import MOG_demo4
from LDA3 import LDA3
from LDA_creator import *

def test_trust_LDA(trust_count = 10, topic_params=(1,1,0), word_params =(1,1,0), K0=5, K = 5, V = 50, N = 50, D = 20):
	Ns = [N for _ in xrange(D)]
	docs, build_seeds = new_create_gaussian_data(topic_params, word_params, K0, V, Ns)
	docs = np.array(docs)
	voc = np.arange(V)
	m = LDA3(docs, voc, K, make_fns = False)
	out1 = LDA_demo4.run([m], end_gather=['bound'])
	av1 = sum(out['bound'] for out in out1[0][2])/10
	out2 = LDA_demo4.run([m], end_gather=['bound'], trust_count = 10)
	av2 = sum(out['bound'] for out in out2[0][2])/10
	return (av2 - av1)/av1

def test_trust_MOG(trust_count = 10, args = [10, 2, 10, 30, 5]):
	m = MOG_demo4.init([args])[0]
	out1 = MOG_demo4.run([m], end_gather=['bound'])
	av1 = sum(out['bound'] for out in out1[0])/10
	out2 = MOG_demo4.run([m], end_gather=['bound'], trust_count = 10)
	av2 = sum(out['bound'] for out in out2[0])/10
	return (av2 - av1)/av1