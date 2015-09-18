import sys
from LDA_demo4 import *
from LDA_creator import *
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')


def get_data(topic_params, word_params, K0, V, D, N, K, method, lengths, alpha_0 = 200):
	Ns = [N for _ in xrange(D)]
	docs, build_seeds = new_create_gaussian_data(topic_params, word_params, K0, V, Ns)
	docs = np.array(docs)
	voc = np.arange(V)
	m = LDA3(docs, voc, K, alpha_0=alpha_0, make_fns = False)
	m.optimize()
	starts = []
	ends = []
	iterations = []
	for l in lengths:
		m.random_jump(l)
		starts.append(m.bound())
		m.optimize()
		iterations.append(m.iteration)
		ends.append(m.bound())
	diff = sum(abs(starts[i] - ends[i]) for i in xrange(len(starts)))/len(starts)
	std = np.std(np.array(ends))
	ite = sum(iterations)/len(iterations)
	return diff, std, ite


