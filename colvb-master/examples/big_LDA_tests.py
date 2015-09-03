from LDA_demo4 import *
from LDA_creator import *
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA_pickler import *
from itertools import product

K0c = [0.5, 1, 2]
Vc = [2**(i) for i in xrange(-2, 11)]
K = [2**(i) for i in xrange(2, 9)]
D = [10**(i) for i in xrange(1, 4)]
V = [100*2**(i) for i in xrange(8)]
methods = ['steepest', 'FR']

def get_data(topic_params = (1,0,0), word_params = (1,0,0), K0 = 5, V = 50,  D = 10, N = 100, K = None, 
			method = 'steepest', restarts = 10, build_seeds = None, param_seeds = None):
	Ns = [N for _ in xrange(D)]
	if K == None:
		K = K0
	if build_seeds == None:
		build_seeds = [None, None]
	docs, build_seeds = new_create_gaussian_data(topic_params, word_params, K0, V, Ns, build_seeds[0], build_seeds[1])
	docs = np.array(docs)
	voc = np.arange(V)
	m = LDA3(docs, voc, K, make_fns = False)
	res = run([m], end_gather=['bound', 'optimizetime', 'get_seed'], methods=[method], param_seeds=param_seeds)[0][2]
	output = {}
	bounds = [doc['bound'] for doc in res]
	times = [doc['optimizetime'] for doc in res]
	param_seeds = [doc['get_seed'] for doc in res]
	maxruntime = max(times)
	averuntime = sum(times)/len(times)
	maxbounddiff = max(bounds)-min(bounds)
	boundstd = np.std(bounds)
	data = [maxruntime, averuntime, maxbounddiff, boundstd]
	seeds = [build_seeds, param_seeds]
	specs = [topic_params, word_params, K0, V, D, N, K, method]
	output['bounds'] = bounds
	output['optimizetimes'] = times
	output['maxruntime'] = maxruntime
	output['averuntime'] = averuntime
	output['boundstd'] = boundstd
	output['maxbounddiff'] = maxbounddiff
	output['build_seeds'] = build_seeds
	output['param_seeds'] = param_seeds
	output['topic_params'] = topic_params
	output['word_params'] = word_params
	output['K0'] = K0
	output['V'] = V
	output['D'] = D
	output['N'] = N
	output['K'] = K
	output['method'] = method
	output['dimension'] = K*N*D
	return output

def run_spec(spec):
	return get_data(topic_params = spec['topic_params'], word_params = spec['word_params'], K0 = spec['K0'], V = spec['V'],
		D = spec['D'], N = spec['N'], K = spec['K'], method = spec['method'], restarts = spec['restarts'],
		build_seeds = spec['build_seeds'], param_seeds = spec['param_seeds'])

def run_specs(specs):
	outputs = [run_spec(spec) for spec in specs]
	outputs_pickle(outputs)

def run_tests(topic_params = [(1,1,0)], word_params = [(1,1,0)], K0 = [5], V = [50], D = [10], N = [100], K = [None], method = ['steepest'],
			restarts = [10], build_seeds = [None], param_seeds = [None]):
	for tup in product(topic_params, word_params, K0, V, D, N, K, method, restarts, build_seeds, param_seeds):
		run_specs([dict(zip(['topic_params','word_params','K0','V','D','N','K','method','restarts','build_seeds','param_seeds'],list(tup)))])
	
