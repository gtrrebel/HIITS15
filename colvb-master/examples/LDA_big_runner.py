import sys
from LDA_demo4 import *
from LDA_creator import *
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA_pickler import *

arg = sys.argv[1:]
topic_params = [int(i) for i in arg[:3]]
word_params = [int(i) for i in arg[3:6]]
specs = [int(i) for i in arg[6:12]]
restarts = int(arg[12])
build_seeds = [int(arg(13)), int(arg(14))]
param_seeds = [int(i) for i in xrange(15: (15 + restarts))]
dir_name = arg[-1]

K0cl = [0.5, 1, 2]
Vcl = [2**(i) for i in xrange(-2, 11)]
Kl = [2**(i) for i in xrange(2, 9)]
Dl = [10**(i) for i in xrange(1, 4)]
Nl = [100*2**(i) for i in xrange(8)]
methodsl = ['steepest', 'FR']

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

def outstring(spec):
	return '_'.join([str(i) for i in spec])

K0c = KOcl[specs[0]]
Vc = Vcl[specs[1]]
K = Kl[specs[2]]
D = Dl[specs[3]]
N = Nl[specs[4]]
method = methodsl[specs[5]]
V = Vc*K
K0 = K0c*K

data = get_data(topic_params= topic_params, word_params=word_params, K0 = K0, V = V, D = D, N = N, K = K, method = method,
			build_seeds = build_seeds, param_seeds=param_seeds)

outputs_pickle(data, directory = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_outputs/' + dir_name,
					ukko = True, filename = outstring(specs))
