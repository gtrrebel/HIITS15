import sys
from MOG_demo4 import *
import numpy as np
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from MOG_pickler import *

arg = sys.argv[1:]
NClust = int(arg[0])
dim = int(arg[1])
Nmin = int(arg[2])
Nmax = int(arg[3])
meanscoeff = int(arg[4])
method = arg[5]
restarts = int(arg[6])
dir_name = arg[-1]
specs = arg[0:-1]

def get_data(NClust, dim, Nmin, Nmax, meanscoeff, method, restarts):
	m = init([[NClust, dim, Nmin, Nmax, meanscoeff]])
	res = run(m, restarts=restarts, end_gather=['bound', 'optimizetime'], methods=[method])[0]
	output = {}
	bounds = [doc['bound'] for doc in res]
	times = [doc['optimizetime'] for doc in res]
	maxruntime = max(times)
	averuntime = sum(times)/len(times)
	maxbounddiff = max(bounds)-min(bounds)
	boundstd = np.std(bounds)
	data = [maxruntime, averuntime, maxbounddiff, boundstd]
	output['bounds'] = bounds
	output['optimizetimes'] = times
	output['maxruntime'] = maxruntime
	output['averuntime'] = averuntime
	output['boundstd'] = boundstd
	output['maxbounddiff'] = maxbounddiff
	output['method'] = method
	return output

def outstring(spec):
	return '_'.join([str(i) for i in spec])

data = get_data(NClust=NClust, dim=dim, Nmin=Nmin, meanscoeff=meanscoeff, method = method, restarts=restarts)

outputs_pickle(data, directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/' + dir_name,
					ukko = True, filename = outstring(specs))