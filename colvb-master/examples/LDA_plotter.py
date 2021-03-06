import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

def result_filepath(ukko=False, form='txt'):
	return result_dir(ukko, form) + result_filename(form)

def result_dir(ukko=False, form='txt'):
	if ukko:
		results = "/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/"
	else:
		results = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/'
	if form == 'txt':
		return results + 'textdata/'
	else:
		return results + 'picdata/'

def result_filename(form='txt'):
	return "LDA_demo3." + time.strftime("%H:%M:%S-%d.%m.%Y") + "." + form

def get_data(arg):
	end_gather, save_specs, end_returns = arg
	data = {}
	for spec in end_gather:
		data[spec] = []
	for run in end_returns:
		for spec in run:
			data[spec].append(run[spec])
	delta = 0e-2
	randomization = delta*np.random.randn((len(data[end_gather[1]])))
	xs, ys = data[end_gather[0]], data[end_gather[1]]+randomization
	return xs, ys, end_gather, save_specs

def plot(args):
	for arg in args:
		xs, ys, end_gather, save_specs = get_data(arg)
		fig = plt.figure()
		plt.title(save_specs[0])
		plt.xlabel(end_gather[0])
		plt.ylabel(end_gather[1])
		plt.plot(xs, ys, 'or')
		plt.ylim(-1, max(ys) + 1)
		plt.show()

def save(args, form = 'txt', ukko=False):
	if form == 'txt':
		save_data(args, ukko)
	elif form == 'png':
		save_plot(args, ukko)

def save_plot(args, ukko=False):
	for arg in args:
		plot([arg])
		plt.savefig(result_filepath(ukko, 'png'))

def save_data(args, ukko=False):
	for arg in args:
		end_gather, save_specs, end_returns = arg
		savestring = save_specs[0] + '\n' + ' '.join(end_gather) + '\n' + \
					'\n'.join([' '.join([str(run[gather]) for gather in end_gather]) for run in end_returns]) + '\n'
		with open(result_filepath(ukko, 'txt'), 'w') as f:
			f.write(savestring)

def plot_data(filenames):
	args = []
	for filename in filenames:
		with open(interpret_filename(filename), 'r') as f:
			lines = f.readlines()
			save_specs = [lines[0]]
			end_gather = lines[1].split()
			end_returns = [dict(zip(end_gather, [float(i) for i in line.split()])) for line in lines[2:]]
			args.append((end_gather, save_specs, end_returns))
	plot(args)

def interpret_filename(filename, ukko=False):
	if filename[0] == '/':
		return filename
	elif ukko:
		return '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/textdata/' + filename
	else:
		return '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/textdata/' + filename

def bound_plot(bounds, dims):
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	for arg in zip(bounds, dims):
		plt.plot([arg[1]]*len(arg[0]), [a - min(arg[0]) for a in arg[0]],'ro')
	plt.show()

def hessian_bound_plot(good, bad, neutral):
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	hessian_bound_plot_helper('neutral', neutral)
	hessian_bound_plot_helper('good', good)
	hessian_bound_plot_helper('bad', bad)
	plt.show()

def hessian_bound_plot_helper(al, data):
	ys = [bou[0] for bou in data]
	xs = [bou[1] for bou in data]
	if al == 'good':
		plt.plot(xs, ys, 'go')
	elif al == 'bad':
		plt.plot(xs, ys, 'ro')
	elif al == 'neutral':
		plt.plot(xs, ys, 'yo')

def bound_optimizetime_plot(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c = 50
	d = 0.05
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']-minbound+d*np.random.randn(1)[0]], 'o', color=color_point(dic['optimizetime'], maxoptimizetime))


def color_point(runtime = None, maxruntime= None):
	cmap = plt.get_cmap('RdYlGn')
	if runtime == None:
		return cmap(0)
	else:
		return cmap(np.sqrt(1- runtime/maxruntime))

def plot_bhr_lib(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']-minbound+d*np.random.randn(1)[0]], mark_point(dic['index']), color=color_point(dic['optimizetime'], maxoptimizetime))

def plot_bhr_lib_plain(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']-minbound+d*np.random.randn(1)[0]], 'ro')

def plot_bhr_lib_custom(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']-minbound+d*np.random.randn(1)[0]], 'o', color=colour_point(dic['index']))

def plot_bhr_lib_reverse_plain(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		maxbound = max(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']+d*np.random.randn(1)[0]]-maxbound, 'ro')

def plot_bhr_lib_reverse_custom(outs):
	maxoptimizetime = max(max(dic['optimizetime'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		maxbound = max(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']+d*np.random.randn(1)[0]]-maxbound, 'o', color=colour_point(dic['index']))

def plot_bhr_lib_voc(outs):
	vmax = max(max(dic['voc_size'] for dic in out[2]) for out in outs)
	plt.figure()
	plt.xlabel('dimension')
	plt.ylabel('bounds')
	c, d = 50, 0.05
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		for dic in out[2]:
			plt.plot([dic['reduced_dimension']+c*np.random.randn(1)[0]], [dic['bound']-minbound+d*np.random.randn(1)[0]], mark_point(dic['index']), color=color_point_voc(dic['voc_size'], vmax))

def mark_point(index):
	if index == 0:
		return 'o'
	elif index == -1:
		return 'o'
	elif not index:
		return 'o'
	else:
		return '*'

def color_point(runtime = None, maxruntime= None):
	cmap = plt.get_cmap('RdYlGn')
	if runtime == None:
		return cmap(0)
	else:
		return cmap(np.sqrt(1- runtime/maxruntime))

def colour_point(index):
	if index == 0:
		return 'red'
	elif index == -1:
		return 'red'
	elif not index:
		return 'red'
	else:
		return 'blue'

def color_point_voc(v, maxv):
	cmap = plt.get_cmap('RdYlGn')
	if v == None:
		return cmap(0)
	else:
		return cmap(np.sqrt(1- v/vmax))

def plot_outs(outs, spec1, spec2, D = None, N = None, V = None, dim = None, K = None, K0 = None, method = None):
	data2 = [d for d in outs]
	if D != None:
		data2 = [d for d in data2 if d['D'] == D]
	if N != None:
		data2 = [d for d in data2 if d['N'] == N]
	if V != None:
		data2 = [d for d in data2 if d['V'] == V]
	if dim != None:
		data2 = [d for d in data2 if d['dimension'] == dim]
	if K0 != None:
		data2 = [d for d in data2 if d['K0'] == K0]
	if method != None:
		data2 = [d for d in data2 if d['method'] == method]
	plt.figure()
	plt.xlabel(spec1)
	plt.ylabel(spec2)
	xs, ys = [], []
	for out in data2:
		if spec1 in out:
			xs.append(out[spec1])
		else:
			xs.append(special(out, spec1))
		if spec2 in out:
			ys.append(out[spec2])
		else:
			ys.append(special(out, spec2))
	plt.plot(xs, ys, 'ro')

def special(out, name):
	possibles = globals().copy()
	possibles.update(locals())
	method = possibles.get(name)
	return method(out)

def bNratio(out):
	return out['maxbounddiff']/out['N']

def bDratio(out):
	return out['maxbounddiff']/out['D']

def bdimratio(out):
	return out['maxbounddiff']/out['dimension']

def psort(data, spec):
	data2 = sorted(data, key=lambda run: run[spec])
	toprint = set(['dimension', 'maxbounddiff', 'K', 'V'])
	toprint.add(spec)
	for dic in data2:
		for spe in toprint:
			print spe, dic[spe],
		print

def psort2(data, spec):
	data2 = sorted(data, key=lambda run: run[spec])
	toprint = set(['dimension', 'maxbounddiff', 'K', 'V'])
	toprint.add(spec)
	for dic in data2:
		for spe in toprint:
			print spe,' '*(20-len(spe)), dic[spe],
		print

def psort3(data, spec, D = None, N = None, V = None, dim = None, K = None, K0 = None, method = None):
	data2 = [d for d in data]
	if D != None:
		data2 = [d for d in data2 if d['D'] == D]
	if N != None:
		data2 = [d for d in data2 if d['N'] == N]
	if V != None:
		data2 = [d for d in data2 if d['V'] == V]
	if dim != None:
		data2 = [d for d in data2 if d['dimension'] == dim]
	if K0 != None:
		data2 = [d for d in data2 if d['K0'] == K0]
	if method != None:
		data2 = [d for d in data2 if d['method'] == method]
	data2 = sorted(data2, key=lambda run: run[spec])
	toprint = set(['dimension', 'maxbounddiff', 'K', 'V', 'D', 'N', 'K0', 'method', 'maxrepeatstd', 'maxrepeatite'])
	toprint.add(spec)
	print '    ',
	for spe in toprint:
		if spe == 'maxbounddiff':
			print '     mbf',  '   ',
		elif spe == 'dimension':
			print '     dim',  '   ',
		elif spe == 'method':
			print '     method',  '   ',
		elif spe == 'maxrepeatstd':
			print '     mrs',  '   ',
		elif spe == 'maxrepeatite':
			print '     mri',  '   ',
		else:
			print ' '*(4-len(str(spe))), spe,  '   ',
	print
	counter = 0
	for dic in data2:
		counter += 1
		print ' '*(3 - len(str(counter))) + str(counter), '',
		for spe in toprint:
			if spe == 'maxboundiff':
				print ' '*(3-len(str(int(np.log(dic[spe]))))), int(np.log(dic[spe])),  '   ',
			elif spe == 'maxbounddiff':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'dimension':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'method':
				print ' '*(10-len(str(dic[spe]))), dic[spe],  '   ',
			elif spe == 'maxrepeatstd':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'maxrepeatite':
				print "{:8}".format(dic[spe]), '   ',
			else:
				print ' '*(4-len(str(dic[spe]))), dic[spe],  '   ',
		print
	print '    ',
	for spe in toprint:
		if spe == 'maxbounddiff':
			print '     mbf',  '   ',
		elif spe == 'dimension':
			print '     dim',  '   ',
		elif spe == 'method':
			print '     method',  '   ',
		elif spe == 'maxrepeatstd':
			print '     mrs',  '   ',
		elif spe == 'maxrepeatite':
			print '     mri',  '   ',
		else:
			print ' '*(4-len(str(spe))), spe,  '   ',
	print

def psort4(data, spec):
	data2 = [d for d in data]
	data2 = sorted(data2, key=lambda run: run[spec])
	toprint = set(['maxbounddiff', 'method', 'maxruntime'])
	toprint.add(spec)
	print '    ',
	for spe in toprint:
		if spe == 'maxbounddiff':
			print '     mbf',  '   ',
		elif spe == 'dimension':
			print '     dim',  '   ',
		elif spe == 'method':
			print '     method',  '   ',
		elif spe == 'maxrepeatstd':
			print '     mrs',  '   ',
		elif spe == 'maxrepeatite':
			print '     mri',  '   ',
		elif spe == 'maxruntime':
			print '     mrt',  '   ',
		else:
			print ' '*(4-len(str(spe))), spe,  '   ',
	print
	counter = 0
	for dic in data2:
		counter += 1
		print ' '*(3 - len(str(counter))) + str(counter), '',
		for spe in toprint:
			if spe == 'maxboundiff':
				print ' '*(3-len(str(int(np.log(dic[spe]))))), int(np.log(dic[spe])),  '   ',
			elif spe == 'maxbounddiff':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'dimension':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'method':
				print ' '*(10-len(str(dic[spe]))), dic[spe],  '   ',
			elif spe == 'maxrepeatstd':
				print "{:.2E}".format(dic[spe]), '   ',
			elif spe == 'maxrepeatite':
				print "{:8}".format(dic[spe]), '   ',
			elif spe == 'maxruntime':
				print "{:8}".format(round(dic[spe])), '   ',
			else:
				print ' '*(4-len(str(dic[spe]))), dic[spe],  '   ',
		print
	print '    ',
	for spe in toprint:
		if spe == 'maxbounddiff':
			print '     mbf',  '   ',
		elif spe == 'dimension':
			print '     dim',  '   ',
		elif spe == 'method':
			print '     method',  '   ',
		elif spe == 'maxrepeatstd':
			print '     mrs',  '   ',
		elif spe == 'maxrepeatite':
			print '     mri',  '   ',
		elif spe == 'maxruntime':
			print '     mrt',  '   ',
		else:
			print ' '*(4-len(str(spe))), spe,  '   ',
	print

def testi4():
	return 0

def testi5():
	return 0

