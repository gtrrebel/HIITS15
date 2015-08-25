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

def mark_point(index):
	if index == 0:
		return 'o'
	elif index == -1:
		return 'o'
	elif not index:
		return 'o'
	else:
		return '*'

def colour_point(index):
	if index == 0:
		return 'red'
	elif index == -1:
		return 'red'
	elif not index:
		return 'red'
	else:
		return 'blue'
