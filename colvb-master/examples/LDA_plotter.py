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

def get_data(end_gather, end_returns):
	data = {}
	for spec in end_gather:
		data[spec] = []
	for run in end_returns:
		for spec in run:
			data[spec[0]].append(spec[1])
	delta = 0e-2
	randomization = delta*np.random.randn((len(data[end_gather[1]])))
	xs, ys = data[end_gather[0]], data[end_gather[1]]+randomization
	return xs, ys, end_gather

def plot(end_gather, end_returns):
	xs, ys, end_gather = get_data(end_gather, end_returns)
	fig = plt.figure()
	plt.title('testi')
	plt.xlabel(end_gather[0])
	plt.ylabel(end_gather[1])
	plt.plot(xs, ys, 'or')
	plt.ylim(-1, max(ys) + 1)
	plt.show()

def save_plot(end_gather, end_returns, ukko=False):
	plot(end_gather, end_returns)
	plt.savefig(result_filepath(ukko, 'png'))

def save_data(end_gather, end_returns, ukko=False):
	f = open(result_filepath(ukko, 'txt'), 'w')
	print >> f, 'Testi:', end_gather, end_returns
	f.close()