import pickle
import time
from LDA3 import LDA3

def LDA3_pickle_file_name():
	name = 'LDA3_'
	t = time.strftime("%H:%M:%S-%d.%m.%Y")
	return name + t

def LDA3_pickle(ms, directory = None, ukko = False):
	if directory == None:
		if ukko:
			directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
		else:
			directory = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
	else:
		if directory[-1] != '/':
			directory += '/'
	data = [m.pickle_data for m in ms]
	f = open(directory + LDA3_pickle_file_name(), 'a+')
	pickle.dump(data, f)

def LDA3_unpickle(filename, ukko=False):
	if filename[0] != '/':
		if ukko:
			filename = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/' + filename
		else:
			filename =  '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/' + filename
	f = open(filename, 'r')
	data = pickle.load(f)
	ms = []
	for d in data:
		ms.append(LDA3(d[1],d[2],d[3], alpha_0=d[4], beta_0=d[5], eps=d[6]))
		ms[-1].set_vb_param(d[0])
	return ms
