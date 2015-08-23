import pickle
import time
from LDA3 import LDA3

def LDA3_pickle_file_name():
	name = 'LDA3_'
	t = time.strftime("%H_%M_%S_%d_%m_%Y")
	return name + t

def bhr_lib_pickle_file_name():
	name = 'bhr_lib_'
	t = time.strftime("%H_%M_%S_%d_%m_%Y")
	return name + t

def LDA3_compress(m):
	return m.pickle_data

def LDA3_uncompress(d):
	m = LDA3(d[1],d[2],d[3], alpha_0=d[4], beta_0=d[5], eps=d[6], make_fns=False)
	m.set_vb_param(d[0])
	return m

def dics_compress(d):
	gather = d[0]
	stats = d[1]
	ld = []
	for dic in d[2]:
		ld.append({})
		ld[-1]['index'] = dic['index']
		for g in gather:
			if g != 'return_m':
				ld[-1][g] = dic[g]
			else:
				ld[-1]['compress_m'] = LDA3_compress(dic[g])
	return (gather, stats, ld)

def dics_uncompress(d):
	gather = d[0]
	stats = d[1]
	ld = []
	for dic in d[2]:
		ld.append({})
		for g in gather:
			ld[-1]['index'] = dic['index']
			if g != 'return_m':
				ld[-1][g] = dic[g]
			else:
				ld[-1]['return_m'] = LDA3_uncompress(dic['compress_m'])
	return (gather, stats, ld)

def LDA3_pickle(ms, directory = None, ukko = False):
	if directory == None:
		if ukko:
			directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
		else:
			directory = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
	else:
		if directory[-1] != '/':
			directory += '/'
	data = [LDA3_compress(m) for m in ms]
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
		ms.append(LDA3_uncompress(d))
	return ms

def bhr_lib_pickle(lib, directory = None, ukko = False):
	if directory == None:
		if ukko:
			directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
		else:
			directory = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/'
	else:
		if directory[-1] != '/':
			directory += '/'
	data = [dics_compress(dics) for dics in lib]
	f = open(directory + bhr_lib_pickle_file_name(), 'a+')
	pickle.dump(data, f)

def bhr_lib_unpickle(filename, ukko=False):
	if filename[0] != '/':
		if ukko:
			filename = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/' + filename
		else:
			filename =  '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA3_pickles/' + filename
	f = open(filename, 'r')
	data = pickle.load(f)
	data = [dics_uncompress(dics) for dics in data]
	return data