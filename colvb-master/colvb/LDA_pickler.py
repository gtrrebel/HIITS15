import pickle
import time

def LDA3_dir_name(m):
	t = time.strftime("%H:%M:%S-%d.%m.%Y")
	stats = '_' + str(m.D) + '_' + str(m.N) + '_' + str(m.K)
	return t + stats

def LDA3_pickle(m, directory):
	pass

def LDA4_data_pickle(data, directory):
	pass