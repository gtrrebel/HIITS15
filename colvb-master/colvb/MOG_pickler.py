import pickle

def outputs_pickle(outputs, directory = None, ukko = False, filename = None):
	if filename == None:
		return 'LDA_' + time.strftime("%H_%M_%S_%d_%m_%Y")
	if directory == None:
		if ukko:
			directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/'
		else:
			directory = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/'
	else:
		if directory[-1] != '/':
			directory += '/'
	f = open(directory + filename, 'a+')
	pickle.dump(outputs, f)
	f.close()

def outputs_unpickle(filename, ukko=False):
	if filename[0] != '/':
		if ukko:
			filename = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/' + filename
		else:
			filename =  '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/' + filename
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

