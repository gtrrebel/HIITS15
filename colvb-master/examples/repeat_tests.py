import LDA_demo4 as LDA4
import MOG_demo4 as MOG4

def test_LDA_repeat1(lengths = [1, 10]):
	ms = LDA4.init(make_fns=False)
	stds = []
	for l in lengths:
		outs = LDA4.run(ms, end_gather=['bound'], repeat = 10, length = l)
		stds.append(sum(out['repeatstd'] for out in outs[0][2])/10)
	return stds

def test_MOG_repeat1(lengths = [1, 10]):
	ms = MOG4.init(make_fns=False)
	stds = []
	for l in lengths:
		outs = MOG4.run(ms, end_gather=['bound'], repeat = 10, length = l)
		stds.append(sum(out['repeatstd'] for out in outs[0])/10)
		print l, stds[-1]
	return stds