import numpy as np
import pylab as pb
import MOG_demo4

def bounddists(args = [10, 2, 10, 30, 5], datacount = 10, restartcount = 10):
	xs, ys = [], []
	for i in xrange(datacount):
		m = MOG_demo4.init([args], make_fns = False)[0]
		for j in xrange(restartcount):
			print i, j
			m.new_param()
			xs.append(m.bound())
			m.optimize()
			ys.append(m.bound())
	pb.plot(xs, ys, 'r*')
	ma = max(ys)
	pb.plot([0, ma*1.2], [0, ma*1.2])
	pb.show()
	return