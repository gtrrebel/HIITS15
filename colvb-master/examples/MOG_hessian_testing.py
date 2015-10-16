from MOG_demo4 import *

def test1():
	m = init(make_fns = True)[0]
	eigs = m.eigenvalues()
	h, h2 = len([i for i in eigs if i > 1e-9]), len(eigs)
	m.optimize()
	eigs2 = m.eigenvalues()
	g, g2, g3 = len([i for i in eigs2 if i > 1e-9]), len([i for i in eigs2 if abs(i) < 1e-9]), len(eigs2)
	return h, h2, g, g2, g3