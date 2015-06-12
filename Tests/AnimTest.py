import numpy as np
import random
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

defaultMatrixSize = 40
over = 0
smalleMatrixSize = 1
tests = 1
alters = 100000
opt = 0
randhop = 3e-2
add = False
coeffType = 2

def index(A, opt = 0):
	if (opt == 0):
		return gaussIndex(A)
	elif (opt == 1):
		return bruteIndex(A)

def bruteIndex(A):
	eigvals = np.linalg.eig(A)[0]
	return sum(1 for eigval in eigvals if eigval < 0)*1./len(A)

def gaussIndex(B, prin = False):
	A = np.copy(B)
	n, neg = len(A), 0
	for i in range(n):
		if (A[i][i] < 0):
			neg += 1
		if (A[i][i] == 0):
			return None
		A[i] /= A[i][i]
		if prin:
			printMatrix(A)
		for j in range(i + 1, n):
			A[j] -= A[i]*A[j][i]
			if prin:
				printMatrix(A)
	return neg*1./n

def makeRandomMatrix(n = defaultMatrixSize, m = defaultMatrixSize, op = 0):
	if op == 0:
		return np.random.random((n, m))
	elif op == 1:
		return np.diag(np.random.normal(size=(n))) 

def data_gen():
	msize = data_gen.msize
	while msize > 1:
		msize -= 1
		pairs = []
		for i in range(tests):
			newPair = createPair(0, coeffType, msize, alters)
			pairs.append(newPair)
		colrat = msize*1./(defaultMatrixSize + over)
		newColor = (colrat, 0, 1 - colrat)
		yield list(zip(*pairs)[0]), list(zip(*pairs)[1]), newColor

def createPair(style, coeffType, ssize, alters):
	A = makeRandomMatrix(defaultMatrixSize, defaultMatrixSize, 1)
	A += np.transpose(A)
	ind = index(A, opt)
	if (ind == None):
		return None
	indsum = 0
	cou = 0
	while cou < alters:
		cou += 1
		C = createC(A, coeffType, ssize)
		newInd = index(C,0)
		if (newInd != None):
			indsum+= newInd
	return [ind + randHop(), indsum/alters+randHop()]

def createC(A, coefType, ssize):
	if (coeffType == 0):
		B = makeRandomMatrix(ssize, len(A))
		C = np.dot(B, np.dot(A, np.transpose(B)))
	elif (coeffType == 1):
		col = np.array(random.sample(xrange(1, len(A)), ssize))
		C = A[col[:, np.newaxis], col]
	elif (coeffType == 2):
		normal_deviates = np.random.normal(size=(len(A)))
		radius = np.sqrt((normal_deviates**2).sum())
		B = normal_deviates/radius
		C = [[np.dot(B, np.dot(A, np.transpose(B)))]]
	return C

def randHop():
	return randhop*(2*np.random.random((1)) -1)

data_gen.msize = defaultMatrixSize + over

fig, ax = plt.subplots()
#plt.plot([0,1])
line, = ax.plot([], [], 'ro')
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
xdata, ydata = [], []

def run(data):
	# update the data
	global xdata, ydata
	newxdata,newydata, newColor = data
	if add:
		xdata.append(newxdata)
		ydata.append(newydata)
	else:
		xdata = newxdata
		ydata = newydata
	line.set_data(xdata, ydata)
	line.set_color(newColor)

	return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,repeat=False)
plt.plot([0,1])

plt.show(block=False)

inp = raw_input('q to quit, s to save the figure and quit\n')
if (inp == 's'):
	name = raw_input('Give name for figure\n')
	if (name == ''):
		name = 'newFigure'
	plt.savefig(name)
	plt.close()
elif (inp == 'q'):
	plt.close()