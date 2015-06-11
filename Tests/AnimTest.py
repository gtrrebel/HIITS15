import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

defaultMatrixSize = 40
over = 10
smalleMatrixSize = 1
tests = 100
alters = 1
opt = 0
randhop = 3e-2

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
		return np.diag(-1 + 2*np.random.random((n))) 

def data_gen():
        msize = data_gen.msize
        while msize > 1:
                msize -= 1
                op = 1
                pairs = []
                for i in range(tests):
                        A = makeRandomMatrix(defaultMatrixSize, defaultMatrixSize, op)
                        A += np.transpose(A)
                        ind = index(A, opt)
                        su = 0
                        count = 0
                        for j in range(alters):
                                B = makeRandomMatrix(msize, defaultMatrixSize)
                                aind = index(np.dot(B, np.dot(A, np.transpose(B))), opt)
                                su+= aind
                        pairs.append([ind + randhop*(2*np.random.random((1)) -1), su/alters+1e-2*(2*np.random.random((1)) -1)])
                yield list(zip(*pairs)[0]), list(zip(*pairs)[1])

data_gen.msize = defaultMatrixSize + over

fig, ax = plt.subplots()
#plt.plot([0,1])
line, = ax.plot([], [], 'ro')
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
xdata, ydata = [], []
def run(data):
    # update the data
    
    xdata,ydata = data
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
    repeat=False)
plt.plot([0,1])
plt.show()
