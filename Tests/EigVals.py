import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

defaultMatrixSize = 20
smalleMatrixSize = 19
tests = 500
alters = 10
opt = 0

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

def printMatrix(A, d = 5):
	n = len(A)
	for i in range(n):
		for j in range(n):
			print (str(A[i][j])+d*' ')[:d],
		print
	print

def test1():
	for i in range(tests):
		A = makeRandomMatrix()
		A += np.transpose(A)
		print 'True: ', index(A, opt), ' alters: ',
		for j in range(alters):
			B = makeRandomMatrix()
			print index(np.dot(B, np.dot(A, np.transpose(B))), opt),
		print

def test2():
	for i in range(tests):
		A = makeRandomMatrix()
		A += np.transpose(A)
		i1, i2 = index(A, 0), index(A, 1)
		print i1, i2
		if (abs(i1 - i2) > 1e-2):
			gaussIndex(A, True)

def test3():
	for i in range(tests):
		A = makeRandomMatrix()
		A += np.transpose(A)
		ind = index(A, opt)
		print 'True: ', 
		prin(ind)
		#print ' alters: ',
		su = 0
		for j in range(alters):
			B = makeRandomMatrix(smalleMatrixSize, defaultMatrixSize)
			aind = index(np.dot(B, np.dot(A, np.transpose(B))), opt)
			su+= aind
			#prin(ind),
		#print 'average: ',
		prin(su/alters)
		print

def test4():
	op = 1
	pairs = []
	for i in range(tests):
		A = makeRandomMatrix(defaultMatrixSize, defaultMatrixSize, op)
		A += np.transpose(A)
		ind = index(A, opt)
		su = 0
		for j in range(alters):
			B = makeRandomMatrix(smalleMatrixSize, defaultMatrixSize)
			aind = index(np.dot(B, np.dot(A, np.transpose(B))), opt)
			su+= aind
		pairs.append([ind, su/alters])
	plotPairs(pairs)

def test5():
	op = 1
	pairs = []
	trupairs = []
	for i in range(tests):
		A = makeRandomMatrix(defaultMatrixSize, defaultMatrixSize, op)
		A += np.transpose(A)
		ind = index(A, opt)
		su = 0
		count = 0
		while True:
			count += 1
			B = makeRandomMatrix(smalleMatrixSize, defaultMatrixSize)
			aind = index(np.dot(B, np.dot(A, np.transpose(B))), opt)
			su += aind
			pairs.append([ind, su/count])
			print str(ind) + ' ' + str(su/count) + '\r',
			if count > 2:
				if abs(pairs[-1][1]-pairs[-2][1]) + abs(pairs[-2][1]-pairs[-3][1]) < 1e-5:
					break
		trupairs.append([ind, su/count])
		pairs = []
		print '\n ',count
	plotPairs(trupairs)

def plotPairs(pairs):
    plt.figure(1)
    plt.plot([0,1])
	plt.plot(*zip(*pairs), marker='o', color='r', ls='')
	plt.axis([0,1,0,1])
	plt.ylabel('test')
	plt.show()
	

def prin(a, d = 3):
	print "%.*f" % (d, a),

test4()
