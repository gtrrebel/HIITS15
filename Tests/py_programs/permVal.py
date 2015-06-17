import itertools

n = 9

def perm_value(perm, k):
	mak = 0
	n = len(perm)
	a = range(k)
	for i in range(n):
		mak = max(mak, sum([perm[(b + i) % n] for b in a]))
	return mak

def f(n, k):
	mik = n*k
	for a in itertools.permutations(range(1, n + 1)):
		mik = min(mik, perm_value(list(a), k))
	return mik

for i in range(1, n + 1):
	for j in range(i):
		print f(i, j + 1),
	print

