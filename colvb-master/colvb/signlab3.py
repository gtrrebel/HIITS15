import scipy

def largest_eigenvalue(A, tol=1E-5, maxiter=5000, sigma=0.01):
	try:
		return scipy.sparse.linalg.eigsh(A, 1, return_eigenvectors = False, which='LM', tol=tol, maxiter=maxiter, sigma=sigma)[0] > 0
	except:
		return -1

def smallest_eigenvalue(A, tol=1E-6, maxiter=5000, sigma=0.1):
	return scipy.sparse.linalg.eigsh(A, 1, return_eigenvectors = False, which='SA', tol=tol, maxiter=maxiter)[0] > 0