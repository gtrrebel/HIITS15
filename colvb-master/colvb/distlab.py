from scipy import linalg

close_eps = 1e-3

def dist(v1, v2):
    return linalg.norm(v1-v2)

def merge(maxs):
	real_maxs = []
	for ma in maxs:
		add = True
		for real_ma in real_maxs:
			if dist(ma[1], real_ma[1]) < close_eps:
				real_ma[2] += 1
				add = False
				break
		if add:
			real_maxs.append([ma[0], ma[1], 1])
	return real_maxs

def all_counts(maxs):
	return [ma[2] for ma in maxs]

def all_bounds(maxs):
	return [ma[0] for ma in maxs]

def all_dists(maxs):
	dists = []
	for ma1 in maxs:
		dists.append([])
		for ma2 in maxs:
			dists[-1].append(dist(ma1[1], ma2[1]))
	return dists

def print_all_dists(maxs):
	dists = all_dists(maxs)
	for row in dists:
		for dist in row:
			print dist,
		print

def tuple_dists(maxs):
	dists = all_dists(maxs)
	return [tuple(row) for row in dists]

def min_max_dist(maxs):
	n = len(maxs)
	if n == 1:
		return None, None
	else:
		mini, maxi = [dist(maxs[0][1], maxs[1][1])]*2
		for i in range(n):
			for j in range(i + 1, n):
				dis = dist(maxs[i][1], maxs[j][1])
				mini = min(mini, dis)
				maxi = max(maxi, dis)
	return mini, maxi

def sorted_dists(maxs):
	n = len(maxs)
	dists = []
	for i in xrange(n):
		for j in xrange(i + 1, n):
			dists.append(dist(maxs[i][1], maxs[j][1]))
	return sorted(dists)   