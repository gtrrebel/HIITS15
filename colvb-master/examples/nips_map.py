import numpy as np
import pylab as pb
from scipy import linalg
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from graph_vis import graph_vis

basic_data = [3, 50, 30, 2]
nips_data = [5, 50, 200, 300]
data_type = 'nips'
run_count = 100
method = 'steepest'

j = 1
while j < len(sys.argv):
    cmd = sys.argv[j]
    j += 1
    if cmd == 'b':
        k = 0
        data_type = 'basic'
        while k < 4:
            k, j = k + 1, j + 1
            basic_data[k] = int(sys.argv[j])
    elif cmd == 'n':
        data_type = 'nips'
        while k < 4:
            k, j = k + 1, j + 1
            nips_data[k] = int(sys.argv[j])
    elif cmd == 'r':
        run_count = int(sys.argv[j])
    else:
        print 'oh no', j
        break
    j += 1

if data_type == 'basic':
    docs, vocab = data_creator.basic_data(*basic_data)
    N_TOPICS = basic_data[0]*basic_data[3]
elif data_type == 'nips':
    docs, vocab = data_creator.nips_data(*nips_data)
    N_TOPICS = basic_data[0]

eps = 1e-14
close_eps = 1

def dist(v1, v2):
    return linalg.norm(v1-v2)

maxs = []
m = LDA3(docs,vocab,N_TOPICS)
m.runspec_set('eps', eps)

for i in range(run_count):
    print i
    m.new_param()
    m.optimize(method=method, maxiter=1e4)
    maxs.append((m.bound(), m.get_param()))

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

maxs = merge(maxs)

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

print_all_dists(maxs)

graph_vis.draw(tuple_dists(maxs), all_bounds(maxs), all_counts(maxs))
'''
print min_max_dist(maxs)
for f in sorted_dists(maxs):
    print f
'''