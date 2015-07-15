import networkx as nx
import numpy as np
import time
import string

class graph_vis():

	@staticmethod
	def draw(A, bounds, counts):
		n = len(bounds)
		bound_order = [i[0] for i in sorted(enumerate(bounds), key=lambda x:x[1])]
		bound_order = sorted(range(len(A)),key=lambda x:bound_order[x], reverse=True)
		lab = [str(order) + ": " + str(count) for order, count in zip(bound_order, counts)]
		dt = [('len', float)]
		print A[0][1]
		A = np.array(A)/10
		A = A.view(dt)

		G = nx.from_numpy_matrix(A)
		G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),lab)))
		for key in G.node.keys():
			G.node[key]['fillcolor'] = graph_vis.color(int(key.split(':')[0]),n)

		G = nx.to_agraph(G)

		G.node_attr.update(color= "#ff0000", style="filled")
		G.edge_attr.update(color="blue", width="0.1")
		G.graph_attr.update(outputorder="edgesfirst", dimen=2, scale=3)
		filename = '/home/othe/Desktop/HIIT/HIITS15/colvb-master/examples/out' + time.strftime("%Y-%m-%d|%H:%M:%S", time.gmtime()) + '.png'

		G.draw(filename, format='png', prog='neato')

	@staticmethod
	def color(i, n):
		return "0.000" + " " + str((n - i - 1)*1.0/n) + " " + "1.000"
