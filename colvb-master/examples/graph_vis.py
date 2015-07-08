import networkx as nx
import numpy as np
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
		A = np.array(A)/100
		A = A.view(dt)

		G = nx.from_numpy_matrix(A)
		G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),lab)))    

		G = nx.to_agraph(G)

		G.node_attr.update(color= "#ff0000", style="filled")
		G.edge_attr.update(color="blue", width="0.1")
		G.graph_attr.update(outputorder="edgesfirst", dimen=2, scale=3)

		G.draw('/home/othe/Desktop/HIIT/HIITS15/colvb-master/examples/out.png', format='png', prog='neato')