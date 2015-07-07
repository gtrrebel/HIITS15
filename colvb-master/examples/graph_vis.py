import networkx as nx
import numpy as np
import string

class graph_vis():

	@staticmethod
	def draw(A, bounds):
		n = len(bounds)
		bound_order = [i[0] for i in sorted(enumerate(bounds), key=lambda x:x[1])]
		bound_order = sorted(range(len(A)),key=lambda x:bound_order[x], reverse=True)
		dt = [('len', float)]
		A = np.array(A)*5
		A = A.view(dt)

		G = nx.from_numpy_matrix(A)
		G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),bound_order)))    

		G = nx.to_agraph(G)

		G.node_attr.update(color= str((n -1.0)/n) + " 0.0 0.0", style="filled")
		G.edge_attr.update(color="blue", width="0.1")

		G.draw('/home/othe/Desktop/HIIT/HIITS15/colvb-master/examples/out.png', format='png', prog='neato')