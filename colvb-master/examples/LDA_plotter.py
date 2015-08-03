import matplotlib
import matplotlib.pyplot as plt

def plot(xs, ys, end_gather):
	fig = plt.figure()
	plt.title('testi')
	plt.xlabel(end_gather[0])
	plt.ylabel(end_gather[1])
	plt.plot(xs, ys, 'or')
	plt.ylim(-1, max(ys) + 1)
	plt.show()

def save(xs, ys, end_gather):
	matplotlib.use('Agg')
	fig = plt.figure()
	plt.title(nips_data)
	plt.xlabel(end_gather[0])
	plt.ylabel(end_gather[1])
	plt.plot(xs, ys, 'or')
	plt.ylim(-1, max(ys) + 1)
	name = "LDA_demo3." + time.strftime("%H:%M:%S-%d.%m.%Y") + ".png"
	plt.savefig(results + name)
	plt.close()