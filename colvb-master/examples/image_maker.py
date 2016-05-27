from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from MOG_demo4 import *

image_path = '/home/othe/Downloads/standard_test_images/'
default_image_name = 'lena_color_256.tif'

def get_image(image_name = default_image_name):
	img = Image.open(image_path + image_name)
	return img

def move_to_cube(image, has_color = True, data_size = None, do_average=False):
	width, height = image.size
	if (data_size == None):
		data_size = width*height
	if (width*height < data_size):
		data_size = width*height
	data = image.getdata()
	pixels = []
	if (has_color):
		for i, pix in enumerate(data):
			xp, yp = i/width, i%width
			x, y = 1.0*xp/height, 1.0*yp/width
			rr,gg,bb = pix
			r,g,b = 1.0*rr/256, 1.0*gg/256, 1.0*bb/256
			pixels.append([x,y,r,g,b])
	pixels = np.array(pixels)
	pixels = np.reshape(pixels, (width, height, 5))
	a = (1.0*data_size/(width*height))**(0.5)
	d = int(1/a)
	if do_average:
		wp, hp = width//d, height//d
		new_pixels = np.zeros(shape = (wp, hp, 5))
		for ii in xrange(wp):
			for jj in xrange(hp):
				rrr, ggg, bbb = 0, 0, 0
				for iii in xrange(d):
					for jjj in xrange(d):
						rrr += pixels[ii*d + iii][jj*d + jjj][2]
						ggg += pixels[ii*d + iii][jj*d + jjj][3]
						bbb += pixels[ii*d + iii][jj*d + jjj][4]
				rrr /= (d*d)
				ggg /= (d*d)
				bbb /= (d*d)
				new_pixels[ii][jj][2] = rrr
				new_pixels[ii][jj][3] = ggg
				new_pixels[ii][jj][4] = bbb
				new_pixels[ii][jj][0] = ii
				new_pixels[ii][jj][1] = jj
	else:
		new_pixels = pixels[::d,::d, :]
	return new_pixels

def make_m(image_name=default_image_name, data_size = None, K = 10):
	img = get_image(image_name)
	data = move_to_cube(img, data_size = data_size)
	return MOG2(data.reshape((data.shape[0]*data.shape[1],5)), K)

def make_figure(m, image, w, h):
	params = m.get_param().copy()
	K = m.K
	N = m.N
	choices = params.reshape((N, K))
	choices = np.array([np.argmax(row) for row in choices])
	choices = choices.reshape((w,h))
	fig = plt.figure()
	plt.imshow(choices)
	return fig


def process_image(image_name = default_image_name, data_size = 100, K = 10, method = 'steepest', do_average = False):
	image = get_image(image_name)
	data = move_to_cube(image, data_size = data_size, do_average = do_average)
	w,h = data.shape[0], data.shape[1]
	m = MOG2(data.reshape((w*h, 5)), K)
	m.optimize_autograd(method='steepest')
	pos = m.epsilon_positive()
	fig = make_figure(m, image, w, h)
	print pos
	plt.show()
