import pylab as pb

class model_display():

	def __init__(self, m, out = 'view'):
		self.m = m
		self.out = out

	def output(self):
		if self.out == 'view':
			pb.show

	def orig_track_display(self):
		pb.figure()
		m.plot_tracks()
		self.output()

	def orig_learned_topics_display(self):
		nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
		pb.figure()
		for i,beta in enumerate(m.beta_p):
			pb.subplot(nrow,ncol,i+1)
			pb.imshow(beta.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
			pb.xticks([])
			pb.yticks([])
		pb.suptitle('inferred topics')
		pb.show()

	def orig_true_topics_display(self):
		nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
		pb.figure()
		for i,topic in enumerate(topics):
			pb.subplot(nrow,ncol,i+1)
			pb.imshow(topic.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
			pb.xticks([])
			pb.yticks([])
		pb.suptitle('true topics')
		pb.show()

	def display(self):
		for disp in self.m.runspecs['display']:
			if self.m.runspecs['display'][disp]:
				if hasattr(self,disp):
					getattr(self, disp)()
