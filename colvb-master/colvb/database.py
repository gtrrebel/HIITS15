

class database():

	def __init__(self, width = 10):
		self.datas = []
		self.width = width

	def basic_format(self):
		return ('{{{{{{}}:{0}}}}} '*4).format(self.width).format(*xrange(4))

	def add(self, documents, topics):
		self.datas.append((documents, topics))

	def size(self):
		return len(self.datas)

	def remove(self, i, j=None):
		if j == None:
			j = i
		del self.datas[i:(j+1)]

	def clear(self):
		self.datas = []

	def new_width(self, width=10):
		self.width = width

	def get_data(self, i):
		return self.datas[i]

	def get_doc(self, i):
		return self.get_data(i)[0]

	def get_words(self, i):
		return sorted(list(set([x for sublist in self.get_data(i)[0] for x in sublist])))

	def get_top(self, i):
		return self.get_data(i)[1]

	def display_documents(self, i):
		print self.get_doc(i)

	def display_words(self, i):
		print self.get_words(i)

	def get_input(self, i):
		N_DOCS = len(self.get_doc(i))
		N_WORDS = len(self.get_words(i))
		DOC_LEN = len(self.get_doc(i)[0])
		N_TOP = self.get_top(i)
		return N_DOCS, N_WORDS, DOC_LEN, N_TOP

	def make_header(self):
		print (self.basic_format()).format('-N_DOCS-', '-N_WORDS-', '-DOC_LEN-', '-N_TOP-')

	def display(self, i):
		self.make_header()
		print (self.basic_format()).format(*self.get_input(i))

	def history(self):
		self.make_header()
		for i in xrange(self.size()):
			print (self.basic_format()).format(*self.get_input(i))




