import numpy as np

class data_creator():

	@staticmethod
	def nips_data(N_TOPICS, NDOCS, NWORDS, DOCUMENT_LENGTH, randomized = False):
		f = file('../data/nips11data/nips11_corpus')
		s = f.read().split('\n')
		f.close()
		vocab = np.array(s[0].split())
		if randomized:
			rawdocs = [s[i] for i in random.sample(xrange(1, len(s)), NDOCS)]
		else:
			rawdocs = s[1:(NDOCS + 1)]
		docs = [np.asarray([int(i) for i in l.split()[1:]],dtype=np.int) for l in rawdocs]
		docs = [d for d in docs if d.size > 1000]
		docs_ = np.hstack(docs)
		wordcounts = np.array([np.sum(docs_==i) for i in range(docs_.max())],dtype=np.int)
		allowed_vocab = np.argsort(wordcounts)[::-1][:NWORDS]
		docs = [np.asarray([w for w in doc if w in allowed_vocab],dtype=np.int) for doc in docs]
		DOCUMENT_LENGTH = min(DOCUMENT_LENGTH, min([len(doc) for doc in docs]))
		docs = [doc[0:DOCUMENT_LENGTH] for doc in docs]
		docs_ = np.hstack(docs)
		return docs, vocab

	@staticmethod
	def basic_data(WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF):
		#generate some documents
		N_TOPICS = WORDSIZE*N_TOPIC_COEFF

		#here's the vocabulary
		V = WORDSIZE**2
		if WORDSIZE==2:
		    vocab = np.array([u'\u25F0',u'\u25F3',u'\u25F1',u'\u25F2'],dtype="<U2")
		else:
		    vocab_ = [np.zeros((WORDSIZE,WORDSIZE)) for v in range(V)]
		    [np.put(v,i,1) for i,v in enumerate(vocab_)]
		    vocab = np.empty(len(vocab_),dtype=np.object)
		    for i,v in enumerate(vocab_):
		        vocab[i] = v

		#generate the topics
		topics = [np.zeros((WORDSIZE,WORDSIZE)) for i in range(N_TOPICS)]
		for i in range(WORDSIZE):
		    topics[i][:,i] = 1
		    topics[i+WORDSIZE][i,:] = 1
		topics = map(np.ravel,topics)
		topics = map(lambda x: x/x.sum(),topics)

		#if the docs are 2x2 square, you'll have as many topics as vocab, which won't work:
		if WORDSIZE==2:
		    topics = topics[:2]
		    N_TOPICS = 2

		#generate the documents
		docs = []
		doc_latents = []
		doc_topic_probs = []
		for d in range(N_DOCS):
		    topic_probs = np.random.dirichlet(np.ones(N_TOPICS)*0.8)
		    latents = np.random.multinomial(1,topic_probs,DOCUMENT_LENGTH).argmax(1)
		    doc_latents.append(latents)
		    doc_topic_probs.append(topic_probs)
		    docs.append(np.array([np.random.multinomial(1,topics[i]).argmax() for i in latents]))
		return docs, vocab
