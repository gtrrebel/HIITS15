import numpy as np
import gensim

def get_corpus():
	return gensim.corpora.MmCorpus('/home/othe/Desktop/HIIT/HIITS15/Gensim/wiki_bow.mm')

def clip_corpus(corpus, length = 100):
	return gensim.utils.ClippedCorpus(corpus, length)

def new_vocab(clipped_corpus):
	vocab = {}
	count = 0
	for doc in clipped_corpus:
		for word in doc:
			if word[0] not in vocab:
				vocab[word[0]] = count
				count += 1
	return vocab

def colvb_transform(clipped_corpus):
	vocab = new_vocab(clipped_corpus)
	docs = []
	for doc in clipped_corpus:
		docs.append([])
		for word in doc:
			for i in xrange(int(word[1])):
				docs[-1].append(vocab[word[0]])
	docs = [np.array(doc) for doc in docs]
	voc = np.array([str(i) for i in xrange(len(vocab))])
	return docs, voc

def get_colvb_corpus(length = 100):
	return colvb_transform(clip_corpus(get_corpus(), length))