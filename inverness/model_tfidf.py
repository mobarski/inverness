from gensim.models import TfidfModel
from tqdm import tqdm
import multiprocessing as mp
import re
from array import array

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet

# ---[ MODEL ]------------------------------------------------------------------

class TFIDF():
	
	@timed
	def init_tfidf(self, **kwargs):
		self.tfidf = TfidfModel(dictionary=self.dictionary, **kwargs)
		self.tfidf.save(self.path+'tfidf.pkl')
	
	def load_tfidf(self):
		self.tfidf = TfidfModel.load(self.path+'tfidf.pkl')


	def init_sparse(self, workers=None, storage=None):
		_workers = workers or self.params.get('sparse__workers') or self.params.get('workers',1)
		_storage = storage or self.params.get('sparse__storage') or self.params.get('storage','disk')
		if _workers>1:
			self._init_sparse_mp(workers=_workers, storage=_storage)
		else:
			self._init_sparse_sp(storage=_storage)

	def _init_sparse_sp(self, storage):
		self.sparse = sorbet(self.path+'sparse', kind=storage).new()
		doc = []
		for doc_id,sen in self.all_sentences(desc='sparse'):
			if not sen: # end of document:
				bow = self.dictionary.doc2bow(doc) or [(0,1)] # 0 -> '<PAD>'
				sparse = self.tfidf[bow]
				a = array('f')
				for i_v in sparse:
					a.extend(i_v)
				self.sparse.append(a)
				doc = []
			else:
				tokens = sen.split(' ')
				doc.extend(tokens)
		self.sparse.save()


	def load_sparse(self, storage='disk'):
		self.sparse = sorbet(self.path+'sparse', kind=storage).load()
	
	def _init_sparse_mp(self, workers, storage):
		chunksize = self.params.get('sparse__chunksize',10)
		s = sorbet(self.path+'sparse').new()
		id_iter = range(len(self.meta))
		id_iter = tqdm(id_iter,'sparse',len(self.meta))
		with mp.Pool(
					workers,
					init_sparse_worker_mk2,
					[
						self.path,
					]
				) as pool:
			sparse = pool.imap(sparse_worker_mk2, id_iter, chunksize)
			for a in sparse:
				#print('mp_sparse',a)
				s.append(a)
		self.sparse = s.save()
	
	# TODO raname dfs_query ???
	def tfidf_query(self, query, exclude=None):
		dfs = []
		token_ids,tokens = self.dictionary_query(query,exclude)
		for i in token_ids:
			df = self.tfidf.dfs[i]
			dfs += [df]
		# sort
		results = list(sorted(zip(token_ids,tokens,dfs),key=lambda x:-x[2]))
		token_ids = [x[0] for x in results]
		tokens = [x[1] for x in results]
		dfs = [x[2] for x in results]
		return token_ids,tokens,dfs
			

# ---[ MULTIPROCESSING ]--------------------------------------------------------

def init_sparse_worker_mk2(*args):
	global model
	from .model_dictionary import Dictionary
	from .model_sentencer import Sentencer
	class Model(TFIDF,Dictionary,Sentencer): pass
	model_path = args[0]
	model = Model()
	model.path = model_path
	model.load_tfidf()
	model.load_dictionary()
	model.load_sentencer()


def sparse_worker_mk2(doc_id):
	tokens = []
	for sen in model.doc_sentences(doc_id):
		tokens.extend(sen.split(' '))
	bow = model.dictionary.doc2bow(tokens) or [(0,1)] # 0 -> '<PAD>'
	sparse = model.tfidf[bow]
	a = array('f')
	for i_v in sparse:
		a.extend(i_v)
	return a


def init_sparse_worker(*args):
	global model
	model_path = args[0]
	model = TFIDF()
	model.path = model_path
	model.bow = sorbet(model.path+'bow').load()
	model.load_tfidf()


def sparse_worker(doc_id):
	bow = model.bow[doc_id]
	sparse = model.tfidf[bow]
	a = array('f')
	for i_v in sparse:
		a.extend(i_v)
	return a


# ---[ DEBUG ]------------------------------------------------------------------

if __name__=="__main__":
	model = TFIDF()
	model.path = '../model_100/'
	model.load_tfidf()
	tfidf = model.tfidf
	#
	from pprint import pprint
	from itertools import islice
	#
	from model_dictionary import Dictionary
	model2 = Dictionary()
	model2.path = model.path
	model2.load_dictionary()
	d = model2.dictionary
	data = ((i,cnt,tfidf.idfs[i],d.get(i,'???')) for i,cnt in tfidf.dfs.items())
	data = sorted(data,key=lambda x:x[1],reverse=True)
	for i,cnt,idf,token in islice(data,40):
		print(i,cnt,
			round(idf,1),
			token,
			sep='\t')
