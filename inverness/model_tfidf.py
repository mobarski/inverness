from gensim.models import TfidfModel
from tqdm import tqdm
import multiprocessing as mp
import re

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

	@timed
	def init_sparse(self, storage='disk'):
		sparse = (self.tfidf[bow] for bow in self.bow)
		sparse = tqdm(sparse, desc='sparse', total=len(self.meta)) # progress bar
		self.sparse = sorbet(self.path+'sparse', kind=storage).dump(sparse)

	@timed
	def init_sparse_mk2(self, storage='disk'):
		sparse = (self.tfidf[bow] for bow in self.bow)
		sparse = tqdm(sparse, desc='sparse', total=len(self.meta)) # progress bar
		self.sparse = sorbet(self.path+'sparse', kind=storage).dump(sparse)


	def load_sparse(self, storage='disk'):
		self.sparse = sorbet(self.path+'sparse', kind=storage).load()
	
	@timed
	def init_sparse_mp(self, workers=4, chunksize=100):
		s = sorbet(self.path+'sparse').new()
		id_iter = range(len(self.meta))
		id_iter = tqdm(id_iter,'sparse',len(self.meta))
		with mp.Pool(
					workers,
					init_sparse_worker,
					[
						self.path,
					]
				) as pool:
			sparse = pool.imap(sparse_worker, id_iter, chunksize)
			for sp in sparse:
				s.append(sp)
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
	return sparse

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
