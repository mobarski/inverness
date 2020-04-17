from gensim.models import LsiModel
from tqdm import tqdm
from array import array
import multiprocessing as mp

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet

# ---[ MODEL ]------------------------------------------------------------------

class LSI():
	
	@timed
	def init_lsi(self, **kwargs):
		corpus = self.sparse
		corpus = tqdm(corpus, desc='lsi_input', total=len(self.sparse)) # progress bar
		self.lsi = LsiModel(corpus, **kwargs)
		self.lsi.save(self.path+'lsi.pkl')
	
	def load_lsi(self):
		self.lsi = LsiModel.load(self.path+'lsi.pkl')

	# TODO use sparse_to_dense
	@timed
	def init_dense(self, storage='disk'):
		corpus = self.sparse
		#num_topics = self.lsi.num_topics
		#dense = (self.lsi[c] for c in corpus)
		#dense = ([dict(d).get(i,0) for i in range(num_topics)] for d in dense)
		dense = (self.sparse_to_dense(c) for c in corpus)
		dense = (array('f',d) for d in dense) # uses 4x less memory on disk
		dense = tqdm(dense, desc='dense', total=len(corpus)) # progress bar
		self.dense = sorbet(self.path+'dense', kind=storage).dump(dense)
		
	def load_dense(self, storage='disk'):
		self.dense = sorbet(self.path+'dense', kind=storage).load()
	
	# TODO gensim.matutils.sparse2full
	def sparse_to_dense(self, sparse):
		dense = self.lsi[sparse]
		dense = [dict(dense).get(i,0) for i in range(self.lsi.num_topics)]
		return dense

	@timed
	def init_dense_mp(self, workers=4, chunksize=100):
		s = sorbet(self.path+'dense').new()
		id_iter = range(len(self.meta))
		id_iter = tqdm(id_iter,'dense',len(self.meta))
		with mp.Pool(
					workers,
					init_dense_worker,
					[
						self.path,
					]
				) as pool:
			dense = pool.imap(dense_worker, id_iter, chunksize)
			for d in dense:
				s.append(d)
		self.dense = s.save()

# ---[ MULTIPROCESSING ]--------------------------------------------------------

def init_dense_worker(*args):
	global model
	model_path = args[0]
	model = LSI()
	model.path = model_path
	model.sparse = sorbet(model.path+'sparse').load()
	model.load_lsi()

# TODO gensim.matutils.sparse2full
def dense_worker(doc_id):
	sparse = model.sparse[doc_id]
	dense = model.lsi[sparse]
	dense = [dict(dense).get(i,0) for i in range(model.lsi.num_topics)]
	dense = array('f',dense)
	return dense


# ---[ DEBUG ]------------------------------------------------------------------

if __name__=="__main__":
	model = LSI()
	model.path = 'model_100/'
	model.load_lsi()
	model.load_dense()
	#
	print(model.dense[0])

