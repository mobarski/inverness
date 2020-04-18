from gensim.models import LsiModel
from gensim.matutils import sparse2full
from tqdm import tqdm
from array import array
import multiprocessing as mp
import sys

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
		# handle onepass=False
		model_sparse = self.sparse
		class __sparse():
			def __iter__(self):
				for a in model_sparse:
					yield [(int(a[i]),a[i+1]) for i in range(0,len(a),2)]

		self.lsi = LsiModel(__sparse(), **kwargs)
		self.lsi.save(self.path+'lsi.pkl')

	
	def load_lsi(self):
		self.lsi = LsiModel.load(self.path+'lsi.pkl')

	def load_dense(self, storage='disk'):
		self.dense = sorbet(self.path+'dense', kind=storage).load()
	
	def sparse_to_dense(self, sparse):
		dense = self.lsi[sparse]
		dense = sparse2full(dense, self.lsi.num_topics)
		dense = array('f',dense)
		return dense

	@timed
	def init_dense(self, storage=None, workers=None):
		_workers = workers or self.params.get('dense__workers') or self.params.get('workers',1)
		_storage = storage or self.params.get('dense__storage') or self.params.get('storage','disk')
		if _workers>1:
			self._init_dense_mp(workers=_workers, storage=_storage)
		else:
			self._init_dense_sp(storage=_storage)

	def _init_dense_sp(self, storage='disk'):
		self.dense = sorbet(self.path+'dense', kind=storage).new()
		for a in self.sparse:
			sparse = [(int(a[i]),a[i+1]) for i in range(0,len(a),2)]
			dense = self.sparse_to_dense(sparse)
			self.dense.append(dense)
		self.dense.save()

	def _init_dense_mp(self, workers, storage):
		chunksize = self.params.get('dense__chunksize',10)
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

def dense_worker(doc_id):
	a = model.sparse[doc_id]
	sparse = [(int(a[i]),a[i+1]) for i in range(0,len(a),2)]
	dense = model.lsi[sparse]
	dense = sparse2full(dense, model.lsi.num_topics)
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

