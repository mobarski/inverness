from tqdm import tqdm

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet


class Meta():

	@timed
	def init_meta(self, storage='disk'):
		self.meta = sorbet(self.path+'meta', kind=storage).new()
		get_meta = self.get_meta
		documents = self.doc_iter()
		documents = tqdm(documents, desc='meta')
		for id,doc in enumerate(documents):
			m = get_meta(id,doc)
			self.meta.append(m)
		self.meta.save()
	
	def load_meta(self, storage='disk'):
		self.meta = sorbet(self.path+'meta', kind=storage).load()
