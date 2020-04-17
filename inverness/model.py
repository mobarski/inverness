import os
import re
import pickle
from types import FunctionType

from tqdm import tqdm
import dill

from .model_params     import Params
from .model_meta       import Meta
from .model_sentencer  import Sentencer
from .model_phraser    import Phraser
from .model_dictionary import Dictionary
from .model_tfidf      import TFIDF
from .model_lsi        import LSI
from .model_ann        import ANN
from .model_subword    import Subword

from .util_time import timed

split_sentences_re = re.compile('[.?!]+ (?=[A-Z])')
split_tokens_re = re.compile('[\s.,;!?()\[\]]+')

class Model(
		Params,
		Meta,
		Sentencer,
		Phraser,
		Dictionary,
		TFIDF,
		LSI,
		ANN,
		Subword
	):
	
	def __init__(self, path):
		Params.__init__(self)
		self.set_params(path=path)
		# create model directory if not exists
		model_dir = os.path.dirname(path)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
	
	@property
	def path(self):
		return self.params['path']
	
	#@timed
	def find_sparse(self, text, k=10):
		sparse = self.text_to_sparse(text)
		if sparse:
			i_list,d_list = self.sparse_ann_query(sparse,k)
			for i,d in zip(i_list,d_list):
				m = self.meta[i]
				yield i,d,m

	#@timed
	def find_dense(self, text, k=10):
		dense = self.text_to_dense(text)
		if dense:
			i_list,d_list = self.dense_ann_query(dense,k)
			for i,d in zip(i_list,d_list):
				m = self.meta[i]
				yield i,d,m

	def text_to_sparse(self, text):
		phrased = self.text_to_phrased(text)
		bow = self.dictionary.doc2bow(phrased)
		return self.tfidf[bow]
	
	def text_to_dense(self, text):
		sparse = self.text_to_sparse(text)
		return self.sparse_to_dense(sparse)

	@staticmethod
	def text_to_sentences(text):
		return split_sentences_re.split(text)

	@staticmethod
	def text_to_tokens(text):
		tokens = split_tokens_re.split(text)
		return [t.lower() for t in tokens]

	@staticmethod
	def doc_to_text(doc):
		values = [x for x in doc.values() if type(x) is str]
		return '\n\n'.join(values)

	def explain(self,id):
		phrased = self.phrased[id]
		ids = [self.dictionary.token2id.get(p,-1) for p in phrased]
		return [p for p,i in zip(phrased,ids) if i>=0]

	# TODO opcja do ladowania wszystkiego co jest zapisane
	# TODO analogiczna funkcja init ala sklearn.model
	@timed
	def load(self, components=[]):
		# TODO wykrywanie ktore komponenty wczytac
		if 'fun'        in components: self.load_fun()
		#
		if 'meta'       in components: self.load_meta()
		if 'sentencer'  in components: self.load_sentencer()
		if 'phraser'    in components: self.load_phraser()
		if 'dictionary' in components: self.load_dictionary()
		if 'tfidf'      in components: self.load_tfidf()
		if 'lsi'        in components: self.load_lsi()
		if 'sparse_ann' in components: self.load_sparse_ann()
		if 'dense_ann'  in components: self.load_dense_ann()
		#
		if 'phrased'    in components: self.load_phrased()
		if 'bow'        in components: self.load_bow()
		if 'sparse'     in components: self.load_sparse()
		if 'dense'      in components: self.load_dense()
		if 'inverted'   in components: self.load_inverted()
		#
		return self

	# EXPERIMENTAL
	def get_doc(self, i):
		"""get single document (paragraph) by index"""
		meta = self.meta[i]
		return self.get_doc_by_meta(meta)

	# EXPERIMENTAL
	def init_fun(self):
		fun_dict = {}
		for name in dir(self):
			if name[0]=='_':continue
			fun = getattr(self, name)
			if type(fun) is FunctionType:
				fun_dict[name] = fun
		dill.dump(fun_dict, open(self.path+'fun.dill','wb'), recurse=True)

	# EXPERIMENTAL
	def load_fun(self):
		fun_dict = dill.load(open(self.path+'fun.dill','rb'))
		for name,fun in fun_dict.items():
			setattr(self, name, fun)

# ------------------------------------------------------------------------------

if __name__=="__main__":
	pass
