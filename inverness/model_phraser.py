from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser as GensimPhraser
from gensim.models.word2vec import LineSentence
from tqdm import tqdm
import multiprocessing as mp
import os

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet

# ---[ MODEL ]------------------------------------------------------------------

class Phraser():

	@timed
	def init_phraser(self, components=False, **kwargs):
		sentences = LineSentence(self.path+'sentences.txt.gz')
		phrases = Phrases(sentences, **kwargs)
		self.phraser = GensimPhraser(phrases)
		self.phraser.components = components
		self.phraser.save(self.path+'phraser.pkl')
		del phrases
	
	def skip_phraser(self):
		self.phraser = None
		open(self.path+'phraser.pkl','wb') # empty file -> skip_phraser

	def load_phraser(self):
		path = self.path+'phraser.pkl'
		if os.path.getsize(path):
			self.phraser = GensimPhraser.load(path)
		else:
			self.phraser = None
	
	def doc_to_phrased(self, doc):
		text = self.doc_to_text(doc)
		yield from self.text_to_phrased(text)
	
	def text_to_phrased(self, text):
		if self.phraser:
			components = self.phraser.components
			sentences = self.text_to_sentences(text)
			for sentence in sentences:
				tokens = self.text_to_tokens(sentence)
				phrased = self.phraser[tokens]
				yield from phrased
				if components:
					yield from set(tokens)-set(phrased)
		else:
			yield from self.text_to_tokens(text)
	
	# TODO jako plik txt.gz
	@timed
	def init_phrased(self, storage='disk'):
		documents = self.doc_iter()
		phrased = (list(self.doc_to_phrased(doc)) for doc in documents)
		phrased = tqdm(phrased, desc='phrased', total=len(self.meta)) # progress bar
		self.phrased = sorbet(self.path+'phrased', kind=storage).dump(phrased)

	@timed
	def init_phrased_mp(self, workers=4, chunksize=100):
		s = sorbet(self.path+'phrased').new()
		doc_id_iter = range(len(self.meta))
		doc_id_iter = tqdm(doc_id_iter,'phrased',len(self.meta))
		with mp.Pool(
					workers,
					init_phrased_worker,
					[
						self.path,
						self.get_doc_by_meta,
						self.doc_to_text,
						self.text_to_sentences,
						self.text_to_tokens,
						self.phraser==None
					]
				) as pool:
			phrased = pool.imap(phrased_worker, doc_id_iter, chunksize)
			for p in phrased:
				s.append(list(p))
		self.phrased = s.save()
	
	def skip_phrased(self):
		doc_to_text = self.doc_to_text
		text_to_tokens = self.text_to_tokens
		doc_iter = self.doc_iter
		class __phrased:
			def __iter__(self):
				return (list(text_to_tokens(doc_to_text(doc))) for doc in doc_iter())
			def delete(self): pass
		self.phrased = __phrased()
	
	def load_phrased(self, storage='disk'):
		self.phrased = sorbet(self.path+'phrased', kind=storage).load()

# ---[ MULTIPROCESSING ]--------------------------------------------------------

def init_phrased_worker(*args):
	global model
	model = Phraser()
	model.path              = args[0]
	model.get_doc_by_meta   = args[1]
	model.doc_to_text       = args[2]
	model.text_to_sentences = args[3]
	model.text_to_tokens    = args[4]
	model.skip_phraser      = args[5]
	model.meta = sorbet(model.path+'meta').load()
	if not model.skip_phraser:
		model.load_phraser()
	
def phrased_worker(doc_id):
	meta = model.meta[doc_id]
	doc = model.get_doc_by_meta(meta)
	text = model.doc_to_text(doc)
	if model.skip_phraser:
		out = model.text_to_tokens(text)
	else:
		out = []
		components = model.phraser.components
		for sen in model.text_to_sentences(text):
			tokens = model.text_to_tokens(sen)
			phrased = model.phraser[tokens]
			out.extend(phrased)
			if components:
				out.extend(set(tokens)-set(phrased)) 
	return out

# ---[ DEBUG ]-------------------------------------------------------------------

def count_digits(tokens):
	x = 0
	for token in tokens:
		for letter in token:
			if letter in "0123456789":
				x += 1
	return x

if __name__=="__main__":
	model = Phraser()
	model.path = 'model_100/'
	model.load_phraser()
	model.load_phrased()
	#
	from pprint import pprint
	#pprint(model.phrased[0])
	#
	#
	from time import time
	t0=time()
	total = 0
	aaa = {}
	aaa['total'] = 0
	if 0: # 4000 -> 35s
		for tokens in model.phrased:
			x = 0
			for token in tokens:
				for letter in token:
					if letter in "0123456789":
						x += 1
			total += x
	if 1: # 4000 -> 2p:22s
		import multiprocessing as mp
		from tqdm import tqdm
		pool = mp.Pool(processes=2)
		def agg(x):
			aaa['total'] += 1
		#results = pool.map(count_digits, tqdm(model.phrased), 100)
		#results = pool.imap_unordered(count_digits, tqdm(model.phrased), 100)
		results = pool.map_async(count_digits, tqdm(model.phrased), 100, agg)
		for x in results.get():
			total += x
	print(total)
	print(aaa['total'])
	print(f"done in {time()-t0:.02f} seconds")
