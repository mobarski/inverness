from gensim.corpora.dictionary import Dictionary as GensimDictionary
from tqdm import tqdm
import re
import pickle
from array import array

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet

# TODO dodatkowa metoda - zamiast doc->bow doc->ids

# ---[ MODEL ]------------------------------------------------------------------

class Dictionary():
	
	@timed
	def init_dictionary(self, save=True):
		import gzip
		from collections import Counter
		doc_id = 0
		num_pos = 0
		num_nnz = 0
		cfs = Counter()
		dfs = Counter()
		f = gzip.open(self.path+'sentences.txt.gz', 'rt',encoding='utf8')
		f = tqdm(f,'dictionary',self.sentences_cnt)
		unique = set()
		for line in f:
			tokens = line.rstrip().split(' ')
			if not tokens: # end of document
				dfs.update(unique)
				num_nnz += len(unique)
				#
				doc_id += 1
				unique = set()
				continue
			cfs.update(tokens)
			num_pos += len(tokens)
			unique.update(tokens)
		f.close()
		#
		token2id = {t:i for i,(t,cnt) in enumerate(cfs.most_common())}
		dictionary = GensimDictionary()
		dictionary.num_pos = num_pos
		dictionary.num_nnz = num_nnz
		dictionary.num_docs = doc_id
		dictionary.token2id = token2id
		dictionary.cfs = {i:cfs[t] for t,i in token2id.items()}
		dictionary.dfs = {i:dfs[t] for t,i in token2id.items()}
		dictionary.patch_with_special_tokens({'<PAD>':0})
		if save:
			dictionary.save(self.path+'dictionary.pkl')
		self.dictionary = dictionary
	
	def load_dictionary(self):
		self.dictionary = GensimDictionary.load(self.path+'dictionary.pkl')

	@timed
	def prune_dictionary(self, stopwords=None, save=True, **kwargs):
		_stopwords = stopwords or self.params.get('dictionary__stopwords')
		self.dictionary.filter_extremes(**kwargs)
		if _stopwords:
			bad_ids = [self.dictionary.token2id.get(t,-1) for t in _stopwords]
			self.dictionary.filter_tokens(bad_ids)
		self.dictionary.compactify()
		if save:
			self.dictionary.save(self.path+'dictionary.pkl')

	@timed
	def init_bow_old(self, storage='disk'):
		self.bow = sorbet(self.path+'bow', kind=storage).new()
		source = self.phrased
		src = tqdm(source, desc='bow', total=len(self.meta))
		for doc in src:
			bow = self.dictionary.doc2bow(doc) or [(0,1)] # 0 -> '<PAD>'
			self.bow.append(bow)
		self.bow.save()

	@timed
	def init_bow(self, storage=None):
		_storage = storage or self.params.get('bow__storage') or self.params.get('storage','disk')
		self.bow = sorbet(self.path+'bow', kind=_storage).new()
		doc = []
		for doc_id,sen in self.all_sentences(desc='bow'):
			if not sen: # end of document:		
				bow = self.dictionary.doc2bow(doc) or [(0,1)] # 0 -> '<PAD>'
				self.bow.append(bow)
				doc = []
			else:
				tokens = sen.split(' ')
				doc.extend(tokens)
		self.bow.save()

	def load_bow(self, storage=None):
		_storage = storage or self.params.get('bow__storage') or self.params.get('storage','disk')
		self.bow = sorbet(self.path+'bow', kind=storage).load()
	
	# TODO opcjonalna inicjalizacja z sentences
	@timed
	def init_inverted(self):
		inverted = {}
		bow = tqdm(self.bow, desc='inverted', total=len(self.bow))
		for i,b in enumerate(bow):
			for t,cnt in b:
				if t in inverted:
					inverted[t] += [i]
				else:
					inverted[t] = [i]
		#
		self.inverted = sorbet(self.path+'inverted').new()
		for t in range(max(inverted)):
			self.inverted.append(set(inverted.get(t,[])))
		self.inverted.save()

	def load_inverted(self):
		self.inverted = sorbet(self.path+'inverted').load()

	# TODO findall vs match
	def dictionary_query(self, query, exclude=None):
		"returns list of ids and list of tokens matching the query"
		q = re.compile(query)
		e = re.compile(exclude) if exclude else None
		ids = []
		tokens = []
		for i,token in self.dictionary.items():
			if q.findall(token):
				if e and e.findall(token): continue
				ids += [i]
				tokens += [token]
		return ids,tokens

	# TODO zliczanie wystapien
	# TODO sortowanie
	def inverted_query(self, query, exclude=None):
		"returns ids of documents containing matching tokens"
		doc_ids = set()
		token_ids,tokens = self.dictionary_query(query, exclude)
		for i in token_ids:
			doc_ids.update(self.inverted[i])
		return doc_ids
	
# ---[ DEBUG ]------------------------------------------------------------------

if __name__=="__main__":
	model = Dictionary()
	model.path = '../model_10/'
	model.load_dictionary()
	#
	d = list(model.dictionary.dfs.items())
	d.sort(key=lambda x:-x[1])
	print(f'NUM_DOCS: {model.dictionary.num_docs}')
	print('\nSINGLE:')
	for id,cnt in d[:60]:
		token = model.dictionary.get(id)
		print(f"{cnt:4d} {token}")
	print('\nPHRASES:')
	i=0
	for id,cnt in d:
		token = model.dictionary.get(id)
		if '__' not in token: continue
		print(f"{cnt:4d} {token}")
		i+=1
		if i>200:break

