from tqdm import tqdm
import pickle
from hashlib import md5
from collections import Counter
import multiprocessing as mp
import gzip

try:
	from .util_time import timed
	from .sorbet import sorbet
except (ModuleNotFoundError,ImportError):
	from util_time import timed
	from sorbet import sorbet


# def my_hash(text):
	# """64bit hash from text"""
	# return int(md5(text.lower().encode()).hexdigest()[:16],16)

class Sentencer:

	# @timed
	# def init_sentencer(self, no_above=None, no_matching=None):
		# self.sentencer = sen = Counter()
		# self.sentencer_no_above = no_above
		# self.sentencer_no_matching = no_matching
		# cnt = 0
		# sentences = self.all_sentences(as_tokens=False, desc='sentencer')
		# for text in sentences:
			# h = my_hash(text)
			# sen[h] += 1
			# cnt += 1
		# with open(self.path+'sentencer.pkl','wb') as f:
			# pickle.dump(sen, f)
			# pickle.dump(no_above, f)
			# pickle.dump(no_matching, f)
	
	# def skip_sentencer(self):
		# self.sentencer = None
		# self.sentencer_no_above = None
		# self.sentencer_no_matching = None
		# pass # TODO
	
	# def load_sentencer(self):
		# with open(self.path+'sentencer.pkl','rb') as f:
			# self.sentencer = pickle.load(f)
			# self.sentencer_no_above = pickle.load(f)
			# self.sentencer_no_matching = pickle.load(f)

	# def all_sentences(self, as_tokens=True, clean=True, desc='sentences'):
		# cnt = self.sentencer
		# no_above = self.sentencer_no_above
		# no_matching = self.sentencer_no_matching
		# doc_iter = self.doc_iter()
		# doc_iter = tqdm(doc_iter, desc=desc, total=len(self.meta))
		# for doc in doc_iter:
			# text = self.doc_to_text(doc)
			# sentences = self.text_to_sentences(text)
			# if cnt:
				# for s in sentences:
					# h = my_hash(s)
					# if clean and no_above and cnt.get(h,0)>no_above: continue # skip common sentences (copyright etc)
					# if clean and no_matching and no_matching.findall(s): continue # skip matching sentences
					# yield self.text_to_tokens(s) if as_tokens else s
			# else:
				# for s in sentences:
					# yield self.text_to_tokens(s) if as_tokens else s

	# def explain_sentencer(self, k):
		# top = self.sentencer.most_common(k)
		# top_ids = set([x[0] for x in top])
		# top_text = {}
		# for text in self.all_sentences(as_tokens=False, clean=False):
			# h = my_hash(text)
			# if h in top_ids:
				# top_text[h] = text
		# for id,cnt in top:
			# print(cnt,id,top_text.get(id,'REMOVED'))

	def all_sentences(self, desc='sentences'):
		f = gzip.open(self.path+'sentences.txt.gz', 'rt', encoding='utf8')
		f = tqdm(f, desc, total=self.sentences_cnt)
		doc_id = 0
		for line in f:
			sen = line.rstrip()
			yield doc_id,sen 
			if not sen: # end of document
				doc_id += 1
		f.close()

	def doc_sentences(self, doc_id):
		pos = self.offset[doc_id]
		f = gzip.open(self.path+'sentences.txt.gz', 'rt', encoding='utf8')
		f.seek(pos)
		for line in f:
			sen = line.rstrip()
			if not sen: break # end of document
			yield sen
		f.close()

	def raw_sentences(self, desc='raw_sentences'):
		"""iterate over doc_id,sentence before tokenization"""
		doc_iter = self.doc_iter()
		doc_iter = tqdm(doc_iter, desc=desc, total=len(self.meta))
		for doc_id,doc in enumerate(doc_iter):
			text = self.doc_to_text(doc)
			sentences = self.text_to_sentences(text)
			for sen in sentences:
				yield doc_id,sen
	
	
	@timed
	def init_sentences(self, workers=None):
		"""store line sentences after tokenization
		
		empty sentences are omitted
		an empty line marks the end of the document
		"""
		_workers = workers or self.params.get('sentences_workers') or self.params.get('workers',1)
		if _workers>1:
			self._init_sentences_mp(_workers)
		else:
			self._init_sentences_sp()
		# TODO store self.sentences_cnt in sentences_meta.pkl


	def _init_sentences_sp(self):
		offset = sorbet(self.path+'offset').new()
		gzip_level = self.params.get('sentences__gzip_level',1) 
		f = gzip.open(self.path+'sentences.txt.gz', 'wt',
		              encoding='utf8', compresslevel=gzip_level)
		text_to_tokens = self.text_to_tokens
		sentences = self.raw_sentences(desc='sentences')
		pos = 0
		sen_cnt = 0
		prev_doc_id = None
		for doc_id,sentence in sentences:
			# end of document
			if doc_id!=prev_doc_id and prev_doc_id is not None:
				f.write('\n')
				offset.append(pos)
				pos = f.tell()
			tokens = text_to_tokens(sentence)
			if not tokens: continue # skip empty sentences
			text = ' '.join(tokens)
			f.write(text)
			f.write('\n')
			prev_doc_id = doc_id
			sen_cnt += 1
		f.close()
		offset.save()
		self.offset = offset
		self.sentences_cnt = sen_cnt
	
	
	def _init_sentences_mp(self, workers=2):
		chunk = self.params.get('sentences__chunk',100)
		offset = sorbet(self.path+'offset').new()
		doc_id_iter = range(len(self.meta))
		doc_id_iter = tqdm(doc_id_iter, 'sentences', len(self.meta))
		compresslevel = 1 # TODO cfg
		f = gzip.open(self.path+'sentences.txt.gz', 'wt',
		              encoding='utf8', compresslevel=compresslevel)
		with mp.Pool(
					workers,
					init_sentences_worker,
					[
						self.path,
						self.get_doc_by_meta,
						self.doc_to_text,
						self.text_to_sentences,
						self.text_to_tokens,
					]
				) as pool:
			doc_sen_iter = pool.imap(sentences_worker, doc_id_iter, chunk)
			sen_cnt = 0
			pos = 0
			for doc_sen in doc_sen_iter:
				offset.append(pos)
				if doc_sen:
					f.write('\n'.join(doc_sen))
					f.write('\n\n')
				else: # empty document
					f.write('\n')
				start_pos = pos
				pos = f.tell()
				length = pos-start_pos
				# TODO store start_pos+length
				sen_cnt += len(doc_sen)
		f.close()
		offset.save()
		self.offset = offset
		self.sentences_cnt = sen_cnt



def init_sentences_worker(*args):
	global model
	model = Sentencer()
	model.path              = args[0]
	model.get_doc_by_meta   = args[1]
	model.doc_to_text       = args[2]
	model.text_to_sentences = args[3]
	model.text_to_tokens    = args[4]
	model.meta = sorbet(model.path+'meta').load()

def sentences_worker(doc_id):
	meta = model.meta[doc_id]
	doc = model.get_doc_by_meta(meta)
	text = model.doc_to_text(doc)
	out = []
	for sentence in model.text_to_sentences(text):
		tokens = model.text_to_tokens(sentence)
		if not tokens: continue # skip empty sentences
		out += [' '.join(tokens)]
	return out

if __name__ == "__main__":
	import data
	model = Sentencer()
	model.path = 'model_100/'
	model.load_sentencer()