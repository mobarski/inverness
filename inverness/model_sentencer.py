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

class sentencer: pass

class Sentencer:

	@timed
	def init_sentencer(self, workers=None):
		"""store line sentences after tokenization
		
		empty sentences are omitted
		an empty line marks the end of the document
		"""
		self.sentencer = sentencer()
		_workers = workers or self.params.get('sentences__workers') or self.params.get('workers',1)
		if _workers>1:
			self._init_sentences_mp(_workers)
		else:
			self._init_sentences_sp()
		pickle.dump(self.sentencer, open(self.path+'sentencer.pkl','wb'))

	def load_sentencer(self):
		self.offset = sorbet(self.path+'offset').load()
		self.sentencer = pickle.load(open(self.path+'sentencer.pkl','rb'))
		return self

	def all_sentences(self, corpus_file=None, desc='sentences'):
		_corpus_file = corpus_file or self.params.get('corpus_file') or 'sentences.txt.gz'
		f = gzip.open(self.path+_corpus_file, 'rt', encoding='utf8')
		f = tqdm(f, desc, total=self.sentencer.sentences_cnt)
		doc_id = 0
		for line in f:
			sen = line.rstrip()
			yield doc_id,sen 
			if not sen: # end of document
				doc_id += 1
		f.close()

	def doc_sentences(self, doc_id, corpus_file=None):
		_corpus_file = corpus_file or self.params.get('corpus_file') or 'sentences.txt.gz'
		try:
			pos = self.offset[doc_id]
		except IndexError:
			raise Exception(f'IndexError: offset[{doc_id}] len(offset)={len(self.offset)}')
		f = gzip.open(self.path+_corpus_file, 'rt', encoding='utf8')
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
	

	def _init_sentences_sp(self):
		offset = sorbet(self.path+'offset').new()
		gzip_level = self.params.get('sentences__gzip_level',1) 
		f = gzip.open(self.path+'sentences.txt.gz', 'wt',
		              encoding='utf8', compresslevel=gzip_level)
		text_to_tokens = self.text_to_tokens
		sentences = self.raw_sentences(desc='sentences')
		pos = 0
		sen_cnt = 0
		tok_cnt = 0
		prev_doc_id = None
		for doc_id,sentence in sentences:
			# end of document
			if doc_id!=prev_doc_id and prev_doc_id is not None:
				f.write('\n')
				offset.append(pos)
				pos = f.tell()
			tokens = text_to_tokens(sentence)
			tok_cnt += len(tokens)
			if not tokens: continue # skip empty sentences
			text = ' '.join(tokens)
			f.write(text)
			f.write('\n')
			prev_doc_id = doc_id
			sen_cnt += 1
		# end of the last document
		f.write('\n')
		offset.append(pos)
		
		f.close()
		offset.save()
		self.offset = offset
		self.sentencer.sentences_cnt = sen_cnt
		self.sentencer.tokens_cnt = tok_cnt
	
	
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
			tok_cnt = 0 # TODO
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
		self.sentencer.sentences_cnt = sen_cnt
		self.sentencer.tokens_cnt = tok_cnt # TODO



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
	tok_cnt = 0
	for sentence in model.text_to_sentences(text):
		tokens = model.text_to_tokens(sentence)
		tok_cnt += len(tokens)
		if not tokens: continue # skip empty sentences
		out += [' '.join(tokens)]
	return out

if __name__ == "__main__":
	import data
	model = Sentencer()
	model.path = 'model_100/'
	model.load_sentencer()