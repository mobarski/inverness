from gensim.models import FastText

try:
	from .util_time import timed
except (ModuleNotFoundError,ImportError):
	from util_time import timed

from time import time

class Subword():

	@timed
	def init_subword(self, size, window=3, min_count=1, epochs=10, workers=None):
		t0 = time()
		_workers = workers or self.params.get('subword__workers') or self.params.get('workers',1)
		corpus_file = self.path+'sentences.txt.gz'
		self.subword = FastText(size=size, window=window, min_count=min_count, workers=_workers)
		self.subword.build_vocab(corpus_file=corpus_file)
		print(f'corpus_count: {self.subword.corpus_count}')
		print(f'sen_cnt: {self.sentences_cnt}')
		print(f'build_vocab {time()-t0:.01f} seconds') # XXX
		self.subword.train(
				corpus_file=corpus_file,
				#total_examples=self.subword.corpus_count,
				total_words=self.tokens_cnt,
				epochs=epochs
			)
		self.subword.save(self.path+'subword')
	
	def load_subword(self):
		self.subword = FastText.load(self.path+'subword')
		return self