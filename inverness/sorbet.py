"""Sorbet - frozen sequence storage engine

Sorbet is fast - on a "standard" laptop it achieves:
- 200k writes per second,
- 300k reads per second.
"""

# ------------------------------------------------------------------------------

def sorbet(path, kind='disk'):
	if kind=='disk':
		return sorbet_on_disk(path)
	elif kind=='mem':
		return sorbet_in_mem(path)
	elif kind=='mem_only':
		return sorbet_in_mem_only(path)
	else:
		raise Exception(f"Invalid sorbet kind: '{kind}'")
		
# ------------------------------------------------------------------------------

from pickle import dump,load,HIGHEST_PROTOCOL
import os

# TODO parallel scan -> zwykly nie ma sensu bo jest funkcja filter
# TODO multiproc scan -> (path,index[lo:hi]->dict)

class sorbet_on_disk:
	"""frozen sequence storage engine"""
	
	def __init__(self, path, mp_pool=None):
		"""configure prefix of sorbet files"""
		self.path = path
		self.mp_pool = mp_pool
		self.protocol = HIGHEST_PROTOCOL
	
	def new(self):
		"""create new storage for appending data"""
		self.f = open(f"{self.path}.data",'wb')
		self.index = []
		return self
	
	def save(self):
		"""save data on disk; removes ability to append new data"""
		dump(self.index, open(f"{self.path}.index",'wb'), protocol=self.protocol)
		self.f.close()
		self.f = open(f"{self.path}.data",'rb')
		print(f"size of {self.path}.data - {os.path.getsize(f'{self.path}.data')/1_000_000:.01f} MB") # XXX
		return self
	
	def load(self):
		"""read data from disk"""
		self.f = open(f"{self.path}.data",'rb')
		self.index = load(open(f"{self.path}.index",'rb'))
		return self
	
	def dump(self, data):
		"""save data on disk"""
		self.new()
		f = self.f
		index = self.index
		p = self.protocol
		for val in data:
			index.append( f.tell() )
			dump(val, f, protocol=p)
		f.close()
		dump(index, open(f"{self.path}.index",'wb'), protocol=self.protocol)
		self.f = open(f"{self.path}.data",'rb')
		print(f"size of {self.path}.data - {os.path.getsize(f'{self.path}.data')/1_000_000:.01f} MB") # XXX
		return self
	
	def append(self, val):
		"""append value (of any type) to the dataset"""
		self.index.append( self.f.tell() )
		dump(val, self.f, protocol=self.protocol)
	
	def delete(self):
		self.f.close()
		os.remove(f"{self.path}.index")
		os.remove(f"{self.path}.data")
	
	def __getitem__(self, key):
		"""return value for given key or iterator for given slice"""
		if type(key) is slice:
			return self.__getslice__(key)
		else:
			pos = self.index[key]
			self.f.seek(pos)
			return load(self.f)
	
	def __getslice__(self, key):
		"""return iterator over given slice"""
		start,stop,step = key.indices(len(self))
		for i in range(start,stop,step):
			yield self[i]
	
	def __len__(self):
		"""return number of items"""
		return len(self.index)
	
	# experimental
	def imap(self, fun):
		yield from self.mp_pool.imap(fun, self)

# ------------------------------------------------------------------------------

class sorbet_in_mem:
	"""frozen sequence storage engine"""
	
	def __init__(self, path):
		"""configure prefix of sorbet files"""
		self.path = path
		self.protocol = HIGHEST_PROTOCOL
	
	def new(self):
		"""create new storage for appending data"""
		self.f = open(f"{self.path}.data",'wb')
		self.data = []
		return self
	
	def save(self):
		"""save data on disk; removes ability to append new data"""
		dump(self.data, self.f, protocol=self.protocol)
		self.f.close()
		print(f"size of {self.path}.data - {os.path.getsize(f'{self.path}.data')/1_000_000:.01f} MB") # XXX
		return self
	
	def load(self):
		"""read data from disk"""
		self.f = open(f"{self.path}.data",'rb')
		self.data = load(open(f"{self.path}.data",'rb'))
		return self
	
	def dump(self, data):
		"""save data on disk"""
		self.new()
		self.data = list(data)
		f = self.f
		p = self.protocol
		dump(self.data, f, protocol=p)
		f.close()
		print(f"size of {self.path}.data - {os.path.getsize(f'{self.path}.data')/1_000_000:.01f} MB") # XXX
		return self
	
	def append(self, val):
		"""append value (of any type) to the dataset"""
		self.data.append(val)
	
	def delete(self):
		del self.data
		os.remove(f"{self.path}.data")
	
	def __getitem__(self, key):
		"""return value for given key or iterator for given slice"""
		return self.data[key]
		
	def __len__(self):
		"""return number of items"""
		return len(self.data)

# ------------------------------------------------------------------------------

class sorbet_in_mem_only:
	"""frozen sequence storage engine"""
	
	def __init__(self, path):
		self.data = []
	
	def new(self):
		return self
	
	def save(self):
		return self
	
	def load(self):
		assert False
	
	def dump(self, data):
		self.data = list(data)
		return self
	
	def append(self, val):
		self.data.append(val)
	
	def delete(self):
		del self.data
	
	def __getitem__(self, key):
		return self.data[key]
		
	def __len__(self):
		return len(self.data)

# ---[ TEST ]-------------------------------------------------------------------

if __name__=="__main__":
	from time import time
	import sys
	label = sys.argv[0][:-3]
	path = f'../data/{label}'
	
	N = 1_100_000
	kind = 'disk'
	if 1:
		data = ({'a':i,'aa':i*10} for i in range(N))
		t0=time()
		db = sorbet(path,kind).dump(data)
		print("write time:",f"{time()-t0:.02f}s",f'{N/(time()-t0):.0f} items/s')
	if 1:
		db = sorbet(path,kind).load()
		t0=time()
		for i in range(N):
			db[i]
		print("read time:",f"{time()-t0:.02f}s",f'{N/(time()-t0):.0f} items/s')
	exit()
	
	print()
	print(list(db[:3]))
	print()
	for x in list(db[:3]):
		print(x)
	print()
	print(db[10])
	print('\nfilter')
	for x in filter(lambda x:x['a']%10000==0, db):
		print(x)
	print('\nfilter - slice')
	for x in filter(lambda x:x['a']%10000==0, db[50000:]):
		print(x)


	
