
class Params:

	def __init__(self):
		self.params = {}

	def set_params(self,**params):
		for k in params:
			self.params[k] = params[k]
	
	def get_params(self, deep=True):
		return self.params if deep else self.extract_params()

	# NOT USED ???
	def extract_params(self, component='', sep='__'):
		out = {}
		for key,val in self.params.items():
			if component:
				if not key.startswith(component): continue
				key = key[len(component)+len(sep):]
			if sep in key or not key: continue
			out[key] = val
		return out


class Model(Params):
	
	def __init__(self):
		print("hello")
		Params.__init__(self)

if __name__ == "__main__":
	p = Model()
	p.set_params(a=1,b=2,c__x=3,c__y=4,d__x=5,d__y__q=6,d__y__w=7)
	print(p.extract_params(''))
