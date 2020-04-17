from time import time
import sys

# TODO zmienna srodowiskowa sterujaca czy ma byc output i gdzie (out/err)
# TODO prefix nazwy funkcji -> nazwa pliku (bez .py)
def timed(fun):
	def wrapped(*args,**kwargs):
		t0 = time()
		out = fun(*args,**kwargs)
		print(f'{fun.__name__} done in {time()-t0:.02f} seconds',file=sys.stderr)
		return out
	return wrapped
