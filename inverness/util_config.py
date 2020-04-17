
def extract_cfg(cfg, prefix, sep='.'):
	out = {}
	for key,val in cfg.items():
		if not key.startswith(prefix): continue
		key = key[len(prefix)+len(sep):]
		if sep in key or not key: continue
		out[key] = val
	return out

if __name__=="__main__":
	cfg = {
		'a.1':'aaa',
		'a.2':'bbb',
		'a.x.1':'ccc',
		'a.x.2':'ddd',
		'b.x':'eee',
	}
	print(extract_cfg(cfg,'a.x'))
	