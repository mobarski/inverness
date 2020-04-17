from setuptools import setup, find_packages

setup(
	name="inverness",
	version="0.1.1",
	packages=find_packages(),
	
	install_requires=["gensim","nmslib","tqdm","dill"],
	
	author="Maciej Obarski",
	author_email="mobarski@gmail.com",
	description=\
		"Natural Language Processing framework "\
		"built on top of gensim and nmslib.",
	keywords="NLP",
	classifiers=[
		"License :: OSI Approved :: MIT License"
	]
)
