from docstr_coverage.coverage import get_docstring_coverage
from pprint import pprint
from pathlib import Path

path_list = [str(p) for p in Path('.').glob('*.py')]
pprint(get_docstring_coverage(path_list))


