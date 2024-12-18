from setuptools import setup
from Cython.Build import cythonize
from os import path
from subprocess import check_output
import numpy
import re

numpy_include = path.join(path.dirname(numpy.__file__), 'core/include')


version = ( check_output('git describe --tags --always'.split())
          .decode()
          .split("-")[0]
          .removeprefix("v")
          )

#                    digits dot digits dot digits
version_pattern = "^ (\d+) (\.) (\d+) (\.) (\d+) $".replace(" ", "")

# HACK:
# in GHA we only pick up the last commit in the repo history and the the command
# above returns the commit hash. If this happens, use a dummy version number.
if not re.match(version, version_pattern):
    version = "0.0.0"

files      = "invisible_cities/*/*.pyx"
directives = dict(language_level=3, embedsignature=True)

setup(name         = 'invisible cities',
      version      = version,
      description  = 'NEXT blah blah',
      url          = 'https://github.com/nextic/IC',
      author       = 'NEXT collaboration',
      author_email = 'nextic@TODO.org',
      license      = 'TODO',
      packages     = ['invisible_cities'],
      ext_modules  = cythonize(files, compiler_directives=directives),
      include_dirs = [numpy.get_include()]
)
