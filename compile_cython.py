# cython
from distutils.core import setup
from Cython.Build import cythonize

if __name__ == '__main__':
    setup(name='PBAS', ext_modules=cythonize('pbas.pyx', annotate=True))
