# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(ext_modules = cythonize(Extension(
    '_quantize',
    sources=['_quantize.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[':libblas.so.3'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)))