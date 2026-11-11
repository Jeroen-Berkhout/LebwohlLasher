# Build with: python setup_LebwohlLasher_cython_openMP.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# -fopenmp is needed at compile and link time, otherwise prange runs serially
ext = Extension("LebwohlLasher_cython_openMP", ["LebwohlLasher_cython_openMP.pyx"],
                extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"])

setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])
