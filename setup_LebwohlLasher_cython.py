# Build with: python setup_LebwohlLasher_cython.py build_ext --inplace
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("LebwohlLasher_cython.pyx"), include_dirs=[numpy.get_include()])
