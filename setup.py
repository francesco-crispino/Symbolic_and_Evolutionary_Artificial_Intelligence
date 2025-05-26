from setuptools import setup
from Cython.Build.Dependencies import cythonize
import numpy as np

setup(
    ext_modules=cythonize("im2col_cython.pyx", language_level=3),
    include_dirs=[np.get_include()],
)