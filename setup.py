from Cython.Build import cythonize
from distutils.core import setup, Extension
ext = Extension(name="implementations", sources=["clip.pyx", "implementations.pyx"])
setup(ext_modules=cythonize(ext))