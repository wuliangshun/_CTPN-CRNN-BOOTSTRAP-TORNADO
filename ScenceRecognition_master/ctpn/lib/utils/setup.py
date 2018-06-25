from Cython.Build import cythonize
import numpy as np
from distutils.core import setup

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
print(numpy_include)
numpy_include = r'C:/Users/Administrator/AppData/Roaming/Python/Python35/site-packages/numpy/core/include/'
#ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"],include_dirs=[numpy_include]),
# include_path=[numpy_include]
setup(
    ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"],include_path=[numpy_include]),
)

