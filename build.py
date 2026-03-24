import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        'src.wavekit.readers.value_change',
        sources=['src/wavekit/readers/value_change.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fpic', '-O3', '-march=native'],
        extra_link_args=['-O3', '-march=native'],
        language='c++',
    ),
    Extension(
        'src.wavekit.readers.fsdb.npi_fsdb_reader',
        sources=['src/wavekit/readers/fsdb/npi_fsdb_reader.pyx'],
        include_dirs=[np.get_include()],
        libraries=['dl'],
        extra_compile_args=['-fpic', '-O3', '-march=native'],
        extra_link_args=['-O3', '-march=native'],
        language='c++',
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'embedsignature': True,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
        },
    ),
    script_args=['build_ext', '--inplace'],
)
