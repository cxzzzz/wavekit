import os
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

for artifact in Path('src/wavekit').glob('**/*.so'):
    artifact.unlink()

if os.environ.get('PYODIDE'):
    compile_args = ['-fpic', '-O3']
    link_args = ['-O3']
    fsdb_libraries = []
elif sys.platform == 'linux':
    compile_args = ['-fpic', '-O3']
    link_args = ['-O3']
    fsdb_libraries = ['dl']
elif sys.platform == 'darwin':
    compile_args = ['-fpic', '-O3']
    link_args = ['-O3']
    fsdb_libraries = []
else:
    raise RuntimeError(f'Unsupported platform: {sys.platform}')

extensions = [
    Extension(
        'wavekit.readers.value_change',
        sources=['src/wavekit/readers/value_change.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++',
    ),
    Extension(
        'wavekit.readers.fsdb.npi_fsdb_reader',
        sources=['src/wavekit/readers/fsdb/npi_fsdb_reader.pyx'],
        include_dirs=[np.get_include()],
        libraries=fsdb_libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++',
    ),
]

setup(
    package_dir={'': 'src'},
    ext_modules=cythonize(
        extensions,
        include_path=['src'],
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
