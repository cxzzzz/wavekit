import os

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
    )
]

if VERDI_HOME := os.environ.get('VERDI_HOME'):
    npi_include_dir = os.path.join(VERDI_HOME, 'share/NPI/inc')
    npi_library_dir = os.path.join(VERDI_HOME, 'share/NPI/lib/LINUX64')

    extensions.append(
        Extension(
            'src.wavekit.readers.fsdb.npi_fsdb_reader',
            sources=['src/wavekit/readers/fsdb/npi_fsdb_reader.pyx'],
            include_dirs=[np.get_include(), npi_include_dir],
            library_dirs=[npi_library_dir],
            libraries=['NPI', 'rt'],
            extra_compile_args=['-fpic', '-O3', '-march=native'],
            extra_link_args=[
                '-O3',
                '-march=native',
                f'-Wl,-rpath,{npi_library_dir}',
            ],
            language='c++',
        )
    )

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
