# setup.py

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
        Extension(
            "src.wavekit.value_change",
            sources=["src/wavekit/value_change.pyx", ],
            include_dirs=[np.get_include()],
            extra_compile_args=["-fpic","-O3", "-march=native"],
            extra_link_args=["-O3", "-march=native"],
            language="c++",
            #define_macros=[('CYTHON_TRACE', '1')] # open profiling
        )
]

if VERDI_HOME := os.environ.get('VERDI_HOME'):

    npi_include_dir = os.path.join(VERDI_HOME,'share/NPI/inc')
    npi_library_dir = os.path.join(VERDI_HOME,'share/NPI/lib/LINUX64')

    extensions = [
        Extension(
            "src.wavekit.npi_fsdb_reader",
            sources=["src/wavekit/npi_fsdb_reader.pyx", ],
            include_dirs=[np.get_include(), npi_include_dir],
            library_dirs=[npi_library_dir],
            libraries=["NPI"],
            extra_compile_args=["-fpic","-O3", "-march=native"],
            extra_link_args=["-O3", "-march=native",f"-Wl,-rpath,{npi_library_dir}"],
            language="c++",
            #define_macros=[('CYTHON_TRACE', '1')] # open profiling
        )
    ]


setup(
    ext_modules=cythonize(extensions,
        compiler_directives={
            "language_level": 3,
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            #'profile': True, # open profiling
            #'linetrace': True
        }
    ),
    script_args=['build_ext','--inplace'],
)
