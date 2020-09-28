#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
import os
from setuptools import setup
from arf import __version__

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 3.5 or greater required.")

cls_txt = """
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: C++
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

short_desc = "Advanced Recording Format for acoustic, behavioral, and physiological data"

long_desc = """
Library for reading and writing Advanced Recording Format files. ARF files
are HDF5 files used to store audio and neurophysiological recordings in a
rational, hierarchical format. Data are organized around the concept of an
entry, which is a set of data channels that all start at the same time.
Supported data types include sampled data and event data (i.e. spike times).
Requires h5py (at least 2.2) and numpy (at least 1.3).
"""

install_requires = ["h5py>=2.10"]
if (os.environ.get('TRAVIS') == 'true' and os.environ.get('TRAVIS_PYTHON_VERSION').startswith('2.6')):
    install_requires.append('unittest2>=0.5.1')

setup(
    name='arf',
    version=__version__,
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    author='Dan Meliza',
    maintainer='Dan Meliza',
    url="https://github.com/melizalab/arf",
    download_url="https://github.com/melizalab/arf/archive/%s.tar.gz" % __version__,
    install_requires=install_requires,

    py_modules=['arf'],
    test_suite='tests'
)
# Variables:
# End:
