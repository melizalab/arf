#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
import os

if sys.hexversion < 0x02060000:
    raise RuntimeError("Python 2.6 or higher required")

# setuptools 0.7+ doesn't play nice with distribute, so try to use existing
# package if possible
try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

VERSION = '2.2.1'

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

install_requires = ["numpy>=1.3", "h5py>=2.2"]
if (os.environ.get('TRAVIS') == 'true' and os.environ.get('TRAVIS_PYTHON_VERSION').startswith('2.6')):
    install_requires.append('unittest2>=0.5.1')

setup(
    name='arf',
    version=VERSION,
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    author='Dan Meliza',
    maintainer='Dan Meliza',
    url="https://github.com/melizalab/arf",
    download_url="https://github.com/melizalab/arf/archive/%s.tar.gz" % VERSION,
    install_requires=install_requires,

    py_modules=['arf'],
    test_suite='nose.collector'
)
# Variables:
# End:
