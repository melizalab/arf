arf
---

|ProjectStatus|_ |Version|_ |BuildStatus|_ |License|_ |PythonVersions|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/arf.svg
.. _Version: https://pypi.python.org/pypi/arf/

.. |BuildStatus| image:: https://github.com/melizalab/arf/actions/workflows/tests-python.yml/badge.svg
.. _BuildStatus: https://github.com/melizalab/arf/actions/workflows/tests-python.yml

.. |License| image:: https://img.shields.io/pypi/l/arf.svg
.. _License: https://opensource.org/license/bsd-3-clause/

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/arf.svg
.. _PythonVersions: https://pypi.python.org/pypi/arf/


The Advanced Recording Format `arf <https://meliza.org/spec:1/arf/>`__
is an open standard for storing data from neuronal, acoustic, and
behavioral experiments in a portable, high-performance, archival format.
The goal is to enable labs to share data and tools, and to allow
valuable data to be accessed and analyzed for many years in the future.

**arf** is built on the the `HDF5 <http://www.hdfgroup.org/HDF5/>`__
format, and all arf files are accessible through standard HDF5 tools,
including interfaces to HDF5 written for other languages (e.g. MATLAB,
Python, etc). **arf** comprises a set of specifications on how different
kinds of data are stored. The organization of arf files is based around
the concept of an *entry*, a collection of data channels associated with
a particular point in time. An entry might contain one or more of the
following:

-  raw extracellular neural signals recorded from a multichannel probe
-  spike times extracted from neural data
-  acoustic signals from a microphone
-  times when an animal interacted with a behavioral apparatus
-  the times when a real-time signal analyzer detected vocalization

Entries and datasets have metadata attributes describing how the data
were collected. Datasets and entries retain these attributes when copied
or moved between arf files, helping to prevent data from becoming
orphaned and uninterpretable.

This repository contains:

-  The specification for arf (in specification.md). This is also hosted
   at https://meliza.org/spec:1/arf/.
-  A fast, type-safe C++ interface for reading and writing arf files
-  A python interface for reading and writing arf files

You don't need the python or C++ libraries to read arf files; they are just
standard HDF5 files that can be accessed with standard tools and libraries, like
h5py (see below).

installation
~~~~~~~~~~~~

ARF files require HDF5>=1.8 (http://www.hdfgroup.org/HDF5).

The python interface requires Python 3.7 or greater and h5py>=3.8. The last
version of this package to support Python 2 was ``2.5.1``. The last version to
support h5py 2 was ``2.6.7``. To install the module:

.. code:: bash

   pip install arf

To use the C++ interface, you need boost>=1.42 (http://boost.org). In
addition, if writing multithreaded code, HDF5 needs to be compiled with
``--enable-threadsafe``. The interface is header-only and does not need
to be compiled. To install:

.. code:: bash

   make install

version information
~~~~~~~~~~~~~~~~~~~

The specification and implementations provided in this project use a
form of semantic versioning (http://semver.org). Specifications receive
a major and minor version number. Changes to minor version numbers must
be backwards compatible (i.e., only added requirements). The current
released version of the ARF specification is ``2.1``.

Implementation versions are synchronized with the major version of the
specification but otherwise evolve independently. For example, the
python ``arf`` package version ``2.1.0`` is compatible with any ARF
version ``2.x``.

There was no public release of arf prior to ``2.0``.

access ARF files with HDF5 tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The structure of an arf file can be explored using the ``h5ls`` tool.
For example, to list entries:

.. code:: bash

   $ h5ls file.arf
   test_0001                Group
   test_0002                Group
   test_0003                Group
   test_0004                Group

Each entry appears as a Group. To list the contents of an entry, use
path notation:

.. code:: bash

   $ h5ls file.arf/test_0001
   pcm                      Dataset {609914}

This shows that the data in ``test_0001`` is stored in a single node,
``pcm``}, with 609914 data points. Typically each channel will have its
own dataset.

The ``h5dump`` command can be used to output data in binary format. See
the HDF5 documentation for details on how to structure the output. For
example, to extract sampled data to a 16-bit little-endian file (i.e.,
PCM format):

.. code:: bash

   h5dump -d /test_0001/pcm -b LE -o test_0001.pcm file.arf

contributing
~~~~~~~~~~~~

ARF is under active development and we welcome comments and
contributions from neuroscientists and behavioral biologists interested
in using it. We’re particularly interested in use cases that don’t fit
the current specification. Please post issues or contact Dan Meliza (dan
at meliza.org) directly.

related projects
~~~~~~~~~~~~~~~~

-  `arfx <https://github.com/melizalab/arfx>`__ is a commandline tool
   for manipulating ARF files.

open data formats
^^^^^^^^^^^^^^^^^

-  `neurodata without borders <http://www.nwb.org>`__ has similar goals
   and also uses HDF5 for storage. The data schema is considerably more
   complex and prescriptive, but it's got a lot of investment from the field,
   so you should consider it first.
-  `NIX <https://github.com/G-Node/nix>`__ was designed by INCF for sharing electrophysiology data.
-  `bark <https://github.com/margoliashlab/bark>`__ is inspired by ARF
   but uses the filesystem directory structure instead of HDF5 to simplify data access.

i/o libraries
^^^^^^^^^^^^^

-  `neo <https://github.com/NeuralEnsemble/python-neo>`__ is a Python
   package for working with electrophysiology data in Python, together
   with support for reading a wide range of neurophysiology file
   formats.
-  `neuroshare <http://neuroshare.org>`__ is a set of routines for
   reading and writing data in various proprietary and open formats.
