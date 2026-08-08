# Release notes

This repository holds a specification and two reference implementations, and
all three version independently. `arf.py` is the published Python package;
`ARF_LIBRARY_VERSION` is the header-only C++ library; `spec_version` is the
specification. A library's major version says nothing about which
specification version it implements — see `supported_spec_versions()`.

## 3.0.0 (2026-08-08)

Both libraries go to 3.0.0, each for its own breaking change, alongside
specification 2.2. The bulk of the work is a test suite for the C++ library and
the audit it made possible: 148 test cases across fourteen suites where there
had been a single `main()` of bare `assert`s, and roughly thirty defects found
and fixed. Much of the C++ functionality for reading arf files had never been
used or tested, and interop between the libraries had never been tested.

### Specification 2.2

Three clarifications. Nothing that conformed to 2.1 stops conforming.

- The `uuid` attribute is specified by its 36-character text form rather than a
  byte count. "36-byte" had been read as a constraint on the buffer, so the C++
  library wrote 37 bytes to leave room for a terminator.
- String attributes may be stored fixed- or variable-length, and readers must
  accept both. This had been left open and the two implementations chose
  differently, which made their files mutually unreadable.
- The `units` attribute is the sole determinant of a dataset's timebase.
  Readers must not infer a discrete timebase from the presence of
  `sampling_rate`.

### Python library — breaking

- Pre-2.0 files are no longer supported. `check_file_version` raises
  `DeprecationWarning` below specification 2.0, up from 1.1. The required
  attributes changed at 2.0. The spec has instructions on how to migrate, but no
  tool is provided, because it hasn't been needed for years.
- The supported range is derived, not hard-coded: `[min_spec_version, next
  major after spec_version)`, today `[2.0, 3.0)`. A file written to any later
  2.x is readable with no new release; a 3.0 file never will be from this one.
- `check_file_version` no longer falls back to `arf_library_version` when the
  library version is 3.0 or greater. That substitution was safe only while the
  library and specification shared a scale, and they no longer do. Files old
  enough to be the reason the fallback exists are unaffected.
- `is_entry` no longer returns `True` for the file root. An `h5py.File` is a
  `Group`, so code walking a file counted it as one of its own entries.
- `create_entry` rejects a name containing `/`. `"a/b"` quietly created a group
  nested inside `"a"`, which is not a top-level entry at all.
- The minimum h5py is now 3.15.0, and numpy's floor varies by interpreter. See
  "Dependencies" below.
- The MATLAB interface has been removed. It was never tested, never distributed,
  and effectively unused for years; `git log -- matlab/` recovers it.

### Python library — added

- `file_version(file)` reports the specification version a file claims without
  judging it. `check_file_version` refuses versions it cannot vouch for, which
  is the wrong answer for a migration tool, whose job is precisely to read
  files that are out of range.
- `supported_spec_versions()` returns the readable range, so it does not have
  to be inferred from the library's own version.
- `check_file_structure(file)` reports violations of the specification's
  structural rules — a dataset linked into more than one entry, an entry linked
  into the root more than once — as a list of problems. Multiple linkage cannot
  be prevented from inside this library, since a caller can alias objects with
  plain h5py long after arf wrote them, so this detects rather than prevents.
  Advisory: nothing calls it for you.
- `create_dataset` accepts any sequence of `units` that h5py accepts, not just
  a `list` or `tuple`. h5py returns this attribute as an object array for this
  library's writes and a fixed-width bytes array for the C++ library's, so
  copying a compound dataset between files no longer needs a conversion step.

### C++ library — breaking

Major changes in the API to reduce allocation overhead and simplify move/copy semantics. 

- `shared_ptr` is gone from every return type. `dataspace()`, `datatype()`,
  `create_dataset()`, `create_packet_table()` and `file()` return values;
  `arf::dataset_ptr` and its siblings are deleted. Write
  `arf::h5d::dataset d = entry.create_dataset(...)`.
- The wrappers have real copy and move semantics. Value-like types
  (`datatype`, `dataspace`, `proplist`) copy via `H5?copy`; handle-like types
  (`file`, `group`, `dataset`, `attribute`, `packet_table`, `entry`) are
  move-only, so a second open handle cannot be created by accident.
  `hid_copy()` is the deliberate escape.
- `arf::handle` owns its identifier and releases it with `H5Idec_ref`. Its
  destructor is protected and non-virtual — it was the library's only virtual
  member, and `delete base_ptr` is now a compile error rather than undefined
  behavior.
- boost is gone entirely, including `boost::uuid`, replaced by
  `c++/arf/uuid.hpp` (~130 lines, no dependency). This cut compile time for a
  translation unit including `arf.hpp` from 564 ms to 176 ms, and the header
  count from 456 boost headers to none.
- `datatype(hid_t)` now adopts its argument and is `explicit`, matching
  `dataspace(hid_t)`. Callers holding a borrowed handle must `H5Tcopy` first.
- `h5f::file::name()` is renamed `filename()`; it collided with
  `handle::name()`.
- `entry::create_packet_table` takes a `std::vector<std::string>` of units, one
  per compound field, as the specification requires.
- `h5g::group::create_link` is removed. It was declared and never defined, so
  no call could have linked.
- Opening a file no longer calls `H5Eset_auto2`, which silenced the HDF5 error
  stack process-wide as a side effect. `arf::h5e::silence_auto_print()` is the
  explicit opt-in.

### C++ library — fixed

These change the bytes written, which is what makes this a major bump.

- `INTRAC_VC` was `7`, against `6` in the specification and in `arf.py`;
  C++-written files mislabeled voltage-clamp data (no files with this data type
  are likely to exist).
- String attributes are sized to exactly their content, with `CSET` declared
  UTF-8 and `STRPAD` `NULLPAD`. The `uuid` falls out at 36 bytes with no
  special case. Previously every string was written one byte long, under a
  `CSET` of ASCII that non-ASCII text violated.
- Zero-length datasets could not be created at all: `guess_chunk` returned a
  chunk of 0, which `H5Pset_chunk` rejects.

And these do not:

- Every packet table leaked two identifiers. `~packet_table` guarded
  `H5PTclose` with `H5PTis_valid(...) > 0`, but that function returns `herr_t`,
  where 0 means valid. The test was false for exactly the tables that
  needed closing. The worst of the leaks, since packet tables are the streaming
  interface.
- `attribute::read(std::string&)` and `dataset::datatype()` each leaked one
  identifier per call. `arf::entry` reads the `uuid` on open, so walking a file
  leaked a handle per entry, and HDF5 will not fully close a file while
  identifiers remain open.
- The C++ library could not read any string attribute `arf.py` wrote. It
  assumed fixed-length storage, allocating `H5Tget_size` bytes (8, the size of
  a pointer) and then read that pointer as though it held the characters.
  Every string attribute was affected, including `units`, which the
  specification requires on every dataset. Reading is now tolerant of
  variable-length, fixed-length with a terminator, and fixed-length without.
- `check_error` silently swallowed a negative return when the HDF5 error stack
  was empty, converting a failure into a plausible-looking `0` that callers
  assigned straight into an `hid_t`. It now always throws.
- `create_packet_table` over an existing name returned a half-built object
  whose writes went nowhere, with no exception raised.
- `group::read_dataset(name, vector&, offset, stride)` could not work: the
  arguments bound to the wrong parameters of the function it forwards to, so it
  never resized the vector and then indexed into an empty one.
- `dataset::read(vector&, count, ...)` wrote through `&data[0]` without
  resizing, resuting in a heap overflow whenever the caller's vector was shorter than
  `count`.
- `entry` left `_uuid` uninitialized when the entry had no `uuid` attribute, so
  reading it was undefined behavior. It now reports the nil uuid, matching
  `arf.py`.
- `arf::file` rewrote the version attributes on mode `"a"`, clobbering the
  provenance of a file another implementation wrote.
- `attribute::write(ptr, size)` asserted against the dataspace size and then
  ignored `size` — an `assert`, so it said nothing under `NDEBUG`.
- Variable-length arrays, `&v[0]` on possibly-empty vectors, `proplist`
  comparing property lists with `H5Tequal`, and an ambiguity in
  `create_dataset<T>` when the storage and memory types coincide.

### Interoperability

The two libraries can now read each other's files. Before this release, a C++
reader could not read any string attribute `arf.py` wrote, and `arf.py`
misclassified C++-written spike trains as sampled data because it compared `str`
against the `bytes` h5py returns for fixed-length strings. Both readers are now
tolerant of either storage, including files generated by pre 3.x versions of
either library; the C++ writer deliberately keeps writing fixed-length, because
it runs during acquisition, and variable-length puts the characters on the global
heap behind a pointer indirection.

`tests/test_interop.py` exercises both directions against binaries built from
`tests/interop/`, and `tests/golden/cxx_writer.txt` pins the structure of a
C++-written file down to dtypes, chunk shapes, filters, and link creation
order.

### Dependencies

Floors are now the oldest release of each package that ships a wheel for the
interpreter, verified across manylinux, musllinux, macOS arm64 and win_amd64.
Below them `pip` and `uv` build from source, which needs a compiler and HDF5
headers; numpy 2.2.1 has no cp314 wheel, so the previous floor built numpy from
source on every Python 3.14 job.

    h5py>=3.15.0
    numpy>=1.26.2; python_version < '3.13'
    numpy>=2.1.0;  python_version == '3.13'
    numpy>=2.3.2;  python_version >= '3.14'

### Packaging

License metadata moves to PEP 639 (`license = "BSD-3-Clause"` with
`license-files`). Every source file carries an SPDX identifier; eleven C++
headers had been carrying GPL v2 boilerplate from before the project was BSD.
The specification itself remains under the GPL v3 — it licenses the format
description, not the implementations.

## 2.7.4 (2026-08-07)

Two regressions in `select_interval`, both introduced by the 2.7.3 fixes.

- Compound data with an integral `start` field raised a `TypeError`. The window
  is only converted to an integer when it gets rescaled, so leaving it in
  seconds meant subtracting a float in place from an integer field, which numpy
  refuses. The offset is now cast to the field's own type.
- Compound datasets carrying a single scalar `units` attribute for the whole
  record stopped being rescaled. The specification asks for one unit per field,
  but jrecord writes the scalar form, so these files exist. Indexing into the
  scalar read its first character, turning `"samples"` into `"s"`, so a window
  given in seconds was applied as though it were samples — a `[0.4, 0.6)`
  second request selected samples 0.4 to 0.6. Wrong data, silently. A scalar is
  now taken to apply to every field.

## 2.7.3 (2026-08-06)

Bug fixes only; no API change.

- `convert_timestamp` discarded `tzinfo`. It called `mktime(obj.timetuple())`,
  which reads the wall-clock fields as local time, so an aware datetime was
  recorded as the wrong instant — off by the local zone offset. It now uses
  `datetime.timestamp()`, which honors the offset and still treats a naive
  datetime as local time, so nothing changes for those. Relatedly, a pre-epoch
  float produced negative microseconds (`-1.5` became `(-1, -500000)`); the
  seconds are now floored rather than truncated.
- `select_interval` rescaled its window by `sampling_rate` whenever the
  attribute was present. The specification permits a real-valued point process
  to carry one, and for those a window of `[0, 1)` seconds was silently
  reinterpreted as `[0, 1000)` samples, returning events from the first
  thousand seconds. It now keys on whether the dataset's own times are in
  samples.
- `select_interval` returned a boolean mask for an empty dataset. The guard
  tested `idx.size`, but `idx` is the mask, so its size is the length of the
  dataset rather than the number of matches.
- `create_dataset` did not reject numpy string arrays. The check sat inside a
  branch that only ran on input needing conversion, so an array that already
  had a dtype skipped arf's validation and failed later inside h5py with an
  opaque "No conversion path for dtype".
- `check_file_version` raised `InvalidVersion` on a malformed version string,
  rather than one of the three warning types it documents.
- `DataTypes.EXTRAC_RAW` (23) was in the specification and in the C++ library
  but missing from `arf.py`.
