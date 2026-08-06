# -*- mode: python -*-
# Copyright (C) 2006-2026 C Daniel Meliza
# SPDX-License-Identifier: BSD-3-Clause
"""
This is ARF, a python library for storing and accessing audio and ephys data in
HDF5 containers.
"""
import contextlib
import numbers
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from time import mktime, struct_time
from typing import Iterator, List, Optional, Tuple, Union
from uuid import UUID

import h5py as h5
import numpy as np
import numpy.typing as npt

try:
    # these symbols were moved in 3.12
    from h5py import INDEX_CRT_ORDER, ITER_INC  # ty: ignore[unresolved-import]
except ImportError:
    from h5py.h5 import INDEX_CRT_ORDER, ITER_INC  # ty: ignore[unresolved-import]

Timestamp = Union[datetime, struct_time, int, float, Tuple[int, int]]
ArfTimeStamp = np.ndarray
Datashape = Tuple[int, ...]

spec_version = "2.2"
__version__ = "3.0.0"
version = __version__

# The oldest specification this library will vouch for. Files older than this
# predate the 2.0 rewrite, which changed which attributes are required; arfx
# ships a script that upgrades them.
min_spec_version = "2.0"


def supported_spec_versions() -> Tuple[str, str]:
    """The range of specification versions this library reads, [min, max).

    The upper bound is the next major version after the one implemented, not a
    fixed number: a minor revision of the specification cannot change or remove
    a required attribute, so files written to any later 2.x are readable
    without a new release. A 3.0 file is not, and never will be from this
    release -- reading it needs a library built after that specification
    exists. The library's own version number says nothing about any of this.

    """
    from packaging.version import Version

    implemented = Version(spec_version)
    return (min_spec_version, f"{implemented.major + 1}.0")


def version_info():
    from h5py.version import hdf5_version
    from h5py.version import version as h5py_version

    low, high = supported_spec_versions()
    return (
        f"Library versions:\n arf: {__version__}\n h5py: {h5py_version}\n"
        f" HDF5: {hdf5_version}\n"
        f" ARF specification: writes {spec_version}, reads >={low},<{high}"
    )


class DataTypes(IntEnum):
    """Available data types, by name and integer code:"""
    UNDEFINED = 0
    ACOUSTIC = 1
    EXTRAC_HP = 2
    EXTRAC_LF = 3
    EXTRAC_EEG = 4
    INTRAC_CC = 5
    INTRAC_VC = 6
    EXTRAC_RAW = 23

    EVENT = 1000
    SPIKET = 1001
    BEHAVET = 1002

    INTERVAL = 2000
    STIMI = 2001
    COMPONENTL = 2002


def open_file(
    path: Union[Path, str],
    mode: Optional[str] = None,
    driver: Optional[str] = None,
    libver: Optional[str] = None,
    userblock_size: Optional[int] = None,
    **kwargs,
) -> h5.File:
    """Open an ARF file, creating as necessary.

    Use this instead of h5py.File to ensure that root-level attributes and group
    creation property lists are set correctly.

    """
    from h5py import File, h5p

    # Caution: This is a private API of h5py, subject to change without notice
    from h5py._hl import files as _files
    from h5py.version import version as h5py_version
    from packaging.version import Version

    path = Path(path)
    exists = path.exists()
    try:
        fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_link_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    except AttributeError:
        # older version of h5py
        fp = File(path, mode=mode, driver=driver, libver=libver, **kwargs)
    else:
        posargs = []
        if Version(h5py_version) >= Version("2.9"):
            posargs += ["rdcc_nslots", "rdcc_nbytes", "rdcc_w0"]
        if Version(h5py_version) >= Version("3.5"):
            posargs += ["locking", "page_buf_size", "min_meta_keep", "min_raw_keep"]
        if Version(h5py_version) >= Version("3.7"):
            # integer is needed
            kwargs.update(
                {
                    arg: kwargs.get(arg, 1)
                    for arg in ["alignment_threshold", "alignment_interval"]
                }
            )
        if Version(h5py_version) >= Version("3.8"):
            posargs += ["meta_block_size"]
        kwargs.update({arg: kwargs.get(arg) for arg in posargs})
        fapl = _files.make_fapl(driver, libver, **kwargs)
        fid = _files.make_fid(
            bytes(path),
            mode,
            userblock_size,
            fapl,
            fcpl=fcpl,
            swmr=kwargs.get("swmr", False),
        )
        fp = File(fid)

    if not exists and fp.mode == "r+":
        set_attributes(
            fp,
            arf_library="python",
            arf_library_version=__version__,
            arf_version=spec_version,
        )
    return fp


def create_entry(
    group: h5.Group, name: str, timestamp: Timestamp, **attributes
) -> h5.Group:
    """Create a new ARF entry under group, setting required attributes.

    An entry is an abstract collection of data which all refer to the same time
    frame. Data can include physiological recordings, sound recordings, and
    derived data such as spike times and labels. See add_data() for information
    on how data are stored.

    name -- the name of the new entry. Any string without a path separator;
            "a/b" would create a group nested inside "a" rather than an entry.

    timestamp -- timestamp of entry (datetime object, or seconds since
               January 1, 1970). Can be an integer, a float, or a tuple
               of integers (seconds, microsceconds)

    Additional keyword arguments are set as attributes on created entry.

    Returns: newly created entry object

    """
    if "/" in name:
        raise ValueError(
            f"entry name {name!r} contains a path separator, which would create "
            "a nested group rather than an entry"
        )
    grp = group.create_group(name, track_order=True)
    set_uuid(grp, attributes.pop("uuid", None))
    set_attributes(grp, timestamp=convert_timestamp(timestamp), **attributes)
    return grp


def create_dataset(
    group: h5.Group,
    name: str,
    data: npt.ArrayLike,
    units: str = "",
    datatype=DataTypes.UNDEFINED,
    chunks: Union[bool, Datashape] = True,
    maxshape: Optional[Datashape] = None,
    compression: Optional[str] = None,
    **attributes,
) -> h5.Dataset:
    """Create an ARF dataset under group, setting required attributes

    Required arguments:
    name --   the name of dataset in which to store the data
    data --   the data to store

    Data can be of the following types:

    * sampled data: an N-D numerical array of measurements
    * "simple" event data: a 1-D array of times
    * "complex" event data: a 1-D array of records, with field 'start' required

    Optional arguments:
    datatype --      a code defining the nature of the data in the channel
    units --         channel units (optional for sampled data, otherwise required)
    sampling_rate -- required for sampled data and event data with units=='samples'

    Arguments passed to h5py:
    maxshape --    make the node resizable up to this shape. Use None for axes that
                   need to be unlimited.
    chunks --      specify the chunk size. The optimal chunk size depends on the
                   intended use of the data. For single-channel sampled data the
                   auto-chunking (True) is probably best.
    compression -- compression strategy. Can be 'gzip', 'szip', 'lzf' or an integer
                   in range(10) specifying gzip(N).  Only gzip is really portable.

    Additional arguments are set as attributes on the created dataset

    Returns the created dataset
    """
    from numpy import asarray

    srate = attributes.get("sampling_rate")
    # check data validity before doing anything
    values = data if hasattr(data, "dtype") else asarray(data)
    # NB: applies to every input, not only to what had to be converted. A
    # numpy string array already has a dtype, and skipping it here only defers
    # the failure to h5py, which reports "No conversion path for dtype".
    if values.dtype.kind in ("S", "O", "U"):  # ty: ignore[unresolved-attribute]
        raise ValueError("data must be in array with numeric or compound type")
    if values.dtype.kind == "V":  # ty: ignore[unresolved-attribute]
        if "start" not in values.dtype.names:  # ty: ignore[unresolved-attribute]
            raise ValueError("complex event data requires 'start' field")
        if not isinstance(units, (list, tuple)):
            raise ValueError("complex event data requires sequence of units")
        if not len(units) == len(values.dtype.names):  # ty: ignore[unresolved-attribute]
            raise ValueError("number of units doesn't match number of fields")
    if units == "":
        if srate is None or not srate > 0:
            raise ValueError(
                "unitless data assumed time series and requires sampling_rate attribute"
            )
    elif units == "samples":
        if srate is None or not srate > 0:
            raise ValueError(
                "data with units of 'samples' requires sampling_rate attribute"
            )
    # NB: can't really catch case where sampled data has units but doesn't
    # have sampling_rate attribute

    dset = group.create_dataset(
        name, data=values, maxshape=maxshape, chunks=chunks, compression=compression
    )
    set_attributes(dset, units=units, datatype=datatype, **attributes)
    return dset


def create_table(
    group: h5.File, name: str, dtype: npt.DTypeLike, **attributes
) -> h5.Dataset:
    """Create a new array dataset under group with compound datatype and maxshape=(None,)

    Intended for top-level tables, such as a log for the whole file, which the
    specification exempts from the dataset requirements. Nothing stops it being
    used inside an entry, but a dataset there must carry `units` and
    `datatype`; pass them as keyword arguments if you do that.

    """
    dset = group.create_dataset(name, shape=(0,), dtype=dtype, maxshape=(None,))
    set_attributes(dset, **attributes)
    return dset


def append_data(dset: h5.Dataset, data: npt.ArrayLike):
    """Append data to dset along axis 0. Data must be a single element or
    a 1D array of the same type as the dataset (including compound datatypes)."""
    # NB: not asarray(data). A tuple is one compound record, but converting it
    # would make it an array of len(fields) elements.
    shape = getattr(data, "shape", None)
    N = shape[0] if shape else 1
    if N == 0:
        return
    oldlen = dset.shape[0]
    newlen = oldlen + N
    dset.resize(newlen, axis=0)
    dset[oldlen:] = data


def select_interval(dset: h5.Dataset, begin: float, end: float):
    """Extracts values from dataset between [begin, end), specified in seconds. For
    point process data, times are offset to the beginning of the interval.
    Returns (values, offset)

    """
    # Rescale the window only when the dataset's own times are in samples.
    # Keying on the presence of sampling_rate instead would be wrong: the spec
    # permits a real-valued point process to carry one, and rescaling those
    # reinterprets a window of [0, 1) seconds as [0, 1000) samples.
    if _sample_timebase(dset):
        Fs = dset.attrs["sampling_rate"]
        begin = int(begin * Fs)
        end = int(end * Fs)

    if is_marked_pointproc(dset):
        t = dset["start"]
        idx = (t >= begin) & (t < end)
        data = dset[idx]
        # NB: cast to the field's own type. begin is only an integer when the
        # window was rescaled, and subtracting a float in place from an integer
        # field raises rather than converting.
        data["start"] -= data.dtype["start"].type(begin)
    elif is_time_series(dset):
        data = dset[slice(begin, end)]
    else:
        t = dset[:]
        idx = (t >= begin) & (t < end)
        # NB: unconditional. Guarding on idx.size would test the length of
        # the mask, not the number of matches, and returning idx for an empty
        # dataset hands back a bool array where the dataset's dtype belongs.
        data = dset[idx] - begin
    return data, begin


def file_version(file: h5.File):
    """Return the specification version a file claims, without judging it.

    check_file_version refuses versions this library cannot vouch for, which is
    the wrong answer for a caller whose whole job is handling old files -- a
    migration tool has to read the version *because* it is out of range. This
    reports what the file says and leaves the decision to the caller.

    Raises UserWarning if the file declares no version, or one that cannot be
    parsed; there is nothing to report in those cases.

    """
    from packaging.version import InvalidVersion, Version

    ver = file.attrs.get("arf_version", None)
    from_library = ver is None
    if from_library:
        try:
            ver = file.attrs["arf_library_version"]
        except KeyError as err:
            raise UserWarning(
                f"Unable to determine ARF version for {file.filename};"
                "created by another program?"
            ) from err
    with contextlib.suppress(LookupError, AttributeError):
        # if the attribute is stored as a string, it's ascii-encoded
        ver = ver.decode("ascii")
    try:
        parsed = Version(ver)
    except InvalidVersion as err:
        raise UserWarning(
            f"Unparseable ARF version {ver!r} for {file.filename};"
            "created by another program?"
        ) from err
    if from_library and parsed >= Version(supported_spec_versions()[1]):
        raise UserWarning(
            f"{file.filename} has no arf_version, and its arf_library_version "
            f"({parsed}) cannot stand in for one: the libraries have "
            "versioned independently of the specification since 3.0"
        )
    return parsed


def check_file_version(file: h5.File):
    """Check the ARF version attribute of file for compatibility.

    Raises DeprecationWarning for files older than this library supports,
    FutureWarning for files from a later major version of the specification,
    and UserWarning for files that may not have been created by an ARF library.

    Returns the version for the file. Use file_version() to read the version
    without the compatibility check.

    """
    from packaging.version import Version

    parsed = file_version(file)
    low, high = supported_spec_versions()
    if parsed < Version(low):
        raise DeprecationWarning(
            f"{file.filename} claims ARF specification {parsed}, which "
            f"predates {low}; the required attributes changed at 2.0. "
            "The arfx package ships a script that upgrades old files."
        )
    elif parsed >= Version(high):
        raise FutureWarning(
            f"{file.filename} claims ARF specification {parsed}, which "
            f"postdates this library's {spec_version}. A major revision may "
            "change required attributes, so its contents cannot be trusted "
            "here; upgrade arf to a release that implements it."
        )
    return parsed


def _link_count(obj: h5.HLObject) -> int:
    """The number of hard links pointing at an object."""
    from h5py import h5o  # ty: ignore[unresolved-import]

    return h5o.get_info(obj.id).rc


def check_file_structure(file: h5.File) -> List[str]:
    """Check the structural rules in the specification.

    These are the rules that cannot be enforced as data is written, because a
    caller can link objects together with plain h5py afterwards:

    - a dataset must not be linked into more than one entry, which would leave
      the time of its data undefined
    - an entry must not be multiply linked to the root

    Returns a list of problems, empty if the file conforms. Like
    check_file_version this is advisory and nothing calls it for you.

    Entries nested inside other entries are not examined: the specification
    says their contents are not part of the ARF data hierarchy. Attributes are
    not checked either -- create_dataset validates those as data is written.

    """
    problems = []
    for name in file:
        entry = file[name]
        if not is_entry(entry):
            continue
        if _link_count(entry) > 1:
            problems.append(f"entry '{name}' is linked into the file more than once")
        for child in entry:
            node = entry[child]
            if isinstance(node, h5.Dataset) and _link_count(node) > 1:
                problems.append(
                    f"dataset '{name}/{child}' is linked into more than one entry"
                )
    return problems


def set_attributes(node: h5.HLObject, overwrite: bool = True, **attributes) -> None:
    """Set multiple attributes on node.

    If overwrite is False, and the attribute already exists, does nothing. If
    the value for a key is None, the attribute is deleted.

    """
    aset = node.attrs
    for k, v in attributes.items():
        if not overwrite and k in aset:
            pass
        elif v is None:
            if k in aset:
                del aset[k]
        else:
            aset[k] = v


def keys_by_creation(group: h5.Group) -> Iterator[str]:
    """Returns a lazy sequence of links in group in order of creation.

    Raises an error if the group was not set to track creation order.

    """
    out: list[bytes] = []
    try:
        group.id.links.iterate(
            out.append, idx_type=INDEX_CRT_ORDER, order=ITER_INC
        )
    except (AttributeError, RuntimeError):
        # pre 2.2 shim
        def f(name):
            if name.find(b"/", 1) == -1:
                out.append(name)

        group.id.links.visit(f, idx_type=INDEX_CRT_ORDER, order=ITER_INC)
    return map(group._d, out)


def convert_timestamp(obj: Timestamp) -> ArfTimeStamp:
    """Make an ARF timestamp from an object.

    Argument can be a datetime.datetime object, a time.struct_time, an integer,
    a float, or a tuple of integers. The returned value is a numpy array with
    the integer number of seconds since the Epoch and any additional
    microseconds.

    Note that because floating point values are approximate, the conversion
    between float and integer tuple may not be reversible.

    """
    from math import floor

    from numpy import zeros

    out = zeros(2, dtype="int64")
    if isinstance(obj, datetime):
        # NB: timestamp(), not mktime(timetuple()). The latter reads the
        # wall-clock fields as *local* time and discards any tzinfo, so an
        # aware datetime was recorded as the wrong instant. For a naive
        # datetime timestamp() also assumes local time, which is what the old
        # code did, so nothing changes there.
        out[0] = int(obj.replace(microsecond=0).timestamp())
        out[1] = obj.microsecond
    elif isinstance(obj, struct_time):
        out[0] = mktime(obj)
    elif isinstance(obj, numbers.Integral):
        out[0] = obj
    elif isinstance(obj, numbers.Real):
        # floor, not truncation: the microseconds are the remainder *after*
        # the second, so truncating a pre-epoch value like -1.5 yields
        # (-1, -500000), which is not a time the spec can express
        seconds = floor(obj)
        out[0] = seconds
        out[1] = round((float(obj) - seconds) * 1e6)  # ty: ignore[unsupported-operator]
    else:
        try:
            out[:2] = obj[:2]  # ty: ignore[not-subscriptable]
        except (IndexError, ValueError) as err:
            raise TypeError(f"unable to convert {obj} to timestamp") from err
    return out


def timestamp_to_datetime(timestamp: ArfTimeStamp) -> datetime:
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta

    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))


def timestamp_to_float(timestamp: ArfTimeStamp) -> float:
    """Convert an ARF timestamp to a floating point (sec since epoch)"""
    return sum(t1 * t2 for t1, t2 in zip(timestamp, (1.0, 1e-6)))


def set_uuid(obj: h5.HLObject, uuid: Union[str, bytes, UUID, None] = None):
    """Set the uuid attribute of an HDF5 object. Use this method to ensure correct dtype"""
    from uuid import uuid4

    if uuid is None:
        uuid = uuid4()
    elif isinstance(uuid, bytes):
        if len(uuid) == 16:
            uuid = UUID(bytes=uuid)
        else:
            uuid = UUID(hex=uuid.decode("ascii"))

    if "uuid" in obj.attrs:
        del obj.attrs["uuid"]
    obj.attrs.create("uuid", str(uuid).encode("ascii"), dtype="|S36")


def get_uuid(obj: h5.HLObject) -> UUID:
    """Return the uuid for obj, or null uuid if none is set"""
    # TODO: deprecate null uuid ret val
    try:
        uuid = obj.attrs["uuid"]
    except KeyError:
        return UUID(int=0)
    return UUID(uuid.decode("ascii"))


def count_children(obj: h5.Group, type=None) -> int:
    """Return the number of children of obj, optionally restricting by class"""
    if type is None:
        return len(obj)
    # hdf5 offers no way to get this without inspecting each child, so this is
    # linear in the number of children
    return sum(1 for x in obj if obj.get(x, getclass=True) is type)


def _decode_attribute(value):
    """Normalize a string attribute for comparison.

    The specification permits either storage for a string attribute, and the
    implementations differ: this library writes variable-length strings, which
    h5py returns as str, while the C++ library writes fixed-length ones, which
    come back as bytes. Comparisons have to normalize first or they silently
    fail against files from another writer.

    """
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _sample_timebase(dset: h5.Dataset) -> bool:
    """Return True if the times in dset are counted in samples rather than seconds.

    Sampled data is indexed by sample by construction. Event data says so in
    its units, which for a compound dataset is the entry for the 'start' field.

    """
    if "sampling_rate" not in dset.attrs:
        return False
    if is_time_series(dset):
        return True
    units = dset.attrs.get("units", None)
    if units is None:
        return False
    if is_marked_pointproc(dset):
        names = dset.dtype.names
        if names is None or "start" not in names:
            return False
        # NB: a compound dataset is supposed to carry one unit per field, but
        # files in the wild carry a single scalar for the whole record. Treat
        # that as applying to every field rather than indexing into the string,
        # which would silently read its first character.
        if not isinstance(units, (str, bytes)):
            try:
                units = units[names.index("start")]
            except (IndexError, KeyError, TypeError):
                return False
    return _decode_attribute(units) == "samples"


def is_time_series(dset: h5.Dataset) -> bool:
    """Return True if dset is a sampled time series (units are not time)"""
    return (
        not is_marked_pointproc(dset)
        and "sampling_rate" in dset.attrs
        and _decode_attribute(dset.attrs.get("units", None)) not in ("s", "samples")
    )


def is_marked_pointproc(dset: h5.Dataset) -> bool:
    """Return True if dset is a marked point process (a complex dtype with 'start' field)"""
    return dset.dtype.names is not None and "start" in dset.dtype.names


def is_entry(obj: h5.HLObject) -> bool:
    """Return True if the object is an entry.

    An entry is an HDF5 group other than the root. The root is excluded
    deliberately: the specification allows a file to hold top-level datasets
    that belong to no entry, and h5py's File *is* a Group, so testing only for
    Group counted the file itself as one of its own entries.

    """
    return isinstance(obj, h5.Group) and obj.name != "/"


def count_channels(dset: h5.Dataset) -> int:
    """Return the number of channels (columns) in dset"""
    try:
        return dset.shape[1]
    except IndexError:
        return 1


# Variables:
# End:
