# -*- mode: python -*-
"""
This is ARF, a python library for storing and accessing audio and ephys data in
HDF5 containers.
"""
import numbers
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from time import mktime, struct_time
from typing import Iterator, Optional, Tuple, Union
from uuid import UUID

import h5py as h5
import numpy as np
import numpy.typing as npt

try:
    # these symbols were moved in 3.12
    from h5py import INDEX_CRT_ORDER, ITER_INC
except ImportError:
    from h5py.h5 import INDEX_CRT_ORDER, ITER_INC

Timestamp = Union[datetime, struct_time, int, float, Tuple[int, int]]
ArfTimeStamp = np.ndarray
Datashape = Tuple[int, ...]

spec_version = "2.1"
__version__ = "2.7.1"
version = __version__


def version_info():
    from h5py.version import hdf5_version
    from h5py.version import version as h5py_version

    return f"Library versions:\n arf: {__version__}\n h5py: {h5py_version}\n HDF5: {hdf5_version}"


class DataTypes(IntEnum):
    """Available data types, by name and integer code:"""
    UNDEFINED = 0
    ACOUSTIC = 1
    EXTRAC_HP = 2
    EXTRAC_LF = 3
    EXTRAC_EEG = 4
    INTRAC_CC = 5
    INTRAC_VC = 6
    
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
        kwargs.update({arg: kwargs.get(arg, None) for arg in posargs})
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

    name -- the name of the new entry. any valid python string.

    timestamp -- timestamp of entry (datetime object, or seconds since
               January 1, 1970). Can be an integer, a float, or a tuple
               of integers (seconds, microsceconds)

    Additional keyword arguments are set as attributes on created entry.

    Returns: newly created entry object

    """
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

    srate = attributes.get("sampling_rate", None)
    # check data validity before doing anything
    if not hasattr(data, "dtype"):
        data = asarray(data)
        if data.dtype.kind in ("S", "O", "U"):
            raise ValueError("data must be in array with numeric or compound type")
    if data.dtype.kind == "V":
        if "start" not in data.dtype.names:
            raise ValueError("complex event data requires 'start' field")
        if not isinstance(units, (list, tuple)):
            raise ValueError("complex event data requires sequence of units")
        if not len(units) == len(data.dtype.names):
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
        name, data=data, maxshape=maxshape, chunks=chunks, compression=compression
    )
    set_attributes(dset, units=units, datatype=datatype, **attributes)
    return dset


def create_table(
    group: h5.File, name: str, dtype: npt.DTypeLike, **attributes
) -> h5.Dataset:
    """Create a new array dataset under group with compound datatype and maxshape=(None,)"""
    dset = group.create_dataset(name, shape=(0,), dtype=dtype, maxshape=(None,))
    set_attributes(dset, **attributes)
    return dset


def append_data(dset: h5.Dataset, data: npt.ArrayLike):
    """Append data to dset along axis 0. Data must be a single element or
    a 1D array of the same type as the dataset (including compound datatypes)."""
    N = data.shape[0] if hasattr(data, "shape") else 1
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
    try:
        Fs = dset.attrs["sampling_rate"]
        begin = int(begin * Fs)
        end = int(end * Fs)
    except KeyError:
        pass

    if is_marked_pointproc(dset):
        t = dset["start"]
        idx = (t >= begin) & (t < end)
        data = dset[idx]
        data["start"] -= begin
    elif is_time_series(dset):
        idx = slice(begin, end)
        data = dset[idx]
    else:
        t = dset[:]
        idx = (t >= begin) & (t < end)
        if idx.size > 0:
            data = dset[idx] - begin
        else:
            data = idx
    return data, begin


def check_file_version(file: h5.File):
    """Check the ARF version attribute of file for compatibility.

    Raises DeprecationWarning for backwards-incompatible files, FutureWarning
    for (potentially) forwards-incompatible files, and UserWarning for files
    that may not have been created by an ARF library.

    Returns the version for the file

    """
    from packaging.version import Version

    try:
        ver = file.attrs.get("arf_version", None)
        if ver is None:
            ver = file.attrs["arf_library_version"]
    except KeyError as err:
        raise UserWarning(
            f"Unable to determine ARF version for {file.filename};"
            "created by another program?"
        ) from err
    try:
        # if the attribute is stored as a string, it's ascii-encoded
        ver = ver.decode("ascii")
    except (LookupError, AttributeError):
        pass
    # should be backwards compatible after 1.1
    file_version = Version(ver)
    if file_version < Version("1.1"):
        raise DeprecationWarning(
            f"ARF library {version} may have trouble reading file "
            f"version {file_version} (< 1.1)"
        )
    elif file_version >= Version("3.0"):
        raise FutureWarning(
            f"ARF library {version} may be incompatible with file "
            f"version {file_version} (>= 3.0)"
        )
    return file_version


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
    from numpy import zeros

    out = zeros(2, dtype="int64")
    if isinstance(obj, datetime):
        out[0] = mktime(obj.timetuple())
        out[1] = obj.microsecond
    elif isinstance(obj, struct_time):
        out[0] = mktime(obj)
    elif isinstance(obj, numbers.Integral):
        out[0] = obj
    elif isinstance(obj, numbers.Real):
        out[0] = obj
        out[1] = (obj - out[0]) * 1e6
    else:
        try:
            out[:2] = obj[:2]
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


def count_children(obj: h5.HLObject, type=None) -> int:
    """Return the number of children of obj, optionally restricting by class"""
    if type is None:
        return len(obj)
    else:
        # there doesn't appear to be any hdf5 function for getting this
        # information without inspecting each child, which makes this somewhat
        # slow
        return sum(1 for x in obj if obj.get(x, getclass=True) is type)


def is_time_series(dset: h5.Dataset) -> bool:
    """Return True if dset is a sampled time series (units are not time)"""
    return (
        not is_marked_pointproc(dset)
        and "sampling_rate" in dset.attrs
        and dset.attrs.get("units", None) not in ("s", "samples")
    )


def is_marked_pointproc(dset: h5.Dataset) -> bool:
    """Return True if dset is a marked point process (a complex dtype with 'start' field)"""
    return dset.dtype.names is not None and "start" in dset.dtype.names


def is_entry(obj: h5.HLObject) -> bool:
    """Return True if the object is an entry (i.e. an hdf5 group)"""
    return isinstance(obj, h5.Group)


def count_channels(dset: h5.Dataset) -> int:
    """Return the number of channels (columns) in dset"""
    try:
        return dset.shape[1]
    except IndexError:
        return 1


# Variables:
# End:
