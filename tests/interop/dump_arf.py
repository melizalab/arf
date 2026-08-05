# -*- mode: python -*-
"""Emit a canonical text description of an ARF file's structure.

Used to pin file layout: dtypes, shapes, chunking, filters, attribute types,
and link order. Values that legitimately vary between runs (uuids, the file
name, the library version) are redacted so the output is stable.

This deliberately reports *structure*, not just data — a refactor that quietly
changes a chunk shape or drops creation-order tracking shows up as a diff.

    python -m tests.interop.dump_arf <path>
"""

import sys

import h5py
import numpy as np

REDACTED = "<redacted>"
# attributes whose values are allowed to change without failing the comparison
VOLATILE = frozenset(("uuid", "arf_library_version"))


def dtype_str(dtype):
    """A stable rendering of a numpy dtype."""
    if dtype.names is not None:
        fields = ", ".join(
            "%s:%s" % (name, dtype_str(dtype.fields[name][0])) for name in dtype.names
        )
        return "compound{%s}" % fields
    if dtype.kind == "S":
        return "S%d" % dtype.itemsize
    if dtype.kind == "O":
        special = h5py.check_string_dtype(dtype)
        if special is not None:
            return "vlen-str(%s)" % special.encoding
        return "object"
    return dtype.str


def value_str(value):
    """A stable rendering of an attribute value."""
    if isinstance(value, bytes):
        return repr(value.decode("ascii", "replace"))
    if isinstance(value, np.ndarray):
        return "[%s]" % ", ".join(value_str(v) for v in value.tolist())
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(value_str(v) for v in value)
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, (np.floating,)):
        return "%.6g" % float(value)
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return repr(value)


def attribute_lines(node, indent):
    out = []
    for name in sorted(node.attrs.keys()):
        value = node.attrs[name]
        dtype = node.attrs.get_id(name).dtype
        shown = REDACTED if name in VOLATILE else value_str(value)
        out.append(
            "%sattr %s = %s (dtype=%s)" % (indent, name, shown, dtype_str(dtype))
        )
    return out


def child_names(group):
    """Children in creation order, falling back to name order."""
    try:
        import arf

        return list(arf.keys_by_creation(group))
    except Exception:
        return sorted(group.keys())


def dataset_lines(dset, path):
    filters = []
    if dset.compression is not None:
        filters.append("%s:%s" % (dset.compression, dset.compression_opts))
    if dset.shuffle:
        filters.append("shuffle")
    if dset.fletcher32:
        filters.append("fletcher32")

    out = [
        "DATASET %s" % path,
        "  dtype=%s shape=%s maxshape=%s chunks=%s filters=[%s]"
        % (
            dtype_str(dset.dtype),
            tuple(dset.shape),
            tuple(dset.maxshape),
            tuple(dset.chunks) if dset.chunks else None,
            ",".join(filters),
        ),
    ]
    out.extend(attribute_lines(dset, "  "))
    if dset.size:
        head = dset[: min(3, dset.shape[0])]
        out.append("  head=%s" % value_str(np.asarray(head)))
    return out


def walk(node, path, out):
    for name in child_names(node):
        child = node[name]
        child_path = "%s/%s" % (path.rstrip("/"), name)
        if isinstance(child, h5py.Group):
            out.append("GROUP %s" % child_path)
            out.extend(attribute_lines(child, "  "))
            walk(child, child_path, out)
        else:
            out.extend(dataset_lines(child, child_path))


def dump(path):
    out = []
    with h5py.File(path, "r") as fp:
        out.append("FILE")
        out.extend(attribute_lines(fp, "  "))
        # Creation-order tracking has to be checked behaviorally. The flag on
        # the file creation property list reads back as 0 once a file has been
        # reopened -- for h5py-written files just as much as C++-written ones --
        # so introspecting it proves nothing. Listing the children in creation
        # order does: the writers deliberately create entries back to front, so
        # this line differs from alphabetical whenever tracking survives.
        out.append(
            "  children-in-creation-order=[%s]" % ", ".join(child_names(fp))
        )
        walk(fp, "", out)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("usage: dump_arf.py <path>\n")
        raise SystemExit(2)
    sys.stdout.write(dump(sys.argv[1]))
