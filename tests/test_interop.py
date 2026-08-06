# -*- mode: python -*-
# Copyright (C) 2026 C Daniel Meliza
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-implementation characterization tests.

Interop is the whole point of ARF, so these are the highest-value
characterization tests in the project: a file written by one implementation is
read by the other, in both directions, and the structure of a C++-written file
is pinned against a golden dump.

This module *runs* the C++ helper binaries but never builds them -- that is
`make interop`'s job, and `make test-interop` does both in order. When the
binaries are absent, everything here skips, so the Python-only CI matrix stays
green.

Regenerate the golden file with `make golden-update` after deliberately
changing what the writer produces.
"""

import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

import arf

from .interop import dump_arf

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "golden" / "cxx_writer.txt"
VARIANT = os.environ.get("ARF_CXX_VARIANT", "plain")
BIN_DIR = REPO_ROOT / "build" / VARIANT

requires_cxx = pytest.mark.skipif(
    not (BIN_DIR / "write_arf").exists() or not (BIN_DIR / "read_arf").exists(),
    reason="C++ interop binaries not built; run `make interop`",
)


def run_tool(name, *args):
    proc = subprocess.run(
        [str(BIN_DIR / name)] + [str(a) for a in args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc


@pytest.fixture(scope="module")
def cxx_file(tmp_path_factory):
    """A canonical file produced by the C++ writer."""
    path = tmp_path_factory.mktemp("interop") / "cxx.arf"
    proc = run_tool("write_arf", path)
    assert proc.returncode == 0, f"write_arf failed:\n{proc.stdout}{proc.stderr}"
    return path


@pytest.fixture(scope="module")
def py_file(tmp_path_factory):
    """A canonical file produced by arf.py, mirroring the C++ writer."""
    path = tmp_path_factory.mktemp("interop") / "py.arf"
    with arf.open_file(path, "w") as fp:
        # back to front, so creation order and name order disagree
        for i in (1, 0):
            entry = arf.create_entry(
                fp,
                "entry_%03d" % i,
                1234567890 + i,
                animal="bird_042",
                experimenter="dmeliza",
            )
            arf.create_dataset(
                entry,
                "pcm",
                np.arange(128, dtype="f8") * 0.5,
                units="mV",
                datatype=arf.DataTypes.ACOUSTIC,
                sampling_rate=20000,
            )
            arf.create_dataset(
                entry,
                "spikes",
                np.arange(16, dtype="f8") * 0.01,
                units="s",
                datatype=arf.DataTypes.SPIKET,
            )
    return path


# --- structural golden ----------------------------------------------------


@requires_cxx
def test_golden_structure(cxx_file):
    """The layout of a C++-written file is exactly what it was."""
    produced = dump_arf.dump(cxx_file)
    if os.environ.get("ARF_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(produced)
        pytest.skip("regenerated %s" % GOLDEN.name)
    assert GOLDEN.exists(), "no golden file; run `make golden-update`"
    expected = GOLDEN.read_text()
    assert produced == expected, (
        "the structure of C++-written files changed.\n"
        "If that was deliberate, run `make golden-update` and review the diff."
    )


# --- arf.py reading a C++ file --------------------------------------------


@requires_cxx
def test_python_accepts_cxx_version(cxx_file):
    with h5py.File(cxx_file, "r") as fp:
        assert str(arf.check_file_version(fp)) == arf.spec_version


@requires_cxx
def test_python_reads_cxx_creation_order(cxx_file):
    with h5py.File(cxx_file, "r") as fp:
        assert list(arf.keys_by_creation(fp)) == ["entry_001", "entry_000"]


@requires_cxx
def test_python_reads_cxx_entry_attributes(cxx_file):
    with h5py.File(cxx_file, "r") as fp:
        entry = fp["entry_000"]
        timestamp = entry.attrs["timestamp"]
        assert timestamp.dtype == np.dtype("int64")
        assert list(timestamp) == [1234567890, 0]
        assert arf.timestamp_to_float(timestamp) == pytest.approx(1234567890.0)
        # the uuid survives as a readable hex string
        assert str(arf.get_uuid(entry)).count("-") == 4


@requires_cxx
def test_python_reads_cxx_payloads(cxx_file):
    with h5py.File(cxx_file, "r") as fp:
        pcm = fp["entry_000"]["pcm"]
        assert pcm.shape == (128,)
        assert pcm[0] == 0.0
        assert pcm[2] == 1.0
        intervals = fp["entry_000"]["intervals"]
        assert intervals["start"][1] == 100
        assert intervals["name"][0] == b"stim_00"


@requires_cxx
def test_python_classifies_cxx_datasets(cxx_file):
    with h5py.File(cxx_file, "r") as fp:
        entry = fp["entry_000"]
        assert arf.is_time_series(entry["pcm"])
        assert not arf.is_marked_pointproc(entry["pcm"])
        # compound records with a 'start' field
        assert arf.is_marked_pointproc(entry["intervals"])
        # real-valued event times: not a time series, no sampling rate
        assert not arf.is_time_series(entry["spikes"])


# --- C++ reading an arf.py file -------------------------------------------


@requires_cxx
def test_cxx_reads_python_file(py_file):
    proc = run_tool("read_arf", py_file)
    assert proc.returncode == 0, (
        "the C++ library's reading of an arf.py file changed:\n"
        f"{proc.stdout}{proc.stderr}"
    )


# --- cross-implementation discrepancies, pinned as they stand -------------


@requires_cxx
def test_cxx_discrete_event_data_classifies_correctly(cxx_file):
    """A spike train on a discrete timebase is event data, not a time series.

    It carries both units="samples" and a sampling_rate, as the spec requires,
    which is the combination easiest to misread as sampled data. The C++
    library writes fixed-length strings, which h5py returns as bytes, so the
    comparison has to normalize or files from the two libraries classify
    differently.
    """
    with h5py.File(cxx_file, "r") as fp:
        discrete = fp["entry_000"]["spike_samples"]
        assert discrete.attrs["units"] == b"samples"
        assert isinstance(discrete.attrs["units"], bytes)
        assert discrete.attrs["sampling_rate"] == 20000
        assert not arf.is_time_series(discrete)
        assert not arf.is_marked_pointproc(discrete)

        # and the sampled dataset next to it is still a time series
        assert arf.is_time_series(fp["entry_000"]["pcm"])


def test_python_units_are_str(py_file):
    """The same attribute written by arf.py compares as arf.py expects."""
    with h5py.File(py_file, "r") as fp:
        units = fp["entry_000"]["pcm"].attrs["units"]
        assert isinstance(units, str)
        assert units == "mV"


@requires_cxx
def test_cxx_uuid_matches_the_spec(cxx_file):
    """The spec asks for a 36-byte string, and both libraries now write one."""
    with h5py.File(cxx_file, "r") as fp:
        assert fp["entry_000"].attrs.get_id("uuid").dtype == np.dtype("S36")
        assert len(str(arf.get_uuid(fp["entry_000"]))) == 36


def test_python_uuid_is_36_bytes(py_file):
    with h5py.File(py_file, "r") as fp:
        assert fp["entry_000"].attrs.get_id("uuid").dtype == np.dtype("S36")


@requires_cxx
def test_cxx_compound_units_are_one_per_field(cxx_file):
    """Complex event data carries one unit per compound field.

    specification.md requires it and arf.create_dataset enforces it;
    entry::create_packet_table took a single std::string, so the C++ library
    could not express it and wrote one scalar for the whole record.
    """
    with h5py.File(cxx_file, "r") as fp:
        intervals = fp["entry_000"]["intervals"]
        units = intervals.attrs["units"]
        assert np.asarray(units).shape == (3,)
        # fixed-length, so h5py hands these back as bytes
        assert list(units) == [b"", b"ms", b"ms"]
        # one per field, in field order
        assert len(units) == len(intervals.dtype.names)

    # and arf.py rejects the scalar form outright
    with pytest.raises(ValueError, match="sequence of units"):
        with arf.open_file("/dev/null", "w", driver="core", backing_store=False) as fp:
            entry = arf.create_entry(fp, "e", 1)
            records = np.rec.fromrecords([(1.0, 2.0)], names=("start", "stop"))
            arf.create_dataset(entry, "intervals", records, units="ms")


@requires_cxx
def test_cxx_datasets_carry_a_deflate_frame(cxx_file):
    """Every C++-written dataset is filtered at deflate level 0, on purpose.

    Level 0 stores the data uncompressed but still frames it through zlib,
    giving each chunk an adler32 for about 16 bytes. That integrity check is
    wanted for recorded data, so it is the default rather than opt-in; a
    negative compression argument writes no filter.
    """
    with h5py.File(cxx_file, "r") as fp:
        for name in ("pcm", "spikes", "spike_samples", "intervals"):
            dset = fp["entry_000"][name]
            assert dset.compression == "gzip"
            assert dset.compression_opts == 0


def test_spec_version_agrees_across_all_three():
    """A fourth thing written out in three places, with the same drift risk.

    specification.md is normative; both libraries must declare the version they
    implement, and the same supported range, or a file will claim to conform to
    something it does not.
    """
    spec = (REPO_ROOT / "specification.md").read_text()
    declared = re.search(r"^-\s+Version:\s+(\S+)", spec, re.M)
    assert declared, "could not find the version in specification.md"
    assert declared.group(1) == arf.spec_version

    types_hpp = (REPO_ROOT / "c++" / "arf" / "types.hpp").read_text()
    assert '#define ARF_VERSION "%s"' % arf.spec_version in types_hpp

    low, high = arf.supported_spec_versions()
    assert '#define ARF_MIN_SPEC_VERSION "%s"' % low in types_hpp
    # the upper bound is derived on both sides, so agreeing on the minimum and
    # the implemented version is enough to make the ranges identical
    assert high == "%d.0" % (int(arf.spec_version.split(".")[0]) + 1)


def test_datatype_codes_agree_across_all_three():
    """specification.md is normative; both libraries must match its table.

    These codes are written out in three places, and have drifted twice: the
    C++ enum gave INTRAC_VC the value 7 where the spec says 6, and arf.py had
    no EXTRAC_RAW at all. Parsing the table keeps every code honest rather than
    spot-checking the two that happened to break.
    """
    table = re.findall(
        r"^\s+(\d+)\s+([A-Z_]+)\s", (REPO_ROOT / "specification.md").read_text(), re.M
    )
    assert len(table) >= 14, "the datatype table in specification.md did not parse"

    types_hpp = (REPO_ROOT / "c++" / "arf" / "types.hpp").read_text()
    for code, name in table:
        assert re.search(
            r"\b%s\s*=\s*%s\b" % (name, code), types_hpp
        ), "c++/arf/types.hpp disagrees with the spec about %s = %s" % (name, code)
        assert hasattr(arf.DataTypes, name), "arf.py has no DataTypes.%s" % name
        assert int(getattr(arf.DataTypes, name)) == int(
            code
        ), "arf.py disagrees with the spec about %s = %s" % (name, code)
