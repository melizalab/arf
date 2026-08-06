# -*- mode: python -*-
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
def test_cxx_bytes_units_classify_correctly(cxx_file):
    """C++ discrete event data is recognized as event data.

    The C++ library stores strings as fixed-length, so h5py hands their values
    back as bytes, while arf.py's variable-length strings come back as str.
    is_time_series tests `units not in ("s", "samples")`, which never matched
    b"samples", so a spike train on a discrete timebase -- which the spec
    requires to carry both units="samples" and a sampling_rate -- was read as
    sampled data. The comparison now normalizes first.
    """
    with h5py.File(cxx_file, "r") as fp:
        discrete = fp["entry_000"]["spike_samples"]
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
def test_cxx_uuid_is_one_byte_too_wide(cxx_file):
    """CHARACTERIZATION: the spec asks for a 36-byte string; C++ writes 37."""
    with h5py.File(cxx_file, "r") as fp:
        dtype = fp["entry_000"].attrs.get_id("uuid").dtype
        assert dtype == np.dtype("S37")
    with h5py.File(cxx_file, "r") as fp:
        py_written = arf.get_uuid(fp["entry_000"])
        assert len(str(py_written)) == 36


def test_python_uuid_is_36_bytes(py_file):
    with h5py.File(py_file, "r") as fp:
        assert fp["entry_000"].attrs.get_id("uuid").dtype == np.dtype("S36")


@requires_cxx
def test_cxx_compound_units_are_scalar(cxx_file):
    """CHARACTERIZATION: complex event data needs one unit per field.

    specification.md requires the units attribute of a compound dataset to be
    an array with an element per field, and arf.create_dataset enforces it.
    entry::create_packet_table takes a single std::string, so the C++ library
    cannot express it and writes one scalar for the whole record.
    """
    with h5py.File(cxx_file, "r") as fp:
        units = fp["entry_000"]["intervals"].attrs["units"]
        assert units == b"ms"
        assert np.asarray(units).shape == ()

    # arf.py refuses to write what C++ just wrote
    with pytest.raises(ValueError, match="sequence of units"):
        with arf.open_file("/dev/null", "w", driver="core", backing_store=False) as fp:
            entry = arf.create_entry(fp, "e", 1)
            records = np.rec.fromrecords([(1.0, 2.0)], names=("start", "stop"))
            arf.create_dataset(entry, "intervals", records, units="ms")


@requires_cxx
def test_cxx_always_applies_a_deflate_filter(cxx_file):
    """CHARACTERIZATION: compress=0 still installs gzip; see backlog item J."""
    with h5py.File(cxx_file, "r") as fp:
        for name in ("pcm", "spikes", "intervals"):
            dset = fp["entry_000"][name]
            assert dset.compression == "gzip"
            assert dset.compression_opts == 0


def test_datatype_codes_diverge():
    """CHARACTERIZATION: the two implementations disagree; see backlog item 1.

    c++/arf/types.hpp gives INTRAC_VC the code 7, while specification.md and
    arf.py both say 6, so voltage-clamp data is mislabeled by C++. arf.py in
    turn has no EXTRAC_RAW, which the spec and the C++ enum both define as 23.
    """
    types_hpp = (REPO_ROOT / "c++" / "arf" / "types.hpp").read_text()
    assert "INTRAC_VC = 7" in types_hpp
    assert arf.DataTypes.INTRAC_CC == 5
    assert not hasattr(arf.DataTypes, "INTRAC_VC") or arf.DataTypes.INTRAC_VC == 6
    assert not hasattr(arf.DataTypes, "EXTRAC_RAW")
    assert "EXTRAC_RAW = 23" in types_hpp
