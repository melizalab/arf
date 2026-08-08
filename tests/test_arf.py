# -*- mode: python -*-
# Copyright (C) 2006-2026 C Daniel Meliza
# SPDX-License-Identifier: BSD-3-Clause

import time
from uuid import UUID, uuid4

import h5py as h5
import numpy as np
import pytest
from numpy.random import randint, randn

import arf

entry_base = "entry_%03d"
tstamp = time.mktime(time.localtime())
entry_attributes = {
    "intattr": 1,
    "vecattr": [1, 2, 3],
    "arrattr": randn(5),
    "strattr": "an attribute",
}
datasets = [
    dict(
        name="acoustic",
        data=randn(100000),
        sampling_rate=20000,
        datatype=arf.DataTypes.ACOUSTIC,
        maxshape=(None,),
        microphone="DK-1234",
        compression=0,
    ),
    dict(
        name="neural",
        data=(randn(100000) * 2**16).astype("h"),
        sampling_rate=20000,
        datatype=arf.DataTypes.EXTRAC_HP,
        compression=9,
    ),
    dict(
        name="multichannel",
        data=randn(10000, 2),
        sampling_rate=20000,
        datatype=arf.DataTypes.ACOUSTIC,
    ),
    dict(
        name="spikes",
        data=randint(0, 100000, 100),
        datatype=arf.DataTypes.SPIKET,
        units="samples",
        sampling_rate=20000,  # required
    ),
    dict(
        name="empty-spikes",
        data=np.array([], dtype="f"),
        datatype=arf.DataTypes.SPIKET,
        method="broken",
        maxshape=(None,),
        units="s",
    ),
    dict(
        name="events",
        data=np.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),  # 'start' required
        datatype=arf.DataTypes.EVENT,
        units=(b"s", b"", b""),
    ),  # only bytes supported by h5py
]

bad_datasets = [
    dict(name="string datatype", data="a string"),
    dict(name="object datatype", data=bytes),
    dict(name="missing samplerate/units", data=randn(1000)),
    dict(
        name="missing samplerate for units=samples", data=randn(1000), units="samples"
    ),
    dict(
        name="missing start field",
        data=np.rec.fromrecords([(1.0, 1), (2.0, 2)], names=("time", "state")),
        units="s",
    ),
    dict(
        name="missing units for complex dtype",
        data=np.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),
    ),
    dict(
        name="wrong length units for complex dtype",
        data=np.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),
        units=("seconds",),
    ),
]


@pytest.fixture
def tmp_file(tmp_path):
    path = tmp_path / "test"
    fp = arf.open_file(path, "w", driver="core", backing_store=False)
    yield fp
    fp.close()


@pytest.fixture
def tmp_entry(tmp_file):
    return arf.create_entry(tmp_file, "entry", tstamp)


@pytest.fixture
def tmp_dataset(tmp_entry):
    return arf.create_dataset(tmp_entry, **datasets[2])


@pytest.fixture
def read_only_file(tmp_path):
    path = tmp_path / "test"
    fp = arf.open_file(path, "w")
    entry = arf.create_entry(fp, "entry", tstamp)
    for dset in datasets:
        _ = arf.create_dataset(entry, **dset)
    fp.close()
    return arf.open_file(path, "r")


def test_created_datasets(read_only_file):
    tmp_entry = read_only_file["/entry"]
    assert len(tmp_entry) == len(datasets)
    assert set(tmp_entry.keys()) == set(dset["name"] for dset in datasets)
    # this will fail if iteration is not in order of creation
    for dset, d in zip(datasets, tmp_entry.values()):
        assert d.shape == dset["data"].shape
        assert not arf.is_entry(d)


def test_child_type_counts(read_only_file):
    assert arf.count_children(read_only_file) == 1
    assert arf.count_children(read_only_file, h5.Group) == 1
    assert arf.count_children(read_only_file, h5.Dataset) == 0
    entry = read_only_file["/entry"]
    assert arf.count_children(entry) == len(datasets)
    assert arf.count_children(entry, h5.Group) == 0
    assert arf.count_children(entry, h5.Dataset) == len(datasets)


def test_channel_counts(read_only_file):
    dset1 = read_only_file["/entry/acoustic"]
    assert arf.count_channels(dset1) == 1
    dset2 = read_only_file["/entry/multichannel"]
    assert arf.count_channels(dset2) == 2


def test_create_entries(tmp_file):
    names = [str(uuid4()).split("-")[0] for _ in range(5)]
    for name in names:
        g = arf.create_entry(tmp_file, name, tstamp, **entry_attributes)
        assert name in tmp_file
        assert arf.is_entry(g)
        assert arf.timestamp_to_float(g.attrs["timestamp"]) > 0
        for k in entry_attributes:
            assert k in g.attrs
    assert len(tmp_file) == len(names)
    # this will fail if creation order is not tracked
    assert list(tmp_file.keys()) == names


def test_create_existing_entry(tmp_file, tmp_entry):
    with pytest.raises(ValueError):
        arf.create_entry(tmp_file, "entry", tstamp, **entry_attributes)


def test_create_bad_dataset(tmp_entry):
    for dset in bad_datasets:
        with pytest.raises(ValueError):
            _ = arf.create_dataset(tmp_entry, **dset)


def test_set_attributes(tmp_entry):
    """tests the set_attributes convenience function"""
    arf.set_attributes(tmp_entry, mystr="myvalue", myint=5000)
    assert tmp_entry.attrs["myint"] == 5000
    assert tmp_entry.attrs["mystr"] == "myvalue"
    arf.set_attributes(tmp_entry, mystr="blah blah", overwrite=False)
    assert tmp_entry.attrs["mystr"] == "myvalue"
    arf.set_attributes(tmp_entry, mystr=None)
    assert "mystr" not in tmp_entry.attrs


def test_set_null_uuid(tmp_entry):
    # nulls in a uuid can make various things barf
    uuid = UUID(bytes=b"".rjust(16, b"\0"))
    arf.set_uuid(tmp_entry, uuid)
    assert arf.get_uuid(tmp_entry) == uuid


def test_get_null_uuid(tmp_entry):
    uuid = UUID(bytes=b"".rjust(16, b"\0"))
    del tmp_entry.attrs["uuid"]
    assert arf.get_uuid(tmp_entry) == uuid


def test_set_uuid_with_bytes(tmp_entry):
    uuid = uuid4()
    arf.set_uuid(tmp_entry, uuid.bytes)
    assert arf.get_uuid(tmp_entry) == uuid


def test_copy_entry_with_attrs(tmp_file, tmp_entry):
    src_entry_attrs = dict(tmp_entry.attrs)
    src_entry_timestamp = src_entry_attrs.pop("timestamp")
    tgt_entry = arf.create_entry(
        tmp_file, "copied_entry", src_entry_timestamp, **src_entry_attrs
    )
    assert tmp_entry.attrs["uuid"] == tgt_entry.attrs["uuid"]


def test_check_file_version(tmp_file):
    arf.check_file_version(tmp_file)


def test_append_to_table(tmp_file):
    dtype = np.dtype({"names": ("f1", "f2"), "formats": [np.uint, np.int32]})
    dset = arf.create_table(tmp_file, "test", dtype=dtype)
    assert dset.shape[0] == 0
    arf.append_data(dset, (5, 10))
    assert dset.shape[0] == 1


def test_append_nothing(tmp_file):
    data = np.random.randn(100)
    dset = arf.create_dataset(tmp_file, "test", data=data, sampling_rate=1)
    arf.append_data(dset, np.random.randn(0))
    assert dset.shape == data.shape


def test_creation_iter(tmp_file):
    # self.fp = arf.open_file("test06", mode="a", driver="core", backing_store=False)
    entry_names = ["z", "y", "a", "q", "zzyfij"]
    for name in entry_names:
        g = arf.create_entry(tmp_file, name, 0)
        arf.create_dataset(g, "dset", (1,), sampling_rate=1)
    assert list(arf.keys_by_creation(tmp_file)) == entry_names


def test_select_from_timeseries(tmp_file):
    entry = arf.create_entry(tmp_file, "entry", tstamp)
    for data in datasets:
        arf.create_dataset(entry, **data)
        dset = entry[data["name"]]
        if data.get("units", None) == "samples":
            selected, offset = arf.select_interval(dset, 0, data["sampling_rate"])
        else:
            selected, offset = arf.select_interval(dset, 0.0, 1.0)
        if arf.is_time_series(dset):
            np.testing.assert_array_equal(
                selected, data["data"][: data["sampling_rate"]]
            )


def test_timestamp_conversion():
    from datetime import datetime

    dt = datetime.now()
    ts = arf.convert_timestamp(dt)
    assert arf.timestamp_to_datetime(ts) == dt
    assert all(arf.convert_timestamp(ts) == ts)
    # lose the sub-second resolution
    assert arf.convert_timestamp(dt.timetuple())[0] == ts[0]
    ts = arf.convert_timestamp(1000)
    assert int(arf.timestamp_to_float(ts)) == 1000
    with pytest.raises(TypeError):
        arf.convert_timestamp("blah blah")


def test_units_stored_as_bytes(tmp_entry):
    """Fixed-length string attributes come back from h5py as bytes.

    That is what the C++ library writes, so classification has to cope with it.
    A discrete-timebase point process carries units of "samples" and a
    sampling_rate, which is the combination easiest to misread as sampled
    data.
    """
    dset = arf.create_dataset(
        tmp_entry,
        "spikes",
        randint(0, 1000, 100),
        units="samples",
        sampling_rate=1000,
        datatype=arf.DataTypes.SPIKET,
    )
    dset.attrs.create("units", b"samples", dtype="|S7")
    assert isinstance(dset.attrs["units"], bytes)
    assert not arf.is_time_series(dset)

    sampled = arf.create_dataset(
        tmp_entry,
        "pcm",
        randn(100),
        units="mV",
        sampling_rate=1000,
        datatype=arf.DataTypes.ACOUSTIC,
    )
    sampled.attrs.create("units", b"mV", dtype="|S3")
    assert arf.is_time_series(sampled)


def test_string_data_is_rejected_with_a_useful_error(tmp_entry):
    """Validation has to reach input that arrives already converted.

    A numpy string array has a dtype, so a check guarded on hasattr(data,
    "dtype") never sees it and the failure surfaces inside h5py instead, as
    "No conversion path for dtype".
    """
    with pytest.raises(ValueError, match="numeric or compound"):
        arf.create_dataset(tmp_entry, "list", ["a", "b"], units="s")
    with pytest.raises(ValueError, match="numeric or compound"):
        arf.create_dataset(tmp_entry, "numpy", np.array(["a", "b"]), units="s")
    with pytest.raises(ValueError, match="numeric or compound"):
        arf.create_dataset(tmp_entry, "object", np.array([object()]), units="s")


def test_compound_units_accepts_any_sequence(tmp_entry):
    """h5py returns the units attribute as an ndarray.

    Requiring a list or tuple meant a compound dataset read from one file could
    not be written to another without converting the attribute by hand -- which
    covers both writers, since this library's units come back as an object
    array and the C++ library's as a fixed-width bytes array.
    """
    dtype = np.dtype([("start", "f8"), ("peak", "f8")])
    data = np.zeros(4, dtype=dtype)
    units = ("s", "mV")

    for name, value in (
        ("tuple", units),
        ("list", list(units)),
        # what h5py hands back for this library's writes
        ("ndarray-object", np.array(units, dtype=object)),
        # and for the C++ library's, which stores fixed-width strings
        ("ndarray-bytes", np.array([u.encode() for u in units], dtype="|S4")),
        # h5py refuses this dtype, so create_dataset converts it
        ("ndarray-unicode", np.array(units)),
    ):
        dset = arf.create_dataset(tmp_entry, name, data, units=value)
        assert len(dset.attrs["units"]) == 2
        assert [arf._decode_attribute(u) for u in dset.attrs["units"]] == list(units)

    # the round trip that motivated it: read the attribute back off a dataset
    # and hand it straight to create_dataset, with nothing in between
    for source_name in ("tuple", "ndarray-bytes"):
        source = tmp_entry[source_name]
        copy = arf.create_dataset(
            tmp_entry, f"copy-of-{source_name}", source[:], units=source.attrs["units"]
        )
        assert [arf._decode_attribute(u) for u in copy.attrs["units"]] == list(units)

    # a bare string is still a scalar, not a sequence of one
    with pytest.raises(ValueError, match="requires sequence of units"):
        arf.create_dataset(tmp_entry, "scalar", data, units="s")
    with pytest.raises(ValueError, match="number of units"):
        arf.create_dataset(tmp_entry, "short", data, units=np.array(["s"]))

    # and the converse: a sequence on data that is not compound. This is
    # non-conforming, and the comparisons downstream would otherwise raise
    # numpy's "truth value of an array is ambiguous" instead.
    with pytest.raises(ValueError, match="only complex event data"):
        arf.create_dataset(
            tmp_entry,
            "sampled",
            np.zeros(4, dtype="f8"),
            units=np.array(["mV"], dtype=object),
            sampling_rate=1000,
        )


def test_select_interval_on_an_empty_dataset(tmp_entry):
    """The guard tested idx.size, which is the length of the mask.

    For an empty dataset that returned the mask itself -- a bool array where
    the caller expects the dataset's dtype.
    """
    dset = arf.create_dataset(
        tmp_entry,
        "empty",
        np.array([], dtype="f8"),
        units="s",
        datatype=arf.DataTypes.SPIKET,
        maxshape=(None,),
    )
    data, offset = arf.select_interval(dset, 0.0, 1.0)
    assert data.dtype == np.dtype("f8")
    assert len(data) == 0
    assert offset == 0


def test_select_interval_respects_units_not_sampling_rate(tmp_entry):
    """A real-valued point process may legitimately carry a sampling_rate.

    Rescaling whenever the attribute is present turns a one-second window over
    times in seconds into the first thousand seconds.

    Another situation to handle is when a non-conforming writer (older versions
    of jrecord) provide a scalar "units" attr for a compound datatype.

    """
    times = np.array([0.5, 5.0, 900.0])
    seconds = arf.create_dataset(
        tmp_entry,
        "in_seconds",
        times,
        units="s",
        sampling_rate=1000,
        datatype=arf.DataTypes.SPIKET,
    )
    data, offset = arf.select_interval(seconds, 0.0, 1.0)
    assert list(data) == [0.5]
    assert offset == 0

    # times in samples are still rescaled, because they have to be
    samples = arf.create_dataset(
        tmp_entry,
        "in_samples",
        np.array([500, 5000, 900000]),
        units="samples",
        sampling_rate=1000,
        datatype=arf.DataTypes.SPIKET,
    )
    data, offset = arf.select_interval(samples, 0.0, 1.0)
    assert list(data) == [500]
    assert offset == 0

    compound = arf.create_dataset(
        tmp_entry,
        "compound",
        np.rec.fromrecords(
            [(200, 1, b"stimulus"), (168505, 0, b"stimulus")],
            names=("start", "status", "message"),
        ),
        sampling_rate=1000,
        datatype=arf.DataTypes.EVENT,
        units=(b"samples", b"", b"")
    )
    data, offset = arf.select_interval(compound, 0.1, 1.0)
    assert data.tolist() == [(100, 1, b"stimulus")]
    assert offset == 100
    # now make the units non-conforming
    compound.attrs["units"] = b"samples"
    data, offset = arf.select_interval(compound, 0.1, 1.0)
    assert data.tolist() == [(100, 1, b"stimulus")]
    assert offset == 100



def test_select_interval_on_a_time_series(tmp_entry):
    dset = arf.create_dataset(
        tmp_entry, "pcm", np.arange(2000, dtype="f8"), units="mV", sampling_rate=1000
    )
    data, offset = arf.select_interval(dset, 0.5, 1.0)
    assert offset == 500
    assert len(data) == 500
    assert data[0] == 500.0


def test_timestamp_honors_timezone():
    """mktime(timetuple()) read the wall-clock fields as local time.

    An aware datetime was therefore recorded as the wrong instant, off by the
    local zone offset.
    """
    from datetime import datetime, timedelta, timezone

    aware = datetime(2020, 1, 1, 12, 0, 30, 123456, tzinfo=timezone.utc)
    ts = arf.convert_timestamp(aware)
    assert ts[0] == int(aware.replace(microsecond=0).timestamp())
    assert ts[1] == 123456
    assert arf.timestamp_to_float(ts) == pytest.approx(aware.timestamp())

    # two aware datetimes naming the same instant convert identically
    other = aware.astimezone(timezone(timedelta(hours=5)))
    assert list(arf.convert_timestamp(other)) == list(ts)

    # a naive datetime is still read as local time, as it always was
    naive = datetime(2020, 1, 1, 12, 0, 30)
    assert arf.convert_timestamp(naive)[0] == int(naive.timestamp())


def test_timestamp_before_the_epoch():
    """The microseconds are the remainder after the second, so never negative."""
    ts = arf.convert_timestamp(-1.5)
    assert ts[1] >= 0
    assert arf.timestamp_to_float(ts) == pytest.approx(-1.5)

    ts = arf.convert_timestamp(1234567890.125)
    assert list(ts) == [1234567890, 125000]


def test_check_file_version_reports_bad_versions(tmp_file):
    """An unparseable version is reported, not raised out of packaging."""
    tmp_file.attrs["arf_version"] = "not-a-version"
    with pytest.raises(UserWarning, match="Unparseable"):
        arf.check_file_version(tmp_file)


def test_is_entry_excludes_the_root(tmp_file):
    """h5py's File is a Group, so testing only for Group counted the file.

    The specification is explicit that a file may hold top-level datasets
    belonging to no entry, so the root is not one.
    """
    assert not arf.is_entry(tmp_file)
    # the same group reached by path rather than as the File object
    assert not arf.is_entry(tmp_file["/"])

    entry = arf.create_entry(tmp_file, "entry", tstamp)
    assert arf.is_entry(entry)

    # datasets are not entries, at the root or inside one
    dset = arf.create_dataset(entry, "pcm", randn(100), units="mV", sampling_rate=1000)
    assert not arf.is_entry(dset)
    top = arf.create_table(tmp_file, "log", dtype=[("message", "S32")])
    assert not arf.is_entry(top)

    # the spec permits an entry to contain another; both are entries
    nested = arf.create_entry(entry, "nested", tstamp)
    assert arf.is_entry(nested)


def test_walking_a_file_finds_only_entries(tmp_file):
    """The shape of the bug: every walk of a file included the file."""
    arf.create_entry(tmp_file, "entry_000", tstamp)
    arf.create_entry(tmp_file, "entry_001", tstamp)
    arf.create_table(tmp_file, "log", dtype=[("message", "S32")])

    entries = [name for name in tmp_file if arf.is_entry(tmp_file[name])]
    assert entries == ["entry_000", "entry_001"]
    assert not arf.is_entry(tmp_file)


def test_library_version_no_longer_stands_in_for_the_spec_version(tmp_path):
    """The two versions parted ways at 3.0.

    Very old files recorded only arf_library_version, back when the libraries
    tracked the spec's major version so one could substitute for the other.
    A 3.x library version says nothing about which spec the file follows.
    """
    path = tmp_path / "legacy.arf"

    # a genuinely old file, where the substitution still holds
    with h5.File(path, "w") as fp:
        fp.attrs["arf_library_version"] = "2.2.0"
    with h5.File(path) as fp:
        assert str(arf.check_file_version(fp)) == "2.2.0"

    # a library version from after the split cannot stand in
    with h5.File(path, "w") as fp:
        fp.attrs["arf_library_version"] = "3.0.0"
    with h5.File(path) as fp:
        with pytest.raises(UserWarning, match="cannot stand in"):
            arf.check_file_version(fp)

    # and an explicit arf_version is always preferred over the fallback
    with h5.File(path, "w") as fp:
        fp.attrs["arf_version"] = "2.1"
        fp.attrs["arf_library_version"] = "3.0.0"
    with h5.File(path) as fp:
        assert str(arf.check_file_version(fp)) == "2.1"


def _file_declaring(path, version):
    """A bare hdf5 file claiming a specification version."""
    with h5.File(path, "w") as fp:
        if version is not None:
            fp.attrs["arf_version"] = version
    return path


@pytest.mark.parametrize("declared", ["2.0", "2.1", "2.2"])
def test_every_version_in_range_is_accepted(tmp_path, declared):
    path = _file_declaring(tmp_path / "v.arf", declared)
    with h5.File(path) as fp:
        assert str(arf.check_file_version(fp)) == declared


def test_a_later_minor_revision_needs_no_release(tmp_path):
    """The point of deriving the upper bound rather than fixing it.

    2.9 does not exist, and by the spec's own rule a minor revision cannot
    change or remove a required attribute, so this library reads it.
    """
    path = _file_declaring(tmp_path / "v.arf", "2.9")
    with h5.File(path) as fp:
        assert str(arf.check_file_version(fp)) == "2.9"


def test_a_later_major_revision_is_refused(tmp_path):
    path = _file_declaring(tmp_path / "v.arf", "3.0")
    with h5.File(path) as fp:
        with pytest.raises(FutureWarning, match="postdates"):
            arf.check_file_version(fp)


def test_files_predating_the_minimum_point_at_the_upgrade_path(tmp_path):
    """Pre-2.0 support was dropped; the message has to say what to do."""
    path = _file_declaring(tmp_path / "v.arf", "1.1")
    with h5.File(path) as fp:
        with pytest.raises(DeprecationWarning, match="arfx"):
            arf.check_file_version(fp)


def test_supported_range_is_derived_from_the_implemented_version():
    from packaging.version import Version

    low, high = arf.supported_spec_versions()
    assert Version(low) <= Version(arf.spec_version) < Version(high)
    # the next major after what is implemented, not a literal
    assert Version(high).major == Version(arf.spec_version).major + 1
    assert Version(high).minor == 0
    assert arf.spec_version in arf.version_info()


def test_unknown_attributes_are_ignored(tmp_file):
    """What minor-version tolerance actually rests on.

    A later minor revision may add optional attributes. Reading a file that
    carries them must not fail, or the tolerance above is worthless.
    """
    entry = arf.create_entry(tmp_file, "entry", tstamp)
    entry.attrs["something_from_the_future"] = "hello"
    dset = arf.create_dataset(
        entry, "pcm", randn(100), units="mV", sampling_rate=1000
    )
    dset.attrs["also_unknown"] = 42

    assert arf.is_entry(entry)
    assert arf.is_time_series(dset)
    assert str(arf.check_file_version(tmp_file)) == arf.spec_version
    assert list(arf.keys_by_creation(tmp_file)) == ["entry"]


def test_entry_names_reject_path_separators(tmp_file):
    """"a/b" made a group nested inside "a", which is not a top-level entry."""
    with pytest.raises(ValueError, match="path separator"):
        arf.create_entry(tmp_file, "a/b", tstamp)
    assert list(tmp_file.keys()) == []

    # an ordinary name still works
    arf.create_entry(tmp_file, "a_b", tstamp)
    assert list(tmp_file.keys()) == ["a_b"]


def test_check_file_structure_passes_a_conforming_file(tmp_file):
    for i in range(2):
        entry = arf.create_entry(tmp_file, "entry_%03d" % i, tstamp)
        arf.create_dataset(entry, "pcm", randn(32), units="mV", sampling_rate=1000)
    arf.create_table(tmp_file, "log", dtype=[("message", "S32")])
    assert arf.check_file_structure(tmp_file) == []


def test_check_file_structure_finds_a_shared_dataset(tmp_file):
    """A dataset in two entries has no defined time, which the spec forbids."""
    first = arf.create_entry(tmp_file, "entry_000", tstamp)
    second = arf.create_entry(tmp_file, "entry_001", tstamp)
    dset = arf.create_dataset(first, "pcm", randn(32), units="mV", sampling_rate=1000)
    second["pcm"] = dset  # a second hard link, which arf cannot prevent

    problems = arf.check_file_structure(tmp_file)
    assert len(problems) == 2
    assert all("linked into more than one entry" in p for p in problems)
    assert any("entry_000/pcm" in p for p in problems)
    assert any("entry_001/pcm" in p for p in problems)


def test_check_file_structure_finds_a_multiply_linked_entry(tmp_file):
    entry = arf.create_entry(tmp_file, "entry_000", tstamp)
    arf.create_dataset(entry, "pcm", randn(32), units="mV", sampling_rate=1000)
    tmp_file["alias"] = entry

    problems = arf.check_file_structure(tmp_file)
    assert any("linked into the file more than once" in p for p in problems)


def test_check_file_structure_ignores_nested_entries(tmp_file):
    """The spec says the contents of a nested entry are outside the hierarchy."""
    outer = arf.create_entry(tmp_file, "outer", tstamp)
    inner = arf.create_entry(outer, "inner", tstamp)
    shared = arf.create_dataset(inner, "pcm", randn(32), units="mV", sampling_rate=1000)
    # aliasing inside a nested entry is not this function's business
    inner["alias"] = shared
    assert arf.check_file_structure(tmp_file) == []


def test_select_interval_with_scalar_units_on_compound_data(tmp_entry):
    """Compound datasets in the wild carry one unit for the whole record.

    The spec asks for one per field, and arfx documents jrecord as a producer
    of the scalar form. Indexing into it would read its first character, so
    "samples" becomes "s" and the window silently stops being rescaled.
    """
    records = np.rec.fromrecords(
        [(500, 1), (5000, 0), (900000, 1)], names=("start", "state")
    )
    dset = tmp_entry.create_dataset("evt", data=records, maxshape=(None,))
    dset.attrs["units"] = b"samples"
    dset.attrs["sampling_rate"] = 1000

    selected, offset = arf.select_interval(dset, 0.4, 0.6)
    assert offset == 400
    assert list(selected["start"]) == [100]


def test_select_interval_offsets_an_integer_start_field(tmp_entry):
    """The offset is subtracted in place, so it has to match the field's type.

    A window that is not rescaled leaves begin as the float the caller passed,
    and numpy will not subtract that from an integer field in place.
    """
    records = np.rec.fromrecords([(1, 0), (5, 1)], names=("start", "state"))
    dset = arf.create_dataset(
        tmp_entry, "evt", records, units=("s", ""), maxshape=(None,)
    )
    selected, offset = arf.select_interval(dset, 0.0, 2.0)
    assert list(selected["start"]) == [1]
    assert selected["start"].dtype == records["start"].dtype


def test_file_version_reports_without_judging(tmp_path):
    """A migration tool has to read the version *because* it is out of range."""
    path = _file_declaring(tmp_path / "old.arf", "1.1")
    with h5.File(path) as fp:
        assert str(arf.file_version(fp)) == "1.1"
        with pytest.raises(DeprecationWarning):
            arf.check_file_version(fp)

    path = _file_declaring(tmp_path / "new.arf", "3.0")
    with h5.File(path) as fp:
        assert str(arf.file_version(fp)) == "3.0"
        with pytest.raises(FutureWarning):
            arf.check_file_version(fp)


def test_file_version_still_refuses_what_it_cannot_read(tmp_path):
    path = _file_declaring(tmp_path / "none.arf", None)
    with h5.File(path) as fp:
        with pytest.raises(UserWarning):
            arf.file_version(fp)

    path = _file_declaring(tmp_path / "bad.arf", "not-a-version")
    with h5.File(path) as fp:
        with pytest.raises(UserWarning, match="Unparseable"):
            arf.file_version(fp)
