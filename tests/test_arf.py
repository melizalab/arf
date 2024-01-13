# -*- mode: python -*-

import time

import numpy as nx
import pytest
from h5py.version import version as h5py_version
from numpy.random import randint, randn
from packaging import version

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
        name="spikes",
        data=randint(0, 100000, 100),
        datatype=arf.DataTypes.SPIKET,
        units="samples",
        sampling_rate=20000,  # required
    ),
    dict(
        name="empty-spikes",
        data=nx.array([], dtype="f"),
        datatype=arf.DataTypes.SPIKET,
        method="broken",
        maxshape=(None,),
        units="s",
    ),
    dict(
        name="events",
        data=nx.rec.fromrecords(
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
        data=nx.rec.fromrecords([(1.0, 1), (2.0, 2)], names=("time", "state")),
        units="s",
    ),
    dict(
        name="missing units for complex dtype",
        data=nx.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),
    ),
    dict(
        name="wrong length units for complex dtype",
        data=nx.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),
        units=("seconds",),
    ),
]


@pytest.fixture
def test_file(tmp_path):
    path = tmp_path / "test"
    fp = arf.open_file(path, "w", driver="core", backing_store=False)
    yield fp
    fp.close()


@pytest.fixture
def test_entry(test_file):
    return arf.create_entry(test_file, "entry", tstamp)


@pytest.fixture
def test_dataset(test_entry):
    return arf.create_dataset(test_entry, **datasets[2])


def test00_create_entries(test_file):
    N = 5
    for i in range(N):
        name = entry_base % i
        g = arf.create_entry(test_file, name, tstamp, **entry_attributes)
        assert name in test_file
        assert arf.is_entry(g)
        assert arf.timestamp_to_float(g.attrs["timestamp"]) > 0
        for k in entry_attributes:
            assert k in g.attrs
    assert len(test_file) == N


def test01_create_existing_entry(test_file, test_entry):
    with pytest.raises(ValueError):
        arf.create_entry(test_file, "entry", tstamp, **entry_attributes)


def test02_create_datasets(test_entry):
    for dset in datasets:
        d = arf.create_dataset(test_entry, **dset)
        assert d.shape == dset["data"].shape
        assert not arf.is_entry(d)
    assert len(test_entry) == len(datasets)
    assert set(test_entry.keys()) == set(dset["name"] for dset in datasets)


def test04_create_bad_dataset(test_entry):
    for dset in bad_datasets:
        with pytest.raises(ValueError):
            _ = arf.create_dataset(test_entry, **dset)


def test05_set_attributes(test_entry):
    """tests the set_attributes convenience function"""
    arf.set_attributes(test_entry, mystr="myvalue", myint=5000)
    assert test_entry.attrs["myint"] == 5000
    assert test_entry.attrs["mystr"] == "myvalue"
    arf.set_attributes(test_entry, mystr=None)
    assert "mystr" not in test_entry.attrs


def test06_null_uuid(test_entry):
    # nulls in a uuid can make various things barf
    from uuid import UUID

    uuid = UUID(bytes=b"".rjust(16, b"\0"))
    arf.set_uuid(test_entry, uuid)
    assert arf.get_uuid(test_entry) == uuid


def test07_copy_entry_with_attrs(test_file, test_entry):
    src_entry_attrs = dict(test_entry.attrs)
    src_entry_timestamp = src_entry_attrs.pop("timestamp")
    tgt_entry = arf.create_entry(
        test_file, "copied_entry", src_entry_timestamp, **src_entry_attrs
    )
    assert test_entry.attrs["uuid"] == tgt_entry.attrs["uuid"]


def test08_check_file_version(test_file):
    arf.check_file_version(test_file)


def test09_append_to_table(test_file):
    dtype = nx.dtype({"names": ("f1", "f2"), "formats": [nx.uint, nx.int32]})
    dset = arf.create_table(test_file, "test", dtype=dtype)
    assert dset.shape[0] == 0
    arf.append_data(dset, (5, 10))
    assert dset.shape[0] == 1


@pytest.mark.skipif(
    version.Version(h5py_version) < version.Version("2.2"),
    reason="not supported on h5py < 2.2",
)
def test01_creation_iter(test_file):
    # self.fp = arf.open_file("test06", mode="a", driver="core", backing_store=False)
    entry_names = ["z", "y", "a", "q", "zzyfij"]
    for name in entry_names:
        g = arf.create_entry(test_file, name, 0)
        arf.create_dataset(g, "dset", (1,), sampling_rate=1)
    assert list(arf.keys_by_creation(test_file)) == entry_names


@pytest.mark.skipif(
    version.Version(h5py_version) < version.Version("2.2"),
    reason="not supported on h5py < 2.2",
)
def test10_select_from_timeseries(test_file):
    entry = arf.create_entry(test_file, "entry", tstamp)
    for data in datasets:
        arf.create_dataset(entry, **data)
        dset = entry[data["name"]]
        if data.get("units", None) == "samples":
            selected, offset = arf.select_interval(dset, 0, data["sampling_rate"])
        else:
            selected, offset = arf.select_interval(dset, 0.0, 1.0)
        if arf.is_time_series(dset):
            nx.testing.assert_array_equal(
                selected, data["data"][: data["sampling_rate"]]
            )


def test01_timestamp_conversion():
    from datetime import datetime

    dt = datetime.now()
    ts = arf.convert_timestamp(dt)
    assert arf.timestamp_to_datetime(ts) == dt
    assert all(arf.convert_timestamp(ts) == ts)
    ts = arf.convert_timestamp(1000)
    assert int(arf.timestamp_to_float(ts)) == 1000


def test99_various():
    # test some functions difficult to cover otherwise
    arf.DataTypes._doc()
    arf.DataTypes._todict()


# # Variables:
# # End:
