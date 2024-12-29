# -*- mode: python -*-

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


def test_datatype_doc():
    arf.DataTypes._doc()


# # Variables:
# # End:
