# -*- coding: utf-8 -*-
# -*- mode: python -*-

# test harness for arf interface. assumes the underlying hdf5 and h5py libraries
# are working.
from __future__ import division
from __future__ import unicode_literals

from nose.tools import *
from nose.plugins.skip import SkipTest
from distutils import version
from h5py.version import version as h5py_version, hdf5_version

import numpy as nx
import arf
import time
from numpy.random import randn, randint

fp = arf.open_file("test", 'w', driver="core", backing_store=False)
entry_base = "entry_%03d"
tstamp = time.mktime(time.localtime())
entry_attributes = {'intattr': 1,
                    'vecattr': [1, 2, 3],
                    'arrattr': randn(5),
                    'strattr': "an attribute",
                    }
datasets = [dict(name="acoustic",
                 data=randn(100000),
                 sampling_rate=20000,
                 datatype=arf.DataTypes.ACOUSTIC,
                 maxshape=(None,),
                 microphone="DK-1234",
                 compression=0),
            dict(name="neural",
                 data=(randn(100000) * 2 ** 16).astype('h'),
                 sampling_rate=20000,
                 datatype=arf.DataTypes.EXTRAC_HP,
                 compression=9),
            dict(name="spikes",
                 data=randint(0, 100000, 100),
                 datatype=arf.DataTypes.SPIKET,
                 units="samples",
                 sampling_rate=20000,  # required
                 ),
            dict(name="empty-spikes",
                 data=nx.array([], dtype='f'),
                 datatype=arf.DataTypes.SPIKET,
                 method="broken",
                 maxshape=(None,),
                 units="s",
                 ),
            dict(name="events",
                 data=nx.rec.fromrecords(
                 [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
                 names=("start", "state", "name")),  # 'start' required
                 datatype=arf.DataTypes.EVENT,
                 units=(b"s",b"",b"")) # only bytes supported by h5py
            ]

bad_datasets = [dict(name="string datatype",
                     data="a string"),
                dict(name="object datatype",
                     data=bytes),
                dict(name="missing samplerate/units",
                     data=randn(1000)),
                dict(name="missing samplerate for units=samples",
                     data=randn(1000),
                     units="samples"),
                dict(name="missing start field",
                     data=nx.rec.fromrecords([(1.0, 1), (2.0, 2)],
                                             names=("time", "state")),
                     units="s"),
                dict(name="missing units for complex dtype",
                     data=nx.rec.fromrecords(
                     [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
                     names=(
                     "start", "state", "name"))),
                dict(name="wrong length units for complex dtype",
                     data=nx.rec.fromrecords(
                     [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
                     names=(
                     "start", "state", "name")),
                     units=("seconds",)),
                ]


def create_entry(name):
    g = arf.create_entry(fp, name, tstamp, **entry_attributes)
    assert_true(name in fp)
    assert_true(arf.timestamp_to_float(g.attrs['timestamp']) > 0)
    for k in entry_attributes:
        assert_true(k in g.attrs)


def create_dataset(g, dset):
    d = arf.create_dataset(g, **dset)
    assert_equal(d.shape, dset['data'].shape)


def test00_create_entries():
    N = 5
    for i in range(N):
        yield create_entry, entry_base % i
    assert_equal(len(fp), N)


@raises(ValueError)
def test01_create_existing_entry():
    arf.create_entry(fp, entry_base % 0, tstamp, **entry_attributes)


def test02_create_datasets():
    for name in arf.keys_by_creation(fp):
        entry = fp[name]
        for dset in datasets:
            yield create_dataset, entry, dset
        assert_equal(len(entry), len(datasets))
        assert_equal(set(entry.keys()), set(dset['name'] for dset in datasets))


def test03_set_attributes():
    # tests the set_attributes convenience function
    arf.set_attributes(fp["entry_001/spikes"], mystr="myvalue", myint=5000)
    assert_equal(fp["entry_001/spikes"].attrs['myint'], 5000)
    assert_equal(fp["entry_001/spikes"].attrs['mystr'], "myvalue")
    arf.set_attributes(fp["entry_001/spikes"], mystr=None)
    assert_false("mystr" in fp["entry_001/spikes"].attrs)


def test04_create_bad_dataset():
    f = raises(ValueError)(create_dataset)
    e = fp['entry_001']
    for dset in bad_datasets:
        yield f, e, dset


def test05_null_uuid():
    # nulls in a uuid can make various things barf
    from uuid import UUID
    uuid = UUID(bytes=b''.rjust(16, b'\0'))
    e = fp['entry_001']
    arf.set_uuid(e, uuid)

    assert_equal(arf.get_uuid(e), uuid)


def test06_creation_iter():
    fp = arf.open_file("test06", mode="a", driver="core", backing_store=False)
    entry_names = ['z', 'y', 'a', 'q', 'zzyfij']
    for name in entry_names:
        g = arf.create_entry(fp, name, 0)
        arf.create_dataset(g, "dset", (1,), sampling_rate=1)
    assert_equal(list(arf.keys_by_creation(fp)), entry_names)

if version.StrictVersion(h5py_version) < version.StrictVersion("2.2"):
    test06_creation_iter = SkipTest(test06_creation_iter)


def test07_append_to_table():
    fp = arf.open_file("test07", mode="a", driver="core", backing_store=False)
    dtype = nx.dtype({'names': ("f1","f2"), 'formats': [nx.uint, nx.int32]})
    dset = arf.create_table(fp, 'test', dtype=dtype)
    assert_equal(dset.shape[0], 0)
    arf.append_data(dset, (5, 10))
    assert_equal(dset.shape[0], 1)


def test08_check_file_version():
    fp = arf.open_file("test08", mode="a", driver="core", backing_store=False)
    arf.check_file_version(fp)


def test09_timestamp_conversion():
    from datetime import datetime

    dt = datetime.now()
    ts = arf.convert_timestamp(dt)
    assert_equal(arf.timestamp_to_datetime(ts), dt)
    assert_true(all(arf.convert_timestamp(ts) == ts))

    ts = arf.convert_timestamp(1000)
    assert_equal(int(arf.timestamp_to_float(ts)), 1000)


def test10_select_from_timeseries():
    entry = fp[entry_base % 0]
    for data in datasets:
        dset = entry[data["name"]]
        if data.get("units", None) == "samples":
            selected = arf.select_interval(dset, 0, data["sampling_rate"])
        else:
            selected = arf.select_interval(dset, 0.0, 1.0)
        if arf.is_time_series(dset):
            nx.testing.assert_array_equal(selected, data["data"][:data["sampling_rate"]])


def test99_various():
    # test some functions difficult to cover otherwise
    arf.DataTypes._doc()
    arf.DataTypes._todict()


# Variables:
# End:
