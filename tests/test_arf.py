# -*- coding: utf-8 -*-
# -*- mode: python -*-

import unittest
from distutils import version
from h5py.version import version as h5py_version

import numpy as nx
import arf
import time
from numpy.random import randn, randint

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
                         names=("start", "state", "name"))),
                dict(name="wrong length units for complex dtype",
                     data=nx.rec.fromrecords(
                         [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
                         names=("start", "state", "name")),
                     units=("seconds",)),
                ]


class TestArfCreation(unittest.TestCase):
    def setUp(self):
        self.fp = arf.open_file("test", 'w', driver="core", backing_store=False)
        self.entry = arf.create_entry(self.fp, "entry", tstamp)
        self.dataset = arf.create_dataset(self.entry, **datasets[2])

    def tearDown(self):
        self.fp.close()

    def create_entry(self, name):
        g = arf.create_entry(self.fp, name, tstamp, **entry_attributes)
        self.assertTrue(name in self.fp)
        self.assertTrue(arf.is_entry(g))
        self.assertTrue(arf.timestamp_to_float(g.attrs['timestamp']) > 0)
        for k in entry_attributes:
            self.assertTrue(k in g.attrs)

    def create_dataset(self, g, dset):
        d = arf.create_dataset(g, **dset)
        self.assertEqual(d.shape, dset['data'].shape)
        self.assertFalse(arf.is_entry(d))

    def test00_create_entries(self):
        N = 5
        for i in range(N):
            yield self.create_entry, entry_base % i
        self.assertEqual(len(self.fp), N)

    def test01_create_existing_entry(self):
        with self.assertRaises(ValueError):
            arf.create_entry(self.fp, "entry", tstamp, **entry_attributes)

    def test02_create_datasets(self):
        for dset in datasets:
            yield self.create_dataset, self.entry, dset
        self.assertEqual(len(self.entry), len(datasets))
        self.assertEqual(set(self.entry.keys()), set(dset['name'] for dset in datasets))

    def test04_create_bad_dataset(self):
        for dset in bad_datasets:
            with self.assertRaises(ValueError):
                self.create_dataset(self.entry, dset)

    def test05_set_attributes(self):
        """ tests the set_attributes convenience function """
        arf.set_attributes(self.entry, mystr="myvalue", myint=5000)
        self.assertEqual(self.entry.attrs['myint'], 5000)
        self.assertEqual(self.entry.attrs['mystr'], "myvalue")
        arf.set_attributes(self.entry, mystr=None)
        self.assertFalse("mystr" in self.entry.attrs)

    def test06_null_uuid(self):
        # nulls in a uuid can make various things barf
        from uuid import UUID
        uuid = UUID(bytes=b''.rjust(16, b'\0'))
        arf.set_uuid(self.entry, uuid)
        self.assertEqual(arf.get_uuid(self.entry), uuid)

    def test07_copy_entry_with_attrs(self):
        src_entry_attrs = dict(self.entry.attrs)
        src_entry_timestamp = src_entry_attrs.pop("timestamp")
        tgt_entry = arf.create_entry(self.fp, "copied_entry", src_entry_timestamp, **src_entry_attrs)
        self.assertEqual(self.entry.attrs['uuid'], tgt_entry.attrs['uuid'])

    def test08_check_file_version(self):
        arf.check_file_version(self.fp)

    def test09_append_to_table(self):
        dtype = nx.dtype({'names': ("f1","f2"), 'formats': [nx.uint, nx.int32]})
        dset = arf.create_table(self.fp, 'test', dtype=dtype)
        self.assertEqual(dset.shape[0], 0)
        arf.append_data(dset, (5, 10))
        self.assertEqual(dset.shape[0], 1)


@unittest.skipIf(version.StrictVersion(h5py_version) < version.StrictVersion("2.2"), "not supported on h5py < 2.2")
class TestArfNavigation(unittest.TestCase):
    def setUp(self):
        self.fp = arf.open_file("test", 'w', driver="core", backing_store=False)

    def tearDown(self):
        self.fp.close()

    def test01_creation_iter(self):
        self.fp = arf.open_file("test06", mode="a", driver="core", backing_store=False)
        entry_names = ['z', 'y', 'a', 'q', 'zzyfij']
        for name in entry_names:
            g = arf.create_entry(self.fp, name, 0)
            arf.create_dataset(g, "dset", (1,), sampling_rate=1)
        self.assertEqual(list(arf.keys_by_creation(self.fp)), entry_names)

    def test10_select_from_timeseries(self):
        entry = arf.create_entry(self.fp, "entry", tstamp)
        for data in datasets:
            arf.create_dataset(entry, **data)
            dset = entry[data["name"]]
            if data.get("units", None) == "samples":
                selected, offset = arf.select_interval(dset, 0, data["sampling_rate"])
            else:
                selected, offset = arf.select_interval(dset, 0.0, 1.0)
            if arf.is_time_series(dset):
                nx.testing.assert_array_equal(selected, data["data"][:data["sampling_rate"]])


class TestArfUtility(unittest.TestCase):

    def test01_timestamp_conversion(self):
        from datetime import datetime
        dt = datetime.now()
        ts = arf.convert_timestamp(dt)
        self.assertEqual(arf.timestamp_to_datetime(ts), dt)
        self.assertTrue(all(arf.convert_timestamp(ts) == ts))
        ts = arf.convert_timestamp(1000)
        self.assertEqual(int(arf.timestamp_to_float(ts)), 1000)

    def test99_various(self):
        # test some functions difficult to cover otherwise
        arf.DataTypes._doc()
        arf.DataTypes._todict()





# # Variables:
# # End:
