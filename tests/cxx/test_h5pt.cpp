/* @file test_h5pt.cpp
 * @brief unit tests for arf/h5pt.hpp -- packet tables
 */

#include <cstring>
#include <vector>

#include <boost/cstdint.hpp>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5f::file;
using arf::h5g::group;

namespace {

/** A compound record, the shape arf uses for marked point processes. */
struct interval {
        char name[64];
        boost::uint32_t start;
        boost::uint32_t stop;
};

}

namespace arf { namespace h5t { namespace detail {

template<>
struct datatype_traits<interval> {
        static hid_t value() {
                hid_t ret = H5Tcreate(H5T_COMPOUND, sizeof(interval));
                hid_t str = H5Tcopy(H5T_C_S1);
                H5Tset_size(str, 64);
                H5Tinsert(ret, "name", HOFFSET(interval, name), str);
                H5Tinsert(ret, "start", HOFFSET(interval, start), H5T_STD_U32LE);
                H5Tinsert(ret, "stop", HOFFSET(interval, stop), H5T_STD_U32LE);
                H5Tclose(str);
                return ret;
        }
};

}}}

TEST_SUITE("h5pt") {

// NB: no case here uses arftest::handle_guard, because every packet table
// leaks two identifiers. That is pinned deliberately in "packet tables are
// never closed"; until it is fixed a guard would fire in every case.

TEST_CASE("a packet table appends across many writes") {
        arftest::scratch_file scratch("pt_append");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        arf::packet_table_ptr pt = entry.create_packet_table<float>("stream");
        CHECK(pt->dataspace()->size() == 0);

        std::vector<float> packet(128, 1.5f);
        for (int i = 0; i < 5; ++i) pt->write(packet);
        CHECK(pt->dataspace()->size() == 5 * 128);
}

TEST_CASE("packets can be written from a bare pointer") {
        arftest::scratch_file scratch("pt_pointer");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        arf::packet_table_ptr pt = entry.create_packet_table<boost::uint32_t>("stream");
        boost::uint32_t values[4] = { 10, 20, 30, 40 };
        pt->write(values, 4);
        CHECK(pt->dataspace()->size() == 4);

        std::vector<boost::uint32_t> read;
        pt->read(read);
        REQUIRE(read.size() == 4);
        CHECK(read[0] == 10);
        CHECK(read[3] == 40);
}

TEST_CASE("data written as packets reads back through the dataset interface") {
        arftest::scratch_file scratch("pt_read");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        std::vector<float> packet;
        for (int i = 0; i < 64; ++i) packet.push_back(i * 0.5f);
        arf::packet_table_ptr pt = entry.create_packet_table<float>("stream");
        pt->write(packet);
        pt->write(packet);

        SUBCASE("whole") {
                std::vector<float> read;
                pt->read(read);
                REQUIRE(read.size() == 128);
                CHECK(read[0] == packet[0]);
                CHECK(read[64] == packet[0]);
        }
        SUBCASE("at an offset, through the parent group") {
                std::vector<float> read(64, -1.0f);
                entry.read_dataset("stream", &read[0], 64, 64);
                CHECK(read == packet);
        }
}

TEST_CASE("compound records round trip") {
        arftest::scratch_file scratch("pt_compound");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        arf::packet_table_ptr pt = entry.create_packet_table<interval>("intervals");
        for (int i = 0; i < 3; ++i) {
                interval record;
                std::memset(&record, 0, sizeof(record));
                std::sprintf(record.name, "label_%03d", i);
                record.start = 100 * i;
                record.stop = 100 * i + 50;
                pt->write(&record, 1);
        }
        REQUIRE(pt->dataspace()->size() == 3);

        std::vector<interval> read(3);
        pt->read(&read[0], 3);
        CHECK(std::string(read[0].name) == "label_000");
        CHECK(read[2].start == 200);
        CHECK(read[2].stop == 250);
}

TEST_CASE("chunk size is configurable") {
        arftest::scratch_file scratch("pt_chunks");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        arf::packet_table_ptr pt =
                entry.create_packet_table<float>("stream", false, 256);
        CHECK(pt->chunks() == std::vector<hsize_t>(1, 256));
}

TEST_CASE("replace drops an existing table") {
        arftest::scratch_file scratch("pt_replace");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        {
                arf::packet_table_ptr first = entry.create_packet_table<float>("stream");
                first->write(std::vector<float>(32, 1.0f));
                CHECK(first->dataspace()->size() == 32);
        }
        arf::packet_table_ptr second =
                entry.create_packet_table<float>("stream", true);
        CHECK(second->dataspace()->size() == 0);
}

TEST_CASE("packet tables are never closed") {
        // CHARACTERIZATION: known bug. ~packet_table guards H5PTclose with
        // `H5PTis_valid(_ptself) > 0`, but H5PTis_valid follows the herr_t
        // convention -- 0 means valid, negative means not -- rather than the
        // htri_t convention of H5Iis_valid, which the guard was clearly copied
        // from. The test is therefore false for exactly the tables that need
        // closing, so H5PTclose never runs. Each table strands its packet-table
        // identifier and the dataset identifier underneath it.
        arftest::scratch_file scratch("pt_leak");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        ssize_t before = arftest::handle_guard::open_handles();
        {
                arf::packet_table_ptr pt = entry.create_packet_table<float>("stream");
                pt->write(std::vector<float>(8, 1.0f));
        }
        CHECK(arftest::handle_guard::open_handles() == before + 2);
}

TEST_CASE("creating over an existing table silently hands back the old one") {
        // CHARACTERIZATION: known bug, and a direct consequence of backlog
        // item B. Unlike create_dataset, create_packet_table never checks for
        // an existing link when replace is false, so H5PTcreate_fl is called
        // over one and fails -- returning -1 while leaving hdf5's error stack
        // *empty*. check_error's auto_throw sees nothing to report and converts
        // the failure into 0, so no exception is raised and _ptself is set to a
        // nonsense identifier. The constructor then opens the pre-existing
        // dataset, and the caller gets a half-built object whose writes go
        // nowhere.
        arftest::scratch_file scratch("pt_noreplace");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        arf::packet_table_ptr first = entry.create_packet_table<float>("stream");
        first->write(std::vector<float>(32, 1.0f));
        REQUIRE(first->dataspace()->size() == 32);

        arf::packet_table_ptr second = entry.create_packet_table<float>("stream");
        // no throw, and it is aliasing the original dataset
        CHECK(second->dataspace()->size() == 32);
}

TEST_CASE("an existing table can be reopened") {
        arftest::scratch_file scratch("pt_reopen");
        file f(scratch.path, "w");
        {
                group entry(f, "entry", true);
                arf::packet_table_ptr pt = entry.create_packet_table<float>("stream");
                pt->write(std::vector<float>(16, 2.0f));
        }
        group entry(f, "entry");
        arf::h5pt::packet_table reopened(entry.hid(), "stream");
        CHECK(reopened.dataspace()->size() == 16);
        reopened.write(std::vector<float>(16, 3.0f));
        CHECK(reopened.dataspace()->size() == 32);
}

}
