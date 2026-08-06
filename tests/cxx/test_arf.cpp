/* @file test_arf.cpp
 * @brief unit tests for arf.hpp -- the arf entry and file wrappers
 *
 * These cover the layer that enforces the format: the attributes an arf file
 * and entry must carry, per specification.md.
 */

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "arf.hpp"
#include "fixtures.hpp"

namespace {

/** Size in bytes of the datatype an attribute is stored with. */
std::size_t
attribute_width(arf::h5a::node & node, char const * name)
{
        arf::h5a::attribute attr(node, name);
        hid_t type = H5Aget_type(attr.hid());
        std::size_t size = H5Tget_size(type);
        H5Tclose(type);
        return size;
}

std::size_t
attribute_count(arf::h5a::node & node, char const * name)
{
        arf::h5a::attribute attr(node, name);
        return attr.dataspace().size();
}

}

TEST_SUITE("arf") {

TEST_CASE("a new file identifies itself as arf") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_version");
        arf::file f(scratch.path, "w");

        CHECK(f.read_attribute<std::string>("arf_version") == ARF_VERSION);
        CHECK(f.read_attribute<std::string>("arf_library") == "c++");
        CHECK(f.read_attribute<std::string>("arf_library_version") == ARF_LIBRARY_VERSION);
}

TEST_CASE("the declared spec version matches the specification") {
        // specification.md says 2.1; bump both together or the file lies about
        // what it conforms to
        CHECK(std::string(ARF_VERSION) == "2.1");
}

TEST_CASE("opening for append preserves another writer's provenance") {
        // Mode "a" used to stamp this library's identity on unconditionally,
        // so appending one entry to a file arf.py wrote relabelled it as a c++
        // file at whatever version this library happened to be.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_append");
        {
                arf::h5f::file plain(scratch.path, "w");
                plain.write_attribute("arf_library", "python");
                plain.write_attribute("arf_library_version", "2.7.2");
                plain.write_attribute("arf_version", "2.1");
        }
        arf::file f(scratch.path, "a");
        CHECK(f.read_attribute<std::string>("arf_library") == "python");
        CHECK(f.read_attribute<std::string>("arf_library_version") == "2.7.2");
}

TEST_CASE("opening for append labels a file that has no version yet") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_append_bare");
        {
                // a plain hdf5 file with nothing of ours in it
                arf::h5f::file plain(scratch.path, "w");
                arf::h5g::group child(plain, "something", true);
        }
        arf::file f(scratch.path, "a");
        CHECK(f.read_attribute<std::string>("arf_library") == "c++");
        CHECK(f.read_attribute<std::string>("arf_version") == ARF_VERSION);
        CHECK(f.contains("something"));
}

TEST_CASE("an entry with no uuid reports the nil uuid") {
        // the member is a POD, so without an initializer it held whatever was
        // on the stack -- reading it was undefined behavior
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_no_uuid");
        {
                arf::h5f::file plain(scratch.path, "w");
                arf::h5g::group bare(plain, "entry_000", true);
        }
        arf::h5f::file f(scratch.path, "r");
        arf::entry e(f, "entry_000");
        CHECK_FALSE(e.has_attribute("uuid"));
        CHECK(e.uuid().is_nil());
}

TEST_CASE("an entry stores the timestamp the spec requires") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_timestamp");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1234567890, 500);

        // "a two-element array ... at least 64-bit integer precision"
        CHECK(attribute_width(e, "timestamp") == 8);
        CHECK(attribute_count(e, "timestamp") == 2);

        std::vector<boost::int64_t> timestamp;
        e.read_attribute("timestamp", timestamp);
        REQUIRE(timestamp.size() == 2);
        CHECK(timestamp[0] == 1234567890);
        CHECK(timestamp[1] == 500);
}

TEST_CASE("a timestamp can also be supplied as a vector") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_timestamp_vec");
        arf::file f(scratch.path, "w");

        std::vector<boost::int32_t> supplied;
        supplied.push_back(99);
        supplied.push_back(1000);
        arf::entry e(f, "entry_000", supplied);

        // narrower input is widened to the required 64 bits on the way in
        CHECK(attribute_width(e, "timestamp") == 8);
        std::vector<boost::int64_t> timestamp;
        e.read_attribute("timestamp", timestamp);
        REQUIRE(timestamp.size() == 2);
        CHECK(timestamp[0] == 99);
        CHECK(timestamp[1] == 1000);
}

TEST_CASE("an entry gets a uuid that survives reopening") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_uuid");
        arf::file f(scratch.path, "w");

        arf::entry created(f, "entry_000", 1, 0);
        REQUIRE(created.has_attribute("uuid"));

        arf::entry reopened(f, "entry_000");
        CHECK(reopened.uuid() == created.uuid());
        CHECK(boost::uuids::to_string(reopened.uuid()).size() == 36);
}

TEST_CASE("the uuid attribute is exactly 36 bytes, as the spec requires") {
        // "a 36-byte H5T_STRING", which is also what arf.py writes as |S36.
        // node::write_attribute sizes strings at value.size()+1 for a
        // terminator, so the uuid goes through the fixed-width path instead.
        // Every other string attribute is variable-length.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_uuid_width");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);

        CHECK(attribute_width(e, "uuid") == 36);
        CHECK(e.read_attribute<std::string>("uuid").size() == 36);
}

TEST_CASE("datasets created through an entry carry units and datatype") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_dataset");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);

        std::vector<double> data = arftest::ramp(256);
        arf::h5d::dataset d = e.create_dataset("pcm", data, "mV", arf::ACOUSTIC);

        CHECK(d.read_attribute<std::string>("units") == "mV");
        CHECK(d.read_attribute<int>("datatype") == arf::ACOUSTIC);
}

TEST_CASE("replace drops an existing dataset") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_replace");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);

        e.create_dataset("pcm", arftest::ramp(256), "mV", arf::ACOUSTIC);
        arf::h5d::dataset replaced =
                e.create_dataset("pcm", arftest::ramp(16), "mV", arf::ACOUSTIC, true);
        CHECK(replaced.dataspace().size() == 16);
}

TEST_CASE("without replace, a second dataset of the same name throws") {
        // the doc comment on create_dataset says the data is appended when
        // replace is false; it is not, the underlying create throws. See
        // backlog item 9.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_noreplace");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);

        e.create_dataset("pcm", arftest::ramp(16), "mV", arf::ACOUSTIC);
        CHECK_THROWS_AS(e.create_dataset("pcm", arftest::ramp(16), "mV", arf::ACOUSTIC),
                        arf::Exception);
}

TEST_CASE("packet tables created through an entry carry units and datatype") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_pt");
        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);

        arf::h5pt::packet_table pt =
                e.create_packet_table<float>("spikes", "s", arf::SPIKET);
        CHECK(pt.read_attribute<std::string>("units") == "s");
        CHECK(pt.read_attribute<int>("datatype") == arf::SPIKET);
}

TEST_CASE("entries are listed in creation order") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("arf_order");
        arf::file f(scratch.path, "w");
        {
                arf::entry third(f, "entry_003", 3, 0);
                arf::entry first(f, "entry_001", 1, 0);
                arf::entry second(f, "entry_002", 2, 0);
        }
        std::vector<std::string> children = f.children();
        REQUIRE(children.size() == 3);
        CHECK(children[0] == "entry_003");
        CHECK(children[1] == "entry_001");
        CHECK(children[2] == "entry_002");
}

TEST_CASE("the datatype codes match the specification") {
        CHECK(arf::UNDEFINED == 0);
        CHECK(arf::ACOUSTIC == 1);
        CHECK(arf::EXTRAC_HP == 2);
        CHECK(arf::EXTRAC_LF == 3);
        CHECK(arf::EXTRAC_EEG == 4);
        CHECK(arf::INTRAC_CC == 5);
        CHECK(arf::EXTRAC_RAW == 23);
        CHECK(arf::EVENT == 1000);
        CHECK(arf::SPIKET == 1001);
        CHECK(arf::BEHAVET == 1002);
        CHECK(arf::INTERVAL == 2000);
        CHECK(arf::STIMI == 2001);
        CHECK(arf::COMPONENTL == 2002);
}

TEST_CASE("INTRAC_VC matches the specification") {
        // specification.md and arf.py both give voltage clamp the code 6; this
        // library said 7 from the day it was written, so anything it labelled
        // as voltage clamp was mislabelled. Nothing was ever recorded with it,
        // so there are no files in the wild to migrate.
        CHECK(arf::INTRAC_VC == 6);
}

}
