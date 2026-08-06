/* @file test_h5a.cpp
 * @brief unit tests for arf/h5a.hpp -- attributes
 */

#include <string>
#include <vector>

#include <cstdint>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5a::attribute;
using arf::h5f::file;
using arf::h5g::group;

namespace {

std::vector<int>
counts()
{
        std::vector<int> out;
        for (int i = 1; i <= 5; ++i) out.push_back(i * 10);
        return out;
}

}

TEST_SUITE("h5a") {

TEST_CASE("scalar attributes round trip") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_scalar");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute("an_int", 42);
        node.write_attribute("a_float", 2.5f);
        node.write_attribute("a_double", 1.0 / 3.0);
        node.write_attribute("a_long", static_cast<std::int64_t>(1) << 40);

        CHECK(node.read_attribute<int>("an_int") == 42);
        CHECK(node.read_attribute<float>("a_float") == 2.5f);
        CHECK(node.read_attribute<double>("a_double") == doctest::Approx(1.0 / 3.0));
        CHECK(node.read_attribute<std::int64_t>("a_long")
              == (static_cast<std::int64_t>(1) << 40));
}

TEST_CASE("vector attributes round trip") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_vector");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute("counts", counts());

        std::vector<int> read;
        node.read_attribute("counts", read);
        CHECK(read == counts());
}

TEST_CASE("string attributes round trip") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_string");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute("who", std::string("an experimenter"));
        node.write_attribute("what", "a c string");

        CHECK(node.read_attribute<std::string>("who") == "an experimenter");
        CHECK(node.read_attribute<std::string>("what") == "a c string");
}

TEST_CASE("string attributes are readable whatever their storage") {
        // arf files contain all three, and the spec requires readers to
        // accept any of them. Handling only fixed-length with a terminator --
        // the form this library writes -- makes units unreadable in every file
        // arf.py produces, and units is required on every dataset.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_storage");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        SUBCASE("variable-length, as arf.py writes most attributes") {
                hid_t type = H5Tcopy(H5T_C_S1);
                H5Tset_size(type, H5T_VARIABLE);
                H5Tset_cset(type, H5T_CSET_UTF8);
                hid_t space = H5Screate(H5S_SCALAR);
                hid_t attr = H5Acreate2(node.hid(), "vlen", type, space,
                                        H5P_DEFAULT, H5P_DEFAULT);
                char const * value = "variable length";
                REQUIRE(H5Awrite(attr, type, &value) >= 0);
                H5Aclose(attr);
                H5Sclose(space);
                H5Tclose(type);

                CHECK(node.read_attribute<std::string>("vlen") == "variable length");
        }

        SUBCASE("fixed-length with no terminator, as arf.py writes a uuid") {
                char const * value = "0123456789abcdef";
                hid_t type = H5Tcopy(H5T_C_S1);
                H5Tset_size(type, 16);  // exactly the characters, no room for a NUL
                hid_t space = H5Screate(H5S_SCALAR);
                hid_t attr = H5Acreate2(node.hid(), "exact", type, space,
                                        H5P_DEFAULT, H5P_DEFAULT);
                REQUIRE(H5Awrite(attr, type, value) >= 0);
                H5Aclose(attr);
                H5Sclose(space);
                H5Tclose(type);

                CHECK(node.read_attribute<std::string>("exact") == "0123456789abcdef");
        }

        SUBCASE("fixed-length with a terminator, as this library writes") {
                node.write_attribute("own", "written here");
                CHECK(node.read_attribute<std::string>("own") == "written here");
        }
}

TEST_CASE("writing again replaces the value") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_overwrite");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute("count", 1);
        node.write_attribute("count", 2);
        CHECK(node.read_attribute<int>("count") == 2);

        SUBCASE("including strings, whose width has to change") {
                node.write_attribute("label", "short");
                node.write_attribute("label", "a considerably longer label");
                CHECK(node.read_attribute<std::string>("label")
                      == "a considerably longer label");
                node.write_attribute("label", "tiny");
                CHECK(node.read_attribute<std::string>("label") == "tiny");
        }
}

TEST_CASE("has_attribute and delete_attribute") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_presence");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        CHECK_FALSE(node.has_attribute("units"));
        node.write_attribute("units", "mV");
        CHECK(node.has_attribute("units"));

        node.delete_attribute("units");
        CHECK_FALSE(node.has_attribute("units"));
        // deleting something absent is a no-op rather than an error
        node.delete_attribute("units");
        CHECK_FALSE(node.has_attribute("units"));
}

TEST_CASE("attributes can be written by chaining") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_chain");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute()
                ("intattr", 1)
                ("vecattr", counts())
                ("strattr", "chained");

        CHECK(node.read_attribute<int>("intattr") == 1);
        CHECK(node.read_attribute<std::string>("strattr") == "chained");
        std::vector<int> read;
        node.read_attribute("vecattr", read);
        CHECK(read == counts());
}

TEST_CASE("the storage type can differ from the memory type") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_convert");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        // store a plain int as 64-bit, the way arf::entry stores timestamps
        node.write_attribute<std::int64_t, int>("stored_wide", 7);
        attribute attr(node, "stored_wide");
        hid_t stored = H5Aget_type(attr.hid());
        CHECK(H5Tget_size(stored) == 8);
        H5Tclose(stored);
        CHECK(node.read_attribute<int>("stored_wide") == 7);

        SUBCASE("and reads convert too") {
                node.write_attribute("an_int", 3);
                CHECK(node.read_attribute<double>("an_int") == doctest::Approx(3.0));
        }
}

TEST_CASE("reading a non-string attribute as a string throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_wrongtype");
        file f(scratch.path, "w");
        group node(f, "entry", true);
        node.write_attribute("count", 5);

        CHECK_THROWS_AS(node.read_attribute<std::string>("count"), arf::Exception);
}

TEST_CASE("reading an attribute that isn't there throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_absent");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        CHECK_THROWS_AS(node.read_attribute<int>("nope"), arf::Exception);
}

TEST_CASE("an attribute reports its own name and shape") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_introspect");
        file f(scratch.path, "w");
        group node(f, "entry", true);
        node.write_attribute("counts", counts());

        attribute attr(node, "counts");
        CHECK(attr.name() == "counts");
        CHECK(attr.dataspace().size() == counts().size());
}

TEST_CASE("writing an attribute with the wrong length throws") {
        // H5Awrite always writes the attribute's whole extent, so a shorter
        // buffer is read past its end. An assert would not catch it in the
        // build where the overrun actually happens.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_width");
        file f(scratch.path, "w");
        group node(f, "entry", true);
        node.write_attribute("counts", counts());

        attribute attr(node, "counts");
        std::vector<int> shorter(2, 1);
        CHECK_THROWS_AS(attr.write(shorter.data(), shorter.size()), arf::Exception);

        std::vector<int> longer(99, 1);
        CHECK_THROWS_AS(attr.write(longer.data(), longer.size()), arf::Exception);

        // the right length still works, and the value is unchanged by the
        // rejected attempts
        std::vector<int> exact = counts();
        attr.write(exact.data(), exact.size());
        std::vector<int> read;
        node.read_attribute("counts", read);
        CHECK(read == counts());
}

TEST_CASE("reading a string attribute releases its datatype") {
        // The datatype H5Aget_type opens has to be released on the throw path
        // as well as the success path. arf::entry's open-existing constructor
        // reads the uuid, so leaking one per read means one per entry walked,
        // and hdf5 will not fully close a file while identifiers remain open.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_leak");
        file f(scratch.path, "w");
        group node(f, "entry", true);
        node.write_attribute("label", "a string");
        node.write_attribute("count", 5);

        ssize_t before = arftest::handle_guard::open_handles();
        node.read_attribute<std::string>("label");
        CHECK(arftest::handle_guard::open_handles() == before);

        before = arftest::handle_guard::open_handles();
        node.read_attribute<int>("count");
        CHECK(arftest::handle_guard::open_handles() == before);

        // including the early return on a type mismatch, which is the path
        // easiest to leave unreleased
        before = arftest::handle_guard::open_handles();
        CHECK_THROWS_AS(node.read_attribute<std::string>("count"), arf::Exception);
        CHECK(arftest::handle_guard::open_handles() == before);
}

TEST_CASE("attributes leak no handles") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("a_handles");
        file f(scratch.path, "w");
        {
                group node(f, "entry", true);
                node.write_attribute("a", 1);
                node.write_attribute("b", "two");
                node.write_attribute("c", counts());
                attribute attr(node, "a");
                CHECK(attr.read<int>() == 1);
                node.delete_attribute("b");
        }
}

}
