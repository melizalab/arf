/* @file test_h5a.cpp
 * @brief unit tests for arf/h5a.hpp -- attributes
 */

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

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
        node.write_attribute("a_long", static_cast<boost::int64_t>(1) << 40);

        CHECK(node.read_attribute<int>("an_int") == 42);
        CHECK(node.read_attribute<float>("a_float") == 2.5f);
        CHECK(node.read_attribute<double>("a_double") == doctest::Approx(1.0 / 3.0));
        CHECK(node.read_attribute<boost::int64_t>("a_long")
              == (static_cast<boost::int64_t>(1) << 40));
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

// NB: cases below that read a string attribute deliberately omit
// arftest::handle_guard, because every such read leaks an hdf5 identifier.
// That leak is pinned once, on purpose, in "reading a string attribute leaks a
// datatype handle".

TEST_CASE("string attributes round trip") {
        arftest::scratch_file scratch("a_string");
        file f(scratch.path, "w");
        group node(f, "entry", true);

        node.write_attribute("who", std::string("an experimenter"));
        node.write_attribute("what", "a c string");

        CHECK(node.read_attribute<std::string>("who") == "an experimenter");
        CHECK(node.read_attribute<std::string>("what") == "a c string");
}

TEST_CASE("writing again replaces the value") {
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
        node.write_attribute<boost::int64_t, int>("stored_wide", 7);
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
        CHECK(attr.dataspace()->size() == counts().size());

        // NB: attribute::write(ptr, size) asserts against this size but then
        // ignores it, writing however many elements the attribute holds. See
        // backlog item 6. Passing a shorter buffer reads out of bounds, so
        // there is deliberately no test that does it.
}

TEST_CASE("reading a string attribute leaks a datatype handle") {
        // CHARACTERIZATION: known bug. attribute::read(std::string&) opens a
        // datatype with H5Aget_type and never closes it -- unlike
        // attribute::write(std::string const&) three lines above, which does.
        // One identifier is leaked per read, on the success path and on the
        // throw path alike. arf::entry's open-existing constructor reads the
        // uuid attribute, so walking a file leaks one handle per entry, and
        // hdf5 will not fully close a file while identifiers remain open.
        arftest::scratch_file scratch("a_leak");
        file f(scratch.path, "w");
        group node(f, "entry", true);
        node.write_attribute("label", "a string");
        node.write_attribute("count", 5);

        ssize_t before = arftest::handle_guard::open_handles();
        node.read_attribute<std::string>("label");
        CHECK(arftest::handle_guard::open_handles() == before + 1);

        // reading an int is clean, so this is specific to the string overload
        before = arftest::handle_guard::open_handles();
        node.read_attribute<int>("count");
        CHECK(arftest::handle_guard::open_handles() == before);

        // and the early return on a type mismatch leaks the same way
        before = arftest::handle_guard::open_handles();
        CHECK_THROWS_AS(node.read_attribute<std::string>("count"), arf::Exception);
        CHECK(arftest::handle_guard::open_handles() == before + 1);
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
