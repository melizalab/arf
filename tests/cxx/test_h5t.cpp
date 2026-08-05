/* @file test_h5t.cpp
 * @brief unit tests for arf/h5t.hpp -- datatypes
 */

#include <string>

#include <boost/cstdint.hpp>
#include <boost/uuid/uuid.hpp>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5t::datatype;
using arf::h5t::wrapper;

namespace {

template <typename Type>
datatype
make()
{
        return datatype(wrapper<Type>());
}

}

TEST_SUITE("h5t") {

TEST_CASE("integer traits map to fixed-width native types") {
        SUBCASE("signed") {
                CHECK(make<boost::int8_t>().size() == 1);
                CHECK(make<boost::int16_t>().size() == 2);
                CHECK(make<boost::int32_t>().size() == 4);
                CHECK(make<boost::int64_t>().size() == 8);
                CHECK(H5Tget_sign(make<boost::int32_t>().hid()) == H5T_SGN_2);
        }
        SUBCASE("unsigned") {
                CHECK(make<boost::uint8_t>().size() == 1);
                CHECK(make<boost::uint16_t>().size() == 2);
                CHECK(make<boost::uint32_t>().size() == 4);
                CHECK(make<boost::uint64_t>().size() == 8);
                CHECK(H5Tget_sign(make<boost::uint32_t>().hid()) == H5T_SGN_NONE);
        }
        SUBCASE("all of them are integers") {
                CHECK(H5Tget_class(make<boost::int16_t>().hid()) == H5T_INTEGER);
                CHECK(H5Tget_class(make<boost::uint64_t>().hid()) == H5T_INTEGER);
        }
}

TEST_CASE("floating point traits") {
        CHECK(make<float>().size() == sizeof(float));
        CHECK(make<double>().size() == sizeof(double));
        CHECK(H5Tget_class(make<float>().hid()) == H5T_FLOAT);
        CHECK(H5Tget_class(make<double>().hid()) == H5T_FLOAT);
}

TEST_CASE("char is a one-byte integer, not a string") {
        datatype t = make<char>();
        CHECK(t.size() == 1);
        CHECK(H5Tget_class(t.hid()) == H5T_INTEGER);
}

TEST_CASE("strings start out one byte wide and are resized on use") {
        datatype t = make<std::string>();
        CHECK(H5Tget_class(t.hid()) == H5T_STRING);
        // H5T_C_S1 is a single character until somebody calls set_size, which
        // is why node::write_attribute has to special-case strings
        CHECK(t.size() == 1);
        t.set_size(37);
        CHECK(t.size() == 37);
}

TEST_CASE("uuids are stored as 16 raw bytes") {
        // the spec prefers the 36-byte hex string, but this trait is the
        // 128-bit form; arf::entry writes the string via boost::uuids::to_string
        datatype t = make<boost::uuids::uuid>();
        CHECK(t.size() == 16);
}

TEST_CASE("copies own independent handles") {
        datatype original = make<std::string>();
        datatype copy(original);
        REQUIRE(copy.hid() != original.hid());

        copy.set_size(64);
        CHECK(copy.size() == 64);
        CHECK(original.size() == 1);
}

TEST_CASE("assignment releases the old handle and takes a fresh copy") {
        datatype target = make<float>();
        datatype source = make<double>();
        hid_t before = target.hid();

        target = source;
        CHECK(target.size() == sizeof(double));
        CHECK(target.hid() != source.hid());
        CHECK(H5Iis_valid(before) == 0);
}

TEST_CASE("constructing from a hid_t copies rather than adopts") {
        hid_t native = H5Tcopy(H5T_NATIVE_INT);
        {
                datatype wrapped(native);
                CHECK(wrapped.hid() != native);
                CHECK(wrapped.size() == H5Tget_size(native));
        }
        // the wrapper's destructor must not have closed our handle
        CHECK(H5Iis_valid(native) > 0);
        H5Tclose(native);
}

TEST_CASE("equality compares the underlying types") {
        H5Eclear2(H5E_DEFAULT);
        CHECK(make<float>() == make<float>());
        CHECK(make<boost::int32_t>() == make<boost::int32_t>());
        CHECK_FALSE(make<float>() == make<double>());
        CHECK(make<float>() != make<double>());
}

TEST_CASE("equality survives a dirty error stack") {
        // operator== routes a plain bool through check_error, which throws on
        // false whenever hdf5's error stack is non-empty (see backlog item B).
        // It stays dormant here only because H5Tequal, like every public hdf5
        // entry point, clears the stack before returning -- so by the time the
        // bool reaches check_error there is nothing left to raise. Worth
        // pinning: any refactor that compares types without going through an
        // hdf5 call first would expose the defect.
        datatype a = make<float>();
        datatype b = make<double>();

        H5Eset_auto2(H5E_DEFAULT, 0, 0);
        H5Fopen("/tmp/arf_no_such_file_exists.arf", H5F_ACC_RDONLY, H5P_DEFAULT);
        CHECK_FALSE(a == b);
        CHECK(a != b);
}

TEST_CASE("datatypes leak no handles") {
        arftest::handle_guard guard;
        H5Eclear2(H5E_DEFAULT);
        {
                datatype a = make<double>();
                datatype b(a);
                datatype c = make<std::string>();
                c = b;
                CHECK(c.size() == sizeof(double));
        }
}

}
