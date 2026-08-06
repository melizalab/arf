/* @file test_h5t.cpp
 * @brief unit tests for arf/h5t.hpp -- datatypes
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <string>

#include <cstdint>

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
                CHECK(make<std::int8_t>().size() == 1);
                CHECK(make<std::int16_t>().size() == 2);
                CHECK(make<std::int32_t>().size() == 4);
                CHECK(make<std::int64_t>().size() == 8);
                CHECK(H5Tget_sign(make<std::int32_t>().hid()) == H5T_SGN_2);
        }
        SUBCASE("unsigned") {
                CHECK(make<std::uint8_t>().size() == 1);
                CHECK(make<std::uint16_t>().size() == 2);
                CHECK(make<std::uint32_t>().size() == 4);
                CHECK(make<std::uint64_t>().size() == 8);
                CHECK(H5Tget_sign(make<std::uint32_t>().hid()) == H5T_SGN_NONE);
        }
        SUBCASE("all of them are integers") {
                CHECK(H5Tget_class(make<std::int16_t>().hid()) == H5T_INTEGER);
                CHECK(H5Tget_class(make<std::uint64_t>().hid()) == H5T_INTEGER);
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

TEST_CASE("strings are fixed-length and declared UTF-8") {
        // The spec constrains the class and CTYPE, and requires the declared
        // CSET to match the encoding -- which ASCII did not, once a
        // std::string held UTF-8. It leaves fixed versus variable length open
        // except for uuid. Fixed-length keeps the characters inline in the
        // object header, with no global-heap indirection and no allocation on
        // read, which is what the acquisition path wants.
        datatype t = make<std::string>();
        CHECK(H5Tget_class(t.hid()) == H5T_STRING);
        CHECK(H5Tis_variable_str(t.hid()) == 0);
        CHECK(H5Tget_cset(t.hid()) == H5T_CSET_UTF8);
        CHECK(H5Tget_strpad(t.hid()) == H5T_STR_NULLPAD);

        datatype c = make<char const *>();
        CHECK(H5Tis_variable_str(c.hid()) == 0);
        CHECK(H5Tget_cset(c.hid()) == H5T_CSET_UTF8);

        // the width belongs to the value, and is set when the attribute is
        // created -- see node::write_attribute
        t.set_size(36);
        CHECK(t.size() == 36);
}

TEST_CASE("uuids are stored as 16 raw bytes") {
        // the spec prefers the 36-byte hex string, but this trait is the
        // 128-bit form; arf::entry writes the string via uuid::str()
        datatype t = make<arf::uuid>();
        CHECK(t.size() == 16);
}

TEST_CASE("copies own independent handles") {
        // a fixed-length string, since resizing is what makes the
        // independence observable
        datatype original(H5Tcopy(H5T_C_S1));
        original.set_size(8);
        datatype copy(original);
        REQUIRE(copy.hid() != original.hid());

        copy.set_size(64);
        CHECK(copy.size() == 64);
        CHECK(original.size() == 8);
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

TEST_CASE("constructing from a hid_t adopts the handle") {
        // Matches h5s::dataspace(hid_t). Callers holding a borrowed handle
        // must copy it themselves. Every in-library call site passes a
        // freshly returned handle, which nothing else owns.
        hid_t native = H5Tcopy(H5T_NATIVE_INT);
        {
                datatype wrapped(native);
                CHECK(wrapped.hid() == native);
                CHECK(wrapped.size() == sizeof(int));
        }
        CHECK(H5Iis_valid(native) == 0);
}

TEST_CASE("equality compares the underlying types") {
        H5Eclear2(H5E_DEFAULT);
        CHECK(make<float>() == make<float>());
        CHECK(make<std::int32_t>() == make<std::int32_t>());
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
