/* @file test_h5f.cpp
 * @brief unit tests for arf/h5f.hpp -- files
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <string>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5f::file;

TEST_SUITE("h5f") {

TEST_CASE("mode w creates a file") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_create");
        REQUIRE_FALSE(scratch.exists());
        {
                file f(scratch.path, "w");
                CHECK(f.nchildren() == 0);
        }
        CHECK(scratch.exists());
}

TEST_CASE("mode w truncates an existing file") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_truncate");
        {
                file f(scratch.path, "w");
                arf::h5g::group child(f, "keep_me", true);
                CHECK(f.nchildren() == 1);
        }
        {
                file f(scratch.path, "w");
                CHECK(f.nchildren() == 0);
        }
}

TEST_CASE("mode a creates when absent and preserves when present") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_append");

        SUBCASE("creates") {
                file f(scratch.path, "a");
                CHECK(f.nchildren() == 0);
        }
        SUBCASE("preserves") {
                {
                        file f(scratch.path, "a");
                        arf::h5g::group child(f, "keep_me", true);
                }
                {
                        file f(scratch.path, "a");
                        CHECK(f.nchildren() == 1);
                        CHECK(f.contains("keep_me"));
                }
        }
}

TEST_CASE("mode r opens an existing file") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_read");
        {
                file f(scratch.path, "w");
                arf::h5g::group child(f, "entry", true);
        }
        file f(scratch.path, "r");
        CHECK(f.contains("entry"));
}

TEST_CASE("mode r on a missing file throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_missing");
        REQUIRE_FALSE(scratch.exists());
        CHECK_THROWS_AS(file(scratch.path, "r"), arf::Exception);
}

TEST_CASE("an unrecognized mode throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_badmode");
        CHECK_THROWS_AS(file(scratch.path, "x"), arf::Exception);
        CHECK_THROWS_AS(file(scratch.path, ""), arf::Exception);
        // and nothing was created along the way
        CHECK_FALSE(scratch.exists());
}

TEST_CASE("name reports the path the file was opened with") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_name");
        file f(scratch.path, "w");
        CHECK(f.filename() == scratch.path);
}

TEST_CASE("size grows as data is written") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_size");
        file f(scratch.path, "w");
        hsize_t empty = f.size();
        CHECK(empty > 0);

        f.create_dataset("data", arftest::ramp(4096));
        f.flush();
        CHECK(f.size() > empty);
}

TEST_CASE("constructing from a hid_t adopts the handle") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_adopt");
        hid_t native = H5Fcreate(scratch.path.c_str(), H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(native >= 0);
        {
                file f(native);
                CHECK(f.file_id() == native);
        }
        CHECK(H5Iis_valid(native) == 0);
}

TEST_CASE("two handles can be open on one file at once") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("f_concurrent");
        file writer(scratch.path, "w");
        arf::h5g::group child(writer, "entry", true);
        writer.flush();

        file second(scratch.path, "a");
        CHECK(second.children() == writer.children());
}

}
