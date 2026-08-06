/* @file test_h5g.cpp
 * @brief unit tests for arf/h5g.hpp -- groups
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <string>
#include <vector>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5f::file;
using arf::h5g::group;

namespace {

std::vector<std::string>
names(char const * a, char const * b)
{
        std::vector<std::string> out;
        out.push_back(a);
        out.push_back(b);
        return out;
}

std::vector<std::string>
names(char const * a, char const * b, char const * c)
{
        std::vector<std::string> out = names(a, b);
        out.push_back(c);
        return out;
}

/** A user-supplied iterator, to prove the iterate() contract is usable. */
struct counting_iterator {
        typedef int return_value;
        return_value value;

        counting_iterator() : value(0) {}

        static herr_t iterate(hid_t, char const *, H5L_info_t const *, void * data) {
                static_cast<counting_iterator *>(data)->value += 1;
                return 0;
        }
};

}

TEST_SUITE("h5g") {

TEST_CASE("groups are created and reopened by name") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_create");
        file f(scratch.path, "w");

        {
                group created(f, "entry", true);
                CHECK(f.contains("entry"));
        }
        group reopened(f, "entry");
        CHECK(reopened.nchildren() == 0);
}

TEST_CASE("opening a group that isn't there throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_missing");
        file f(scratch.path, "w");
        CHECK_THROWS_AS(group(f, "nope"), arf::Exception);
}

TEST_CASE("contains distinguishes present from absent") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_contains");
        file f(scratch.path, "w");
        group child(f, "here", true);

        CHECK(f.contains("here"));
        CHECK_FALSE(f.contains("not_here"));
}

TEST_CASE("children come back in creation order, not alphabetical") {
        // this is the whole reason the file and group creation property lists
        // set H5P_CRT_ORDER_TRACKED; arf.py depends on it too, via
        // keys_by_creation
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_order");
        file f(scratch.path, "w");
        {
                group a(f, "zebra", true);
                group b(f, "apple", true);
                group c(f, "mango", true);
        }

        CHECK(f.children() == names("zebra", "apple", "mango"));
        CHECK(f.nchildren() == 3);

        arf::h5g::detail::name_iterator by_name;
        CHECK(f.iterate(by_name, H5_INDEX_NAME, H5_ITER_INC)
              == names("apple", "mango", "zebra"));

        arf::h5g::detail::name_iterator backwards;
        CHECK(f.iterate(backwards, H5_INDEX_CRT_ORDER, H5_ITER_DEC)
              == names("mango", "apple", "zebra"));
}

TEST_CASE("nested groups track their own creation order") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_nested");
        file f(scratch.path, "w");
        group outer(f, "outer", true);
        {
                group second(outer, "second", true);
                group first(outer, "first", true);
        }
        CHECK(outer.children() == names("second", "first"));
        CHECK(outer.nchildren() == 2);
}

TEST_CASE("a user-supplied functor can drive iteration") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_iterate");
        file f(scratch.path, "w");
        {
                group a(f, "one", true);
                group b(f, "two", true);
        }
        counting_iterator counter;
        CHECK(f.iterate(counter) == 2);
}

TEST_CASE("unlink removes a child") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_unlink");
        file f(scratch.path, "w");
        {
                group doomed(f, "doomed", true);
        }
        REQUIRE(f.contains("doomed"));
        f.unlink("doomed");
        CHECK_FALSE(f.contains("doomed"));
        CHECK(f.nchildren() == 0);
}

TEST_CASE("unlinking something that isn't there throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_unlink_missing");
        file f(scratch.path, "w");
        CHECK_THROWS_AS(f.unlink("nope"), arf::Exception);
}

TEST_CASE("datasets are created under a group") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_dataset");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        std::vector<double> data = arftest::ramp(512);
        arf::h5d::dataset d = entry.create_dataset("pcm", data);
        CHECK(entry.contains("pcm"));
        CHECK(d.dataspace().size() == 512);
}

TEST_CASE("creating a dataset over an existing name throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_dup");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        std::vector<double> data = arftest::ramp(8);
        arf::h5d::dataset first = entry.create_dataset("pcm", data);
        CHECK_THROWS_AS(entry.create_dataset("pcm", data), arf::Exception);
}

TEST_CASE("read_dataset fills a caller-supplied array") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_read_array");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> written = arftest::ramp(64);
        entry.create_dataset("pcm", written);

        SUBCASE("whole dataset") {
                std::vector<double> read(64, -1.0);
                entry.read_dataset("pcm", &read[0], 64);
                CHECK(read == written);
        }
        SUBCASE("from an offset") {
                std::vector<double> read(4, -1.0);
                entry.read_dataset("pcm", &read[0], 4, 10);
                CHECK(read[0] == written[10]);
                CHECK(read[3] == written[13]);
        }
        SUBCASE("with a stride") {
                std::vector<double> read(4, -1.0);
                entry.read_dataset("pcm", &read[0], 4, 0, 2);
                CHECK(read[0] == written[0]);
                CHECK(read[1] == written[2]);
                CHECK(read[3] == written[6]);
        }
}

TEST_CASE("the vector overload of read_dataset resizes and fills") {
        // NB: the count is worked out here rather than forwarded. Passing
        // (offset, stride) straight to dataset::read(vector, count, offset)
        // puts offset in count and stride in offset, which asks for zero
        // elements starting at index one -- and reads nothing, quietly.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("g_read_vector");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> written = arftest::ramp(64);
        entry.create_dataset("pcm", written);

        SUBCASE("the whole dataset, into an empty vector") {
                std::vector<double> read;
                entry.read_dataset("pcm", read);
                CHECK(read == written);
        }
        SUBCASE("from an offset") {
                std::vector<double> read;
                entry.read_dataset("pcm", read, 60);
                REQUIRE(read.size() == 4);
                CHECK(read[0] == written[60]);
                CHECK(read[3] == written[63]);
        }
        SUBCASE("with a stride") {
                std::vector<double> read;
                entry.read_dataset("pcm", read, 0, 8);
                REQUIRE(read.size() == 8);
                CHECK(read[0] == written[0]);
                CHECK(read[1] == written[8]);
                CHECK(read[7] == written[56]);
        }
        SUBCASE("an offset past the end reads nothing") {
                std::vector<double> read(4, -1.0);
                entry.read_dataset("pcm", read, 999);
                CHECK(read.empty());
        }
        SUBCASE("a caller's oversized vector is shrunk, not overrun") {
                std::vector<double> read(4096, -1.0);
                entry.read_dataset("pcm", read);
                CHECK(read.size() == 64);
                CHECK(read == written);
        }
}

}
