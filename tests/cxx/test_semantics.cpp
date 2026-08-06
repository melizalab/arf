/* @file test_semantics.cpp
 * @brief copy, move and lifetime semantics of the wrapper classes
 *
 * The design is deliberate and is easy to undo by accident, so it is pinned
 * here rather than left implicit:
 *
 *   value-like  (datatype, dataspace, proplist) copy by duplicating the
 *               description, and move by stealing the identifier
 *   handle-like (file, group, dataset, attribute, packet_table, entry) are
 *               move-only, because duplicating one means another open handle
 *               and that should never happen by accident
 *
 * arf::handle is not polymorphic: its destructor is protected and non-virtual,
 * so every wrapper is the size of the identifier it holds.
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <type_traits>
#include <utility>
#include <vector>

#include "arf.hpp"
#include "fixtures.hpp"

TEST_SUITE("semantics") {

TEST_CASE("no wrapper carries a vtable pointer") {
        CHECK_FALSE(std::is_polymorphic<arf::handle>::value);
        CHECK(sizeof(arf::h5s::dataspace) == sizeof(hid_t));
        CHECK(sizeof(arf::h5t::datatype) == sizeof(hid_t));
        CHECK(sizeof(arf::h5d::dataset) == sizeof(hid_t));
}

TEST_CASE("value-like classes copy and move") {
        CHECK(std::is_copy_constructible<arf::h5t::datatype>::value);
        CHECK(std::is_move_constructible<arf::h5t::datatype>::value);
        CHECK(std::is_copy_constructible<arf::h5s::dataspace>::value);
        CHECK(std::is_move_constructible<arf::h5s::dataspace>::value);
        CHECK(std::is_copy_constructible<arf::h5p::proplist>::value);
        CHECK(std::is_move_constructible<arf::h5p::proplist>::value);
}

TEST_CASE("handle-like classes move but do not copy") {
        // an accidental copy would be a second open handle to the same object
        CHECK_FALSE(std::is_copy_constructible<arf::h5d::dataset>::value);
        CHECK_FALSE(std::is_copy_constructible<arf::h5g::group>::value);
        CHECK_FALSE(std::is_copy_constructible<arf::h5a::attribute>::value);
        CHECK_FALSE(std::is_copy_constructible<arf::h5f::file>::value);
        CHECK_FALSE(std::is_copy_constructible<arf::entry>::value);

        CHECK(std::is_move_constructible<arf::h5d::dataset>::value);
        CHECK(std::is_move_constructible<arf::h5g::group>::value);
        CHECK(std::is_move_constructible<arf::h5f::file>::value);
}

namespace {

/**
 * Assign one object to another through references.
 *
 * Written this way on purpose. clang's -Wself-assign-overloaded rejects the
 * literal `x = x`, and with -Werror that is a build failure -- but the
 * behavior still needs testing, and this is closer to how self-assignment
 * actually reaches a class anyway: two names for one object, not a visibly
 * silly statement.
 */
template <typename Type>
void
assign(Type & target, Type const & source)
{
        target = source;
}

}

TEST_CASE("self-assignment is safe") {
        // closing _self before copying from the argument looks equivalent to
        // copy-and-swap and is not: when they are the same object it copies
        // from the handle it just closed
        arftest::handle_guard guard;

        arf::h5s::dataspace space(std::vector<hsize_t>(1, 8));
        assign(space, space);
        CHECK(space.size() == 8);

        arf::h5t::datatype type{arf::h5t::wrapper<double>()};
        assign(type, type);
        CHECK(type.size() == sizeof(double));

        arf::h5p::proplist plist(H5P_FILE_CREATE);
        assign(plist, plist);
        CHECK(plist.valid());
}

TEST_CASE("copies of a value are independent") {
        arftest::handle_guard guard;
        arf::h5s::dataspace original(std::vector<hsize_t>(1, 4));
        arf::h5s::dataspace copy(original);
        CHECK(copy.hid() != original.hid());
        CHECK(copy.size() == original.size());
}

TEST_CASE("a moved-from object is empty and safe to destroy") {
        arftest::handle_guard guard;
        {
                arf::h5s::dataspace source(std::vector<hsize_t>(1, 4));
                hid_t original = source.hid();
                arf::h5s::dataspace sink(std::move(source));

                CHECK(sink.hid() == original);
                CHECK(sink.valid());
                CHECK_FALSE(source.valid());
        }
        // both destructors ran; only one release happened, which the guard
        // would catch either way
}

TEST_CASE("moving a dataset transfers the handle") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("sem_dataset");
        arf::h5f::file f(scratch.path, "w");
        arf::h5g::group entry(f, "entry", true);
        entry.create_dataset("pcm", arftest::ramp(32));

        arf::h5d::dataset source(entry.hid(), "pcm");
        hid_t original = source.hid();

        arf::h5d::dataset sink(std::move(source));
        CHECK(sink.hid() == original);
        CHECK(sink.dataspace().size() == 32);
        CHECK_FALSE(source.valid());

        // and move assignment releases what the target already held
        arf::h5d::dataset other(entry.hid(), "pcm");
        other = std::move(sink);
        CHECK(other.dataspace().size() == 32);
        CHECK_FALSE(sink.valid());
}

TEST_CASE("moving a file carries both of its identifiers") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("sem_file");
        {
                arf::h5f::file source(scratch.path, "w");
                arf::h5g::group child(source, "entry", true);
        }
        arf::h5f::file source(scratch.path, "r");
        hid_t file_id = source.file_id();

        arf::h5f::file sink(std::move(source));
        CHECK(sink.file_id() == file_id);
        CHECK(sink.contains("entry"));
        CHECK_FALSE(source.valid());
}

TEST_CASE("moving a packet table carries both of its identifiers") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("sem_pt");
        arf::h5f::file f(scratch.path, "w");
        arf::h5g::group entry(f, "entry", true);

        arf::h5pt::packet_table pt = entry.create_packet_table<float>("stream");
        pt.write(std::vector<float>(8, 1.0f));

        arf::h5pt::packet_table sink(std::move(pt));
        CHECK(sink.dataspace().size() == 8);
        // the moved-from table still destructs cleanly when pt goes away
        sink.write(std::vector<float>(8, 2.0f));
        CHECK(sink.dataspace().size() == 16);
}

}
