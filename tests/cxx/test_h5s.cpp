/* @file test_h5s.cpp
 * @brief unit tests for arf/h5s.hpp -- dataspaces and chunk guessing
 */

#include <vector>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5s::dataspace;
using arf::h5s::detail::guess_chunk;

namespace {

std::vector<hsize_t>
dims(hsize_t a)
{
        return std::vector<hsize_t>(1, a);
}

std::vector<hsize_t>
dims(hsize_t a, hsize_t b)
{
        std::vector<hsize_t> out;
        out.push_back(a);
        out.push_back(b);
        return out;
}

}

TEST_SUITE("h5s") {

TEST_CASE("the default dataspace is scalar") {
        dataspace space;
        CHECK(space.ndims() == 0);
        // a scalar holds exactly one element, which size() special-cases
        CHECK(space.size() == 1);
}

TEST_CASE("a simple dataspace reports its extent") {
        dataspace space(dims(17));
        CHECK(space.ndims() == 1);
        CHECK(space.dims() == dims(17));
        CHECK(space.size() == 17);
}

TEST_CASE("size is the product of the dimensions") {
        dataspace space(dims(6, 7));
        CHECK(space.ndims() == 2);
        CHECK(space.size() == 42);
}

TEST_CASE("maxdims may be unlimited") {
        dataspace space(dims(10), dims(H5S_UNLIMITED));
        CHECK(space.dims() == dims(10));
        CHECK(space.maxdims() == dims(H5S_UNLIMITED));
}

TEST_CASE("an empty maxdims means fixed extent") {
        dataspace space(dims(10), std::vector<hsize_t>());
        CHECK(space.dims() == dims(10));
        CHECK(space.maxdims() == dims(10));
}

TEST_CASE("copies own independent handles") {
        dataspace original(dims(4));
        dataspace copy(original);
        CHECK(copy.hid() != original.hid());
        CHECK(copy.dims() == original.dims());
}

TEST_CASE("assignment releases the old handle") {
        dataspace target(dims(4));
        dataspace source(dims(9));
        hid_t before = target.hid();

        target = source;
        CHECK(target.size() == 9);
        CHECK(target.hid() != source.hid());
        CHECK(H5Iis_valid(before) == 0);
}

TEST_CASE("constructing from a hid_t adopts the handle") {
        // NB: the opposite of h5t::datatype(hid_t), which copies. See backlog
        // item 8 -- the two wrappers disagree about ownership.
        hid_t native = H5Screate_simple(1, dims(5).data(), NULL);
        {
                dataspace wrapped(native);
                CHECK(wrapped.hid() == native);
                CHECK(wrapped.size() == 5);
        }
        CHECK(H5Iis_valid(native) == 0);
}

TEST_CASE("a hyperslab selects a strided subset") {
        dataspace whole(dims(100));
        dataspace slab(whole, dims(10), dims(2), dims(20));
        CHECK(slab.dims() == dims(100));
        CHECK(H5Sget_select_npoints(slab.hid()) == 20);
        CHECK(H5Sget_select_type(slab.hid()) == H5S_SEL_HYPERSLABS);
}

TEST_CASE("selections can be reset") {
        dataspace space(dims(10));
        space.select_none();
        CHECK(H5Sget_select_npoints(space.hid()) == 0);
        space.select_all();
        CHECK(H5Sget_select_npoints(space.hid()) == 10);
}

TEST_CASE("guess_chunk leaves a small dataset in one chunk") {
        // 40 bytes is far under the 8k soft minimum, so the loop exits at once
        CHECK(guess_chunk(dims(10), sizeof(float)) == dims(10));
}

TEST_CASE("guess_chunk halves a large dataset toward the target size") {
        CHECK(guess_chunk(dims(1 << 20), sizeof(float)) == dims(8192));
}

TEST_CASE("guess_chunk walks the axes in turn") {
        std::vector<hsize_t> chunks = guess_chunk(dims(4096, 4), sizeof(double));
        REQUIRE(chunks.size() == 2);
        CHECK(chunks[0] <= 4096);
        CHECK(chunks[1] <= 4);
        CHECK(chunks[0] * chunks[1] * sizeof(double) <= 1024 * 1024);
}

TEST_CASE("guess_chunk refuses a scalar") {
        CHECK_THROWS_AS(guess_chunk(std::vector<hsize_t>(), sizeof(float)),
                        arf::Exception);
}

TEST_CASE("guess_chunk returns a zero chunk for an empty dataset") {
        // CHARACTERIZATION: a zero-length extent has zero size, which is
        // already under the target, so the loop exits before adjusting
        // anything and hands back a chunk of 0. H5Pset_chunk rejects that, so
        // creating an empty dataset fails downstream rather than here.
        CHECK(guess_chunk(dims(0), sizeof(float)) == dims(0));
}

TEST_CASE("dataspaces leak no handles") {
        arftest::handle_guard guard;
        {
                dataspace a(dims(4));
                dataspace b(a);
                dataspace c;
                c = b;
                dataspace slab(a, dims(1), dims(1), dims(2));
                CHECK(slab.size() == 4);
        }
}

}
