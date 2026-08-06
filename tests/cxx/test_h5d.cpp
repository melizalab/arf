/* @file test_h5d.cpp
 * @brief unit tests for arf/h5d.hpp -- datasets
 */

#include <vector>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5d::dataset;
using arf::h5f::file;
using arf::h5g::group;

namespace {

int
filter_count(dataset const & d)
{
        hid_t dcpl = H5Dget_create_plist(d.hid());
        int n = H5Pget_nfilters(dcpl);
        H5Pclose(dcpl);
        return n;
}

}

TEST_SUITE("h5d") {

TEST_CASE("data round trips through a dataset") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_roundtrip");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> written = arftest::ramp(1024);

        arf::dataset_ptr d = entry.create_dataset("pcm", written);
        std::vector<double> read;
        d->read(read);
        CHECK(read == written);
}

TEST_CASE("the vector read resizes to the dataset") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_resize_read");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(37));

        std::vector<double> read;
        REQUIRE(read.empty());
        d->read(read);
        CHECK(read.size() == 37);
}

TEST_CASE("partial reads honor count, offset, and stride") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_partial");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> written = arftest::ramp(64);
        arf::dataset_ptr d = entry.create_dataset("pcm", written);

        SUBCASE("count") {
                std::vector<double> read(8, -1.0);
                d->read(&read[0], 8);
                CHECK(read[0] == written[0]);
                CHECK(read[7] == written[7]);
        }
        SUBCASE("offset") {
                std::vector<double> read(8, -1.0);
                d->read(&read[0], 8, 20);
                CHECK(read[0] == written[20]);
                CHECK(read[7] == written[27]);
        }
        SUBCASE("stride") {
                std::vector<double> read(8, -1.0);
                d->read(&read[0], 8, 0, 4);
                CHECK(read[1] == written[4]);
                CHECK(read[7] == written[28]);
        }
        SUBCASE("a pre-sized vector works, since it is never resized") {
                // the (vector, count, offset, stride) overload writes through
                // &data[0] without resizing -- see backlog item 3. Sized
                // correctly by the caller it behaves; sized short it overflows,
                // which is why no test passes it a short one.
                std::vector<double> read(8, -1.0);
                d->read(read, 8, 16);
                CHECK(read[0] == written[16]);
        }
}

TEST_CASE("writing resizes the dataset to match") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_rewrite");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(100));
        REQUIRE(d->dataspace()->size() == 100);

        std::vector<double> shorter = arftest::ramp(10);
        d->write(shorter);
        CHECK(d->dataspace()->size() == 10);

        std::vector<double> longer = arftest::ramp(500);
        d->write(longer);
        CHECK(d->dataspace()->size() == 500);

        std::vector<double> read;
        d->read(read);
        CHECK(read == longer);
}

TEST_CASE("set_extent grows and shrinks") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_extent");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(50));

        d->set_extent(std::vector<hsize_t>(1, 200));
        CHECK(d->dataspace()->size() == 200);
        d->set_extent(std::vector<hsize_t>(1, 5));
        CHECK(d->dataspace()->size() == 5);
}

TEST_CASE("datasets are chunked, and report their chunk shape") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_chunks");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(1 << 20));

        std::vector<hsize_t> chunks = d->chunks();
        REQUIRE(chunks.size() == 1);
        CHECK(chunks == arf::h5s::detail::guess_chunk(std::vector<hsize_t>(1, 1 << 20),
                                                      sizeof(double)));
}

TEST_CASE("every dataset carries a deflate filter, even at level zero") {
        // CHARACTERIZATION: create_dataset applies H5Pset_deflate whenever
        // compress > -1, and the default argument is 0. So the default is not
        // "no compression" but "deflate at level 0", which still writes filter
        // metadata and routes data through zlib. Only a negative value skips
        // the filter entirely.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_filters");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> data = arftest::ramp(4096);

        arf::dataset_ptr defaulted = entry.create_dataset("defaulted", data);
        CHECK(filter_count(*defaulted) == 1);

        arf::dataset_ptr squeezed = entry.create_dataset("squeezed", data, 9);
        CHECK(filter_count(*squeezed) == 1);

        arf::dataset_ptr raw = entry.create_dataset("raw", data, -1);
        CHECK(filter_count(*raw) == 0);
}

TEST_CASE("compression shrinks a compressible dataset") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_compress");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        std::vector<double> flat(1 << 16, 1.0);

        arf::dataset_ptr squeezed = entry.create_dataset("squeezed", flat, 9);
        arf::dataset_ptr raw = entry.create_dataset("raw", flat, -1);
        f.flush();
        CHECK(H5Dget_storage_size(squeezed->hid()) < H5Dget_storage_size(raw->hid()));
}

TEST_CASE("the storage type can differ from the memory type") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_convert");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        std::vector<float> written;
        for (int i = 0; i < 64; ++i) written.push_back(i * 0.25f);
        arf::dataset_ptr d = entry.create_dataset<double, float>("pcm", written);
        CHECK(d->datatype()->size() == sizeof(double));

        std::vector<double> read;
        d->read(read);
        REQUIRE(read.size() == written.size());
        CHECK(read[7] == doctest::Approx(written[7]));
}

TEST_CASE("opening a dataset that isn't there throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_missing");
        file f(scratch.path, "w");
        CHECK_THROWS_AS(dataset(f.hid(), "nope"), arf::Exception);
}

TEST_CASE("creating a dataset over an existing name throws") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_dup");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        entry.create_dataset("pcm", arftest::ramp(8));
        CHECK_THROWS_AS(entry.create_dataset("pcm", arftest::ramp(8)), arf::Exception);
}

TEST_CASE("an empty dataset cannot be created") {
        // CHARACTERIZATION: guess_chunk hands back a chunk of 0 for a
        // zero-length extent (see the h5s suite), and H5Pset_chunk rejects it.
        // Writing an empty channel therefore fails at creation time. arf.py
        // allows empty datasets -- test_arf.py has an "empty-spikes" case -- so
        // the two implementations disagree here.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_empty");
        file f(scratch.path, "w");
        group entry(f, "entry", true);

        std::vector<double> nothing;
        CHECK_THROWS_AS(entry.create_dataset("empty", nothing), arf::Exception);
}

TEST_CASE("introspection releases the handles it opens") {
        // dataset::datatype() hands the result of H5Dget_type to
        // h5t::datatype(hid_t), which used to *copy* it, leaving the original
        // unowned. dataspace() next door was always clean because
        // h5s::dataspace(hid_t) adopts. Both wrappers adopt now.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_typeleak");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(8));

        ssize_t before = arftest::handle_guard::open_handles();
        d->datatype();
        CHECK(arftest::handle_guard::open_handles() == before);

        before = arftest::handle_guard::open_handles();
        d->dataspace();
        CHECK(arftest::handle_guard::open_handles() == before);
}

TEST_CASE("datasets expose their dataspace and datatype") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("d_introspect");
        file f(scratch.path, "w");
        group entry(f, "entry", true);
        arf::dataset_ptr d = entry.create_dataset("pcm", arftest::ramp(12));

        CHECK(d->dataspace()->ndims() == 1);
        CHECK(d->dataspace()->size() == 12);
        CHECK(d->dataspace()->maxdims() == std::vector<hsize_t>(1, H5S_UNLIMITED));
        CHECK(d->datatype()->size() == sizeof(double));
}

}
