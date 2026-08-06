/* @file test_smoke.cpp
 * @brief Minimal end-to-end checks that the build and headers are sound.
 *
 */

#include "doctest.h"

#include <cstdio>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "arf.hpp"

namespace {

/** A unique scratch path, removed by the destructor. */
struct scratch_file {
        std::string path;

        explicit scratch_file(char const * tag) {
                std::ostringstream os;
                os << "/tmp/arf_smoke_" << tag << "_" << getpid() << ".arf";
                path = os.str();
                std::remove(path.c_str());
        }
        ~scratch_file() { std::remove(path.c_str()); }
};

}

TEST_SUITE("smoke") {

TEST_CASE("a new file carries the arf version attributes") {
        scratch_file scratch("file");
        arf::file f(scratch.path, "w");

        CHECK(f.read_attribute<std::string>("arf_version") == ARF_VERSION);
        CHECK(f.read_attribute<std::string>("arf_library") == "c++");
        CHECK(f.read_attribute<std::string>("arf_library_version") == ARF_LIBRARY_VERSION);
        CHECK(f.name() == scratch.path);
}

TEST_CASE("an entry stores its timestamp and uuid") {
        scratch_file scratch("entry");
        arf::file f(scratch.path, "w");

        arf::entry created(f, "entry_000", 1234567890, 500);
        REQUIRE(created.has_attribute("timestamp"));
        REQUIRE(created.has_attribute("uuid"));

        std::vector<boost::int64_t> timestamp;
        created.read_attribute("timestamp", timestamp);
        REQUIRE(timestamp.size() == 2);
        CHECK(timestamp[0] == 1234567890);
        CHECK(timestamp[1] == 500);

        // reopening reads the uuid back off the file rather than regenerating it
        arf::entry reopened(f, "entry_000");
        CHECK(reopened.uuid() == created.uuid());
}

TEST_CASE("a dataset round trips through an entry") {
        scratch_file scratch("dataset");
        std::vector<double> written;
        for (int i = 0; i < 1024; ++i) written.push_back(i * 0.5);

        arf::file f(scratch.path, "w");
        arf::entry e(f, "entry_000", 1, 0);
        // NB: no explicit template argument. create_dataset<double>(...) is
        // ambiguous when the storage and memory types coincide -- see the
        // phase 5 backlog in CLAUDE.md.
        arf::h5d::dataset d = e.create_dataset("pcm", written, "mV", arf::ACOUSTIC);
        d.write_attribute("sampling_rate", 20000);

        // NB: reads through the dataset, not through e.read_dataset(name, vec),
        // which is broken -- see the phase 5 backlog in CLAUDE.md.
        std::vector<double> read;
        d.read(read);
        REQUIRE(read.size() == written.size());
        CHECK(read == written);
}

TEST_CASE("check_error passes valid return values through") {
        CHECK(arf::h5e::check_error(3) == 3);
        CHECK(arf::h5e::check_error(0) == 0);
}

TEST_CASE("check_error throws on a negative return with an empty stack") {
        H5Eclear2(H5E_DEFAULT);
        CHECK_THROWS_AS(arf::h5e::check_error(-1), arf::Exception);
}

TEST_CASE("opening a missing file throws") {
        CHECK_THROWS_AS(arf::h5f::file("/tmp/arf_smoke_does_not_exist.arf", "r"),
                        arf::Exception);
}

TEST_CASE("guess_chunk returns a chunk no larger than the dataset") {
        std::vector<hsize_t> shape(1, 1u << 20);
        std::vector<hsize_t> chunks = arf::h5s::detail::guess_chunk(shape, sizeof(float));
        REQUIRE(chunks.size() == 1);
        CHECK(chunks[0] > 0);
        CHECK(chunks[0] <= shape[0]);
}

}
