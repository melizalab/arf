/* @file test_version.cpp
 * @brief unit tests for arf/version.hpp -- specification version checking
 *
 * The library writes one specification version and reads a range. The range's
 * upper bound is derived from what is implemented rather than hard-coded, so a
 * later *minor* revision of the specification needs no release here: by the
 * specification's own rule, a minor revision cannot change or remove a
 * required attribute.
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <string>

#include "arf.hpp"
#include "fixtures.hpp"

namespace {

/** A bare hdf5 file carrying whatever version attribute we want to test. */
void
write_version(std::string const & path, char const * declared)
{
        arf::h5f::file f(path, "w");
        if (declared) f.write_attribute("arf_version", declared);
}

}

TEST_SUITE("version") {

TEST_CASE("versions parse and order by major then minor") {
        using arf::spec_version;
        CHECK(spec_version::parse("2.2").major == 2);
        CHECK(spec_version::parse("2.2").minor == 2);
        CHECK(spec_version::parse("2.10").minor == 10);
        // trailing components are ignored; only major.minor decides
        CHECK(spec_version::parse("2.2.3") == spec_version(2, 2));

        CHECK(spec_version(2, 1) < spec_version(2, 2));
        CHECK(spec_version(2, 9) < spec_version(3, 0));
        CHECK(spec_version(2, 10) >= spec_version(2, 9));
        CHECK(spec_version(3, 0) >= spec_version(2, 99));
}

TEST_CASE("malformed versions are rejected") {
        CHECK_THROWS_AS(arf::spec_version::parse(""), arf::Exception);
        CHECK_THROWS_AS(arf::spec_version::parse("2"), arf::Exception);
        CHECK_THROWS_AS(arf::spec_version::parse("2."), arf::Exception);
        CHECK_THROWS_AS(arf::spec_version::parse(".2"), arf::Exception);
        CHECK_THROWS_AS(arf::spec_version::parse("two.two"), arf::Exception);
}

TEST_CASE("the supported range is derived from what is implemented") {
        std::pair<arf::spec_version, arf::spec_version> range =
                arf::supported_spec_versions();
        CHECK(range.first == arf::spec_version::parse(ARF_MIN_SPEC_VERSION));
        // the next major after the implemented version, not a literal
        CHECK(range.second == arf::spec_version(
                      arf::spec_version::parse(ARF_VERSION).major + 1, 0));
        CHECK(arf::spec_version::parse(ARF_VERSION) < range.second);
        CHECK(arf::spec_version::parse(ARF_VERSION) >= range.first);
}

TEST_CASE("a file this library wrote passes its own check") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_own");
        {
                arf::file f(scratch.path, "w");
        }
        arf::h5f::file f(scratch.path, "r");
        CHECK(arf::check_file_version(f) == arf::spec_version::parse(ARF_VERSION));
}

TEST_CASE("every version in the supported range is accepted") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_range");
        char const * accepted[] = { "2.0", "2.1", "2.2" };
        for (std::size_t i = 0; i < sizeof(accepted) / sizeof(*accepted); ++i) {
                CAPTURE(accepted[i]);
                write_version(scratch.path, accepted[i]);
                arf::h5f::file f(scratch.path, "r");
                CHECK(arf::check_file_version(f) == arf::spec_version::parse(accepted[i]));
        }
}

TEST_CASE("a later minor revision is accepted without a release") {
        // the point of deriving the upper bound: 2.9 does not exist yet, and
        // by the specification's own rule it cannot have changed anything
        // required, so this library reads it
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_future_minor");
        write_version(scratch.path, "2.9");
        arf::h5f::file f(scratch.path, "r");
        CHECK(arf::check_file_version(f) == arf::spec_version(2, 9));
}

TEST_CASE("a later major revision is refused") {
        // a major revision may change required attributes, so nothing written
        // before it exists can be trusted to read it
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_future_major");
        write_version(scratch.path, "3.0");
        arf::h5f::file f(scratch.path, "r");
        CHECK_THROWS_AS(arf::check_file_version(f), arf::Exception);
}

TEST_CASE("a file predating the supported minimum is refused") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_old");
        write_version(scratch.path, "1.1");
        arf::h5f::file f(scratch.path, "r");
        CHECK_THROWS_AS(arf::check_file_version(f), arf::Exception);
}

TEST_CASE("a file with no version attribute is refused") {
        // NB: unlike arf.py this does not fall back to arf_library_version.
        // That attribute is provenance, and since 3.0 the libraries version
        // independently of the specification, so it cannot stand in for one.
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_none");
        write_version(scratch.path, 0);
        arf::h5f::file f(scratch.path, "r");
        CHECK_THROWS_AS(arf::check_file_version(f), arf::Exception);
}

TEST_CASE("the check is advisory: opening a file never runs it") {
        // someone holding a future-spec file can still read it by not asking
        arftest::handle_guard guard;
        arftest::scratch_file scratch("ver_advisory");
        write_version(scratch.path, "9.9");
        arf::h5f::file f(scratch.path, "r");
        CHECK(f.read_attribute<std::string>("arf_version") == "9.9");
        CHECK_THROWS_AS(arf::check_file_version(f), arf::Exception);
}

}
