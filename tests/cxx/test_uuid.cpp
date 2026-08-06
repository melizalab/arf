/* @file test_uuid.cpp
 * @brief unit tests for arf/uuid.hpp
 */

#include <set>
#include <string>

#include "arf.hpp"
#include "fixtures.hpp"

TEST_SUITE("uuid") {

TEST_CASE("a default uuid is nil") {
        arf::uuid u;
        CHECK(u.is_nil());
        CHECK(u.str() == "00000000-0000-0000-0000-000000000000");
        CHECK(u == arf::uuid());
}

TEST_CASE("generated uuids are version 4, variant 1, and not nil") {
        arf::uuid u = arf::uuid::generate();
        CHECK_FALSE(u.is_nil());
        REQUIRE(u.size() == 16);

        // RFC 4122 4.4: version in the high nibble of octet 6
        CHECK((u.data()[6] >> 4) == 4);
        // and the variant bits are 10x in octet 8
        CHECK((u.data()[8] & 0xc0) == 0x80);
}

TEST_CASE("generated uuids do not repeat") {
        std::set<std::string> seen;
        for (int i = 0; i < 500; ++i) seen.insert(arf::uuid::generate().str());
        CHECK(seen.size() == 500);
}

TEST_CASE("the text form is 36 characters in 8-4-4-4-12 groups") {
        std::string s = arf::uuid::generate().str();
        REQUIRE(s.size() == arf::uuid::string_size);
        CHECK(s[8] == '-');
        CHECK(s[13] == '-');
        CHECK(s[18] == '-');
        CHECK(s[23] == '-');
        for (std::size_t i = 0; i < s.size(); ++i) {
                if (i == 8 || i == 13 || i == 18 || i == 23) continue;
                CAPTURE(i);
                CHECK(std::string("0123456789abcdef").find(s[i]) != std::string::npos);
        }
}

TEST_CASE("parsing round trips") {
        arf::uuid original = arf::uuid::generate();
        arf::uuid parsed = arf::uuid::parse(original.str());
        CHECK(parsed == original);
        CHECK(parsed.str() == original.str());
}

TEST_CASE("parsing accepts upper case and normalizes to lower") {
        std::string upper = "550E8400-E29B-41D4-A716-446655440000";
        arf::uuid u = arf::uuid::parse(upper);
        CHECK(u.str() == "550e8400-e29b-41d4-a716-446655440000");
}

TEST_CASE("a known value parses to the expected octets") {
        arf::uuid u = arf::uuid::parse("00112233-4455-6677-8899-aabbccddeeff");
        CHECK(u.data()[0] == 0x00);
        CHECK(u.data()[3] == 0x33);
        CHECK(u.data()[6] == 0x66);
        CHECK(u.data()[15] == 0xff);
}

TEST_CASE("malformed text is rejected") {
        CHECK_THROWS_AS(arf::uuid::parse(""), arf::Exception);
        CHECK_THROWS_AS(arf::uuid::parse("too short"), arf::Exception);
        // 36 characters, but the separators are in the wrong places
        CHECK_THROWS_AS(arf::uuid::parse("550e8400e29b-41d4-a716-4466554400-0"),
                        arf::Exception);
        // 36 characters, correct separators, but a non-hex digit
        CHECK_THROWS_AS(arf::uuid::parse("550e8400-e29b-41d4-a716-44665544000g"),
                        arf::Exception);
}

TEST_CASE("uuids compare by value") {
        arf::uuid a = arf::uuid::parse("550e8400-e29b-41d4-a716-446655440000");
        arf::uuid b = arf::uuid::parse("550e8400-e29b-41d4-a716-446655440000");
        arf::uuid c = arf::uuid::generate();
        CHECK(a == b);
        CHECK_FALSE(a != b);
        CHECK(a != c);
}

TEST_CASE("an entry's uuid survives the round trip through a file") {
        // the whole point: 36 characters out, 36 characters back
        arftest::handle_guard guard;
        arftest::scratch_file scratch("uuid_roundtrip");
        std::string written;
        {
                arf::file f(scratch.path, "w");
                arf::entry e(f, "entry_000", 1, 0);
                written = e.uuid().str();
                CHECK(written.size() == 36);
        }
        arf::h5f::file f(scratch.path, "r");
        arf::entry reopened(f, "entry_000");
        CHECK(reopened.uuid().str() == written);
        CHECK_FALSE(reopened.uuid().is_nil());
}

}
