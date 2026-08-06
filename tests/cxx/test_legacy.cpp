/* @file test_legacy.cpp
 * @brief the pre-doctest tests/test_arf.cpp scenario, with real assertions
 *
 * The original was a single main() driven by bare assert(), which compiled
 * away under NDEBUG. Everything it covered is either here or in the per-header
 * suites; this file keeps the part the unit tests don't reach, which is the
 * bulk round trip: many entries, each carrying attributes, a converted sampled
 * dataset, a packet table, and a table of compound records, written in one
 * order and read back in another.
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "arf.hpp"
#include "fixtures.hpp"

namespace {

// the original used 256; the point is bulk, not any particular count, and a
// smaller number keeps the sanitizer job quick
int const nentries = 64;
int const nsamples = 1 << 12;
int const npackets = 5;

struct interval {
        char name[64];
        boost::uint32_t start;
        boost::uint32_t stop;
};

std::string
entry_name(int i)
{
        char buf[64];
        std::sprintf(buf, "entry_%03d", i);
        return buf;
}

/** Deterministic stand-in for the original's nrand48() noise. */
std::vector<float>
samples()
{
        std::vector<float> out;
        out.reserve(nsamples);
        for (int i = 0; i < nsamples; ++i)
                out.push_back(static_cast<float>((i * 2654435761u) % 10007) / 10007.0f);
        return out;
}

std::vector<int>
vector_attribute()
{
        return std::vector<int>(5, 10);
}

}

namespace arf { namespace h5t { namespace detail {

template<>
struct datatype_traits<interval> {
        static hid_t value() {
                hid_t ret = H5Tcreate(H5T_COMPOUND, sizeof(interval));
                hid_t str = H5Tcopy(H5T_C_S1);
                H5Tset_size(str, 64);
                H5Tinsert(ret, "name", HOFFSET(interval, name), str);
                H5Tinsert(ret, "start", HOFFSET(interval, start), H5T_STD_U32LE);
                H5Tinsert(ret, "stop", HOFFSET(interval, stop), H5T_STD_U32LE);
                H5Tclose(str);
                return ret;
        }
};

}}}

TEST_SUITE("legacy") {

TEST_CASE("a file of many entries round trips") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("legacy_bulk");
        std::vector<float> const data = samples();

        {
                arf::file f(scratch.path, "w");

                // written back to front, so that creation order and name order
                // disagree and lookups can't accidentally rely on either
                for (int i = 0; i < nentries; ++i) {
                        std::string name = entry_name(nentries - i - 1);
                        arf::entry e(f, name, 1234567890 + i, i);

                        e.write_attribute()
                                ("intattr", 1)
                                ("vecattr", vector_attribute())
                                ("strattr", "blahdeblah");

                        // stored as double though held as float, which is also
                        // what keeps this call out of the ambiguity in item C
                        arf::h5d::dataset d =
                                e.create_dataset<double>("dataset", data, "mV",
                                                         arf::ACOUSTIC);
                        d.write_attribute("sampling_rate", 1000);

                        arf::h5pt::packet_table pt =
                                e.create_packet_table<float>("apackettable", "mV",
                                                             arf::ACOUSTIC);
                        pt.write_attribute("sampling_rate", 1000);
                        for (int p = 0; p < npackets; ++p) pt.write(data);

                        arf::h5pt::packet_table intervals =
                                e.create_packet_table<interval>("intervals", "ms",
                                                                arf::STIMI);
                        for (int p = 0; p < npackets; ++p) {
                                interval record;
                                std::memset(&record, 0, sizeof(record));
                                std::sprintf(record.name, "label_%03d", p);
                                record.start = 100 * p;
                                record.stop = 100 * p + 123;
                                intervals.write(&record, 1);
                        }
                }
                CHECK(f.nchildren() == static_cast<hsize_t>(nentries));
                CHECK(f.children().size() == static_cast<std::size_t>(nentries));

                // a second handle on the open file agrees about the contents
                arf::h5f::file second(scratch.path, "a");
                CHECK(second.children() == f.children());
        }

        {
                arf::h5f::file f(scratch.path, "r");
                CHECK(f.nchildren() == static_cast<hsize_t>(nentries));

                // read front to back: the opposite of the write order
                for (int i = 0; i < nentries; ++i) {
                        arf::entry e(f, entry_name(i));

                        CHECK(e.read_attribute<int>("intattr") == 1);
                        CHECK(e.read_attribute<std::string>("strattr") == "blahdeblah");
                        std::vector<int> vec;
                        e.read_attribute("vecattr", vec);
                        CHECK(vec == vector_attribute());

                        std::vector<boost::int64_t> timestamp;
                        e.read_attribute("timestamp", timestamp);
                        REQUIRE(timestamp.size() == 2);
                        CHECK(timestamp[0] == 1234567890 + (nentries - i - 1));

                        std::vector<double> sampled(nsamples, -1.0);
                        e.read_dataset("dataset", &sampled[0], nsamples);
                        CHECK(sampled[0] == doctest::Approx(data[0]));
                        CHECK(sampled[nsamples - 1] == doctest::Approx(data[nsamples - 1]));

                        // the packet table holds npackets copies; read the
                        // second one back, as the original did
                        std::vector<float> packet(nsamples, -1.0f);
                        e.read_dataset("apackettable", &packet[0], nsamples, nsamples);
                        CHECK(packet == data);
                }
        }
}

TEST_CASE("entries keep their uuid across a close and reopen") {
        arftest::handle_guard guard;
        arftest::scratch_file scratch("legacy_uuid");
        std::string first;
        {
                arf::file f(scratch.path, "w");
                arf::entry e(f, "entry_000", 1, 0);
                first = boost::uuids::to_string(e.uuid());
        }
        arf::h5f::file f(scratch.path, "r");
        arf::entry reopened(f, "entry_000");
        CHECK(boost::uuids::to_string(reopened.uuid()) == first);
}

}
