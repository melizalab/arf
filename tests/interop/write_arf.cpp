/* @file write_arf.cpp
 * @brief Write a canonical arf file with the C++ library.
 *
 * Used two ways: as the input to the structural golden comparison, and as the
 * file that arf.py is asked to read in the interop tests. Everything it writes
 * is fixed, so the only thing that varies between runs is the uuid of each
 * entry, which the dump script redacts.
 *
 *     write_arf <path>
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "arf.hpp"

namespace {

int const nentries = 2;
int const nsamples = 128;
int const nspikes = 16;
int const nintervals = 4;

struct interval {
        char name[32];
        boost::uint32_t start;
        boost::uint32_t stop;
};

}

namespace arf { namespace h5t { namespace detail {

template<>
struct datatype_traits<interval> {
        static hid_t value() {
                hid_t ret = H5Tcreate(H5T_COMPOUND, sizeof(interval));
                hid_t str = H5Tcopy(H5T_C_S1);
                H5Tset_size(str, 32);
                H5Tinsert(ret, "name", HOFFSET(interval, name), str);
                H5Tinsert(ret, "start", HOFFSET(interval, start), H5T_STD_U32LE);
                H5Tinsert(ret, "stop", HOFFSET(interval, stop), H5T_STD_U32LE);
                H5Tclose(str);
                return ret;
        }
};

}}}

int
main(int argc, char ** argv)
{
        arf::h5e::silence_auto_print();

        if (argc < 2) {
                std::fprintf(stderr, "usage: %s <path>\n", argv[0]);
                return 2;
        }

        try {
                arf::file f(argv[1], "w");

                // deliberately back to front, so that creation order and name
                // order disagree and the golden dump can prove which one the
                // file preserves
                for (int i = nentries - 1; i >= 0; --i) {
                        char name[32];
                        std::sprintf(name, "entry_%03d", i);
                        arf::entry e(f, name, 1234567890 + i, 1000 * i);
                        e.write_attribute()
                                ("animal", "bird_042")
                                ("experimenter", "dmeliza")
                                ("protocol", "playback");

                        // sampled data: a ramp, so the values are obvious
                        std::vector<double> pcm;
                        for (int j = 0; j < nsamples; ++j)
                                pcm.push_back(j * 0.5 + i);
                        arf::dataset_ptr sampled =
                                e.create_dataset("pcm", pcm, "mV", arf::ACOUSTIC);
                        sampled->write_attribute("sampling_rate", 20000);

                        // simple event data: times in seconds
                        std::vector<double> spikes;
                        for (int j = 0; j < nspikes; ++j)
                                spikes.push_back(j * 0.01 + i);
                        e.create_dataset("spikes", spikes, "s", arf::SPIKET);

                        // simple event data on a discrete timebase: the spec
                        // requires units of "samples" plus a sampling_rate
                        std::vector<boost::int32_t> ticks;
                        for (int j = 0; j < nspikes; ++j)
                                ticks.push_back(j * 200 + i);
                        arf::dataset_ptr discrete =
                                e.create_dataset("spike_samples", ticks, "samples",
                                                 arf::SPIKET);
                        discrete->write_attribute("sampling_rate", 20000);

                        // complex event data: compound records with a start
                        // field. The spec wants one unit per field, in field
                        // order -- the label carries none.
                        std::vector<std::string> interval_units;
                        interval_units.push_back("");
                        interval_units.push_back("ms");
                        interval_units.push_back("ms");
                        arf::packet_table_ptr intervals =
                                e.create_packet_table<interval>("intervals", interval_units,
                                                                arf::STIMI);
                        for (int j = 0; j < nintervals; ++j) {
                                interval record;
                                std::memset(&record, 0, sizeof(record));
                                std::sprintf(record.name, "stim_%02d", j);
                                record.start = 100 * j;
                                record.stop = 100 * j + 50;
                                intervals->write(&record, 1);
                        }
                }
        }
        catch (arf::Exception const & err) {
                std::fprintf(stderr, "arf error: %s\n", err.what());
                return 1;
        }
        return 0;
}
