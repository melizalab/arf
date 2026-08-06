/* @file read_arf.cpp
 * @brief Read a file written by arf.py and report what the C++ library makes of it.
 *
 * The other direction of the interop test. Exits 0 when the file matches
 * expectations, 1 with detail on stderr otherwise.
 *
 * Some of what it asserts is current *broken* behavior, marked CHARACTERIZATION
 * below: arf.py stores strings as variable-length, and this library can only
 * read fixed-length ones. Those checks assert the breakage on purpose, so that
 * fixing it turns this program red rather than leaving the gap unnoticed.
 *
 *     read_arf <path>
 */

#include <cstdio>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "arf.hpp"

namespace {

int failures = 0;

void
check(bool ok, char const * what)
{
        if (!ok) {
                std::fprintf(stderr, "FAIL: %s\n", what);
                failures += 1;
        }
}

}

int
main(int argc, char ** argv)
{
        if (argc < 2) {
                std::fprintf(stderr, "usage: %s <path>\n", argv[0]);
                return 2;
        }

        try {
                arf::h5f::file f(argv[1], "r");

                // --- structure, which does travel between implementations ---

                std::vector<std::string> children = f.children();
                check(children.size() == 2, "expected two entries");
                // arf.py wrote these back to front, so creation order tracking
                // is what makes this differ from alphabetical
                check(children.size() == 2 && children[0] == "entry_001",
                      "entries should come back in creation order");

                for (std::size_t i = 0; i < children.size(); ++i) {
                        arf::entry e(f, children[i]);

                        check(e.has_attribute("timestamp"), "entry needs a timestamp");
                        check(e.has_attribute("uuid"), "entry needs a uuid");

                        std::vector<boost::int64_t> timestamp;
                        e.read_attribute("timestamp", timestamp);
                        check(timestamp.size() == 2,
                              "timestamp should have two elements");
                        check(timestamp.size() == 2 && timestamp[0] > 1000000000,
                              "timestamp should be seconds since the epoch");

                        check(e.contains("pcm"), "entry should contain pcm");
                        arf::h5d::dataset pcm(e.hid(), "pcm");
                        check(pcm.dataspace()->size() == 128,
                              "pcm should hold 128 samples");
                        check(pcm.read_attribute<int>("datatype") == arf::ACOUSTIC,
                              "pcm datatype should be ACOUSTIC");
                        check(pcm.read_attribute<int>("sampling_rate") == 20000,
                              "pcm should carry a sampling rate");

                        std::vector<double> samples;
                        pcm.read(samples);
                        check(samples.size() == 128, "should read 128 samples back");
                        check(samples.size() == 128 && samples[0] == 0.0,
                              "first sample should be 0");
                        check(samples.size() == 128 && samples[2] == 1.0,
                              "third sample should be 1.0");

                        check(e.contains("spikes"), "entry should contain spikes");
                        arf::h5d::dataset spikes(e.hid(), "spikes");
                        check(spikes.dataspace()->size() == 16,
                              "spikes should hold 16 times");
                }

                // --- strings, which arf.py stores as variable-length ---

                // These were unreadable until the reader learned to handle
                // variable-length strings: it assumed fixed-length, allocating
                // H5Tget_size() bytes -- 8, the size of a pointer -- and then
                // read the returned char* as if the buffer held the characters.
                // units is required on every dataset by the specification, so
                // this made arf.py's files largely uninterpretable here.
                check(f.read_attribute<std::string>("arf_library") == "python",
                      "root arf_library should say python");
                check(f.read_attribute<std::string>("arf_version").size() > 0,
                      "root arf_version should be readable");
                arf::entry first(f, children[0]);
                check(first.read_attribute<std::string>("animal") == "bird_042",
                      "entry animal should be readable");
                check(first.read_attribute<std::string>("experimenter") == "dmeliza",
                      "entry experimenter should be readable");
                arf::h5d::dataset pcm(first.hid(), "pcm");
                check(pcm.read_attribute<std::string>("units") == "mV",
                      "pcm units should be readable");
                arf::h5d::dataset spikes_units(first.hid(), "spikes");
                check(spikes_units.read_attribute<std::string>("units") == "s",
                      "spike units should be readable");

                // the uuid is fixed-length and exactly 36 bytes, with no room
                // for a terminator -- the case that used to read past the end
                check(first.has_attribute("uuid"), "entry needs a uuid");
                check(first.read_attribute<std::string>("uuid").size() == 36,
                      "uuid should read back as 36 characters");
        }
        catch (arf::Exception const & err) {
                std::fprintf(stderr, "FAIL: threw arf::Exception: %s\n", err.what());
                return 1;
        }
        catch (std::exception const & err) {
                std::fprintf(stderr, "FAIL: threw std::exception: %s\n", err.what());
                return 1;
        }

        if (failures > 0) {
                std::fprintf(stderr, "%d check(s) failed\n", failures);
                return 1;
        }
        std::printf("ok\n");
        return 0;
}
