/* @file version.hpp
 * @brief checking a file's specification version
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ARF_VERSION_HPP
#define ARF_VERSION_HPP

#include <string>
#include <utility>

#include "types.hpp"
#include "hdf5.hpp"
#include "h5a.hpp"

namespace arf {

/**
 * A specification version, compared by major and minor number.
 *
 * Only the two leading components matter: the specification versions
 * semantically, so a difference in major number means required attributes may
 * have changed, while a difference in minor number cannot.
 */
struct spec_version {
        int major;
        int minor;

        explicit spec_version(int major_ = 0, int minor_ = 0)
                : major(major_), minor(minor_) {}

        /**
         * Parse a leading "major.minor" from text, ignoring anything after.
         *
         * @throws Exception if the text does not start with two numbers
         */
        static spec_version parse(std::string const & text) {
                std::size_t dot = text.find('.');
                if (dot == std::string::npos || dot == 0 || dot + 1 >= text.size())
                        throw Exception("version is not in major.minor form");
                spec_version out;
                try {
                        out.major = std::stoi(text.substr(0, dot));
                        out.minor = std::stoi(text.substr(dot + 1));
                }
                catch (std::exception const &) {
                        throw Exception("version is not in major.minor form");
                }
                return out;
        }

        std::string str() const {
                return std::to_string(major) + "." + std::to_string(minor);
        }

        bool operator<(spec_version const & other) const {
                if (major != other.major) return major < other.major;
                return minor < other.minor;
        }
        bool operator>=(spec_version const & other) const { return !(*this < other); }
        bool operator==(spec_version const & other) const {
                return major == other.major && minor == other.minor;
        }
};

/**
 * The range of specification versions this library reads, [first, second).
 *
 * The upper bound is the next major version after the one implemented, not a
 * fixed number: a minor revision cannot change or remove a required attribute,
 * so files written to any later 2.x are readable without a new release. A 3.0
 * file is not, and never will be from this release.
 */
inline std::pair<spec_version, spec_version>
supported_spec_versions()
{
        spec_version implemented = spec_version::parse(ARF_VERSION);
        return std::make_pair(spec_version::parse(ARF_MIN_SPEC_VERSION),
                              spec_version(implemented.major + 1, 0));
}

/**
 * Check that a file claims a specification version this library understands.
 *
 * Advisory, and not called when opening a file: a caller who knows what they
 * are doing can read a file this would decline. Note that unlike arf.py, this
 * does not fall back to arf_library_version -- that attribute is provenance,
 * and since 3.0 the libraries version independently of the specification, so it
 * cannot stand in for one.
 *
 * @return the version the file claims
 * @throws Exception if the file declares no version, or one outside the
 *         supported range
 */
inline spec_version
check_file_version(h5a::node const & file)
{
        if (!file.has_attribute("arf_version"))
                throw Exception("file declares no arf_version; "
                                "was it written by another program?");
        std::string declared;
        file.read_attribute("arf_version", declared);
        spec_version claimed = spec_version::parse(declared);

        std::pair<spec_version, spec_version> range = supported_spec_versions();
        if (claimed < range.first)
                throw Exception("file predates ARF specification "
                                ARF_MIN_SPEC_VERSION
                                ", when the required attributes changed; "
                                "the arfx package can upgrade it");
        if (claimed >= range.second)
                throw Exception("file postdates ARF specification " ARF_VERSION
                                "; a major revision may change required "
                                "attributes, so upgrade this library to read it");
        return claimed;
}

}

#endif /* ARF_VERSION_HPP */
