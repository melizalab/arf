/* @file fixtures.hpp
 * @brief Shared helpers for the arf C++ test suite.
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ARF_TEST_FIXTURES_HH
#define ARF_TEST_FIXTURES_HH 1

#include <cstdio>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <hdf5.h>

#include "doctest.h"

namespace arftest {

/**
 * A unique scratch path, removed on destruction. Unique per process and per
 * tag, so cases can't collide even when the suite is run concurrently.
 */
struct scratch_file {
        std::string path;

        explicit scratch_file(char const * tag) {
                std::ostringstream os;
                os << "/tmp/arf_test_" << tag << "_" << getpid() << ".arf";
                path = os.str();
                std::remove(path.c_str());
        }
        ~scratch_file() { std::remove(path.c_str()); }

        /** True if the file exists on disk. */
        bool exists() const {
                FILE * fp = fopen(path.c_str(), "r");
                if (fp == 0) return false;
                fclose(fp);
                return true;
        }

private:
        scratch_file(scratch_file const &);
        scratch_file & operator=(scratch_file const &);
};

/**
 * Fails the enclosing test case if it leaks HDF5 identifiers.
 *
 * This is the only handle-leak check that runs everywhere: the sanitizers need
 * glibc, and valgrind is noisy against an hdf5 built without
 * --enable-using-memchecker. Declare one at the top of any case that opens
 * files, groups, datasets, or attributes.
 */
struct handle_guard {
        ssize_t before;

        handle_guard() : before(open_handles()) {}

        ~handle_guard() {
                ssize_t after = open_handles();
                // CHECK, not REQUIRE: this runs in a destructor, and REQUIRE
                // throws
                CHECK_MESSAGE(after == before,
                              "leaked hdf5 handles: " << before << " open before the case, "
                                                      << after << " after");
        }

        static ssize_t open_handles() {
                return H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL);
        }

private:
        handle_guard(handle_guard const &);
        handle_guard & operator=(handle_guard const &);
};

/** A ramp, for data that is easy to eyeball in a failure message. */
inline std::vector<double>
ramp(std::size_t n, double step = 0.5)
{
        std::vector<double> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) out.push_back(i * step);
        return out;
}

}  // namespace arftest

#endif /* ARF_TEST_FIXTURES_HH */
