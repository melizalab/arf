/* @file main.cpp
 * @brief doctest entry point for the arf C++ test suite
 *
 * Defines the test runner. Every other file in this directory only registers
 * test cases; this is the single translation unit that instantiates doctest.
 *
 * H5close() is called after the run so that HDF5 releases its internal free
 * lists. Without it, leak checkers report those caches as leaks from the
 * library rather than from anything the tests did.
 *
 * Copyright (C) 2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <hdf5.h>

#include "arf.hpp"

int
main(int argc, char ** argv)
{
        // many cases provoke hdf5 failures on purpose, and each would dump
        // an error stack to stderr. The library will not silence this on a
        // caller's behalf, so the runner asks for it.
        arf::h5e::silence_auto_print();

        doctest::Context context;
        context.applyCommandLine(argc, argv);
        int result = context.run();
        H5close();
        return result;
}
