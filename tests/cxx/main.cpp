/* @file main.cpp
 * @brief doctest entry point for the arf C++ test suite
 *
 * Defines the test runner. Every other file in this directory only registers
 * test cases; this is the single translation unit that instantiates doctest.
 *
 * H5close() is called after the run so that HDF5 releases its internal free
 * lists. Without it, leak checkers report those caches as leaks from the
 * library rather than from anything the tests did.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <hdf5.h>

int
main(int argc, char ** argv)
{
        doctest::Context context;
        context.applyCommandLine(argc, argv);
        int result = context.run();
        H5close();
        return result;
}
