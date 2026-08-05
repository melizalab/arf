/* @file test_h5e.cpp
 * @brief unit tests for arf/h5e.hpp -- error handling
 */

#include <string>

#include "arf.hpp"
#include "fixtures.hpp"

using arf::h5e::check_error;

namespace {

/** Make a real HDF5 call fail, leaving something on the error stack. */
hid_t
failing_call()
{
        H5Eset_auto2(H5E_DEFAULT, 0, 0);
        return H5Fopen("/tmp/arf_no_such_file_exists.arf", H5F_ACC_RDONLY, H5P_DEFAULT);
}

}

TEST_SUITE("h5e") {

TEST_CASE("valid return values pass through untouched") {
        H5Eclear2(H5E_DEFAULT);
        CHECK(check_error(0) == 0);
        CHECK(check_error(1) == 1);
        CHECK(check_error(12345) == 12345);
        CHECK(check_error(true) == true);
}

TEST_CASE("a failed hdf5 call becomes an arf::Exception") {
        hid_t bad = failing_call();
        REQUIRE(bad < 0);
        CHECK_THROWS_AS(check_error(bad), arf::Exception);
}

TEST_CASE("the exception carries hdf5's own message") {
        hid_t bad = failing_call();
        REQUIRE(bad < 0);
        std::string message;
        try {
                check_error(bad);
                FAIL("expected check_error to throw");
        }
        catch (arf::Exception const & err) {
                message = err.what();
        }
        CHECK(message.size() > 0);
        // and it is a std::runtime_error, so callers can catch it generically
        CHECK_THROWS_AS(check_error(failing_call()), std::runtime_error);
}

TEST_CASE("a negative return with an empty error stack is swallowed") {
        // CHARACTERIZATION: known bug, see backlog item B. The contract is
        // "throw if and only if the return value is invalid", but auto_throw()
        // returns 0 when nothing was pushed onto the stack, so a failure that
        // did not go through hdf5 turns into a plausible-looking success value.
        H5Eclear2(H5E_DEFAULT);
        CHECK(check_error(-1) == 0);
        H5Eclear2(H5E_DEFAULT);
        CHECK(check_error(-9999) == 0);
}

TEST_CASE("the bool overload throws only when the stack is dirty") {
        // CHARACTERIZATION: same defect as above, seen through the bool
        // specialization. Whether `false` raises depends on whether some
        // *earlier* hdf5 call happened to leave an entry on the stack, which
        // makes the behavior of == and != on datatypes order-dependent.
        H5Eclear2(H5E_DEFAULT);
        CHECK(check_error(false) == false);

        failing_call();  // dirties the stack
        CHECK_THROWS_AS(check_error(false), arf::Exception);
        H5Eclear2(H5E_DEFAULT);
}

TEST_CASE("checking errors leaks no handles") {
        arftest::handle_guard guard;
        H5Eclear2(H5E_DEFAULT);
        CHECK(check_error(1) == 1);
        hid_t bad = failing_call();
        CHECK_THROWS_AS(check_error(bad), arf::Exception);
        H5Eclear2(H5E_DEFAULT);
}

}
