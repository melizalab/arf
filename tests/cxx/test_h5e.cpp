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

TEST_CASE("an invalid return throws even when the stack is empty") {
        // The contract is "throw if and only if the return value is invalid".
        // auto_throw used to return 0 when nothing had been pushed onto the
        // stack, so a failure hdf5 declined to describe became a
        // plausible-looking success value -- which callers then assigned into
        // _self as an identifier.
        H5Eclear2(H5E_DEFAULT);
        CHECK_THROWS_AS(check_error(-1), arf::Exception);
        H5Eclear2(H5E_DEFAULT);
        CHECK_THROWS_AS(check_error(-9999), arf::Exception);
}

TEST_CASE("the message says whether hdf5 explained itself") {
        std::string described, undescribed;

        H5Eclear2(H5E_DEFAULT);
        try { check_error(-1); } catch (arf::Exception const & e) { undescribed = e.what(); }

        hid_t bad = failing_call();
        try { check_error(bad); } catch (arf::Exception const & e) { described = e.what(); }

        CHECK(undescribed.size() > 0);
        CHECK(described.size() > 0);
        // the described case carries hdf5's own text, so the two differ
        CHECK(described != undescribed);
        H5Eclear2(H5E_DEFAULT);
}

TEST_CASE("the bool overload throws whatever the stack holds") {
        // Whether `false` raised used to depend on whether some *earlier* call
        // had left an entry on the stack, which made == and != on datatypes
        // order-dependent. Now it is unconditional -- which is why those
        // operators check the htri_t from H5Tequal rather than passing the
        // comparison's result through here.
        H5Eclear2(H5E_DEFAULT);
        CHECK_THROWS_AS(check_error(false), arf::Exception);

        failing_call();
        CHECK_THROWS_AS(check_error(false), arf::Exception);
        H5Eclear2(H5E_DEFAULT);

        CHECK(check_error(true) == true);
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
