/* @file h5e.hpp
 * @brief arf c++ interface: error handling
 *
 * Copyright (C) 2011-2026 C Daniel Meliza <dan||meliza.org>
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ARF_H5E_HPP
#define ARF_H5E_HPP

#include "hdf5.hpp"

namespace arf {

/**
 * @namespace h5e
 * @brief error handling
 *
 * The HDF5 library has its own error system; the functions and
 * classes in this namespace help to translate error messages from
 * this system in C++ exceptions.  Functions that make calls to HDF5
 * functions should pass the return value through check_error, which
 * will throw an exception if and only if the return value is invalid;
 * otherwise it will pass the return value.
 */
namespace h5e {


namespace detail {


/** The error callback just stores the last error on the stack */
inline int walk_cb(unsigned int, H5E_error2_t const * desc, void *data)
{
	H5E_error2_t *e = static_cast<H5E_error2_t*>(data);
	*e = *desc;
	return 0;
}

/**
 * Check the HDF5 error stack and throw Exception with a useful error
 * message if there are errors on the stack. In theory this can be
 * used as the auto handler, but that's a C function and we can't
 * throw exceptions back up to the caller.
 */
inline herr_t auto_throw(hid_t estack, void *) {
	H5E_error_t err;
	// An invalid return value is an error whether or not hdf5 saw fit to
	// describe it. Returning 0 for an undescribed failure would be worse
	// than useless: callers assign the result straight into _self as an
	// identifier, so a swallowed error becomes a plausible-looking handle.
	if (H5Eget_num(estack) > 0) {
		if (H5Ewalk2(estack, H5E_WALK_DOWNWARD, walk_cb, &err) >= 0 && err.desc)
			throw Exception(err.desc);
		throw Exception("hdf5 call failed; its error stack could not be read");
	}
	throw Exception("hdf5 call failed without reporting an error");
}

}

/**
 * Stop hdf5 printing its error stack to stderr.
 *
 * Failures are reported as exceptions regardless, so the printout is usually
 * redundant noise. This is *not* called for you: it changes a global setting
 * in a library the rest of the program may also be using, which is not a
 * decision a constructor should make on the caller's behalf. Call it once at
 * startup if you want it.
 */
inline void
silence_auto_print()
{
	H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
}

/**
 * Call this function on any returned HDF5 value to check for an error
 * and throw one if it exists.
 */
template <typename T> inline
T check_error(T retval)
{
	if (retval < 0)
		return detail::auto_throw(H5E_DEFAULT, nullptr);
	return retval;
}

template<> inline
bool check_error(bool retval)
{
	if (!retval)
		return detail::auto_throw(H5E_DEFAULT, nullptr);
	return retval;
}

}}

#endif /* ARF_H5E_HPP */
