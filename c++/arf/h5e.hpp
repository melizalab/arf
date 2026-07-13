/* @file h5e.hpp
 * @brief arf c++ interface: error handling
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5E_H
#define _H5E_H 1

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
static int walk_cb(unsigned int n, H5E_error2_t const * desc, void *data)
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
static herr_t auto_throw(hid_t estack, void*) {
	H5E_error_t err;
	if (H5Eget_num(estack)<=0)
		return 0;

	if (H5Ewalk2(estack, H5E_WALK_DOWNWARD, walk_cb, &err) < 0)
		throw Exception("Failed to walk error stack");

	if (err.desc)
		throw Exception(err.desc);
	else
		throw Exception("Failed to extract detailed error description");
	return 0;
}

}

/**
 * Call this function on any returned HDF5 value to check for an error
 * and throw one if it exists.
 */
template <typename T> inline
T check_error(T retval)
{
	if (retval < 0)
		return detail::auto_throw(H5E_DEFAULT,0);
	return retval;
}

template<>
bool check_error(bool retval)
{
	if (!retval)
		return detail::auto_throw(H5E_DEFAULT,0);
	return retval;
}

}}

#endif /* _H5E_H */

