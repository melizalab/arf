/* @file h5t.hpp
 * @brief C++ arf interface: datatypes
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5T_H
#define _H5T_H 1

#include <boost/static_assert.hpp>
#include <boost/integer_traits.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/uuid/uuid.hpp>
#include "hdf5.hpp"
#include "h5e.hpp"

namespace arf {
namespace h5t {

/**
 * Traits classes to convert C types to HDF5 types.  This approach is
 * adapted from the hdf5 C++ interface by James Sharpe, except it only
 * supports simple datatypes.
 */
namespace detail {

template<int ValueBits>
struct int_dtype_traits{};

template<>
struct int_dtype_traits<7> {
	static hid_t value() { return H5T_NATIVE_INT8; }
};

template<>
struct int_dtype_traits<8> {
	static hid_t value() { return H5T_NATIVE_UINT8; }
};

template<>
struct int_dtype_traits<15> {
	static hid_t value() { return H5T_NATIVE_INT16; }
};

template<>
struct int_dtype_traits<16> {
	static hid_t value() { return H5T_NATIVE_UINT16; }
};

template<>
struct int_dtype_traits<31> {
	static hid_t value() { return H5T_NATIVE_INT32; }
};

template<>
struct int_dtype_traits<32> {
	static hid_t value() { return H5T_NATIVE_UINT32; }
};

template<>
struct int_dtype_traits<63> {
	static hid_t value() { return H5T_NATIVE_INT64; }
};

template<>
struct int_dtype_traits<64> {
	static hid_t value() { return H5T_NATIVE_UINT64; }
};

template <typename T>
struct datatype_traits {
	typedef typename boost::integer_traits<T> traits;
	BOOST_STATIC_ASSERT((traits::is_integral));
	static hid_t value() { return H5Tcopy(int_dtype_traits<traits::digits>::value()); }
};

template<>
struct datatype_traits<std::string> {
	static hid_t value() {
                hid_t str = H5Tcopy(H5T_C_S1);
                // H5Tset_cset(str, H5T_CSET_UTF8);
                return str;
        }
};

template<>
struct datatype_traits<char const *> {
	static hid_t value() {
                hid_t str = H5Tcopy(H5T_C_S1);
                // H5Tset_cset(str, H5T_CSET_UTF8);
                return str;
        }
};

/**
 * uuids can be stored directly as a 128-bit integer, but the preferred format
 * in the specification is as a hex-encoded string.
 */
template<>
struct datatype_traits<boost::uuids::uuid> {
        static hid_t value() {
                hid_t v = H5Tcopy(H5T_NATIVE_CHAR); // 128-bit integer
                H5Tset_size(v, 16);
                return v;
        }
};

template<>
struct datatype_traits<char> {
	static hid_t value() { return H5Tcopy(H5T_NATIVE_CHAR); }
};

template<>
struct datatype_traits<float> {
	static hid_t value() { return H5Tcopy(H5T_NATIVE_FLOAT); }
};

template<>
struct datatype_traits<double> {
	static hid_t value() { return H5Tcopy(H5T_NATIVE_DOUBLE); }
};

} // detail namespace

/**
 * Use wrapper class to pass type as object without having to
 * instantiate the type itself.
 */
template <typename Type>
class wrapper {};

/**
 * Base class for HDF5 data types. This is a fairly simple wrapper
 * with copy semantics: on initialization the object creates a new
 * HDF5 handle and releases it on destruction.
 */
class datatype : public handle {
public:
	typedef boost::shared_ptr<datatype> ptr_type;

	/** Create a datatype from a C type */
	template <typename Type>
	explicit datatype(wrapper<Type>) {
		_self = h5e::check_error(detail::datatype_traits<typename boost::remove_cv<Type>::type>::value());
	}

	datatype(datatype const & other) {
		_self = h5e::check_error(H5Tcopy(other.hid()));
	}

	/**
	 * Take ownership of a datatype handle, as h5s::dataspace does. Callers
	 * holding a borrowed handle must H5Tcopy it themselves. This used to
	 * copy, which left the caller's handle unowned and leaking at every
	 * call site -- all of which pass a freshly returned handle.
	 */
	explicit datatype(hid_t dtype_id) : handle(h5e::check_error(dtype_id)) {}

	datatype & operator= (datatype const & other) {
		// release old handle
		H5Tclose(_self);
		_self = h5e::check_error(H5Tcopy(other.hid()));
		return *this;
	}

	~datatype() {
		H5Tclose(_self);
	}

	bool operator==(datatype const & other) const {
		return h5e::check_error(H5Tequal(_self, other.hid()) > 0);
	}
	bool operator!=(datatype const & other) const {
		return h5e::check_error(H5Tequal(_self, other.hid()) <= 0);
	}

	hsize_t size() const { return H5Tget_size(_self); }

	void set_size(hsize_t size) { h5e::check_error(H5Tset_size(_self, size)); }
};


} // namespace h5t
} // namespace arf


#endif /* _H5T_H */

