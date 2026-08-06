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

#include <algorithm>
#include <limits>
#include <type_traits>
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
	static_assert(std::numeric_limits<T>::is_integer,
		      "no datatype_traits specialization for this type");
	static hid_t value() {
		return H5Tcopy(int_dtype_traits<std::numeric_limits<T>::digits>::value());
	}
};

/**
 * Strings are fixed-length and declared UTF-8.
 *
 * The specification constrains the class and CTYPE, and requires the declared
 * CSET to match the encoding -- which ASCII did not, once a std::string held
 * UTF-8. It does not constrain fixed versus variable length, except for uuid,
 * where it demands a 36-byte string. Fixed-length is what this library writes:
 * the characters live inline in the object header rather than on the file's
 * global heap, with no pointer indirection and no per-read allocation, which
 * matters because this library runs during acquisition. Variable-length
 * attributes, which arf.py writes, are still *readable* -- see attribute::read.
 *
 * The width is set per value when the attribute is created; see
 * node::write_attribute.
 */
template<>
struct datatype_traits<std::string> {
	static hid_t value() {
                hid_t str = H5Tcopy(H5T_C_S1);
                H5Tset_cset(str, H5T_CSET_UTF8);
                H5Tset_strpad(str, H5T_STR_NULLPAD);
                return str;
        }
};

template<>
struct datatype_traits<char const *> {
	static hid_t value() {
                hid_t str = H5Tcopy(H5T_C_S1);
                H5Tset_cset(str, H5T_CSET_UTF8);
                H5Tset_strpad(str, H5T_STR_NULLPAD);
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
	/** Create a datatype from a C type */
	template <typename Type>
	explicit datatype(wrapper<Type>)
		: handle(h5e::check_error(
				 detail::datatype_traits<typename std::remove_cv<Type>::type>::value())) {}

	/**
	 * Take ownership of a datatype handle, as h5s::dataspace does. Callers
	 * holding a borrowed handle must H5Tcopy it themselves.
	 */
	explicit datatype(hid_t dtype_id) : handle(h5e::check_error(dtype_id)) {}

	/** Copying duplicates the description with H5Tcopy. */
	datatype(datatype const & other)
		: handle(h5e::check_error(H5Tcopy(other.hid()))) {}

	datatype(datatype && other) = default;

	/**
	 * Copy-and-swap: one operator serves copy and move assignment, and
	 * self-assignment is safe. Assigning used to close _self and then copy
	 * from other, so `a = a` copied from the handle it had just closed.
	 */
	datatype & operator= (datatype other) {
		swap(other);
		return *this;
	}

	void swap(datatype & other) { std::swap(_self, other._self); }

	hsize_t size() const { return H5Tget_size(_self); }

	void set_size(hsize_t size) { h5e::check_error(H5Tset_size(_self, size)); }

	bool operator==(datatype const & other) const {
		return h5e::check_error(H5Tequal(_self, other.hid())) > 0;
	}
	bool operator!=(datatype const & other) const {
		return h5e::check_error(H5Tequal(_self, other.hid())) <= 0;
	}

};


} // namespace h5t
} // namespace arf


#endif /* _H5T_H */

