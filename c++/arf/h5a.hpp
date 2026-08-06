/* @file h5a.hpp
 * @brief C++ arf interface: attributes
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef ARF_H5A_HPP
#define ARF_H5A_HPP

#include <algorithm>
#include <cstring>
#include <vector>
#include "hdf5.hpp"
#include "h5t.hpp"
#include "h5s.hpp"

namespace arf {

namespace h5a {

/** Base class for HDF5 attributes */
class attribute : public handle {

public:
	/** Open an existing attribute */
	attribute(hid_t parent, std::string const & name)
		: handle(h5e::check_error(H5Aopen(parent, name.c_str(), H5P_DEFAULT))) {}

        attribute(handle const & parent, std::string const & name)
                : handle(h5e::check_error(H5Aopen(parent.hid(), name.c_str(), H5P_DEFAULT))) {}

	/** Open an existing attribute or create a new attribute */
	attribute(hid_t parent, std::string const & name,
		  h5s::dataspace const & dspace,
		  h5t::datatype const & type) {
		if (H5Aexists(parent, name.c_str()) > 0)
			_self = h5e::check_error(H5Aopen(parent, name.c_str(), H5P_DEFAULT));
		else {
			_self = h5e::check_error(H5Acreate2(parent, name.c_str(),
                                                            type.hid(), dspace.hid(),
                                                            H5P_DEFAULT, H5P_DEFAULT));
		}
	}

	attribute(attribute && other) = default;
	attribute & operator=(attribute && other) = default;
	attribute(attribute const &) = delete;
	attribute & operator=(attribute const &) = delete;

	/** assign value to attribute. automatic type conversion */
	template <typename Type>
	void write(Type const & value) {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
		h5e::check_error(H5Awrite(_self, type.hid(), &value));
	}

	template <typename Type>
	void write(Type const * arr, std::size_t size) {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
		// H5Awrite always writes the attribute's whole extent, so a
		// buffer of any other length is a mistake: a shorter one gets
		// read past its end. Throwing rather than asserting, because an
		// assert says nothing in the build where it matters.
		if (dataspace().size() != static_cast<hsize_t>(size))
			throw Exception("attribute write size does not match its extent");
		h5e::check_error(H5Awrite(_self, type.hid(), arr));
	}

	template <typename Type>
	void write(std::vector<Type> const & value) {
                write(value.data(), value.size());
	}

	/**
	 * Write a string. This library creates fixed-length attributes sized to
	 * their value, but an attribute written by another implementation may
	 * be variable-length, so both are handled.
	 */
	void write(std::string const & value) {
		// the wrapper owns the handle, so it is released even if the
		// write throws
		h5t::datatype type(h5e::check_error(H5Aget_type(_self)));
		if (H5Tis_variable_str(type.hid()) > 0) {
			// hdf5 wants the address of the pointer, not the
			// characters
			char const * p = value.c_str();
			h5e::check_error(H5Awrite(_self, type.hid(),
						  static_cast<void const *>(&p)));
			return;
		}
		// A fixed-length write consumes exactly the datatype's width.
		// The common case is an attribute this library just sized to
		// fit, where the value can go straight across; only a wider
		// attribute needs a padded copy.
		std::size_t width = type.size();
		if (width <= value.size()) {
			h5e::check_error(H5Awrite(_self, type.hid(), value.data()));
			return;
		}
		std::vector<char> buf(width, '\0');
		std::memcpy(buf.data(), value.data(), value.size());
		h5e::check_error(H5Awrite(_self, type.hid(), buf.data()));
	}


	/** read value from attribute. automatic type conversion */
        template <typename Type>
        Type read() const {
                Type t;
                read(t);
                return t;
        }


	template <typename Type>
	void read(Type & out) const {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
		h5e::check_error(H5Aread(_self, type.hid(), &out));
	}

	template <typename Type>
	void read(std::vector<Type> & out) const {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
                out.resize(dataspace().size());
		h5e::check_error(H5Aread(_self, type.hid(), out.data()));
	}

	/**
	 * Read a string attribute, whatever its storage.
	 *
	 * Three forms turn up in arf files and all of them must work:
	 * variable-length, which is what arf.py writes for most attributes;
	 * fixed-length with no terminator, which is what arf.py writes for a
	 * uuid; and fixed-length with one, which is what this library writes.
	 */
	void read(std::string & str) const {
		// the wrapper owns the handle, so it is released on the throw
		// path as well as the success path
		h5t::datatype type(h5e::check_error(H5Aget_type(_self)));
		if (H5Tget_class(type.hid())!=H5T_STRING)
			throw Exception("Attempt to read non-string attribute into string");

		if (H5Tis_variable_str(type.hid()) > 0) {
			// hdf5 allocates the characters and hands us the pointer
			char * buf = nullptr;
			herr_t rc = H5Aread(_self, type.hid(), static_cast<void *>(&buf));
			if (buf) {
				str.assign(buf);
				H5free_memory(buf);
			}
			else {
				str.clear();
			}
			h5e::check_error(rc);
			return;
		}

		std::size_t size = type.size();
		std::vector<char> buf(size, '\0');
		h5e::check_error(H5Aread(_self, type.hid(), buf.data()));
		// take everything up to the first NUL, or the whole buffer if
		// there isn't one. Scanning for a terminator that may not be
		// there would read past the end.
		std::size_t len = 0;
		while (len < size && buf[len] != '\0') ++len;
		str.assign(buf.data(), len);
	}

	/** Return the attribute's dataspace */
	h5s::dataspace dataspace() const {
		return h5s::dataspace(h5e::check_error(H5Aget_space(_self)));
	}

	/** Return the name of the attribute */
	std::string name() const {
		ssize_t sz = H5Aget_name(_self, 0, nullptr);
		if (sz <= 0) return std::string();
		std::vector<char> buf(sz + 1, '\0');
		H5Aget_name(_self, buf.size(), buf.data());
		return std::string(buf.data());
	}

private:

};


/** Base class for any HDF5 node (an object that can have attributes) */
class node : public handle {

public:

	node(node && other) = default;
	node & operator=(node && other) = default;
	node(node const &) = delete;
	node & operator=(node const &) = delete;


        /** Determine whether an attribute with a given name exists on the object */
        bool has_attribute(std::string const & name) const {
                return (h5e::check_error(H5Aexists(_self, name.c_str())));
        }

	/** Set/create an attribute.
	 *  If the attribute doesn't exist, it's created using the type of the data.
	 *  If the attribute already exists, it's updated. If the data can't be converted
	 *  to the attribute data type, an error is thrown.
	 *
	 *  Explicitly specify the first template parameter to force a
	 *  particular storage type.
	 */
	template <typename StorageType, typename MemType>
	void write_attribute(std::string const & name, MemType const & value) {
		h5t::wrapper<StorageType> t;
		h5t::datatype type(t);
		h5s::dataspace dspace;
		attribute attr(_self, name, dspace, type);
		attr.write<MemType>(value);
	}

	template <typename StorageType, typename MemType>
	void write_attribute(std::string const & name, MemType const * arr, std::size_t size) {
		h5t::wrapper<StorageType> t;
		h5t::datatype type(t);
		std::vector<hsize_t> dims(1,size);
		h5s::dataspace dspace(dims);
		attribute attr(_self, name, dspace, type);
		attr.write<MemType>(arr, size);
	}

	/**
	 * Write a string attribute, sized to exactly the characters it holds.
	 *
	 * Strings are handled separately because the datatype's width is a
	 * property of the value, so the attribute has to be recreated whenever
	 * the length changes. Sizing to `value.size()` rather than
	 * `value.size() + 1` is what makes a uuid come out as the 36-byte
	 * string the specification asks for, and matches arf.py's |S36. An empty
	 * value still needs a byte to live in.
	 */
	void write_attribute(std::string const & name, std::string const & value) {
		delete_attribute(name);
		h5t::wrapper<std::string> t;
		h5t::datatype type(t);
		type.set_size(value.empty() ? 1 : value.size());
		h5s::dataspace dspace;
		attribute attr(_self, name, dspace, type);
		attr.write(value);
	}

	/**
	 * Write an array of strings, all one width. The specification requires
	 * this shape for the units of complex event data, one element per
	 * compound field.
	 */
	void write_attribute(std::string const & name,
			     std::vector<std::string> const & values) {
		delete_attribute(name);
		std::size_t width = 1;
		for (std::size_t i = 0; i < values.size(); ++i)
			width = std::max(width, values[i].size());

		h5t::wrapper<std::string> t;
		h5t::datatype type(t);
		type.set_size(width);
		std::vector<hsize_t> dims(1, values.size());
		h5s::dataspace dspace(dims);
		attribute attr(_self, name, dspace, type);
		if (values.empty()) return;

		// one packed buffer, NUL-padded to the common width
		std::vector<char> buf(values.size() * width, '\0');
		for (std::size_t i = 0; i < values.size(); ++i)
			std::memcpy(&buf[i * width], values[i].data(), values[i].size());
		h5e::check_error(H5Awrite(attr.hid(), type.hid(), buf.data()));
	}

	template <typename Type>
	void write_attribute(std::string const & name, Type const & value) {
		write_attribute<Type,Type>(name, value);
	}

	template <typename Type>
	void write_attribute(std::string const & name, Type const * arr, std::size_t size) {
		write_attribute<Type,Type>(name, arr, size);
	}

	template <typename StorageType, typename MemType>
	void write_attribute(std::string const & name, std::vector<MemType> const & value) {
                write_attribute<StorageType,MemType>(name, value.data(), value.size());
        }

	template <typename Type>
	void write_attribute(std::string const & name, std::vector<Type> const & value) {
		write_attribute<Type,Type>(name, value);
	}

	void write_attribute(std::string const & name, char const * value) {
		write_attribute(name, std::string(value));
	}

        template <typename Type>
        void write_attribute(std::pair<const std::string, Type> const &p) {
                write_attribute(p.first, p.second);
        }

        struct attr_writer {
                explicit attr_writer (node & n) : _n(n) {}
                node & _n;

                template <typename Type>
                attr_writer & operator() (std::string const & name, Type const & value) {
                        _n.write_attribute(name, value);
                        return *this;
                }

                template <typename Type>
                attr_writer & operator() (std::pair<const std::string, Type> const &p) {
                        _n.write_attribute(p);
                        return *this;
                }
        };

        /** Write a series of attributes using chaining */
        attr_writer write_attribute() { return attr_writer(*this); }

	/** Read an attribute's value */
	template <typename T>
        T read_attribute(std::string const & name) const {
                T ret;
                read_attribute(name, ret);
                return ret;
        }

	template <typename T>
	void read_attribute(std::string const & name, T & value) const {
		attribute attr(_self, name);
		attr.read(value);
	}

	/** Delete an attribute */
	void delete_attribute(std::string const & name) {
		if (H5Aexists(_self, name.c_str()) > 0)
			h5e::check_error(H5Adelete(_self, name.c_str()));
	}

protected:
	explicit node(hid_t hid = H5I_INVALID_HID) : handle(hid) {}
};

} // namespace h5a
} // namespace arf

#endif /* ARF_H5A_HPP */
