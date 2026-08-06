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

#ifndef _H5A_H
#define _H5A_H 1

#include <boost/scoped_array.hpp>
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
	attribute(hid_t parent, std::string const & name) {
		_self = h5e::check_error(H5Aopen(parent, name.c_str(), H5P_DEFAULT));
	}

        attribute(handle const & parent, std::string const & name) {
                _self = h5e::check_error(H5Aopen(parent.hid(), name.c_str(), H5P_DEFAULT));
        }

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

	~attribute() {
		H5Aclose(_self);
	}

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
		assert(dataspace()->size() >= size);
		h5e::check_error(H5Awrite(_self, type.hid(), arr));
	}

	template <typename Type>
	void write(std::vector<Type> const & value) {
                write(&value[0], value.size());
	}

	void write(std::string const & value) {
		// the wrapper owns the handle, so it is released even if the
		// write throws
		h5t::datatype type(h5e::check_error(H5Aget_type(_self)));
		h5e::check_error(H5Awrite(_self, type.hid(), value.c_str()));
	}


	/** read value from attribute. automatic type conversion */
        template <typename Type>
        Type read() {
                Type t;
                read(t);
                return t;
        }


	template <typename Type>
	void read(Type & out) {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
		h5e::check_error(H5Aread(_self, type.hid(), &out));
	}

	template <typename Type>
	void read(std::vector<Type> & out) {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
                out.resize(dataspace()->size());
		h5e::check_error(H5Aread(_self, type.hid(), &out[0]));
	}

	void read(std::string & str) {
		// the wrapper owns the handle, so it is released on the throw
		// path as well as the success path
		h5t::datatype type(h5e::check_error(H5Aget_type(_self)));
		if (H5Tget_class(type.hid())!=H5T_STRING)
			throw Exception("Attempt to read non-string attribute into string");
		boost::scoped_array<char> buf(new char[type.size()]);
		h5e::check_error(H5Aread(_self, type.hid(), buf.get()));
		str.assign(buf.get());
	}

	/** Return the attributes's dataspace */
	h5s::dataspace::ptr_type dataspace() const {
		return boost::make_shared<h5s::dataspace>(h5e::check_error(H5Aget_space(_self)));
	}

	/** Return the name of the attribute */
	std::string name() const {
		ssize_t sz = H5Aget_name(_self, 0, 0);
		char name[sz+1];
		H5Aget_name(_self, sz+1, name);
		return name;
	}

private:

};


/** Base class for any HDF5 node (an object that can have attributes) */
class node : public handle {

public:

        /** Determine whether an attribute with a given name exists on the object */
        bool has_attribute(std::string const & name) {
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

	// strings have to be handled differently because we need to
	// set the size of the datatype when the attribute is created.
	void write_attribute(std::string const & name, std::string const & value) {
		delete_attribute(name);
		h5t::wrapper<std::string> t;
		h5t::datatype type(t);
		type.set_size(value.size()+1);
		h5s::dataspace dspace;
		attribute attr(_self, name, dspace, type);
		attr.write(value);
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
                write_attribute<StorageType,MemType>(name, &value[0], value.size());
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
                attr_writer (node & n) : _n(n) {}
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
        T read_attribute(std::string const & name) {
                T ret;
                read_attribute(name, ret);
                return ret;
        }

	template <typename T>
	void read_attribute(std::string const & name, T & value) {
		attribute attr(_self, name);
		attr.read(value);
	}

	/** Delete an attribute */
	void delete_attribute(std::string const & name) {
		if (H5Aexists(_self, name.c_str()) > 0)
			h5e::check_error(H5Adelete(_self, name.c_str()));
	}

};

} // namespace h5a
} // namespace arf

#endif /* _H5A_H */

