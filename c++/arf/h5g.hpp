/* @file h5g.hpp
 * @brief C++ arf interface: groups
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5G_H
#define _H5G_H 1

#include "hdf5.hpp"
#include "h5p.hpp"
#include "h5a.hpp"
#include "h5d.hpp"
#include "h5pt.hpp"

namespace arf { namespace h5g {

namespace detail {

struct name_iterator {

	typedef std::vector<std::string> return_value;
	return_value value;
	static herr_t iterate(hid_t id, const char * name, const H5L_info_t *link_info, void *data) {
		name_iterator *_this = static_cast<name_iterator*>(data);
		_this->value.push_back(std::string(name));
		return 0;
	}
};

}

/**
 * Base class for an HDF5 group. Functions are provided for creating
 * subgroups and datasets.
 */
class group : public h5a::node {

public:
	typedef boost::shared_ptr<group> ptr_type;

	/** Open or create a group object by name */
	group(h5a::node const & parent, std::string const & path) {
		open_group(parent, path);
	}

	group(h5a::node const & parent, std::string const & path, bool create) {
		if (!create) {
			open_group(parent, path);
			return;
		}
		h5p::proplist gcpl(H5P_GROUP_CREATE);
		h5e::check_error(H5Pset_link_creation_order(gcpl.hid(),
							    H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED));
		_self = h5e::check_error(H5Gcreate2(parent.hid(), path.c_str(), H5P_DEFAULT,
						   gcpl.hid(), H5P_DEFAULT));
	}

	~group() {
                if (H5Iis_valid(_self) > 0) {
                        H5Gclose(_self);
                }
	}

	/** Link an existing node under this group. */
	void create_link(h5a::node const & subgroup);

	/**
	 * Create a new dataset and add data to it.  Currently only 1D
	 * data is supported.
	 *
	 * @param name  the name of the dataset
	 * @param data  the data to store in the dataset
	 * @param compression deflate level. 0, the default, stores the data
	 *        uncompressed but still frames it through zlib, which costs
	 *        about 16 bytes a chunk and gives every chunk an adler32.
	 *        1 through 9 compress; a negative value writes no filter.
	 */
	template <typename StorageType, typename MemType>
	h5d::dataset::ptr_type
	create_dataset(std::string const & name,
		       std::vector<MemType> const & data,
		       int compression=0) {
		if (H5Lexists(_self, name.c_str(), H5P_DEFAULT) > 0)
			throw Exception("Object already exists with that name");

		h5t::wrapper<StorageType> t;
		h5t::datatype type(t);
		h5s::dataspace dspace(std::vector<hsize_t>(1,data.size()), std::vector<hsize_t>(1,H5S_UNLIMITED));
		h5d::dataset::ptr_type ds = boost::make_shared<h5d::dataset>(_self, name.c_str(), dspace,
									     type, compression);
		ds->write(data);
		return ds;
	}

	template <typename Type>
	boost::shared_ptr<h5d::dataset>
	create_dataset(std::string const & name,
		       std::vector<Type> const & data,
		       int compression=0) {
		return create_dataset<Type,Type>(name,data,compression);
	}

	/**
	 * Create a new packet table dataset. Packet table datasets
	 * are useful for writing a stream of data. No type conversion
	 * is supported, but the interface is extremely simple.
	 *
	 * @param name  the name of the dataset
	 * @param replace if true, drop existing dataset; otherwise appends data
	 * @param compression integer code indicating compression ratio
	 */
	template <typename StorageType>
	typename h5pt::packet_table::ptr_type
	create_packet_table(std::string const & name,
			    bool replace=false,
			    hsize_t chunk_size=1024,
			    int compression=0) {
		h5t::wrapper<StorageType> t;
		h5t::datatype type(t);
		// same contract as create_dataset: without replace, an existing
		// name is an error. H5PTcreate_fl fails on one anyway, but it
		// fails without pushing anything onto hdf5's error stack, so
		// the diagnosis has to happen here.
		if (H5Lexists(_self, name.c_str(), H5P_DEFAULT) > 0) {
			if (!replace)
				throw Exception("Object already exists with that name");
			unlink(name);
		}

                // H5PTcreate_fl uses the same convention as create_dataset:
                // -1 for no filter, 0 through 9 for a deflate level
                h5pt::packet_table::ptr_type pt =
                        boost::make_shared<h5pt::packet_table>(_self, name, type, chunk_size,
                                                               compression);
		return pt;
	}


	template <typename Type>
	void read_dataset(std::string const & name, std::vector<Type> & data,
                          hsize_t offset=0, hsize_t stride=1) {
		h5d::dataset(_self, name).read(data, offset, stride);
	}

	template <typename Type>
	void read_dataset(std::string const & name, Type * data, hsize_t size,
                          hsize_t offset=0, hsize_t stride=1) {
		h5d::dataset(_self, name).read(data, size, offset, stride);
	}

	/** Delete a child */
	void unlink(std::string const & name) {
		h5e::check_error(H5Ldelete(_self, name.c_str(), H5P_DEFAULT));
	}

	/**
	 * Apply an iterator to the children of the group.
	 *
	 * @param functor  function object with the following members:
	 *                 return_value: type of data managed by the iterator
	 *                 value: member variable
	 *                 iterate(): static member function, must be of type
	 *                 H5L_iterate_t. This is called for each child of the group.
	 * @param index_type  can be H5_INDEX_CRT_ORDER (default) or H5_INDEX_NAME
	 * @param order       iteration order: H5_ITER_INC (def), H5_ITER_DEC, or H5_ITER_NATIVE
	 * @param idx         input/output value specifiying where to start the
	 *                    iteration, and at the end, the last position.
	 */
	template <typename F>
	typename F::return_value
	iterate(F & functor,
                H5_index_t index_type=H5_INDEX_CRT_ORDER,
                H5_iter_order_t order=H5_ITER_INC,
                hsize_t *idx=0) const {
		h5e::check_error(H5Literate(_self, index_type, order, idx,
					    &F::iterate,
					    static_cast<void*>(&functor)));
		return functor.value;
	}

	/** Return a list of the names of the children of this group */
	std::vector<std::string> children() const {
		detail::name_iterator it;
		return iterate(it);
	}

	hsize_t nchildren() const {
		H5G_info_t inf;
		h5e::check_error(H5Gget_info(_self, &inf));
		return inf.nlinks;
	}

	bool contains(std::string const & name) const {
		return (h5e::check_error(H5Lexists(_self, name.c_str(), H5P_DEFAULT)) > 0);
	}


protected:
	group(hid_t group_id=-1) {_self = group_id;}

private:
	void open_group(h5a::node const & parent, std::string const & path) {
		_self = h5e::check_error(H5Gopen2(parent.hid(), path.c_str(), H5P_DEFAULT));
	}

};


}}

#endif /* _H5G_H */

