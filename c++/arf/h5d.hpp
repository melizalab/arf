/* @file h5d.hpp
 * @brief C++ arf interface: datasets
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5D_H
#define _H5D_H 1

#include <cassert>

#include "hdf5.hpp"
#include "h5a.hpp"
#include "h5p.hpp"

namespace arf { namespace h5d {

/**
 * Represents a dataset object, which derives from a node and can store data in
 * arrays. Currently only 1 dimensional datasets are supported.
 */
class dataset : public h5a::node {

public:
	/** Open an existing dataset */
	dataset(hid_t parent, std::string const & name) {
		if (H5Lexists(parent, name.c_str(), H5P_DEFAULT) <= 0)
			throw Exception("No such dataset");
		open_dataset(parent, name);
	}

	/** Create a new dataset */
	dataset(hid_t parent, std::string const & name,
		h5s::dataspace const & dspace,
		h5t::datatype const & type,
		int compress) {
		std::vector<hsize_t> chunks = h5s::detail::guess_chunk(dspace.dims(), type.size());
		create_dataset(parent, name, dspace, type, chunks, compress);
	}

	template <typename Type>
	dataset(hid_t parent, std::string const & name,
		h5s::dataspace const & dspace,
		h5t::datatype const & type,
		std::vector<hsize_t> const & chunkdims,
		int compress) {
		create_dataset(parent, name, dspace, type, chunkdims, compress);
	}

	dataset(dataset && other) = default;
	dataset & operator=(dataset && other) = default;

	/** Write data to the current dataset. The extent of the dataset is resized to match. */
	template <typename Type>
	void write(Type const * data, hsize_t size) {
		h5t::wrapper<Type> t;
		h5t::datatype memtype(t);
		std::vector<hsize_t> extent(1,size);
		set_extent(extent);
		h5s::dataspace memspace(extent);
		h5e::check_error(H5Dwrite(_self, memtype.hid(), memspace.hid(),
					  H5S_ALL, H5P_DEFAULT, data));
	}

	template <typename Type>
	void write(std::vector<Type> const & data) {
                write(data.empty() ? 0 : data.data(), data.size());
        }

        /**
         * @brief read data into an array or vector
         *
         * Copies data from the dataset into an array, converting data type as
         * necessary. The default is to read the full dataset, but
         * offset/stride/count arguments can be set to read only part of it.
         *
         * @note Although this function can be used to read data types that have
         * variable length components, this will create memory leaks.
         *
         * @param data the destination array or vector
         * @param count the number of elements to read (and size of the
         *              destination array). Required for array arguments or for
         *              reading partial datasets
         * @param offset the offset at which to start reading
         * @param stride the number of elements to step between reads
         *
         * @throws exception if the size of the dest array exceeds the amount of
         * available data
         *
         */
	template <typename Type>
	void read(Type * data, hsize_t size, hsize_t offset=0, hsize_t stride=1) {
		h5t::wrapper<Type> t;
		h5t::datatype type(t);
                h5s::dataspace filespace(dataspace(),
                                         std::vector<hsize_t>(1,offset),
                                         std::vector<hsize_t>(1,stride),
                                         std::vector<hsize_t>(1,size));
		h5s::dataspace memspace(std::vector<hsize_t>(1,size));
		h5e::check_error(H5Dread(_self, type.hid(), memspace.hid(),
					 filespace.hid(), H5P_DEFAULT, data));
	}

	template <typename Type>
	void read(std::vector<Type> & data, hsize_t count, hsize_t offset=0, hsize_t stride=1) {
                // resize first: this used to write through the caller's
                // existing buffer whatever its length, overflowing whenever it
                // held fewer than count elements
                data.resize(count);
                if (count == 0) return;
                read(data.data(), count, offset, stride);
        }

	template <typename Type>
	void read(std::vector<Type> & data) {
                data.resize(dataspace().size());
                if (data.empty()) return;
                read(data.data(), data.size());
        }


	/**
	 * Resize the dataset. Datasets must be chunked, and the new
	 * sizes must be less than or equal to the dataspace's
	 * maxsize
	 */
	void set_extent(std::vector<hsize_t> const & size) {
		assert(size.size() >= dataspace().ndims());
		h5e::check_error(H5Dset_extent(_self, size.data()));
	}

	h5s::dataspace dataspace() const {
		return h5s::dataspace(h5e::check_error(H5Dget_space(_self)));
	}

	h5t::datatype datatype() const {
		return h5t::datatype(h5e::check_error(H5Dget_type(_self)));
	}

	std::vector<hsize_t> chunks() const {
		int ndims = dataspace().ndims();
		hid_t plist = H5Dget_create_plist(_self);
		std::vector<hsize_t> out(ndims);
		h5e::check_error(H5Pget_chunk(plist, ndims, out.data()));
		H5Pclose(plist);
		return out;
	}

protected:
	dataset() {}

	void open_dataset(hid_t parent, std::string const & name) {
		_self = h5e::check_error(H5Dopen2(parent, name.c_str(), H5P_DEFAULT));
	}

private:
	void create_dataset(hid_t parent, std::string const & name,
			    h5s::dataspace const & dspace, h5t::datatype const & dtype,
			    std::vector<hsize_t> const & chunkdims, int compress=0) {
		if (H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0)
			throw Exception("Dataset already exists");
		h5p::proplist dcpl(H5P_DATASET_CREATE);
		h5e::check_error(H5Pset_layout(dcpl.hid(), H5D_CHUNKED));
		h5e::check_error(H5Pset_chunk(dcpl.hid(), dspace.ndims(), chunkdims.data()));
		// 0 is a real setting, not "off": zlib stores the data
		// uncompressed but still frames it, giving a per-chunk adler32
		// for about 16 bytes a chunk. H5Pset_deflate rejects a negative
		// level, which is what makes it the sentinel for no filter.
		if (compress > -1)
			h5e::check_error(H5Pset_deflate(dcpl.hid(), compress));

		_self = h5e::check_error(H5Dcreate2(parent, name.c_str(), dtype.hid(),
						   dspace.hid(), H5P_DEFAULT,
						   dcpl.hid(), H5P_DEFAULT));
	}

};

}}

#endif /* _H5D_H */

