/* @file h5s.hpp
 * @brief C++ arf interface: dataspaces
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5S_H
#define _H5S_H 1

#include <vector>
#include <cmath>
#include <numeric>
#include "hdf5.hpp"

namespace arf { namespace h5s {

namespace detail {

/** Guess a good chunk size */
inline std::vector<hsize_t> guess_chunk(std::vector<hsize_t> const & shape, int typesize)
{
	static const int CHUNK_BASE = 16*1024;    // Multiplier by which chunks are adjusted
	static const int CHUNK_MIN = 8*1024;      // Soft lower limit (8k)
	static const int CHUNK_MAX = 1024*1024;   // Hard upper limit (1M)
	int idx = 0, ndims = shape.size();

	if (ndims==0)
		throw Exception("Scalar datasets can't be chunked");
	std::vector<hsize_t> chunks(shape);
	// H5Pset_chunk rejects a zero-length chunk, so an empty dataset -- which
	// arf.py allows -- could not be created at all. Clamp before measuring.
	for (int i = 0; i < ndims; ++i)
		if (chunks[i] == 0) chunks[i] = 1;
	float dset_size = std::accumulate(chunks.begin(), chunks.end(), 1.0, std::multiplies<float>()) * typesize;
	float target_size = CHUNK_BASE * pow(2, log10(dset_size/(1024.0*1024)));
	if (target_size > CHUNK_MAX)
		target_size = CHUNK_MAX;
	else if (target_size < CHUNK_MIN)
		target_size = CHUNK_MIN;

	while (1) {
		dset_size = std::accumulate(chunks.begin(), chunks.end(), 1.0, std::multiplies<float>()) * typesize;
		if (dset_size < target_size || (fabs(dset_size-target_size)/target_size < 0.5 &&
						dset_size < CHUNK_MAX))
			break;
		chunks[idx%ndims] = static_cast<hsize_t>(ceil(chunks[idx%ndims] / 2.0));
		++idx;
	}
	return chunks;
}

}

/**
 * Base class for HDF5 data spaces. This is a fairly simple wrapper
 * with copy semantics: on initialization the object creates a new
 * HDF5 handle and releases it on destruction.
 */
class dataspace : public handle {
public:
	typedef boost::shared_ptr<dataspace> ptr_type;
	/**
	 * The default dataspace is a scalar.
	 */
	dataspace() {
		_self = h5e::check_error(H5Screate(H5S_SCALAR));
	}

	dataspace(dataspace const & other) {
		_self = h5e::check_error(H5Scopy(other.hid()));
	}

	/** Take ownership of a dataspace handle */
	explicit dataspace(hid_t hid) : handle(hid) {}

	explicit dataspace(std::vector<hsize_t> const & dims) {
		_self = h5e::check_error(H5Screate_simple(dims.size(), &dims[0], NULL));
	}

	dataspace & operator= (dataspace const & other) {
		// release old handle
		H5Sclose(_self);
		_self = h5e::check_error(H5Scopy(other.hid()));
		return *this;
	}

	dataspace(std::vector<hsize_t> const & dims,
		  std::vector<hsize_t> const & maxdims) {
		assert(dims.size() == maxdims.size() || maxdims.empty());
		if(maxdims.empty())
			_self = h5e::check_error(H5Screate_simple(dims.size(), &dims[0], NULL));
		else
			_self = h5e::check_error(H5Screate_simple(dims.size(), &dims[0], &maxdims[0]));
	}

	dataspace(dataspace const & orig,
		  std::vector<hsize_t> const & offset,
		  std::vector<hsize_t> const & stride,
		  std::vector<hsize_t> const & count) {
		assert((offset.size() == stride.size()) && (stride.size() == count.size()));
		_self = h5e::check_error(H5Scopy(orig.hid()));
		h5e::check_error(H5Sselect_hyperslab(_self, H5S_SELECT_SET, &offset[0],
						     &stride[0], &count[0], NULL));
	}

	dataspace(dataspace const & orig,
		  std::vector<hsize_t> const & offset,
		  std::vector<hsize_t> const & stride,
		  std::vector<hsize_t> const & count,
		  std::vector<hsize_t> const & block) {
		assert(offset.size() == stride.size() && stride.size() == count.size() &&
		       count.size() == block.size());
		_self = h5e::check_error(H5Scopy(orig.hid()));
		h5e::check_error(H5Sselect_hyperslab(_self, H5S_SELECT_SET, &offset[0],
						     &stride[0], &count[0], &block[0]));
	}

	~dataspace() {
		H5Sclose(_self);
	}


	hsize_t ndims() const {
		return h5e::check_error(H5Sget_simple_extent_ndims(_self));
	}

	std::vector<hsize_t> dims() const {
		std::vector<hsize_t> dims(ndims());
		h5e::check_error(H5Sget_simple_extent_dims(_self, &dims[0], 0));
		return dims;
	}

	std::vector<hsize_t> maxdims() const {
		std::vector<hsize_t> dims(ndims());
		h5e::check_error(H5Sget_simple_extent_dims(_self, 0, &dims[0]));
		return dims;
	}

	hsize_t size() const {
		if (ndims()==0) return 1;
		std::vector<hsize_t> shape(dims());
		return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<hsize_t>());
	}

	void select_all() {
		h5e::check_error(H5Sselect_all(_self));
	}

	void select_none() {
		h5e::check_error(H5Sselect_none(_self));
	}

};

}} // namespace h5s // namespace arf

#endif /* _H5S_H */

