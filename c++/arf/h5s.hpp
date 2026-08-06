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

#ifndef ARF_H5S_HPP
#define ARF_H5S_HPP

#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>
#include "hdf5.hpp"
#include "h5e.hpp"

namespace arf { namespace h5s {

namespace detail {

/** Multiply an accumulated double by an extent, without a narrowing warning. */
struct multiply_as_double {
	double operator()(double acc, hsize_t extent) const {
		return acc * static_cast<double>(extent);
	}
};

/** Guess a good chunk size */
inline std::vector<hsize_t> guess_chunk(std::vector<hsize_t> const & shape, int typesize)
{
	static const int CHUNK_BASE = 16*1024;    // Multiplier by which chunks are adjusted
	static const int CHUNK_MIN = 8*1024;      // Soft lower limit (8k)
	static const int CHUNK_MAX = 1024*1024;   // Hard upper limit (1M)
	std::size_t idx = 0, ndims = shape.size();

	if (ndims==0)
		throw Exception("Scalar datasets can't be chunked");
	std::vector<hsize_t> chunks(shape);
	// H5Pset_chunk rejects a zero-length chunk, so an empty dataset -- which
	// arf.py allows -- could not be created at all. Clamp before measuring.
	for (std::size_t i = 0; i < ndims; ++i)
		if (chunks[i] == 0) chunks[i] = 1;
	double dset_size = std::accumulate(chunks.begin(), chunks.end(), 1.0,
					   multiply_as_double()) * typesize;
	double target_size = CHUNK_BASE * std::pow(2, std::log10(dset_size / (1024.0 * 1024)));
	if (target_size > CHUNK_MAX)
		target_size = CHUNK_MAX;
	else if (target_size < CHUNK_MIN)
		target_size = CHUNK_MIN;

	while (1) {
		dset_size = std::accumulate(chunks.begin(), chunks.end(), 1.0,
					    multiply_as_double()) * typesize;
		if (dset_size < target_size ||
		    (std::fabs(dset_size - target_size) / target_size < 0.5 &&
		     dset_size < CHUNK_MAX))
			break;
		chunks[idx % ndims] = static_cast<hsize_t>(
			std::ceil(static_cast<double>(chunks[idx % ndims]) / 2.0));
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
	/**
	 * The default dataspace is a scalar.
	 */
	/** The default dataspace is a scalar. */
	dataspace() : handle(h5e::check_error(H5Screate(H5S_SCALAR))) {}

	/** Take ownership of a dataspace handle */
	explicit dataspace(hid_t hid) : handle(hid) {}

	explicit dataspace(std::vector<hsize_t> const & dims)
		: handle(h5e::check_error(H5Screate_simple(static_cast<int>(dims.size()),
							   dims.data(), nullptr))) {}

	dataspace(std::vector<hsize_t> const & dims,
		  std::vector<hsize_t> const & maxdims)
		: handle(h5e::check_error(
				 H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
						  maxdims.empty() ? nullptr : maxdims.data()))) {
		if (!maxdims.empty() && dims.size() != maxdims.size())
			throw Exception("dims and maxdims must have the same rank");
	}

	dataspace(dataspace const & orig,
		  std::vector<hsize_t> const & offset,
		  std::vector<hsize_t> const & stride,
		  std::vector<hsize_t> const & count)
		: handle(h5e::check_error(H5Scopy(orig.hid()))) {
		if (offset.size() != stride.size() || stride.size() != count.size())
			throw Exception("hyperslab arguments must have the same rank");
		h5e::check_error(H5Sselect_hyperslab(_self, H5S_SELECT_SET, offset.data(),
						     stride.data(), count.data(), NULL));
	}

	dataspace(dataspace const & orig,
		  std::vector<hsize_t> const & offset,
		  std::vector<hsize_t> const & stride,
		  std::vector<hsize_t> const & count,
		  std::vector<hsize_t> const & block)
		: handle(h5e::check_error(H5Scopy(orig.hid()))) {
		if (offset.size() != stride.size() || stride.size() != count.size() ||
		    count.size() != block.size())
			throw Exception("hyperslab arguments must have the same rank");
		h5e::check_error(H5Sselect_hyperslab(_self, H5S_SELECT_SET, offset.data(),
						     stride.data(), count.data(), block.data()));
	}

	/** Copying duplicates the description with H5Scopy. */
	dataspace(dataspace const & other)
		: handle(h5e::check_error(H5Scopy(other.hid()))) {}

	dataspace(dataspace && other) = default;

	/** Copy-and-swap; see the note on h5t::datatype::operator=. */
	dataspace & operator= (dataspace other) {
		swap(other);
		return *this;
	}

	void swap(dataspace & other) noexcept { std::swap(_self, other._self); }

	hsize_t ndims() const {
		return h5e::check_error(H5Sget_simple_extent_ndims(_self));
	}

	std::vector<hsize_t> dims() const {
		std::vector<hsize_t> extents(ndims());
		h5e::check_error(H5Sget_simple_extent_dims(_self, extents.data(), nullptr));
		return extents;
	}

	std::vector<hsize_t> maxdims() const {
		std::vector<hsize_t> extents(ndims());
		h5e::check_error(H5Sget_simple_extent_dims(_self, nullptr, extents.data()));
		return extents;
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

#endif /* ARF_H5S_HPP */
