/* @file h5f.hpp
 * @brief C++ arf interface: files
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef ARF_H5F_HPP
#define ARF_H5F_HPP

#include "hdf5.hpp"
#include "h5g.hpp"
#include "h5p.hpp"

namespace arf { namespace h5f {

/**
 * Represents an HDF5 file, as well as the root group of the file
 */
class file : public h5g::group {

public:
	/**
	 * Open or create an HDF5 file.  File access mode can be one
	 * of the following values:
	 * 'r' : read-only; file must exist
	 * 'a' : read-write access, creating file if necessary
	 * 'w' : read-write access; truncates file if it exists
	 *
	 * @note destruction may not fully close the file, if objects in the
	 *       file remain open
	 *
	 * @param name  the path of the file to open/create
	 * @param mode  the mode to open the file
	 */
	file(std::string const & path, std::string const & mode) {
		const char * name = path.c_str();
		h5p::proplist fapl(H5P_FILE_ACCESS);
                h5p::proplist fcpl(H5P_FILE_CREATE);

#ifdef H5_HAVE_PARALLEL
		H5Pset_fapl_mpiposix(fapl.hid(), MPI_COMM_WORLD, false);
#endif
                h5e::check_error(H5Pset_link_creation_order(fcpl.hid(),
                                                            H5P_CRT_ORDER_TRACKED|H5P_CRT_ORDER_INDEXED));

		if(mode == "r")
			_file_id = h5e::check_error(H5Fopen(name, H5F_ACC_RDONLY, fapl.hid()));
		else if (mode == "a") {
                        // test for existence (HDF5is_hdf5 may not work)
                        FILE *fp = fopen(name,"r");
                        if (fp == nullptr)
				_file_id = h5e::check_error(H5Fcreate(name, H5F_ACC_TRUNC,
								      fcpl.hid(), fapl.hid()));
                        else {
                                fclose(fp);
				_file_id = h5e::check_error(H5Fopen(name, H5F_ACC_RDWR, fapl.hid()));
                        }
		}
		else if (mode == "w")
			_file_id = h5e::check_error(H5Fcreate(name, H5F_ACC_TRUNC, fcpl.hid(), fapl.hid()));
		else
			throw Exception("Invalid mode");

		_self = h5e::check_error(H5Gopen2(_file_id, "/", H5P_DEFAULT));
	}

	/** Wrap file hid_t object. Takes ownership of handle */
	explicit file(hid_t file_id) : _file_id(file_id) {
		_self = h5e::check_error(H5Gopen2(_file_id, "/", H5P_DEFAULT));
	}

	~file() { close_file(); }

	file(file const &) = delete;
	file & operator=(file const &) = delete;

	// a second identifier to carry, so the moves are written out rather
	// than defaulted; the base takes care of _self, the root group
	file(file && other) noexcept
		: h5g::group(std::move(other)), _file_id(H5I_INVALID_HID) {
		// clang-tidy reports a use-after-move here. It is a false
		// positive: the base's move touches only its own _self, so
		// other._file_id is untouched, and swapping leaves the source
		// invalid either way.
		// NOLINTNEXTLINE(bugprone-use-after-move)
		std::swap(_file_id, other._file_id);
	}

	file & operator=(file && other) noexcept {
		if (this != &other) {
			close_file();
			h5g::group::operator=(std::move(other));
			// NOLINTNEXTLINE(bugprone-use-after-move)
			std::swap(_file_id, other._file_id);
		}
		return *this;
	}

	void flush() {
                if (H5Iget_type(_file_id)==H5I_FILE)
                        H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
        }

	/** size of the file, in bytes */
	hsize_t size() const {
		hsize_t v = 0;
		h5e::check_error(H5Fget_filesize(_file_id, &v));
		return v;
	}

	/**
	 * The path of the file on disk, or an empty string if the handle is
	 * invalid.
	 *
	 * NB: not called name(). handle::name() returns the object's path
	 * *within* the file, which for the root group is always "/", and having
	 * both under one name meant the answer depended on the static type of
	 * the reference you happened to hold.
	 */
	std::string filename() const {
		ssize_t sz = H5Fget_name(_file_id, nullptr, 0);
		if (sz <= 0) return std::string();
		std::vector<char> buf(sz + 1, '\0');
		H5Fget_name(_file_id, buf.data(), buf.size());
		return std::string(buf.data());
	}

        /** the identifier for the file */
        hid_t file_id() const { return _file_id; }

private:
	void close_file() {
		if (H5Iget_type(_file_id) == H5I_FILE) {
#ifdef H5_HAVE_PARALLEL
			H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
#endif
			H5Fclose(_file_id);
		}
		_file_id = H5I_INVALID_HID;
	}

	hid_t _file_id;

};

}

inline h5f::file
handle::file() const {
	return h5f::file(H5Iget_file_id(_self));
}

}




#endif /* ARF_H5F_HPP */
