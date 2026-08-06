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

#ifndef _H5F_H
#define _H5F_H 1

#include "hdf5.hpp"
#include "h5g.hpp"
#include "h5p.hpp"

namespace arf { namespace h5f {

/**
 * Represents an HDF5 file, as well as the root group of the file
 */
class file : public h5g::group {

public:
	typedef boost::shared_ptr<file> ptr_type;

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
                        if (fp == 0)
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
	file(hid_t file_id) {
		_file_id = file_id;
		_self = h5e::check_error(H5Gopen2(_file_id, "/", H5P_DEFAULT));
	}

	~file() {
		if (H5Iget_type(_file_id)==H5I_FILE) {
#ifdef H5_HAVE_PARALLEL
                        H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
#endif
			H5Fclose(_file_id);
                }
	}

	void flush() {
                if (H5Iget_type(_file_id)==H5I_FILE)
                        H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
        }

	/** size of the file, in bytes */
	hsize_t size() const {
		hsize_t v;
		h5e::check_error(H5Fget_filesize(_file_id, &v));
		return v;
	}

        /** name of the file, or an empty string if handl is invalid */
	std::string name() const {
		ssize_t sz = H5Fget_name(_file_id, 0, 0);
                if (sz < 0) return "";
		char name[sz+1];
		H5Fget_name(_file_id, name, sz+1);
		return name;
	}

        /** the identifier for the file */
        hid_t file_id() const { return _file_id; }

private:
	hid_t _file_id;

};

}

inline h5f::file::ptr_type
handle::file() const {
	return boost::make_shared<h5f::file>(H5Iget_file_id(_self));
}

}




#endif /* _H5F_H */

