/* @file h5pt.hpp
 * @brief C++ arf interface: packet tables
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _H5PT_H
#define _H5PT_H 1

#include <hdf5_hl.h>
#include "hdf5.hpp"
#include "h5d.hpp"

namespace arf { namespace h5pt {

/**
 * Represents a packet table. This is a specialized dataset that can append data
 * quickly. There is no implicit conversion, so it's up to the user to supply
 * the correct type of data in write() calls.
 *
 * The object maintains two handles, one to access the node as a packet table
 * and the other to access it as a dataset; this is necessary for writing
 * attributes and for reading data and doesn't appear to ccause any problems.
 * The write function will use the packet table interface.
 */
class packet_table : public h5d::dataset {

public:
	typedef boost::shared_ptr<packet_table> ptr_type;

	/** Open an existing packet table */
	packet_table(hid_t parent, std::string const & name)
		: h5d::dataset(parent, name) {
		_ptself = h5e::check_error(H5PTopen(parent, name.c_str()));
	}

	/** Create a new packet table */
	packet_table(hid_t parent, std::string const & name,
                     h5t::datatype const & type,
		     hsize_t chunk_size, int compression) {
		_ptself = h5e::check_error(H5PTcreate_fl(parent, name.c_str(), type.hid(),
							 chunk_size, compression));
		open_dataset(parent, name);
	}

	~packet_table() {
                // NB: H5PTis_valid returns herr_t, so any non-negative value
                // means valid. It is not htri_t like H5Iis_valid, where only a
                // positive value does.
                if (H5PTis_valid(_ptself) >= 0) {
                        H5PTclose(_ptself);
                }
	}

	/**
         * Appends data to the packet table. It's up to the user to ensure that
         * the data type matches the data type of the object.
         */
        void write(void const * data, hsize_t nitems) {
                h5e::check_error(H5PTappend(_ptself, nitems, data));
        }

	/** Appends data to the packet table */
        template <typename Type>
	void write(std::vector<Type> const & data) {
                write(reinterpret_cast<void const *>(&data[0]), data.size());
	}

protected:

	hid_t _ptself;
};

}}

#endif /* _H5PT_H */

