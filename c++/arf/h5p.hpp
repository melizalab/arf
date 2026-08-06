/* @file h5p.hpp
 * @brief C++ arf interface: property lists
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef ARF_H5P_HPP
#define ARF_H5P_HPP

#include <algorithm>
#include <hdf5.h>
#include "hdf5.hpp"
#include "h5e.hpp"

namespace arf { namespace h5p {

/**
 * A property list. Value-like, as datatypes and dataspaces are: copying
 * duplicates the list with H5Pcopy.
 */
class proplist : public handle {

public:
        /** Create a new property list of the given class */
        explicit proplist(hid_t cls_id) : handle(h5e::check_error(H5Pcreate(cls_id))) {}

        proplist(proplist const & other)
                : handle(h5e::check_error(H5Pcopy(other.hid()))) {}

        proplist(proplist && other) = default;

        /** Copy-and-swap; see the note on h5t::datatype::operator=. */
        proplist & operator= (proplist other) {
                swap(other);
                return *this;
        }

        void swap(proplist & other) noexcept { std::swap(_self, other._self); }

        bool operator==(proplist const & other) const {
                return h5e::check_error(H5Pequal(_self, other.hid())) > 0;
        }
        bool operator!=(proplist const & other) const {
                return h5e::check_error(H5Pequal(_self, other.hid())) <= 0;
        }
};

}}

#endif /* ARF_H5P_HPP */
