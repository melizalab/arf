/* @file types.hpp
 * @brief C++ arf interface: type and forward declarations
 *
 * Copyright (C) 2011-2013 C Daniel Meliza <dan||meliza.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef _ARF_TYPES_HH
#define _ARF_TYPES_HH 1

#include <cstdint>
#include <hdf5.h>

#define ARF_VERSION "2.1"
#define ARF_LIBRARY_VERSION "3.0.0"

namespace arf {

// NB: these used to be shared_ptr aliases. The wrappers are move-only values
// now, returned by value, so there is nothing to point at -- write
// `arf::h5d::dataset d = entry.create_dataset(...)` and let it move.
class entry;
class file;

namespace h5d {
        class dataset;
}

namespace h5pt {
        class packet_table;
}

/** defines the type of data stored in a dataset */
enum DataType {
        UNDEFINED = 0,
        ACOUSTIC = 1,
        EXTRAC_HP = 2,
        EXTRAC_LF = 3,
        EXTRAC_EEG = 4,
        INTRAC_CC = 5,
        INTRAC_VC = 6,
        EXTRAC_RAW = 23,
        EVENT = 1000,
        SPIKET = 1001,
        BEHAVET = 1002,
        INTERVAL = 2000,
        STIMI = 2001,
        COMPONENTL = 2002
};

}

#endif
