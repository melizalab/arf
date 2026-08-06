/* @(#)hdf5.hpp
 * @brief C++ arf interface: hdf5 C header plus some exceptions, etc
 *
 */

#ifndef _HDF5_HH
#define _HDF5_HH 1

#include <algorithm>
#include <string>
#include <stdexcept>
#include <vector>

#include <hdf5.h>

namespace arf {

namespace h5f { class file; }

/** Base class for runtime HDF5 errors. */
struct Exception : public std::runtime_error {
	Exception(char const * what) : std::runtime_error(what) { }
};

/**
 * Owns an HDF5 identifier and releases it on destruction.
 *
 * Every wrapper in this library derives from this. HDF5 identifiers are
 * reference counted, and H5Idec_ref releases any of them -- dataspace,
 * datatype, attribute, property list, dataset, group, or file -- so one
 * destructor here serves them all. Only packet tables need their own, since
 * H5PTclose does more than drop a reference.
 *
 * The class is **move-only**. Duplicating an identifier is never free: for a
 * description it means H5Tcopy or H5Scopy, and for anything in a file it means
 * another open handle. Neither should happen because a value was passed by
 * accident, so a copy has to be written deliberately -- by the value-like
 * subclasses, which define their own, or through hid_copy().
 *
 * The destructor is protected and non-virtual, following C++ Core Guidelines
 * C.35: nothing here is ever owned through a base pointer, and making the
 * destructor virtual would put a vtable pointer in every wrapper -- doubling
 * the size of objects that hold a single 8-byte identifier -- to support a
 * polymorphic delete nobody performs. Protected makes `delete base_ptr` a
 * compile error rather than undefined behavior.
 */
class handle {
public:
	/**
	 * The underlying identifier, borrowed. It is valid only while this
	 * object is, and must not be closed by the caller.
	 */
	hid_t hid() const { return _self; }

	/**
	 * The identifier with its reference count raised, for a caller that
	 * needs it to outlive this object. The caller owns the reference and
	 * must release it.
	 */
	hid_t hid_copy() const {
		H5Iinc_ref(_self);
		return _self;
	}

	/** True if this object holds a live identifier. */
	bool valid() const { return H5Iis_valid(_self) > 0; }

	/** The path of the object within its file. */
	std::string name() const {
		ssize_t sz = H5Iget_name(_self, 0, 0);
		if (sz <= 0) return std::string();
		std::vector<char> buf(sz + 1, '\0');
		if (H5Iget_name(_self, buf.data(), buf.size()) < 0) return std::string();
		return std::string(buf.data());
	}

	/** A handle on the file containing this object. */
	h5f::file file() const;
	// has to be defined in h5f.hpp

protected:
	explicit handle(hid_t hid = H5I_INVALID_HID) : _self(hid) {}

	~handle() { close(); }

	handle(handle && other) : _self(other._self) {
		other._self = H5I_INVALID_HID;
	}

	handle & operator=(handle && other) {
		if (this != &other) {
			close();
			_self = other._self;
			other._self = H5I_INVALID_HID;
		}
		return *this;
	}

	handle(handle const &) = delete;
	handle & operator=(handle const &) = delete;

	/** Release the identifier, if this object holds a live one. */
	void close() {
		if (H5Iis_valid(_self) > 0) H5Idec_ref(_self);
		_self = H5I_INVALID_HID;
	}

	hid_t _self;
};

}

#endif /* _HDF5_HH */
