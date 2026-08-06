/* @file uuid.hpp
 * @brief a minimal RFC 4122 version 4 uuid
 *
 * The library needs six things from a uuid: a 128-bit value, random
 * generation, parsing, formatting, equality, and a nil check. boost::uuid
 * supplies those, but pulls in 456 headers to do it and roughly quadruples the
 * compile time of any translation unit that includes arf.hpp. Everything here
 * fits in one header with no dependency beyond the standard library.
 */

#ifndef _ARF_UUID_HH
#define _ARF_UUID_HH 1

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "hdf5.hpp"

namespace arf {

/**
 * A 128-bit universally unique identifier.
 *
 * Default-constructed instances are nil (all zero), which is what an entry
 * reports when its group carries no uuid attribute. Use generate() for a new
 * random one.
 *
 * The canonical text form is 36 characters, 8-4-4-4-12 lowercase hex digits
 * separated by dashes -- which is exactly what the specification asks entries
 * to store, in a 36-byte string.
 */
class uuid {

public:
	typedef std::uint8_t value_type;

	// NB: an enum rather than `static const` members, which would need
	// out-of-line definitions the moment anything bound them to a reference
	// -- and a header-only library has nowhere to put those.
	enum : std::size_t {
		static_size = 16,  ///< bytes in the identifier
		string_size = 36   ///< characters in the canonical text form
	};

	/** The nil uuid. */
	uuid() {
		std::fill(_bytes, _bytes + static_size, static_cast<value_type>(0));
	}

	/**
	 * A new random (version 4) identifier.
	 *
	 * Entropy comes from std::random_device, which is backed by the
	 * operating system on every platform this library targets. Note that a
	 * few freestanding implementations make it deterministic; if this is
	 * ever ported to one of those, this is the function to revisit.
	 */
	static uuid generate() {
		uuid out;
		std::random_device source;
		for (std::size_t i = 0; i < static_size; i += 4) {
			std::uint32_t bits = static_cast<std::uint32_t>(source());
			out._bytes[i]     = static_cast<value_type>(bits & 0xff);
			out._bytes[i + 1] = static_cast<value_type>((bits >> 8) & 0xff);
			out._bytes[i + 2] = static_cast<value_type>((bits >> 16) & 0xff);
			out._bytes[i + 3] = static_cast<value_type>((bits >> 24) & 0xff);
		}
		// RFC 4122 4.4: version 4 in the high nibble of octet 6, and the
		// variant bits 10x in the high bits of octet 8
		out._bytes[6] = static_cast<value_type>((out._bytes[6] & 0x0f) | 0x40);
		out._bytes[8] = static_cast<value_type>((out._bytes[8] & 0x3f) | 0x80);
		return out;
	}

	/**
	 * Parse the canonical text form. Accepts either case.
	 *
	 * @throws Exception if the text is not 36 characters of correctly
	 *         placed hex digits and dashes
	 */
	static uuid parse(std::string const & text) {
		if (text.size() != string_size)
			throw Exception("a uuid is 36 characters");
		if (text[8] != '-' || text[13] != '-' ||
		    text[18] != '-' || text[23] != '-')
			throw Exception("uuid separators are misplaced");

		uuid out;
		std::size_t byte = 0;
		for (std::size_t i = 0; i < string_size; ++i) {
			if (text[i] == '-') continue;
			int hi = hex_value(text[i]);
			int lo = hex_value(text[i + 1]);
			out._bytes[byte++] = static_cast<value_type>((hi << 4) | lo);
			++i;
		}
		return out;
	}

	/** The canonical text form: 36 lowercase hex characters with dashes. */
	std::string str() const {
		static char const * const digits = "0123456789abcdef";
		std::string out;
		out.reserve(string_size);
		for (std::size_t i = 0; i < static_size; ++i) {
			if (i == 4 || i == 6 || i == 8 || i == 10) out.push_back('-');
			out.push_back(digits[_bytes[i] >> 4]);
			out.push_back(digits[_bytes[i] & 0x0f]);
		}
		return out;
	}

	/** True if every octet is zero. */
	bool is_nil() const {
		for (std::size_t i = 0; i < static_size; ++i)
			if (_bytes[i] != 0) return false;
		return true;
	}

	/** The raw octets, for writing the 128-bit form. */
	value_type const * data() const { return _bytes; }
	std::size_t size() const { return static_size; }

	bool operator==(uuid const & other) const {
		return std::equal(_bytes, _bytes + static_size, other._bytes);
	}
	bool operator!=(uuid const & other) const { return !(*this == other); }

private:
	static int hex_value(char c) {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		throw Exception("uuid contains a character that is not a hex digit");
	}

	value_type _bytes[static_size];
};

}

#endif /* _ARF_UUID_HH */
