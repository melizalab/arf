# arf C++ build and test driver.
#
# The library is header-only; everything built here is a test.
#
#   make test              build and run the doctest suite (VARIANT=plain)
#   make test-release      same, compiled -O2 -DNDEBUG (proves the suite does
#                          not depend on assert() for its checks)
#   make test-sanitize     same, under AddressSanitizer + UBSan
#   make test-all          all three
#   make lint-strict       -Wpedantic -Werror syntax check
#   make install           install the headers under $(PREFIX)
#
# Objects go to build/$(VARIANT)/. Sanitizers require a glibc target.

PREFIX     ?= /usr/local
CXX        ?= g++
CXXSTD     ?= -std=c++11
WARN       ?= -Wall -Wextra -Wpedantic -Werror
PKG_CONFIG ?= pkg-config

# Debian and Ubuntu ship the serial build as hdf5-serial.pc, with headers under
# /usr/include/hdf5/serial; everyone else provides hdf5.pc. Probe for both.
# Override HDF5_CFLAGS/HDF5_LIBS directly if your hdf5 has no pkg-config file.
HDF5_PC ?= $(shell for m in hdf5 hdf5-serial; do \
	$(PKG_CONFIG) --exists $$m 2>/dev/null && { echo $$m; break; }; done)
HDF5_CFLAGS ?= $(shell $(PKG_CONFIG) --cflags $(HDF5_PC) 2>/dev/null)
# the -L from the .pc file is what makes -lhdf5_hl resolvable on Debian, where
# libhdf5_hl.so lives in the versioned serial directory rather than on the
# default search path
HDF5_LIBS   ?= $(shell $(PKG_CONFIG) --libs $(HDF5_PC) 2>/dev/null) -lhdf5_hl

# boost ships no pkg-config file; pick up the usual MacPorts/Homebrew prefixes
# when they are actually present.
ifneq ($(wildcard /opt/local/include/boost),)
BOOST_CFLAGS  ?= -I/opt/local/include
BOOST_LDFLAGS ?= -L/opt/local/lib
endif
ifneq ($(wildcard /opt/homebrew/include/boost),)
BOOST_CFLAGS  ?= -I/opt/homebrew/include
BOOST_LDFLAGS ?= -L/opt/homebrew/lib
endif

ifeq ($(strip $(HDF5_PC)$(HDF5_CFLAGS)),)
  $(warning no hdf5 pkg-config module found: tried hdf5 and hdf5-serial.)
  $(warning Install the hdf5 development package, or set HDF5_CFLAGS and HDF5_LIBS.)
endif

INCLUDES = -Ic++ -Itests/vendor $(HDF5_CFLAGS) $(BOOST_CFLAGS)

VARIANT ?= plain
ifeq ($(VARIANT),plain)
  VARIANT_CXXFLAGS := -g -O0
  VARIANT_LDFLAGS  :=
else ifeq ($(VARIANT),release)
  VARIANT_CXXFLAGS := -O2 -DNDEBUG
  VARIANT_LDFLAGS  :=
else ifeq ($(VARIANT),asan)
  VARIANT_CXXFLAGS := -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
                      -fno-sanitize-recover=undefined -D_GLIBCXX_ASSERTIONS
  VARIANT_LDFLAGS  := -fsanitize=address,undefined
else
  $(error unknown VARIANT '$(VARIANT)': use plain, release, or asan)
endif

ALL_CXXFLAGS = $(CXXSTD) $(WARN) $(VARIANT_CXXFLAGS) $(INCLUDES) -MMD -MP $(CXXFLAGS)
ALL_LDFLAGS  = $(VARIANT_LDFLAGS) $(BOOST_LDFLAGS) $(LDFLAGS)

OBJDIR     := build/$(VARIANT)
TEST_SRCS  := $(sort $(wildcard tests/cxx/*.cpp))
TEST_OBJS  := $(patsubst tests/cxx/%.cpp,$(OBJDIR)/%.o,$(TEST_SRCS))
TEST_BIN   := $(OBJDIR)/arf_test

# standalone programs for the cross-implementation tests; they have their own
# main(), so they live outside tests/cxx and never join the doctest binary
INTEROP_SRCS := $(sort $(wildcard tests/interop/*.cpp))
INTEROP_BINS := $(patsubst tests/interop/%.cpp,$(OBJDIR)/%,$(INTEROP_SRCS))

SAN_PROBE := build/.san_probe

# Relink when a test source is added or removed.
SRC_STAMP := $(OBJDIR)/.sources
$(shell mkdir -p $(OBJDIR) && printf '%s\n' $(TEST_SRCS) | \
	cmp -s - $(SRC_STAMP) 2>/dev/null || printf '%s\n' $(TEST_SRCS) > $(SRC_STAMP))

.PHONY: all test test-release test-sanitize test-all build-tests interop \
        test-interop golden-update check-sanitizers lint-strict clean install

all: test

# --- doctest suite --------------------------------------------------------

build-tests: $(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS) $(SRC_STAMP)
	$(CXX) $(ALL_LDFLAGS) -o $@ $(TEST_OBJS) $(HDF5_LIBS)

$(OBJDIR)/%.o: tests/cxx/%.cpp | $(OBJDIR)
	$(CXX) $(ALL_CXXFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $@

test: $(TEST_BIN)
	$(TEST_BIN)

test-release:
	$(MAKE) VARIANT=release test

test-sanitize: check-sanitizers
	$(MAKE) VARIANT=asan test

# each variant is a separate submake so the pattern rules aren't duplicated.
# asan is skipped rather than fatal here so the target stays usable on musl;
# CI calls test-sanitize directly, where a missing runtime must fail loudly.
test-all: test
	$(MAKE) VARIANT=release test
	@if $(MAKE) --no-print-directory check-sanitizers >/dev/null 2>&1; then \
		$(MAKE) VARIANT=asan test; \
	else \
		echo "note: skipping asan variant, $(CXX) has no sanitizer runtime"; \
	fi

# --- cross-implementation tests -------------------------------------------

interop: $(INTEROP_BINS)

$(OBJDIR)/%: tests/interop/%.cpp | $(OBJDIR)
	$(CXX) $(ALL_CXXFLAGS) -MF $(OBJDIR)/$*.d $(ALL_LDFLAGS) -o $@ $< $(HDF5_LIBS)

# needs the python environment as well as the C++ one
test-interop: interop
	uv run pytest tests/test_interop.py

golden-update: interop
	ARF_UPDATE_GOLDEN=1 uv run pytest tests/test_interop.py -k golden

# Fail with something more useful than "cannot find -lasan".
check-sanitizers:
	@mkdir -p build
	@printf 'int main(){return 0;}\n' > $(SAN_PROBE).cpp
	@if $(CXX) -fsanitize=address,undefined -o $(SAN_PROBE) $(SAN_PROBE).cpp >/dev/null 2>&1; then \
		rm -f $(SAN_PROBE) $(SAN_PROBE).cpp; \
	else \
		rm -f $(SAN_PROBE) $(SAN_PROBE).cpp; \
		echo "error: $(CXX) cannot link -fsanitize=address,undefined."; \
		echo "       Alpine/musl ships no libasan/libubsan, and LeakSanitizer needs glibc."; \
		echo "       Run this variant on a glibc host or in CI; see CLAUDE.md."; \
		exit 1; \
	fi

# --- checks and housekeeping ----------------------------------------------

# The ordinary build already compiles with -Wpedantic -Werror. This target keeps
# the strict flags explicit and independent of WARN, so it still means something
# if someone loosens the default.
lint-strict:
	$(CXX) $(CXXSTD) -Wall -Wextra -Wpedantic -Werror -fsyntax-only \
		$(INCLUDES) $(TEST_SRCS)

clean:
	rm -rf build
	rm -f tests/*.o test.arf tests/test.arf

install:
	install -d $(PREFIX)/include/arf
	install -m 644 -o root c++/*.hpp $(PREFIX)/include/
	install -m 644 -o root c++/arf/*.hpp $(PREFIX)/include/arf/

# both sets: without the interop dep files, those binaries would not rebuild
# when a header changes, and `make golden-update` would regenerate the golden
# from a stale writer
-include $(TEST_OBJS:.o=.d)
-include $(INTEROP_BINS:=.d)
