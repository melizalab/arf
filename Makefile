# arf C++ build and test driver.
#
# The library is header-only; everything built here is a test.
#
#   make test              build and run the doctest suite (VARIANT=plain)
#   make test-release      same, compiled -O2 -DNDEBUG (proves the suite does
#                          not depend on assert() for its checks)
#   make test-sanitize     same, under AddressSanitizer + UBSan
#   make test-all          all three
#   make lint-strict       informational -Wpedantic -Werror syntax check
#   make install           install the headers under $(PREFIX)
#
# Objects go to build/$(VARIANT)/. Sanitizers require a glibc target.

PREFIX     ?= /usr/local
CXX        ?= g++
CXXSTD     ?= -std=c++11
WARN       ?= -Wall -Wextra
PKG_CONFIG ?= pkg-config

HDF5_CFLAGS ?= $(shell $(PKG_CONFIG) --cflags hdf5)
HDF5_LIBS   ?= $(shell $(PKG_CONFIG) --libs hdf5) -lhdf5_hl

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
# the pre-doctest test program, kept building until phase 3 ports it
LEGACY_BIN := tests/test_arf

SAN_PROBE := build/.san_probe

.PHONY: all test test-release test-sanitize test-all build-tests legacy \
        check-sanitizers lint-strict clean install

all: test

# --- doctest suite --------------------------------------------------------

build-tests: $(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS)
	$(CXX) $(ALL_LDFLAGS) -o $@ $^ $(HDF5_LIBS)

$(OBJDIR)/%.o: tests/cxx/%.cpp | $(OBJDIR)
	$(CXX) $(ALL_CXXFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $@

test: $(TEST_BIN) $(LEGACY_BIN)
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

# --- legacy test program --------------------------------------------------

legacy: $(LEGACY_BIN)

$(LEGACY_BIN): tests/test_arf.cpp | $(OBJDIR)
	$(CXX) $(ALL_CXXFLAGS) -MF $(OBJDIR)/test_arf.d $(ALL_LDFLAGS) -o $@ $< $(HDF5_LIBS)

# --- checks and housekeeping ----------------------------------------------

# Informational only: the VLAs and unused parameters in the headers still fail
# this.
lint-strict:
	-$(CXX) $(CXXSTD) -Wall -Wextra -Wpedantic -Werror -fsyntax-only \
		$(INCLUDES) $(TEST_SRCS) tests/test_arf.cpp

clean:
	rm -rf build
	rm -f $(LEGACY_BIN) tests/*.o test.arf tests/test.arf

install:
	install -d $(PREFIX)/include/arf
	install -m 644 -o root c++/*.hpp $(PREFIX)/include/
	install -m 644 -o root c++/arf/*.hpp $(PREFIX)/include/arf/

-include $(TEST_OBJS:.o=.d)
