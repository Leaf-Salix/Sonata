# Makefile for sonata_hook fail-open test harness
#
# Builds test_hook_harness by compiling sonata_hook.cpp with a stub
# aicpu_entry.  Run from pypto-sonata/ root:
#   make -C tests/sonata/tools test
#
# Requires: g++ or clang++ with C++17 support, and GoogleTest (gtest +
# gtest_main). Mirrors upstream/pypto/runtime/tests/ut/cpp/CMakeLists.txt's
# search order: system install first (apt libgtest-dev / brew googletest),
# falling back to /usr/local. Unlike the upstream CMake build, this Makefile
# has no FetchContent fallback — install gtest locally first
# (`brew install googletest` on macOS, `apt install libgtest-dev` on Linux).

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O0 -g
RM       = rm -f

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  GTEST_SEARCH_PATHS = /opt/homebrew /usr/local
else
  GTEST_SEARCH_PATHS = /usr/local
endif

# Prefix (e.g. /opt/homebrew) whose lib/libgtest.a exists; empty if not found.
GTEST_ROOT := $(firstword $(foreach p,$(GTEST_SEARCH_PATHS),$(if $(wildcard $(p)/lib/libgtest.a),$(p))))

# Paths (relative to this Makefile's directory)
HOOK_CPP  = ../../../runtime/sonata_tmarb/aicpu/sonata_hook.cpp
HOOK_H    = ../../../runtime/sonata_tmarb/include/sonata_hook.h
SCHED_H   = ../../../runtime/sonata_tmarb/runtime/flat_schedule.h

TARGET    = test_hook_harness
SOURCES   = test_hook_harness.cpp $(HOOK_CPP)
INCLUDES  = -I../../../runtime/sonata_tmarb/include \
            -I../../../runtime/sonata_tmarb/runtime
GTEST_LIBS = -lgtest -lgtest_main -lpthread

.PHONY: all test clean

all: $(TARGET)

$(TARGET): test_hook_harness.cpp $(HOOK_CPP) $(HOOK_H) $(SCHED_H)
	@if [ -z "$(GTEST_ROOT)" ]; then \
		echo "ERROR: GoogleTest not found under $(GTEST_SEARCH_PATHS)."; \
		echo "Install it first: brew install googletest (macOS) or apt install libgtest-dev (Linux)."; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(GTEST_ROOT)/include \
		$(SOURCES) -L$(GTEST_ROOT)/lib $(GTEST_LIBS) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
