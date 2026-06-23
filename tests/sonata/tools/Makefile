# Makefile for sonata_hook fail-open test harness
#
# Builds test_hook_harness by compiling sonata_hook.cpp with a stub
# aicpu_entry.  Run from pypto-sonata/ root:
#   make -C tests/sonata/tools -f Makefile test
#
# Requires: g++ or clang++ with C++17 support.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O0 -g
RM       = rm -f

# Paths (relative to this Makefile's directory)
HOOK_CPP  = ../../../runtime/sonata_tmarb/aicpu/sonata_hook.cpp
HOOK_H    = ../../../runtime/sonata_tmarb/include/sonata_hook.h
SCHED_H   = ../../../runtime/sonata_tmarb/runtime/flat_schedule.h

TARGET    = test_hook_harness
SOURCES   = test_hook_harness.cpp $(HOOK_CPP)
INCLUDES  = -I../../../runtime/sonata_tmarb/include \
            -I../../../runtime/sonata_tmarb/runtime

.PHONY: all test clean

all: $(TARGET)

$(TARGET): test_hook_harness.cpp $(HOOK_CPP) $(HOOK_H) $(SCHED_H)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
