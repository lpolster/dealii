# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation

# Include any dependencies generated for this target.
include CMakeFiles/step-3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-3.dir/flags.make

CMakeFiles/step-3.dir/step-3.cc.o: CMakeFiles/step-3.dir/flags.make
CMakeFiles/step-3.dir/step-3.cc.o: step-3.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/step-3.dir/step-3.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/step-3.dir/step-3.cc.o -c /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation/step-3.cc

CMakeFiles/step-3.dir/step-3.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-3.dir/step-3.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation/step-3.cc > CMakeFiles/step-3.dir/step-3.cc.i

CMakeFiles/step-3.dir/step-3.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-3.dir/step-3.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation/step-3.cc -o CMakeFiles/step-3.dir/step-3.cc.s

CMakeFiles/step-3.dir/step-3.cc.o.requires:
.PHONY : CMakeFiles/step-3.dir/step-3.cc.o.requires

CMakeFiles/step-3.dir/step-3.cc.o.provides: CMakeFiles/step-3.dir/step-3.cc.o.requires
	$(MAKE) -f CMakeFiles/step-3.dir/build.make CMakeFiles/step-3.dir/step-3.cc.o.provides.build
.PHONY : CMakeFiles/step-3.dir/step-3.cc.o.provides

CMakeFiles/step-3.dir/step-3.cc.o.provides.build: CMakeFiles/step-3.dir/step-3.cc.o

# Object files for target step-3
step__3_OBJECTS = \
"CMakeFiles/step-3.dir/step-3.cc.o"

# External object files for target step-3
step__3_EXTERNAL_OBJECTS =

step-3: CMakeFiles/step-3.dir/step-3.cc.o
step-3: CMakeFiles/step-3.dir/build.make
step-3: /home/lisa/Desktop/dealii/lib/libdeal_II.g.so.8.4.pre
step-3: /usr/lib/liblapack.so.3gf
step-3: /usr/lib/libblas.so.3gf
step-3: /usr/lib/x86_64-linux-gnu/libz.so
step-3: CMakeFiles/step-3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable step-3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-3.dir/build: step-3
.PHONY : CMakeFiles/step-3.dir/build

CMakeFiles/step-3.dir/requires: CMakeFiles/step-3.dir/step-3.cc.o.requires
.PHONY : CMakeFiles/step-3.dir/requires

CMakeFiles/step-3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-3.dir/clean

CMakeFiles/step-3.dir/depend:
	cd /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation /home/lisa/Desktop/dealii/examples/myCodes/laser-stimulation/CMakeFiles/step-3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-3.dir/depend

