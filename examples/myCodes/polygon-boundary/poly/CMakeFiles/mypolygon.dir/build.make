# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly

# Include any dependencies generated for this target.
include CMakeFiles/mypolygon.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mypolygon.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mypolygon.dir/flags.make

CMakeFiles/mypolygon.dir/mypolygon.cc.o: CMakeFiles/mypolygon.dir/flags.make
CMakeFiles/mypolygon.dir/mypolygon.cc.o: mypolygon.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mypolygon.dir/mypolygon.cc.o"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/bin/mpic++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mypolygon.dir/mypolygon.cc.o -c /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly/mypolygon.cc

CMakeFiles/mypolygon.dir/mypolygon.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mypolygon.dir/mypolygon.cc.i"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/bin/mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly/mypolygon.cc > CMakeFiles/mypolygon.dir/mypolygon.cc.i

CMakeFiles/mypolygon.dir/mypolygon.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mypolygon.dir/mypolygon.cc.s"
	/Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/bin/mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly/mypolygon.cc -o CMakeFiles/mypolygon.dir/mypolygon.cc.s

CMakeFiles/mypolygon.dir/mypolygon.cc.o.requires:
.PHONY : CMakeFiles/mypolygon.dir/mypolygon.cc.o.requires

CMakeFiles/mypolygon.dir/mypolygon.cc.o.provides: CMakeFiles/mypolygon.dir/mypolygon.cc.o.requires
	$(MAKE) -f CMakeFiles/mypolygon.dir/build.make CMakeFiles/mypolygon.dir/mypolygon.cc.o.provides.build
.PHONY : CMakeFiles/mypolygon.dir/mypolygon.cc.o.provides

CMakeFiles/mypolygon.dir/mypolygon.cc.o.provides.build: CMakeFiles/mypolygon.dir/mypolygon.cc.o

# Object files for target mypolygon
mypolygon_OBJECTS = \
"CMakeFiles/mypolygon.dir/mypolygon.cc.o"

# External object files for target mypolygon
mypolygon_EXTERNAL_OBJECTS =

mypolygon: CMakeFiles/mypolygon.dir/mypolygon.cc.o
mypolygon: CMakeFiles/mypolygon.dir/build.make
mypolygon: /Applications/deal.II.app/Contents/Resources/lib/libdeal_II.g.8.2.1.dylib
mypolygon: /usr/lib/libbz2.dylib
mypolygon: /usr/lib/libz.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtrilinoscouplings.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libpiro.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libmoochothyra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libmoocho.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/librythmos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libmuelu-adapters.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libmuelu.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/liblocathyra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/liblocaepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/liblocalapack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libloca.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libnoxepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libnoxlapack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libnox.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libintrepid.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikosbelos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikosaztecoo.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikosamesos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikosml.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libstratimikosifpack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libifpack2-adapters.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libifpack2.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libanasazitpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libModeLaplace.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libanasaziepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libanasazi.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libbelostpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libbelosepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libbelos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libml.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkomplex.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libifpack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libpamgen_extras.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libpamgen.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libamesos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libgaleri-xpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libgaleri.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libaztecoo.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libisorropia.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/liboptipack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libthyratpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libthyraepetraext.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libthyraepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libthyracore.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libxpetra-sup.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libxpetra-ext.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libxpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libepetraext.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtpetraext.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtpetrainout.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtpetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtriutils.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libglobipack.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libshards.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libzoltan.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libepetra.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libsacado.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkosdisttsqr.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkosnodetsqr.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkoslinalg.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkosnodeapi.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkos.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libkokkosTPL_unused_dummy.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/librtop.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libtpi.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libteuchosremainder.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libteuchosnumerics.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libteuchoscomm.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libteuchosparameterlist.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/trilinos-70824c5/lib/libteuchoscore.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKBO.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKBool.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKBRep.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKernel.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKFeat.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKFillet.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKG2d.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKG3d.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKGeomAlgo.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKGeomBase.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKHLR.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKIGES.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKMath.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKMesh.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKOffset.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKPrim.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKShHealing.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKSTEP.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKSTEPAttr.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKSTEPBase.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKSTL.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKTopAlgo.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/oce-6d4ba0e/lib/libTKXSBase.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/p4est-fb278b3/lib/libp4est.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/p4est-fb278b3/lib/libsc.dylib
mypolygon: /usr/lib/libm.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/slepc-c81b9e0/lib/libslepc.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libpetsc.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libHYPRE.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libcmumps.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libdmumps.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsmumps.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libzmumps.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libmumps_common.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libpord.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libscalapack.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsundials_cvode.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsundials_nvecserial.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsundials_nvecparallel.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsuperlu_4.3.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libsuperlu_dist_3.3.a
mypolygon: /usr/lib/liblapack.dylib
mypolygon: /usr/lib/libblas.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libparmetis.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libmetis.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libhdf5hl_fortran.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libhdf5_fortran.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libhdf5_hl.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libhdf5.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/petsc-38ae631/lib/libhwloc.dylib
mypolygon: /usr/lib/libssl.dylib
mypolygon: /usr/lib/libcrypto.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/lib/libmpi_f90.a
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/lib/libmpi_f77.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/gfortran/lib/libgcc_ext.10.5.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/lib/libmpi_cxx.dylib
mypolygon: /usr/lib/libc++.dylib
mypolygon: /Applications/deal.II.app/Contents/Resources/opt/openmpi-1.6.5/lib/libmpi.dylib
mypolygon: CMakeFiles/mypolygon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable mypolygon"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mypolygon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mypolygon.dir/build: mypolygon
.PHONY : CMakeFiles/mypolygon.dir/build

CMakeFiles/mypolygon.dir/requires: CMakeFiles/mypolygon.dir/mypolygon.cc.o.requires
.PHONY : CMakeFiles/mypolygon.dir/requires

CMakeFiles/mypolygon.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mypolygon.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mypolygon.dir/clean

CMakeFiles/mypolygon.dir/depend:
	cd /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly /Users/lisa/Desktop/dealii/examples/myCodes/polygon-boundary/poly/CMakeFiles/mypolygon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mypolygon.dir/depend

