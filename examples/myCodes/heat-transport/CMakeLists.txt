#SET(TARGET "heat-transport")
##SET(TARGET_SRC
##    heat-transport.cpp
##    LaplaceProblemFEM.cpp
##         )

#FILE(GLOB sources *.cpp)

#cmake_minimum_required(VERSION 2.8.8)

#SET(DEAL_II_DIR ../../../build/lib/cmake/deal.II/)

#FIND_PACKAGE(deal.II 8.4 QUIET
#  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
#  )
#IF(NOT ${deal.II_FOUND})
#  MESSAGE(FATAL_ERROR "\n"
#    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
#    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
#    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
#    )
#ENDIF()

#DEAL_II_INITIALIZE_CACHED_VARIABLES()
#PROJECT(${TARGET})
##aux_source_directory(. SRC_LIST)
##add_executable(${PROJECT_NAME} ${SRC_LIST})
#ADD_EXECUTABLE(heat-transport ${sources})

#INCLUDE_DIRECTORIES(/Users/lisa/Desktop/dealii/include)

##DEAL_II_INVOKE_AUTOPILOT()



SET(TARGET heat-transport)

SET(TARGET_SRC
  ${TARGET}.cpp
  )


CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.2 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()



