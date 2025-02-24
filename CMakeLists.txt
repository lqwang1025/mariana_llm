cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

if(NOT DEFINED MAR_VERSION_MAJOR)
  set(MAR_VERSION_MAJOR 1)
endif()
if(NOT DEFINED MAR_VERSION_MINOR)
  set(MAR_VERSION_MINOR 0)
endif()
if(NOT DEFINED MAR_VERSION_PATCH)
  set(MAR_VERSION_PATCH 0)
endif()
if(NOT DEFINED MAR_VERSION_BUILD)
  set(MAR_VERSION_BUILD 0)
endif()

set(MAR_VERSION ${MAR_VERSION_MAJOR}.${MAR_VERSION_MINOR}.${MAR_VERSION_PATCH}.${MAR_VERSION_BUILD} CACHE STRING "Mariana-llm logical version")
set(MAR_SOVERSION ${MAR_VERSION_MAJOR}.${MAR_VERSION_MINOR}.${MAR_VERSION_PATCH}.${MAR_VERSION_BUILD} CACHE STRING "Mariana-llm soname version")

include(cmake/git.cmake)
configure_file(cmake/version.h.in generated/version.h @ONLY)

IF (NOT DEFINED CMAKE_INSTALL_PREFIX)
   SET (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Mariana Installation Directory")
 ENDIF()
 
project(mariana_llm VERSION ${MAR_VERSION} LANGUAGES C CXX)

option(OPEN_PROFILER "Profiler this library" OFF)
option(USE_CUDA "Compile with cuda" OFF)

if (OPEN_PROFILER)
  add_definitions(-DOPEN_TIME_TRACE)
endif()
if (USE_CUDA)
  add_definitions(-DMLM_USE_CUDA)
endif()

IF("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  add_definitions(-DDEBUG)
  message(STATUS "building for: debugging")
  set(CXX_FORMAT_SIGNEDNESS "-Wformat-signedness")
  set(CXX_FORMAT_SECURITY "-Werror=format-security")
  set(CXX_STACK_PROTECTOR "-fstack-protector-all")
  set(CXX_FLAGS_DEBUG "-O0")
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -ggdb -Wl,--no-undefined -Wall -Wextra -DNETDATA_INTERNAL_CHECKS=1 -fPIC -DNETDATA_VERIFY_LOCKS=1 ${CXX_FORMAT_SIGNEDNESS} ${CXX_FORMAT_SECURITY} ${CXX_STACK_PROTECTOR} ${CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined -Wall -fPIC -O0 -ggdb -std=c++11 -lrt -ldl -ffast-math")
ELSE()
  message(STATUS "building for: release")
  cmake_policy(SET CMP0069 "NEW")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT error)
  IF(${ipo_supported})
    message(STATUS "link time optimization: supported")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  ELSE()
    message(STATUS "link time optimization: not supported")
  ENDIF()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined -Wall -fPIC -O3 -std=c++11 -lrt -ldl -ffast-math")
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined -Wall -fPIC -O3 -lrt -ldl -ffast-math")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARIANA_TOOLCHAIN_FLAG}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MARIANA_TOOLCHAIN_FLAG}")

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(3RD_PARTY ${CMAKE_SOURCE_DIR}/3rd_party)
add_compile_options(-fPIC)
add_compile_options(-mfma)
add_compile_options(-mavx2)
# include(cmake/torch.cmake)

include_directories(${CMAKE_BINARY_DIR}/generated)
include_directories(${CMAKE_SOURCE_DIR}/include)
include(${CMAKE_SOURCE_DIR}/src/CMakeLists.txt)


add_library(${CMAKE_PROJECT_NAME} SHARED)
target_sources(${CMAKE_PROJECT_NAME} PUBLIC ${MARIANA_LLM_SRC})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${MARIANA_EXTERN_LIB} pthread)

file(GLOB INSTALL_HEADERS ${CMAKE_CURRENT_LIST_DIR}/include/mariana_llm/*.h)
set_target_properties(${CMAKE_PROJECT_NAME}
  PROPERTIES
  VERSION ${MAR_VERSION}
  SOVERSION ${MAR_SOVERSION}
  OUTPUT_NAME "mariana_llm"
  POSITION_INDEPENDENT_CODE 1
  PUBLIC_HEADER "${INSTALL_HEADERS}"
  DEBUG_POSTFIX ".d"
)

include(GNUInstallDirs)
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")

install(
  TARGETS
  ${CMAKE_PROJECT_NAME}
  ARCHIVE
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  RUNTIME
  DESTINATION ${INSTALL_BINDIR}
  COMPONENT bin
  LIBRARY
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  PUBLIC_HEADER
  DESTINATION ${INSTALL_INCLUDEDIR}/mariana_llm
  COMPONENT dev
)
