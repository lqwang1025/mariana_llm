set(ABSEIL_ROOT ${3RD_PARTY}/abseil-cpp)

if(NOT EXISTS ${ABSEIL_ROOT}/include AND NOT EXISTS ${ABSEIL_ROOT}/lib)
  include(ExternalProject)

  set(ABSEIL_GIT_TAG 20220623.rc1)
  set(ABSEIL_GIT_URL https://github.com/abseil/abseil-cpp)
  set(ABSEIL_CONFIGURE cd ${ABSEIL_ROOT}/src/abseil-cpp && cmake -B build -D CMAKE_INSTALL_PREFIX=${ABSEIL_ROOT} -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
	-D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CMAKE_CXX_STANDARD=17 -D ABSL_PROPAGATE_CXX_STD=ON .)
  set(ABSEIL_MAKE  cd ${ABSEIL_ROOT}/src/abseil-cpp/build && make -j8)

  set(ABSEIL_INSTALL cd ${ABSEIL_ROOT}/src/abseil-cpp/build &&
	find ./ -name "*.o" | xargs ar cr libabsl.a && make install &&
	cd ${ABSEIL_ROOT}/lib && find ./ -name "*.a" | xargs -l rm &&
	mv ${ABSEIL_ROOT}/src/abseil-cpp/build/libabsl.a ./
	)

  ExternalProject_Add(abseil-cpp
	PREFIX            ${ABSEIL_ROOT}
	GIT_REPOSITORY    ${ABSEIL_GIT_URL}
	GIT_TAG           ${ABSEIL_GIT_TAG}
	CONFIGURE_COMMAND ${ABSEIL_CONFIGURE}
	BUILD_COMMAND     ${ABSEIL_MAKE}
	INSTALL_COMMAND   ${ABSEIL_INSTALL}
	BUILD_ALWAYS FALSE
	)
endif()

set(ABSEIL_LIB_DIR ${ABSEIL_ROOT}/lib)
set(ABSEIL_INCLUDE_DIR ${ABSEIL_ROOT}/include)
link_directories(${ABSEIL_LIB_DIR})
include_directories(${ABSEIL_INCLUDE_DIR})

list(APPEND MARIANA_EXTERN_LIB -labsl)
