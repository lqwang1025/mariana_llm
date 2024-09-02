if(NOT DEFINED ENV{CUDA_HOME})
  message(FATAL_ERROR "Please export CUDA_HOME=/your/cuda/home")  
endif()
if(NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
  set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME}) # 
endif()

if(NOT DEFINED ENV{CUDA_INCLUDE_DIRS})
  set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
if(NOT DEFINED ENV{CUDA_TOOLKIT_INCLUDE})
  set(CUDA_TOOLKIT_INCLUDE ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
if(NOT DEFINED ENV{CUDA_CUDART_LIBRARY})
  list(APPEND CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
  list(APPEND CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas.so)
  list(APPEND CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt.so)
  
  list(APPEND CUDA_STATIC_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a)
  list(APPEND CUDA_STATIC_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a)
  list(APPEND CUDA_STATIC_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt_static.a)
  list(APPEND CUDA_STATIC_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a)
endif()

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) 
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

if(NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
else()
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w --default-stream per-thread ${EXTRA_LIBS}")
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
  else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
  
  list(LENGTH CUDA_ARCH_FLAGS_readable_code arch_count)
  # Current Supported Arch List 
  IF (${arch_count} EQUAL 1)
    set(support_archs 60 61 62 70 72 75 80 86 89)
    list(FIND support_archs ${CUDA_ARCH_FLAGS_readable_code} list_index)
    IF (${list_index} EQUAL -1)
      message(FATAL_ERROR "Please add your own sm arch ${CUDA_ARCH_FLAGS_readable_code} to CmakeLists.txt!")
    ENDIF()
  ENDIF()

  IF ((CUDA_VERSION VERSION_GREATER "8.0") OR (CUDA_VERSION VERSION_EQUAL "8.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
  ENDIF()
  
  IF ((CUDA_VERSION VERSION_GREATER "10.1") OR (CUDA_VERSION VERSION_EQUAL "10.1"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
  ENDIF()

  IF ((CUDA_VERSION VERSION_GREATER "10.2") OR (CUDA_VERSION VERSION_EQUAL "10.2"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
  ENDIF()

  IF ((CUDA_VERSION VERSION_GREATER "11.2") OR (CUDA_VERSION VERSION_EQUAL "11.2"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
  ENDIF()

  # Limit minimum cuda version for each archs
  IF (${arch_count} EQUAL 1)
    IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "80") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "80"))
      IF (CUDA_VERSION VERSION_LESS "11.2")
        message(FATAL_ERROR "Please update cuda version to 11.2 or higher!")
      ENDIF()
    ENDIF()

    IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "75") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "75"))
      IF (CUDA_VERSION VERSION_LESS "10.2")
        message(FATAL_ERROR "Please update cuda version to 10.2 or higher!")
      ENDIF()
    ENDIF()

    IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "70") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "70"))
      IF (CUDA_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR "Please update cuda version to 10.1 or higher!")
      ENDIF()
    ENDIF()
  ENDIF()

  message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
    " archs: ${CUDA_ARCH_FLAGS_readable})")
endif()

list(APPEND MARIANA_EXTERN_LIB ${CUDA_CUDART_LIBRARY})
