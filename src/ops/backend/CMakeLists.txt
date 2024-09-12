include(${CMAKE_CURRENT_LIST_DIR}/cpu/CMakeLists.txt)
if (USE_CUDA)
  include(${CMAKE_CURRENT_LIST_DIR}/gpu/CMakeLists.txt)
endif()
