cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)
project(app LANGUAGES C CXX)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/app.cpp)

include_directories(${CMAKE_SOURCE_DIR}/3rd_party/mariana_llm/include)
link_directories(${CMAKE_SOURCE_DIR}/3rd_party/mariana_llm/lib)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rd_party/opencv_cuda/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(app ${SOURCES} )
#file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/test/single_op_run.cpp)
#add_executable(op_test ${SOURCES} ) 
target_link_libraries(app PUBLIC mariana_llm ${OpenCV_LIBS})

#target_link_libraries(op_test PUBLIC ${CMAKE_PROJECT_NAME})
