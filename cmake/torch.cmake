find_package(PythonInterp REQUIRED)

execute_process(COMMAND
	${PYTHON_EXECUTABLE} "-c" "import re, torch; print(re.compile('/__init__.py.*').sub('', torch.__file__))"
	RESULT_VARIABLE _torch_status
	OUTPUT_VARIABLE _torch_location
	ERROR_QUIET
	OUTPUT_STRIP_TRAILING_WHITESPACE)
	
if(NOT _torch_status)
   set(TORCH ${_torch_location} CACHE STRING "Location of Numpy")
endif()

execute_process(COMMAND
	${PYTHON_EXECUTABLE} "-c" "import torch; print(torch.__version__)"
	OUTPUT_VARIABLE _torch_version
	ERROR_QUIET
	OUTPUT_STRIP_TRAILING_WHITESPACE)

set(Torch_DIR ${_torch_location}/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
list(APPEND MARIANA_EXTERN_LIB ${TORCH_LIBRARIES})