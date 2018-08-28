cmake_minimum_required(VERSION 3.0)

project(tf_kaldi_io)

set(PYTHON_EXECUTABLE "python" CACHE STRING "python implementation")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

if ("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif ()

include(ops/kaldi-io.cmake)

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE TF_COMPILE_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

# retrieve tensorflow include dir and lib dir
execute_process( COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')"
        OUTPUT_VARIABLE TENSORFLOW_LIB_DIR )
find_library( TENSORFLOW_FRAMEWORK_LIB tensorflow_framework PATHS "${TENSORFLOW_LIB_DIR}" NO_DEFAULT_PATH )
message( STATUS "${TENSORFLOW_LIB_DIR}" )
message( STATUS "${TENSORFLOW_FRAMEWORK_LIB}" )

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS}")

set(SOURCE_FILES ops/kaldi_reader_dataset_op.cc)

add_library(tf_kaldi_io SHARED ${SOURCE_FILES})
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

target_link_libraries(tf_kaldi_io
        KALDI_IO_LIBS)

if( TENSORFLOW_FRAMEWORK_LIB )
    target_link_libraries( tf_kaldi_io ${TENSORFLOW_FRAMEWORK_LIB} )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set_target_properties(tf_kaldi_io PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../tf_kaldi_io")
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set_target_properties(tf_kaldi_io PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/../tf_kaldi_io")
else()
    message( FATAL_ERROR "Just support linux or darwin, CMake will exit." )
endif()
