include(ExternalProject)

if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/kaldi-io/include/kaldi-io.h")
    ExternalProject_Add(
            kaldi-io
            GIT_REPOSITORY https://github.com/open-speech/kaldi-io.git
            GIT_TAG "tf_kaldi_io"
            STAMP_DIR "kaldi-io-stamp"
            DOWNLOAD_DIR "kaldi-io"
            SOURCE_DIR "kaldi-io"
            INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kaldi-io
            BUILD_IN_SOURCE 1
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND mkdir -p build
            COMMAND cmake -DCMAKE_SHARED_LIBRARY_SUFFIX_CXX=.so -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -Bbuild -H.
            COMMAND cd build && make -j 4
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
    )
endif()

set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/kaldi-io)

include_directories(${INSTALL_DIR})
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_library(KALDI_IO_LIBS INTERFACE)

add_dependencies(KALDI_IO_LIBS kaldi-io)

target_link_libraries(KALDI_IO_LIBS INTERFACE kaldi_io_shared)
