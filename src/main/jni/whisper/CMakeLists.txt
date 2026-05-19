cmake_minimum_required(VERSION 3.10)

project(whisper.cpp)

set(CMAKE_CXX_STANDARD 17)

# Path to whisper.cpp source tree
# When building via Gradle, this resolves via the 6-level relative path.
# For standalone builds, set -DWHISPER_CPP_DIR=/path/to/whisper.cpp
if(NOT DEFINED WHISPER_CPP_DIR)
    set(WHISPER_CPP_DIR "${CMAKE_SOURCE_DIR}/../../../../../../..")
endif()
get_filename_component(WHISPER_CPP_DIR "${WHISPER_CPP_DIR}" ABSOLUTE)

message(STATUS " WHISPER_CPP_DIR: ${WHISPER_CPP_DIR}")

# Get whisper.cpp version
file(READ "${WHISPER_CPP_DIR}/CMakeLists.txt" MAIN_CMAKE_CONTENT)
string(REGEX MATCH "project\\(\"whisper\\.cpp\" VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)\\)" VERSION_MATCH "${MAIN_CMAKE_CONTENT}")
if(CMAKE_MATCH_1)
    set(WHISPER_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
    message(STATUS " Whisper version: ${WHISPER_VERSION}")
else()
    set(WHISPER_VERSION "unknown" PARENT_SCOPE)
    message(WARNING " Could not detect whisper.cpp version")
endif()

option(GGML_HOME "whisper: Path to external GGML source" OFF)

set(
    SOURCE_FILES
    ${WHISPER_CPP_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/jni.c
    )

find_library(LOG_LIB log)

include(FetchContent)

function(build_library target_name)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_compile_definitions(${target_name} PUBLIC GGML_USE_CPU)
    target_compile_definitions(${target_name} PRIVATE WHISPER_VERSION="${WHISPER_VERSION}")

    # ARM-specific optimizations
    if (${target_name} STREQUAL "whisper_v8fp16_va")
        target_compile_options(${target_name} PRIVATE -march=armv8.2-a+fp16)
        set(GGML_COMPILE_OPTIONS                      -march=armv8.2-a+fp16)
    elseif (${target_name} STREQUAL "whisper_vfpv4")
        target_compile_options(${target_name} PRIVATE -mfpu=neon-vfpv4)
        set(GGML_COMPILE_OPTIONS                      -mfpu=neon-vfpv4)
    endif ()

    # Release optimizations: -O3, LTO, dead code stripping
    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -O3)
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)
        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        target_link_options(${target_name} PRIVATE -flto)
    endif ()

    # Fetch GGML
    if (GGML_HOME)
        FetchContent_Declare(ggml SOURCE_DIR ${GGML_HOME})
    else()
        FetchContent_Declare(ggml SOURCE_DIR ${WHISPER_CPP_DIR}/ggml)
    endif()
    FetchContent_MakeAvailable(ggml)
    target_compile_options(ggml PRIVATE ${GGML_COMPILE_OPTIONS})
    target_link_libraries(${target_name} ${LOG_LIB} android ggml)

endfunction()

# Build ABI-specific optimized variants (ARM only)
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("whisper_v8fp16_va")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("whisper_vfpv4")
endif ()

# Always build the default portable target
build_library("whisper")

include_directories(
    ${WHISPER_CPP_DIR}
    ${WHISPER_CPP_DIR}/src
    ${WHISPER_CPP_DIR}/include
    ${WHISPER_CPP_DIR}/ggml/include
    ${WHISPER_CPP_DIR}/ggml/src
    ${WHISPER_CPP_DIR}/ggml/src/ggml-cpu
)
