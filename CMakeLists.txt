cmake_minimum_required(VERSION 3.21)
project(knndbscan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(KNNDBSCAN_BUILD_LIBRARY "Build the shared library the Python package loads" ON)
option(KNNDBSCAN_BUILD_CLI "Build the MPI command-line program" OFF)

# On macOS with Apple Clang, point OpenMP_ROOT at libomp (e.g. $(brew --prefix libomp)).
find_package(OpenMP REQUIRED COMPONENTS CXX)

# clusters.cpp includes the rest of cpp/core, so the core is one translation unit. It is
# compiled into each target separately because the library and the CLI see different MPIs.
set(KNNDBSCAN_CORE_SOURCE ${PROJECT_SOURCE_DIR}/cpp/core/clusters.cpp)
# The core is the published SC20 implementation; its known-benign warnings are silenced.
set(KNNDBSCAN_CORE_WARNINGS
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-unused-variable -Wno-unused-value
  -Wno-sign-compare -Wno-unused-but-set-variable>)

if(KNNDBSCAN_BUILD_LIBRARY)
  add_library(knndbscan SHARED cpp/capi.cpp ${KNNDBSCAN_CORE_SOURCE})
  target_include_directories(knndbscan PRIVATE cpp/include cpp cpp/core)
  target_link_libraries(knndbscan PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(knndbscan PRIVATE ${KNNDBSCAN_CORE_WARNINGS})
  # Only the knndbscan_* functions of knndbscan.h are exported.
  set_target_properties(knndbscan PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  if(SKBUILD)
    install(TARGETS knndbscan
      LIBRARY DESTINATION knndbscan
      RUNTIME DESTINATION knndbscan)
  endif()
endif()

if(KNNDBSCAN_BUILD_CLI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(knndbscan_cli cpp/cli/knndbscan_cli.cpp ${KNNDBSCAN_CORE_SOURCE})
  set_target_properties(knndbscan_cli PROPERTIES OUTPUT_NAME knndbscan)
  target_include_directories(knndbscan_cli PRIVATE cpp cpp/core)
  target_compile_definitions(knndbscan_cli PRIVATE KNNDBSCAN_WITH_MPI)
  target_link_libraries(knndbscan_cli PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_compile_options(knndbscan_cli PRIVATE ${KNNDBSCAN_CORE_WARNINGS})
endif()
