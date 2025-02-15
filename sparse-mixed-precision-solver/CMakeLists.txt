cmake_minimum_required(VERSION 3.16)
project(SolverProject)

# Debug/Release settings for MSVC
if(MSVC)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=2)
  else()
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
  endif()
endif()

# Include directories (Eigen is in thirdparty/eigen)
include_directories(include thirdparty/eigen)

# Sources for both static and shared libraries
set(SOURCES src/Solver.cpp src/SparseMatrix.cpp)

# Static library
add_library(SolverLib STATIC ${SOURCES})
target_include_directories(SolverLib PUBLIC include)
target_compile_definitions(SolverLib PRIVATE SOLVER_STATIC)

# Shared library (DLL) ----------------------------------------
add_library(SolverDLL SHARED ${SOURCES})
target_include_directories(SolverDLL PUBLIC include)

# Set output directory for DLL and test executable
set(SHARED_LIB_DIR "${CMAKE_BINARY_DIR}/shared_library")
file(MAKE_DIRECTORY ${SHARED_LIB_DIR})

# For DLL
set_target_properties(SolverDLL PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${SHARED_LIB_DIR}  # DLL on Windows
    LIBRARY_OUTPUT_DIRECTORY ${SHARED_LIB_DIR}  # .so on Linux/macOS
)

# For test executable
add_executable(test_Solver test/test_Solver.cpp)
set_target_properties(test_Solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${SHARED_LIB_DIR}  # Force test exe to shared_library
)

# Windows-specific settings
if(WIN32)
  target_compile_definitions(SolverDLL PRIVATE SOLVER_DLL_EXPORTS)
  set_target_properties(SolverDLL PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()

# Tests
enable_testing()
add_subdirectory(thirdparty/googletest)
target_link_libraries(test_Solver PRIVATE SolverDLL gtest gtest_main)
add_test(NAME SolverTests COMMAND test_Solver)