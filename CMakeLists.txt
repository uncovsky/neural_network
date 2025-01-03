cmake_minimum_required( VERSION 3.12 )

project( neuronka )
 
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_BUILD_TYPE Release )

# default compiler set to gcc
set( CMAKE_CXX_COMPILER "/usr/bin/g++" )

set( CXX_OPTIONS -Wall -Wextra -Werror -funroll-loops -march=native )
set( CXX_DEBUG_OPTIONS -g )
set( CXX_RELEASE_OPTIONS -O3 )

# Add include
include_directories(${CMAKE_SOURCE_DIR}/include)

# set build folder as default destination for generated binaries
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )

add_subdirectory( src )

