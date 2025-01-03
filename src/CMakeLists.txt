add_library( rng random.cpp )
add_executable( neuronka main.cpp )

target_include_directories( neuronka PRIVATE testing )

target_link_libraries( neuronka rng )
