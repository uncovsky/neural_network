add_library( rng random.cpp )
add_library( dependencies trainer.cpp model.cpp optimizer.cpp )

add_executable( neural-net main.cpp )

target_include_directories( neural-net PRIVATE testing )
target_link_libraries( neural-net rng dependencies )
