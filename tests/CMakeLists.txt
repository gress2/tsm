include_directories(include)
add_executable(generic_game generic_game.cc)
target_link_libraries(generic_game ${LIBS})
add_test(NAME generic_game COMMAND generic_game)