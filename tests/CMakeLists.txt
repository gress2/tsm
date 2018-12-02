include_directories(include)

add_executable(generic_game_tests generic_game_tests.cc)
target_link_libraries(generic_game_tests ${LIBS})
add_test(NAME generic_game_tests COMMAND generic_game_tests)

add_executable(util_tests util_tests.cc)
target_link_libraries(util_tests ${LIBS})
add_test(NAME util_tests COMMAND util_tests)