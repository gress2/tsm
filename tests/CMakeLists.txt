include_directories(include)

add_executable(generic_game_tests generic_game_tests.cc)
target_link_libraries(generic_game_tests ${LIBS})
add_test(NAME generic_game_tests COMMAND generic_game_tests)

add_executable(same_game_tests same_game_tests.cc)
target_link_libraries(same_game_tests ${LIBS})
add_test(NAME same_game_tests COMMAND same_game_tests)

add_executable(util_tests util_tests.cc)
target_link_libraries(util_tests ${LIBS})
add_test(NAME util_tests COMMAND util_tests)

add_executable(mcts_tests mcts_tests.cc)
target_link_libraries(mcts_tests ${LIBS})
add_test(NAME mcts_tests COMMAND mcts_tests)

add_executable(sample_finite_mixture_tests sample_finite_mixture_tests.cc)
target_link_libraries(sample_finite_mixture_tests ${LIBS})
add_test(NAME sample_finite_mixture_tests COMMAND sample_finite_mixture_tests)
