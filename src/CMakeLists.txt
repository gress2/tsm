include_directories(include)
add_executable(generic_game_mcts generic_game_mcts.cc)
add_library(logger logger.cc)
target_link_libraries(generic_game_mcts ${LIBS})