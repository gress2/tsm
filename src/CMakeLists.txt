include_directories(include)

add_library(logger logger.cc)
add_library(random_engine random_engine.cc)

add_executable(generic_game_mcts generic_game_mcts.cc)
target_link_libraries(generic_game_mcts ${LIBS} ${TORCH_LIBRARIES})

add_executable(same_game_mcts same_game_mcts.cc)
target_link_libraries(same_game_mcts ${LIBS})

add_executable(same_game_pts same_game_pts.cc)
target_link_libraries(same_game_pts ${LIBS})

add_executable(same_game_dts same_game_dts.cc)
target_link_libraries(same_game_dts ${LIBS})
