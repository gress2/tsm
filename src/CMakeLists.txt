include_directories(include)

add_library(logger logger.cc)
add_library(random_engine random_engine.cc)

add_executable(generic_game_mcts generic_game_mcts.cc)
target_link_libraries(generic_game_mcts ${LIBS} ${TORCH_LIBRARIES})

add_executable(generic_game_dts generic_game_dts.cc)
target_link_libraries(generic_game_dts ${LIBS} ${TORCH_LIBRARIES})

add_executable(generic_game_pts generic_game_pts.cc)
target_link_libraries(generic_game_pts ${LIBS} ${TORCH_LIBRARIES})

add_executable(generic_game_rw generic_game_rw.cc)
target_link_libraries(generic_game_rw ${LIBS} ${TORCH_LIBRARIES})

add_executable(same_game_mcts same_game_mcts.cc)
target_link_libraries(same_game_mcts ${LIBS})

add_executable(same_game_pts same_game_pts.cc)
target_link_libraries(same_game_pts ${LIBS})

add_executable(same_game_dts same_game_dts.cc)
target_link_libraries(same_game_dts ${LIBS})

add_executable(same_game_rw same_game_rw.cc)
target_link_libraries(same_game_rw ${LIBS})

add_executable(same_game_cli same_game_cli.cc)
target_link_libraries(same_game_cli ${LIBS})
