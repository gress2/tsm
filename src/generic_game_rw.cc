#include <cstdlib>
#include <ctime>

#include "generic_game.hpp"

int main() {
  std::srand(std::time(0));
  std::string cfg_toml_path = "../cfg/generic_game.toml";
  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  using game = generic_game::game;

  std::ofstream rw_stats("generic_game_rw_stats");
  std::ofstream td_stats("generic_game_td");

  game g(cfg);
  int num_iters = 1e4;

  for (int i = 0; i < num_iters; i++) {
    if (i % 1000 == 0) {
      std::cout << "[" << i << "/" << num_iters << "]" << "\n";
    }

    game cur(g);
    auto moves = cur.get_available_moves();
    while (!moves.empty()) {
      int random_idx = std::rand() % moves.size();
      cur = cur.make_move(moves[random_idx]);
      double mean = cur.get_mean();
      double sd = cur.get_sd();
      int d = cur.get_num_moves_made();
      int k = cur.get_child_means().size();
      rw_stats << mean << ", " << sd << ", " << d << ", " << k << "\n";
      moves = cur.get_available_moves();
    }
    td_stats << cur.get_num_moves_made() << "\n";
  }

}
