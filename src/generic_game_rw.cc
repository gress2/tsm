#include <cstdlib>
#include <ctime>

#include "generic_game.hpp"

int main() {
  std::srand(std::time(0));
  std::string cfg_toml_path = "../cfg/generic_game.toml";
  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  using game = generic_game::game;

  std::ofstream rw_stats("rw_stats");

  game g(cfg);

  for (int i = 0; i < 1000; i++) {
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
  }

}
