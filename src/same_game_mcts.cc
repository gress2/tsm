#include <cstdlib>
#include <ctime>

#include "mcts.hpp"
#include "same_game.hpp"

int main() {
  std::srand(std::time(0));

  std::string cfg_toml_path = "../cfg/same_game.toml";
  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  same_game::game game(cfg);
  game.print();

  auto moves = game.get_available_moves();

  while (!moves.empty()) {
    int rand_idx = std::rand() % moves.size();
    auto move = moves[rand_idx];
    std::cout << "(" << move.first << ", " << move.second << ")" << std::endl;
    game = game.make_move(move);
    game.print();
    moves = game.get_available_moves();
  }

  return 0;
}

