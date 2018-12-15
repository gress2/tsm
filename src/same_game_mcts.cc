#include <cstdlib>
#include <ctime>

#include "mcts.hpp"
#include "same_game.hpp"

int main() {
  std::srand(std::time(0));

  std::string cfg_toml_path = "../cfg/same_game.toml";
  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  same_game::game game(cfg);
  
  mcts::node<same_game::game> node(game);
  mcts::uct uct(node, 1e6);

  uct.search();

  return 0;
}

