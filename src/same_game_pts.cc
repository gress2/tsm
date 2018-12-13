#include <cstdlib>
#include <ctime>

#include "partial_tree_simulator.hpp"
#include "same_game.hpp"

int main() {
  std::srand(std::time(0));

  std::string cfg_toml_path = "../cfg/same_game.toml";
  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  same_game::game game(cfg);

  pts::simulator<same_game::game> sim(game, 1e5);

  sim.simulate();
  return 0;
}
