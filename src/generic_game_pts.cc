#include <cstdlib>
#include <ctime>

#include "partial_tree_simulator.hpp"
#include "generic_game.hpp"

int main() {
  std::srand(std::time(0));

  std::string cfg_toml_path = "../cfg/generic_game.toml";
  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  generic_game::game game(cfg);

  simulator::partial_tree_simulator<generic_game::game> sim(game, 10000);
  sim.simulate();

  return 0;
}
