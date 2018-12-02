#include <iostream>

#include "generic_game.hpp"

int main() {

  std::string cfg_toml_path = "../cfg/generic_game.toml";
  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  generic_game::game game(cfg);

  return 0;
}