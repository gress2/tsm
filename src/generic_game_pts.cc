#include <cstdlib>
#include <ctime>

#include "cxxopts.hpp"
#include "partial_tree_simulator.hpp"
#include "generic_game.hpp"

int main(int argc, char** argv) {
  std::srand(std::time(0));
  
  cxxopts::Options options("generic_game_pts", "Performs partial tree search on the generic game");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/generic_game.toml"))
    ("n,num_iters", "Number of random walks to perform", cxxopts::value<int>()->default_value("1000"))
    ("s,sd_model_path", "Path to pytorch saved SD model", cxxopts::value<std::string>()->default_value("../models/sd_model.pt"))
    ("v,varphi_model_path", "Path to pytorch saved varphi model", cxxopts::value<std::string>()->default_value("../models/varphi_model.pt"))
  ;

  auto result = options.parse(argc, argv);

  std::string cfg_toml_path = result["cfg"].as<std::string>();
  int num_iters = result["num_iters"].as<int>();
  std::string sd_model_path = result["sd_model_path"].as<std::string>();
  std::string varphi_model_path = result["varphi_model_path"].as<std::string>();

  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  generic_game::game game(cfg, sd_model_path, varphi_model_path);

  simulator::partial_tree_simulator<generic_game::game> sim(game, num_iters);
  sim.simulate();

  return 0;
}
