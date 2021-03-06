#include <cstdlib>
#include <ctime>

#include "cxxopts.hpp"
#include "deep_tree_simulator.hpp"
#include "generic_game.hpp"

int main(int argc, char** argv) {
  std::srand(std::time(0));
  cxxopts::Options options("generic_game_dts", "Performs deep tree simulation on the generic game");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/generic_game.toml"))
    ("n,num_iters", "Number of iterations to perform", cxxopts::value<int>()->default_value("1000"))
    ("s,sd_model_path", "Path to pytorch saved SD model", cxxopts::value<std::string>()->default_value("../models/sd_model.pt"))
    ("v,varphi_model_path", "Path to pytorch saved varphi model", cxxopts::value<std::string>()->default_value("../models/varphi_model.pt"))
    ("d,delta_model_path", "Path to pytorch saved delta model", cxxopts::value<std::string>()->default_value("../models/delta_model.pt"))
  ;

  auto result = options.parse(argc, argv);

  std::string cfg_toml_path = result["cfg"].as<std::string>();
  int num_iters = result["num_walks"].as<int>();
  std::string sd_model_path = result["sd_model_path"].as<std::string>();
  std::string varphi_model_path = result["varphi_model_path"].as<std::string>();
  std::string delta_model_path = result["delta_model_path"].as<std::string>();
  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);

  generic_game::game game(cfg, sd_model_path, varphi_model_path, delta_model_path);

  simulator::deep_tree_simulator<generic_game::game> sim(game, num_iters, "main.generic_game.csv");
  sim.simulate();

  return 0;
}
