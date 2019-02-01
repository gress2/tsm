#include <cstdlib>
#include <ctime>

#include "cxxopts.hpp"
#include "mcts.hpp"
#include "same_game.hpp"

int main(int argc, char** argv) {
  std::srand(std::time(0));

  cxxopts::Options options("same_game_mcts", "Performs Monte Carlo on samegame");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/same_game.toml"))
    ("n,num_iters", "Number of iterations to perform", cxxopts::value<int>()->default_value("1000"))
  ;

  auto result = options.parse(argc, argv);

  std::string cfg_toml_path = result["cfg"].as<std::string>();
  int num_iters = result["num_iters"].as<int>();

  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  same_game::game game(cfg);
  
  mcts::node<same_game::game> node(game);
  mcts::uct uct(node, num_iters);

  uct.search();

  return 0;
}

