#include <cstdlib>
#include <ctime>

#include "cxxopts.hpp"
#include "generic_game.hpp"

int main(int argc, char** argv) {
  std::srand(std::time(0));
  cxxopts::Options options("generic_game_rw", "Performs random walks on the generic game");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/same_game.toml"))
    ("n,num_walks", "Number of random walks to perform", cxxopts::value<int>()->default_value("1000"))
    ("s,sd_model_path", "Path to pytorch saved SD model", cxxopts::value<std::string>()->default_value("../models/sd_model.pt"))
    ("v,varphi_model_path", "Path to pytorch saved varphi model", cxxopts::value<std::string>()->default_value("../models/varphi_model.pt"))
  ;

  auto result = options.parse(argc, argv);

  std::string cfg_toml_path = result["cfg"].as<std::string>();
  int num_walks = result["num_walks"].as<int>();

  using game = generic_game::game;

  std::ofstream dk_f("dk.sg.csv");
  std::ofstream td_f("td.sg.csv");

  std::string sd_model_path = result["sd_model_path"].as<std::string>();
  std::string varphi_model_path = result["varphi_model_path"].as<std::string>();

  generic_game::config cfg = generic_game::get_config_from_toml(cfg_toml_path);
  game g(cfg, sd_model_path, varphi_model_path);

  for (int i = 0; i < num_walks; i++) {
    if (i % 1000 == 0) {
      std::cout << "[" << i << "/" << num_walks << "]" << "\n";
    }

    game cur(g);
    auto moves = cur.get_available_moves();
    while (!moves.empty()) {
      int random_idx = std::rand() % moves.size();
      cur = cur.make_move(moves[random_idx]);
      int d = cur.get_num_moves_made();
      int k = cur.get_child_means().size();
      dk_f << d << ", " << k << "\n";
      moves = cur.get_available_moves();
    }
    td_f << cur.get_num_moves_made() << "\n";
  }

}
