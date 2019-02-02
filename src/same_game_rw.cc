#include <cstdlib>
#include <ctime>

#include "cxxopts.hpp"
#include "same_game.hpp"

int main(int argc, char** argv) {
  std::srand(std::time(0));

  cxxopts::Options options("same_game_rw", "Performs random walks on samegame");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/same_game.toml"))
    ("n,num_walks", "Number of random walks to perform", cxxopts::value<int>()->default_value("1000"))
  ;

  auto result = options.parse(argc, argv);

  std::string cfg_toml_path = result["cfg"].as<std::string>();
  int num_walks = result["num_walks"].as<int>();


  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  using game = same_game::game;
  
  std::ofstream dk_f("dk.same_game.csv");
  std::ofstream td_f("td.same_game.csv");

  game g(cfg);

  for (int i = 0; i < num_walks; i++) {
    if (i % 1000 == 0) {
      std::cout << "[" << i << "/" << num_walks << "]" << "\n";
    }

    game cur(g);
    auto moves = cur.get_available_moves();
    while (!moves.empty()) {
      int random_idx = std::rand() % moves.size();
      cur = cur.make_move(moves[random_idx]);
      moves = cur.get_available_moves();
      int d = cur.get_num_moves_made();
      int k = moves.size();
      dk_f << d << ", " << k << "\n";
    }
    td_f << cur.get_num_moves_made() << "\n"; 
  }

  return 0;
}
