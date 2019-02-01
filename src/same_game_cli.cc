#include <iostream>
#include "cxxopts.hpp"
#include "same_game.hpp"

int main(int argc, char** argv) {
  cxxopts::Options options("same_game_cli", "Allows user to play samegame from the commmand line");
  options.add_options()
    ("c,cfg", "Path to game config", cxxopts::value<std::string>()
      ->default_value("../cfg/same_game.toml"))
  ;

  auto result = options.parse(argc, argv);
  std::string cfg_toml_path = result["cfg"].as<std::string>();

  same_game::config cfg = same_game::get_config_from_toml(cfg_toml_path);

  same_game::game game(cfg);
  game.print();

  auto moves = game.get_available_moves();
  int prev_score = 0;

  while (!moves.empty()) {
    std::cout << "Available moves: ";
    for (auto& move : moves) {
      std::cout << "(" << move.first << ", " << move.second << ") "; 
    }
    std::cout << "\n";
    int x, y;
    std::cin >> x >> y;
    auto move = std::make_pair(x, y);
    game = game.make_move(move);
    int score = game.get_cumulative_reward() - prev_score;

    if (score >= 0) {
      std::cout << "Scored: " << score << " (" << (std::sqrt(score) + 2) << " tiles)\n";
    } else {
      std::cout << "Penalized: " << score << " (" << (std::sqrt(-score) + 2) << " tiles)\n";
    }

    std::cout << "Total: " << game.get_cumulative_reward() << "\n";
    game.print();
    moves = game.get_available_moves();
    prev_score = game.get_cumulative_reward();
  }

  return 0;
}
