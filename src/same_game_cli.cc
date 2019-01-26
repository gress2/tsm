#include <iostream>
#include "same_game.hpp"

int main() {
  std::string cfg_toml_path = "../cfg/same_game.toml"; 
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
