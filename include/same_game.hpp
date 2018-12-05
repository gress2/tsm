#pragma once

#include <set>

#include "cpptoml.hpp"
#include "util.hpp"

namespace same_game
{

struct config {
  int width;
  int height;
};

config get_config_from_toml(std::string toml_file_path) {
  auto tbl = cpptoml::parse_file(toml_file_path);
  config cfg;
  cfg.width = get_from_toml<decltype(cfg.width)>(tbl, "width");
  cfg.height = get_from_toml<decltype(cfg.height)>(tbl, "height");
  return cfg;
}

class game {
  public:
    using move_type = std::pair<int, int>;
  private:
    using board_type = std::vector<std::vector<short>>;
    config cfg_;
    board_type board_;
    int num_moves_made_;
    std::vector<move_type> available_moves_;
    double cumulative_reward_;

    static board_type collapse(board_type board) {
      int col_sz = board[0].size();

      for (int i = 0; i < board.size(); i++) {
        int num_removed = 0; 
        for (auto it = board[i].begin(); it != board[i].end();) {
          if (*it == 0) {
            it = board[i].erase(it);
            num_removed++;
          } else {
            ++it;
          }
        }
        board[i].insert(board[i].end(), num_removed, 0);
      }

      int vectors_removed = 0;
      bool have_seen_non_empty = false;
      for (auto it = board.rbegin(); it != board.rend();) {
        bool is_all_zero = std::all_of(it->begin(), it->end(), [](short i) { return i==0; }); 
        if (is_all_zero) {
          if (have_seen_non_empty) {
            it = static_cast<decltype(it)>(board.erase(std::next(it).base()));
            vectors_removed++;
            continue;
          }        
        } else {
          have_seen_non_empty = true;
        }
        ++it;
      }

      board.insert(board.end(), vectors_removed, std::vector<short>(col_sz, 0));
      return board;
    }

    static std::pair<board_type, double> transform_board(board_type board, move_type move) {
      std::set<move_type> adj = get_color_adjacent_tiles(board, move);
      for (auto& m : adj) {
        board[m.first][m.second] = 0;
      }
      double move_reward = std::pow(adj.size() - 2, 2);
      return std::make_pair(collapse(board), move_reward);
    }

    static bool is_valid_position(const board_type& board, move_type move) noexcept {
      return move.first >= 0 && move.first < board.size()
        && move.second >= 0 && move.second < board[0].size();
    }

    static short get_value(const board_type& board, move_type move) noexcept {
      return board[move.first][move.second];
    }

    static void get_color_adjacent_tiles(const board_type& board, move_type move, 
        short target_value, std::set<move_type>& adj) {
      if (is_valid_position(board, move) && get_value(board, move) == target_value
          && adj.find(move) == adj.end()) {
        adj.insert(move);
        move_type left(move.first, move.second - 1); 
        move_type right(move.first, move.second + 1);
        move_type up(move.first + 1, move.second);
        move_type down(move.first - 1, move.second);

        get_color_adjacent_tiles(board, left, target_value, adj);
        get_color_adjacent_tiles(board, right, target_value, adj);
        get_color_adjacent_tiles(board, up, target_value, adj);
        get_color_adjacent_tiles(board, down, target_value, adj);
      } 
    }

    static std::set<move_type> get_color_adjacent_tiles(const board_type& board, move_type move) {
      std::set<move_type> adj;
      get_color_adjacent_tiles(board, move, get_value(board, move), adj);
      return adj;
    }

    static std::vector<move_type> find_available_moves(const board_type& board) {
      std::vector<std::set<move_type>> adj_sets;
      std::set<move_type> covered;
      std::vector<move_type> moves;

      for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[i].size(); j++) {
          move_type pos(i, j);
          if (get_value(board, pos) == 0) {
            break;
          }
          if (covered.find(pos) == covered.end()) {
            std::set<move_type> adj = get_color_adjacent_tiles(board, pos);
            if (adj.size() > 1) {
              adj_sets.push_back(adj);
              covered.insert(adj.begin(), adj.end());
            }
          }
        }
      }

      for (auto& s : adj_sets) {
        moves.push_back(*(s.begin()));
      }
      return moves;
    }

    static double get_final_score(const board_type& board) {
      int width = board.size();
      int height = board[0].size();

      int num_zeroes = 0;
      for (int i = 0; i < width; i++) {
        num_zeroes += std::count(board[i].begin(), board[i].end(), 0);
        if (num_zeroes == height) {
          break;
        }
      }

      int tiles_remaining = width * height - num_zeroes;
      return tiles_remaining ? -std::pow(tiles_remaining - 2, 2) : 1000;
    }

    game(const game& other, move_type move)
      : cfg_(other.cfg_),
        num_moves_made_(other.num_moves_made_ + 1)
    {
      std::pair<board_type, double> move_result = transform_board(other.board_, move);
      board_ = move_result.first;
      cumulative_reward_ = other.cumulative_reward_ + move_result.second;
      available_moves_ = find_available_moves(board_);
      if (available_moves_.empty()) {
        cumulative_reward_ += get_final_score(board_);
      }
    }

  public:
    game(config cfg)
      : cfg_(cfg),
        num_moves_made_(0),
        cumulative_reward_(0)
    {
      for (int i = 0; i < cfg_.width; i++) {
        std::vector<short> col;
        for (int j = 0; j < cfg_.height; j++) {
          short space_value = (std::rand() % 5) + 1;
          col.push_back(space_value);
        }
        board_.push_back(col);
      } 

      available_moves_ = find_available_moves(board_);
    }

    void print() const {
      for (int row = cfg_.height - 1; row >= 0; row--) {
        for (int col = 0; col < cfg_.width; col++) {
          auto value = board_[col][row];
          std::cout << "\033[9" << (value == 0 ? 8 : value) << "m" << value << "\033[0m ";
        }
        std::cout << std::endl;
      }
    }

    int get_num_moves_made() const noexcept {
      return num_moves_made_;
    }

    std::vector<move_type> get_available_moves() const noexcept {
      return available_moves_;
    }

    double get_cumulative_reward() const noexcept {
      return cumulative_reward_;
    }

    game make_move(move_type move) const {
      return game(*this, move);
    }
};

}
