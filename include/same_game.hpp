#pragma once

#include <set>

#include "cpptoml.hpp"
#include "util.hpp"

namespace same_game
{

/**
 * a simple struct for storing a same game configuration
 */
struct config {
  int width;
  int height;
};

/**
 * builds a same game config object given a path to a .toml file
 *
 * @param toml file path (relative to where the binary is invoked)
 * @return a complete same game config object
 */
config get_config_from_toml(std::string toml_file_path) {
  auto tbl = cpptoml::parse_file(toml_file_path);
  config cfg;
  cfg.width = get_from_toml<decltype(cfg.width)>(tbl, "width");
  cfg.height = get_from_toml<decltype(cfg.height)>(tbl, "height");
  return cfg;
}

class game;
/**
 * exposes internals of same game instances for testing purposes
 */
class game_exposer {
  private:
    game& game_;
  public:
    game_exposer(game& game)
      : game_(game)
    {}
};

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
    friend game_exposer;

    /**
     * reduces the board by moving 0's within a column to the top of
     * the column and by moving columns with only 0's to the right-most
     * end of the board.
     *
     * @param board a copy of a board
     * @return the copied board with collapsing applied
     */
    static board_type collapse(board_type board) {
      int col_sz = board[0].size();

      for (board_type::size_type i = 0; i < board.size(); i++) {
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

    /**
     * Essentially, this handles three things;
     *   1) given a selected tile, removes the tile and all adjacent tiles of the same
     *      color from the board
     *   2) collapses the board to move zero'd tiles
     *   3) determines a reward for the move
     *
     * @param board a copy of a board
     * @param move the selected tile to remove
     * @return a pair consisting of the board with the move applied as well as the a double
     * holding the score of the move
     */
    static std::pair<board_type, double> transform_board(board_type board, move_type move) {
      std::set<move_type> adj = get_color_adjacent_tiles(board, move);
      for (auto& m : adj) {
        board[m.first][m.second] = 0;
      }
      double move_reward = std::pow(adj.size() - 2, 2);
      return std::make_pair(collapse(board), move_reward);
    }

    /**
     * determines if a board position (move position) is in bounds of the board. that is,
     * it makes sure if the position is on the grid.
     *
     * @param board a const reference to a board instance
     * @param move the position to be checked on the board
     * @return true if the move is valid, false otherwise
     */
    static bool is_valid_position(const board_type& board, move_type move) noexcept {
      return move.first >= 0 && move.first < static_cast<int>(board.size())
        && move.second >= 0 && move.second < static_cast<int>(board[0].size());
    }

    /**
     * retrieves the board value at a given position
     *
     * @param board a const reference to a board instance
     * @param move a board position with which to retrieve a value
     */
    static short get_value(const board_type& board, move_type move) noexcept {
      return board[move.first][move.second];
    }


    /**
     * This is a helper function for finding the set of contiguous same-colored tiles
     * starting from a given board position. If the passed board position has the correct
     * target value, we recurse up, right, down, and left on the board and check if they
     * also have the corrrect target value. To avoid visiting the same position more than
     * once with this recursion, we pass around a reference to a visited tile set which
     * keeps track of where the algorithm has already been.
     *
     * @param board a const reference to a board instance
     * @param move a position on the board to check
     * @param target_value the color which we are interested in matching with
     * @param adj a reference to the set of color adjacent tiles we are building
     * @param visited a reference to the set of tiles where we have already been
     */
    static void get_color_adjacent_tiles(const board_type& board, move_type move,
        short target_value, std::set<move_type>& adj, std::set<move_type>& visited) {
      if (visited.find(move) == visited.end()) {
        visited.insert(move);
        if (is_valid_position(board, move) && get_value(board, move) == target_value) {
          adj.insert(move);
          move_type left(move.first, move.second - 1);
          move_type right(move.first, move.second + 1);
          move_type up(move.first + 1, move.second);
          move_type down(move.first - 1, move.second);

          get_color_adjacent_tiles(board, left, target_value, adj, visited);
          get_color_adjacent_tiles(board, right, target_value, adj, visited);
          get_color_adjacent_tiles(board, up, target_value, adj, visited);
          get_color_adjacent_tiles(board, down, target_value, adj, visited);
        }
      }
    }

    /**
     * This function dispatches a series of recursive calls to find the set of tiles from
     * (and including) the passed position, move, which form a chain of adjacent tiles sharing
     * the same color as move.
     *
     * @param board a const reference to a board instance
     * @param move the position on the board to find the color adjacent set from
     * @return a set representing the chain of color adjacent tiles starting from (and including)
     * the passed position, move.
     */
    static std::set<move_type> get_color_adjacent_tiles(const board_type& board, move_type move) {
      std::set<move_type> adj;
      std::set<move_type> visited;
      get_color_adjacent_tiles(board, move, get_value(board, move), adj, visited);
      return adj;
    }

    /**
     * finds the set of moves which can be made on the board. A move (a board position)
     * may be made if there is at least one tile bordering it of the same color. This property
     * is recursive meaning that there can be a chain of color adjacent tiles such that two
     * non-adjacent tiles can belong to the same chain if there are tiles of the same color
     * bridging the gap between them. Only the left-most, down-most tile in each chain is
     * returned as a move. This is to prevent having multiple tiles available as moves which
     * all have the same consequence when taken, thus greatly reducing the branching factor
     * of the game.
     *
     * @param a const reference to a board instance
     * @return a vector of available moves which can be made on the passed board
     */
    static std::vector<move_type> find_available_moves(const board_type& board) {
      std::vector<std::set<move_type>> adj_sets;
      std::set<move_type> covered;
      std::vector<move_type> moves;

      for (board_type::size_type i = 0; i < board.size(); i++) {
        for (board_type::size_type j = 0; j < board[i].size(); j++) {
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

    /**
     * This function should only be called when the game is essentially over, i.e,
     * there are no more possible moves to be taken. If the board has been cleared,
     * the player is rewarded +1000 points. Otherwise, they are penalized based on
     * the number of remaining tiles on the board.
     *
     * @param board a const reference to a board instance
     * @return the end of game penalty/reward for the player as a double
     */
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

    /**
     * Constructs a game instance given a const reference to another game and
     * a move to be taken on that game.
     *
     * @param other a board instance to base the new game on
     * @param move the move to be taken on a copy of the passed board instance
     */
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
    /**
     * basic constructor for a game
     *
     * @param cfg a same game config instance
     */
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

    /**
     * writes the game state to standard out
     */
    void print() const {
      for (int row = cfg_.height - 1; row >= 0; row--) {
        for (int col = 0; col < cfg_.width; col++) {
          short value = board_[col][row];
          std::cout << "\033[9" << (value == 0 ? 8 : value) << "m" << value << "\033[0m ";
        }
        std::cout << std::endl;
      }
    }

    /**
     * gets the number of moves which have occurred so far in this game.
     *
     * @return the number of moves which have occurred
     */
    int get_num_moves_made() const noexcept {
      return num_moves_made_;
    }

    /**
     * Getter for the vector of available moves which may be taken in this game
     *
     * @return the vector of available moves
     */
    std::vector<move_type> get_available_moves() const noexcept {
      return available_moves_;
    }

    int get_num_available_moves() const noexcept {
      return available_moves_.size();
    }

    bool has_available_moves() const noexcept {
      return !available_moves_.empty();
    }

    /**
     * getter for the cumulative reward of this game. returns the sum of
     * rewards for all moves taken thus far in the game.
     *
     * @return the cumululative reward of the game
     */
    double get_cumulative_reward() const noexcept {
      return cumulative_reward_;
    }

    /**
     * Returns a newly constructed game state by taking a move on
     * the current game state.
     *
     * @param move the move to be taken on the current state
     * @return a newly constructed game instance
     */
    game make_move(move_type move) const {
      return game(*this, move);
    }
};

}
